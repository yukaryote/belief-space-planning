import sys
from enum import Enum
from copy import copy
import numpy as np

from manipulation.scenarios import AddIiwaDifferentialIK
from pydrake.all import (AbstractValue, RollPitchYaw, PiecewisePose, DiagramBuilder,
                         PiecewisePolynomial, InputPortIndex, LeafSystem, MathematicalProgram,
                         MeshcatVisualizer, Simulator, Solve, StartMeshcat, eq, ge,
                         le, RigidTransform, PortSwitch)
from pydrake.symbolic import Variable
from manipulation.pick import MakeGripperFrames, MakeGripperPoseTrajectory, MakeGripperCommandTrajectory
from directives import robot_directives
from utils.make_station import MakeManipulationStationCustom
from utils.add_bodies import BOX_SIZE, WALL_SIZE
from utils.histogram_filter import HistogramFilter

from scipy.special import kl_div
import scipy.stats as stats

np.set_printoptions(threshold=sys.maxsize)

meshcat = StartMeshcat()

rs = np.random.RandomState()  # this is for python


class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    GO_HOME = 2


class Planner(LeafSystem):
    def __init__(self, plant, field, x_g, T=100, alpha=0.0085, N=10, noise_std=0.1, lower_bound=None, upper_bound=None):
        LeafSystem.__init__(self)
        self.time_step = plant.time_step
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))
        self._laser_observation = self.DeclareAbstractInputPort(
            "laser_observation", AbstractValue.Make(
                (np.inf, RigidTransform()))).get_index()
        self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE))
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self._times_index = self.DeclareAbstractState(AbstractValue.Make({"initial": 0.0}))
        self._attempts_index = self.DeclareDiscreteState(1)

        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # For GoHome mode.
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions).get_index()
        self.DeclareAbstractOutputPort(
            "control_mode", lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode)
        self.DeclareAbstractOutputPort(
            "reset_diff_ik", lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset)
        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self.DeclareVectorOutputPort("iiwa_position_command", num_positions,
                                     self.CalcIiwaPosition)
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        # For SQP solver
        self.T = T
        self.alpha = alpha
        self.h = field
        self.H = np.zeros_like(self.h)
        prev = self.h[0]
        # h is a square wave, so its derivative is two impulses of opposite sign
        for i in range(len(self.h)):
            if prev < self.h[i]:
                self.H[i] = 1
            elif prev > self.h[i]:
                self.H[i] = -1
            prev = self.h[i]
        self.Q = np.random.normal(loc=0, scale=noise_std)
        self.histogram_filter = HistogramFilter(N, field, noise_std)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def dirtran(self, x_samples, x_g=None):
        # Discrete-time approximation of the double integrator.
        K = len(x_samples)

        prog = MathematicalProgram()

        # Create decision variables

        # u is T-dim velocity in the y-axis
        u = np.empty((self.T - 1), dtype=Variable)
        # x is kxT-dim positions in the y-axis from our k samples
        x = np.empty((K, self.T), dtype=Variable)
        for t in range(self.T - 1):
            u[t] = prog.NewContinuousVariables(1, 'u' + str(t))
            for k in range(K):
                x[k, t] = prog.NewContinuousVariables(1, 'x' + str(k) + str(t))

        for k in range(K):
            x[k, self.T - 1] = prog.NewContinuousVariables(1, 'x' + str(k) + str(self.T))

        # Add costs and constraints
        x0 = x_samples
        J = np.mean([self.calc_weights(x_samples[0], k, x_samples, u, self.T) ** 2 for k in range(K)])
        cost2 = self.alpha * sum([np.linalg.norm(u[t]) ** 2 for t in range(self.T)])
        prog.AddCost(J + cost2)
        prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])
        for t in range(self.T - 1):
            for k in range(K):
                w_next = self.calc_weights(x_samples[0], k, x_samples, u, t + 1)
                w_cur = self.calc_weights(x_samples[0], k, x_samples, u, t)
                prog.AddConstraint(eq(x[k, t + 1], self.f(x[k, t], u[t])))
                prog.AddConstraint(eq(w_next, w_cur * np.e ** self.phi(x[k, t], x_samples[0])))
                prog.AddBoundingBoxConstraint(x[k, t], self.lower_bound, self.upper_bound)
        if x_g is not None:
            prog.AddConstraint(eq(x[0, self.T - 1], x_g))

        result = Solve(prog)

        u_sol = result.GetSolution(u)
        assert (result.is_success()), "Optimization failed"
        return u_sol

    def calc_weights(self, x_1, i, x_samples, u, T):
        weight = 1
        for t in range(T):
            weight *= np.e ** (self.phi(self.F(x_samples[i], u[:t]), self.F(x_1, u[:t])))
        return weight

    def phi(self, x, y):
        """
        Weighting function
        """
        return 1 / 2 * (self.h[x] - self.h[y]) * 1 / (2 * self.Q +
                       self.H[x] * self.H[x] + self.H[y] * self.H[y]) * (
                       self.h[x] - self.h[y])

    def f(self, x, u):
        """
        Returns next state if we are in state x and take action u
        0.001 is the timestep of the system
        """
        return x + u * self.time_step

    def F(self, x, u):
        """
        Returns next state if we are in state x and take actions u (a T-dim vector of actions)
        """
        state = x
        for i in range(len(u)):
            state += self.f(state, u[i])
        return state

    def theta_cap(self, belief_state, r, x_g):
        """
        Probability that we are in a ball of radius r around x_g.
        Amounts to calculating the CDF difference.
        """
        cdf = np.cumsum(belief_state)
        # calculate which bin x_g +/- r is in.
        lower_bound = int((x_g - r - self.histogram_filter.field[0]) / self.histogram_filter.bin_size)
        upper_bound = int((x_g + r - self.histogram_filter.field[0]) / self.histogram_filter.bin_size)
        return cdf[lower_bound] - cdf[upper_bound]

    def J(self, x_samples, u, t):
        return np.mean([self.calc_weights(x_samples[0], k, x_samples, u, t) ** 2 for k in range(len(x_samples))])

    def create_plan(self, x_samples, x_g, omega=0.5, r=0.5):
        belief_state = self.histogram_filter.p[:]
        u = self.dirtran(x_samples, x_g=x_g)
        belief_states = np.ndarray(shape=(belief_state.shape[0], self.T))
        belief_states[0] = belief_state
        for t in range(self.T - 1):
            belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        if self.theta_cap(belief_state, r, x_g) <= omega:
            u = self.dirtran(x_samples, self.T)
            belief_states[0] = belief_state
            for t in range(self.T - 1):
                belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        return belief_states, u

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(
            self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return
        elif mode == PlannerState.GO_HOME:
            traj_q = context.get_mutable_abstract_state(int(
                self._traj_q_index)).get_value()
            if not traj_q.is_time_in_range(current_time):
                self.Plan(context, state)
            return

        # If we are between pick and place and the gripper is closed, then
        # we've missed or dropped the object. Time to replan.
        if times["postpick"] < current_time < times["preplace"]:
            wsg_state = self.get_input_port(self._wsg_state_index).Eval(context)
            if wsg_state[0] < 0.01:
                attempts = state.get_mutable_discrete_state(int(self._attempts_index)).get_mutable_value()
                if attempts[0] > 5:
                    # If I've failed 5 times in a row, then switch bins.
                    print("Switching to the other bin after 5 consecutive failed attempts")
                    attempts[0] = 0
                    if mode == PlannerState.PICKING_FROM_X_BIN:
                        state.get_mutable_abstract_state(int(
                            self._mode_index)).set_value(
                            PlannerState.PICKING_FROM_Y_BIN)
                    else:
                        state.get_mutable_abstract_state(int(
                            self._mode_index)).set_value(
                            PlannerState.PICKING_FROM_X_BIN)
                    self.Plan(context, state)
                    return

                attempts[0] += 1
                state.get_mutable_abstract_state(int(
                    self._mode_index)).set_value(
                    PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                times = {"initial": current_time}
                state.get_mutable_abstract_state(int(
                    self._times_index)).set_value(times)
                X_G = self.get_input_port(0).Eval(context)[int(
                    self._gripper_body_index)]
                state.get_mutable_abstract_state(int(
                    self._traj_X_G_index)).set_value(PiecewisePose.MakeLinear([current_time, np.inf], [X_G, X_G]))
                return

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index)).get_value()
        if not traj_X_G.is_time_in_range(current_time):
            self.Plan(context, state)
            return

        X_G = self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]

        if np.linalg.norm(traj_X_G.GetPose(current_time).translation() - X_G.translation()) > 0.2:
            # If my trajectory tracking has gone this wrong, then I'd better stop and re-plan.
            self.GoHome(context, state)
            return

    def GoHome(self, context, state):
        print("Replanning due to large tracking error.")
        state.get_mutable_abstract_state(int(
            self._mode_index)).set_value(
            PlannerState.GO_HOME)
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = copy(context.get_discrete_state(self._q0_index).get_value())
        q0[0] = q[0]  # Safer to not reset the first joint.

        current_time = context.get_time()
        q_traj = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time + 5.0], np.vstack((q, q0)).T)
        state.get_mutable_abstract_state(int(
            self._traj_q_index)).set_value(q_traj)

    def Plan(self, context, state):
        mode = copy(
            state.get_mutable_abstract_state(int(self._mode_index)).get_value())

        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index)]
        }

        cost = np.inf
        for i in range(5):
            if mode == PlannerState.PICKING_FROM_Y_BIN:
                cost, X_G["pick"] = self.get_input_port(
                    self._y_bin_grasp_index).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_X_BIN
            else:
                cost, X_G["pick"] = self.get_input_port(
                    self._x_bin_grasp_index).Eval(context)
                if np.isinf(cost):
                    mode = PlannerState.PICKING_FROM_Y_BIN
                else:
                    mode = PlannerState.PICKING_FROM_X_BIN

            if not np.isinf(cost):
                break

        assert not np.isinf(cost), "Could not find a valid grasp in either bin after 5 attempts"
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(mode)

        # TODO(russt): The randomness should come in through a random input
        # port.
        if mode == PlannerState.PICKING_FROM_X_BIN:
            # Place in Y bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, 0),
                [rs.uniform(-.25, .15),
                 rs.uniform(-.6, -.4), .3])
        else:
            # Place in X bin:
            X_G["place"] = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2),
                [rs.uniform(.35, .65),
                 rs.uniform(-.12, .28), .3])

        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(
            self._times_index)).set_value(times)

        if False:  # Useful for debugging
            AddMeshcatTriad(meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index)).set_value(traj_wsg_command)

    def start_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()

    def CalcGripperPose(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(int(
            self._traj_X_G_index)).get_value()
        if (traj_X_G.get_number_of_segments() > 0 and
                traj_X_G.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(
                    self._traj_X_G_index)).get_value().GetPose(
                    context.get_time()))
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(self.get_input_port(0).Eval(context)
                         [int(self._gripper_body_index)])

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

        if mode == PlannerState.GO_HOME:
            # Command the open position
            output.SetFromVector([opened])
            return

        traj_wsg = context.get_abstract_state(int(
            self._traj_wsg_index)).get_value()
        if (traj_wsg.get_number_of_segments() > 0 and
                traj_wsg.is_time_in_range(context.get_time())):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))  # Go Home
        else:
            output.set_value(InputPortIndex(1))  # Diff IK

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context))

    def CalcIiwaPosition(self, context, output):
        traj_q = context.get_mutable_abstract_state(int(
            self._traj_q_index)).get_value()

        output.SetFromVector(traj_q.value(context.get_time()))


def clutter_clearing_demo():
    meshcat.Delete()
    builder = DiagramBuilder()

    station = builder.AddSystem(
        MakeManipulationStationCustom(robot_directives, time_step=0.001))
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.CreateDefaultContext()

    table_frame = plant.GetFrameByName("top_center")

    X_WorldTable = table_frame.CalcPoseInWorld(plant_context)

    # size of gap between the boxes
    gap = 0.05
    box_1 = plant.GetBodyByName("box_1")
    X_TableBox1 = RigidTransform(
        RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[-0.2, -BOX_SIZE[1] / 2 - gap / 2, BOX_SIZE[2] / 2])
    X_WorldBox1 = X_WorldTable.multiply(X_TableBox1)
    plant.SetDefaultFreeBodyPose(box_1, X_WorldBox1)

    box_2 = plant.GetBodyByName("box_2")
    X_TableBox2 = RigidTransform(
        RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[-0.2, BOX_SIZE[1] / 2 + gap / 2, BOX_SIZE[2] / 2])
    X_WorldBox2 = X_WorldTable.multiply(X_TableBox2)
    plant.SetDefaultFreeBodyPose(box_2, X_WorldBox2)

    wall = plant.GetBodyByName("wall")
    X_TableWall = RigidTransform(
        RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[0, 0, WALL_SIZE[2] / 2])
    X_WorldWall = X_WorldTable.multiply(X_TableWall)
    plant.SetDefaultFreeBodyPose(wall, X_WorldWall)

    # constrain the robot to move in the y direction
    box1_y = X_WorldBox1.translation()[1]
    box2_y = X_WorldBox2.translation()[1]
    y_min = box1_y - BOX_SIZE[1] / 2
    y_max = box2_y + BOX_SIZE[1] / 2
    lower_bound = y_min
    upper_bound = y_max

    # make the observation field
    N = 100
    gripper_frame = plant.GetFrameByName("body")
    wall_frame = plant.GetFrameByName("wall")
    box_frame = plant.GetFrameByName("box_1")
    X_WallGripper = gripper_frame.CalcPose(plant_context, wall_frame)
    X_BoxGripper = gripper_frame.CalcPose(plant_context, box_frame)
    dist_to_wall = X_WallGripper.translation()[0]
    dist_to_box =  X_BoxGripper.translation()[0]
    boxes_span = y_max - y_min
    step = boxes_span / N
    field = np.zeros((N,))
    for i in range(N):
        if BOX_SIZE[1] < i * step < BOX_SIZE[1] + gap:
            field[i] = dist_to_wall
        else:
            field[i] = dist_to_box

    planner = builder.AddSystem(Planner(plant, field, 0., N=N, lower_bound=lower_bound, upper_bound=upper_bound))
    builder.Connect(station.GetOutputPort("body_poses"),
                    planner.GetInputPort("body_poses"))
    builder.Connect(station.GetOutputPort("camera_depth_image"),
                    planner.GetInputPort("laser_observation"))
    builder.Connect(station.GetOutputPort("wsg_state_measured"),
                    planner.GetInputPort("wsg_state"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    planner.GetInputPort("iiwa_position"))

    robot = station.GetSubsystemByName(
        "iiwa_controller").get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                    diff_ik.GetInputPort("robot_state"))
    builder.Connect(planner.GetOutputPort("reset_diff_ik"),
                    diff_ik.GetInputPort("use_robot_state"))
    builder.Connect(planner.GetOutputPort("wsg_position"),
                    station.GetInputPort("wsg_position"))

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(),
                    switch.DeclareInputPort("diff_ik"))
    builder.Connect(planner.GetOutputPort("iiwa_position_command"),
                    switch.DeclareInputPort("position"))
    builder.Connect(switch.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(planner.GetOutputPort("control_mode"),
                    switch.get_port_selector_input_port())

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()

    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

    simulator.set_target_realtime_rate(1.0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")


clutter_clearing_demo()
