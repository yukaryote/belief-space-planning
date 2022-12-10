import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

import pydot
from IPython.display import SVG, display
from manipulation import running_as_notebook, FindResource
from pydrake.all import (BasicVector, RollPitchYaw, ConstantVectorSource, DiagramBuilder,
                         GenerateHtml, Integrator, JacobianWrtVariable,
                         LeafSystem, MathematicalProgram, MeshcatVisualizer,
                         Simulator, SnoptSolver, Solve, StartMeshcat, eq, ge,
                         le, RigidTransform, RenderCameraCore, CameraInfo, ClippingRange, DepthRange)
from directives import robot_directives
from utils.make_station import MakeManipulationStationCustom
from utils.add_bodies import add_boxes, BOX_SIZE, WALL_SIZE
from utils.sqp import SolveSQP

meshcat = StartMeshcat()


class DifferentialIKSystem(LeafSystem):
    """Wrapper system for Differential IK.
        @param plant MultibodyPlant of the simulated plant.
        @param diffik_fun function object that handles diffik. Must have the signature
               diffik_fun(J_G, V_G_desired, q_now, v_now, X_now)
    """

    def __init__(self, plant, diffik_fun, lower_bound=None, upper_bound=None):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._diffik_fun = diffik_fun
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.DeclareVectorInputPort("spatial_velocity", BasicVector(6))
        self.DeclareVectorInputPort("iiwa_position_measured", BasicVector(7))
        self.DeclareVectorInputPort("iiwa_velocity_measured", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity_command", BasicVector(7),
                                     self.CalcOutput)

    def CalcOutput(self, context, output):
        V_G_desired = self.get_input_port(0).Eval(context)
        q_now = self.get_input_port(1).Eval(context)
        v_now = self.get_input_port(2).Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context, JacobianWrtVariable.kQDot,
            self._G, [0, 0, 0], self._W, self._W)
        J_G = J_G[:, 0:7]  # Ignore gripper terms

        X_now = self._plant.CalcRelativeTransform(self._plant_context,
                                                  self._W, self._G)
        p_now = X_now.translation()
        if self._diffik_fun == "zeros":
            v = self.DiffIK_Zero()
        elif self._diffik_fun == "pseudoinv":
            v = self.DiffIKPseudoInverse(J_G, V_G_desired, q_now, v_now, p_now)
        else:
            v = self.DiffIKQP_Wall(J_G, V_G_desired, q_now, v_now, p_now)
        output.SetFromVector(v)

    def DiffIK_Zero(self):
        return np.zeros(7)

    def DiffIKPseudoInverse(self, J_G, V_G_desired, q_now, v_now, p_now):
        v = np.linalg.pinv(J_G).dot(V_G_desired)
        return v

    def DiffIKQP_Wall(self, J_G, V_G_desired, q_now, v_now, p_now):
        prog = MathematicalProgram()
        v = prog.NewContinuousVariables(7, 'joint_velocities')
        v_max = 4.0
        h = 4e-3

        sub = J_G.dot(v) - V_G_desired
        expr = 0
        for i in sub:
            expr += pow(i, 2)
        prog.AddCost(expr)
        prog.AddConstraint(le(v, v_max * np.ones(7)))
        prog.AddConstraint(ge(v, -v_max * np.ones(7)))
        prog.AddConstraint(ge(J_G.dot(v)[3:], (self.lower_bound - p_now) / h))
        prog.AddConstraint(le(J_G.dot(v)[3:], (self.upper_bound - p_now) / h))

        solver = SnoptSolver()
        result = solver.Solve(prog)

        if not (result.is_success()):
            raise ValueError("Could not find the optimal solution.")

        v_solution = result.GetSolution(v)
        return v_solution


def BuildAndSimulate(diffik_fun, V_d):
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeManipulationStationCustom(model_directives=robot_directives, prefinalize_callback=add_boxes))
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
    lower_bound = [-100, y_min, -100]
    upper_bound = [100, y_max, 100]

    controller = builder.AddSystem(DifferentialIKSystem(plant,
                                                        diffik_fun,
                                                        lower_bound=lower_bound,
                                                        upper_bound=upper_bound))
    integrator = builder.AddSystem(Integrator(7))
    desired_vel = builder.AddSystem(ConstantVectorSource(V_d))

    builder.Connect(controller.get_output_port(),
                    integrator.get_input_port())
    builder.Connect(integrator.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),
                    controller.get_input_port(1))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    controller.get_input_port(2))
    builder.Connect(desired_vel.get_output_port(),
                    controller.get_input_port(0))

    visualizer = MeshcatVisualizer.AddToBuilder(builder,
                                                station.GetOutputPort("query_object"), meshcat)

    diagram = builder.Build()
    diagram.set_name("diagram")

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(
        context)
    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros((7, 1)))
    station.GetInputPort("wsg_position").FixValue(station_context, [0.1])

    integrator.set_integral_value(
        integrator.GetMyMutableContextFromRoot(context),
        plant.GetPositions(plant.GetMyContextFromRoot(context),
                           plant.GetModelInstanceByName("iiwa")))
    laser_img = station.GetOutputPort("camera_depth_image").Eval(station_context).data.squeeze()
    middle = laser_img.shape[1] // 2
    print(laser_img[0][middle], laser_img.shape)

    # SQPsolver = SolveSQP(500, 0.0085, 10, )

    simulator.set_target_realtime_rate(1.0)
    return simulator


# Corresponds to [wx, wy, wz, vx, vy, vz]
V_d = np.array([0., 0., 0., 0.0, 0.1, 0])
simulator = BuildAndSimulate("wall", V_d)

i = 0.01
while i < 5.0:
    simulator.AdvanceTo(i)
    i += 0.01
