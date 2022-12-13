import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from pydrake.all import (AbstractValue, RollPitchYaw, ConstantVectorSource, DiagramBuilder,
                         PiecewisePose, JacobianWrtVariable,
                         LeafSystem, MathematicalProgram, MeshcatVisualizer,
                         Simulator, SnoptSolver, StartMeshcat, ge,
                         le, RigidTransform, RotationMatrix, MeshcatVisualizerParams)
from directives import robot_directives
from utils.make_station import MakeManipulationStationCustom
from utils.add_bodies import add_boxes, BOX_SIZE, WALL_SIZE
from manipulation.scenarios import AddIiwaDifferentialIK
from sqp import SolveSQP

import matplotlib.pyplot as plt

meshcat = StartMeshcat()


class PoseTrajectorySource(LeafSystem):
    def __init__(self, pose_trajectory):
        LeafSystem.__init__(self)
        self._pose_trajectory = pose_trajectory
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcPose)

    def CalcPose(self, context, output):
        output.set_value(self._pose_trajectory.GetPose(context.get_time()))


class TrajectorySimulator:
    def __init__(self, traj=None):
        builder = DiagramBuilder()
        self.station = builder.AddSystem(
            MakeManipulationStationCustom(model_directives=robot_directives, prefinalize_callback=add_boxes))
        self.plant = self.station.GetSubsystemByName("plant")
        controller_plant = self.station.GetSubsystemByName(
            "iiwa_controller").get_multibody_plant_for_control()
        plant_context = self.plant.CreateDefaultContext()

        table_frame = self.plant.GetFrameByName("top_center")

        X_WorldTable = table_frame.CalcPoseInWorld(plant_context)

        # size of gap between the boxes
        gap = 0.05
        box_1 = self.plant.GetBodyByName("box_1")
        X_TableBox1 = RigidTransform(
            RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[-0.2, -BOX_SIZE[1] / 2 - gap / 2, BOX_SIZE[2] / 2])
        X_WorldBox1 = X_WorldTable.multiply(X_TableBox1)
        self.plant.SetDefaultFreeBodyPose(box_1, X_WorldBox1)

        box_2 = self.plant.GetBodyByName("box_2")
        X_TableBox2 = RigidTransform(
            RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[-0.2, BOX_SIZE[1] / 2 + gap / 2, BOX_SIZE[2] / 2])
        X_WorldBox2 = X_WorldTable.multiply(X_TableBox2)
        self.plant.SetDefaultFreeBodyPose(box_2, X_WorldBox2)

        wall = self.plant.GetBodyByName("wall")
        X_TableWall = RigidTransform(
            RollPitchYaw(np.asarray([0, 0, 0]) * np.pi / 180), p=[0, 0, WALL_SIZE[2] / 2])
        X_WorldWall = X_WorldTable.multiply(X_TableWall)
        self.plant.SetDefaultFreeBodyPose(wall, X_WorldWall)

        self.gripper_frame = self.plant.GetFrameByName('body')
        X_WorldGripper = self.gripper_frame.CalcPoseInWorld(plant_context)
        print(X_WorldGripper)
        gripper_x = X_WorldGripper.translation()[0]
        gripper_z = X_WorldGripper.translation()[2]
        self.world_frame = self.plant.world_frame()

        # constrain the robot to move in the y direction
        box1_y = X_WorldBox1.translation()[1]
        box2_y = X_WorldBox2.translation()[1]
        y_min = box1_y - BOX_SIZE[1] / 2
        y_max = box2_y + BOX_SIZE[1] / 2
        lower_bound = [-100, y_min, -100]
        upper_bound = [100, y_max, 100]

        if traj is not None:
            transforms = []
            for i in range(len(traj)):
                transforms.append(RigidTransform(X_WorldGripper.rotation(), [gripper_x, traj[i], gripper_z]))
            print(transforms)
            times = np.linspace(0, 20, len(transforms))
            traj = PiecewisePose.MakeLinear(times, transforms)
            traj_source = builder.AddSystem(PoseTrajectorySource(traj))
            self.controller = AddIiwaDifferentialIK(
                builder,
                controller_plant,
                frame=controller_plant.GetFrameByName("body"))
            builder.Connect(traj_source.get_output_port(),
                            self.controller.get_input_port(0))
            builder.Connect(self.station.GetOutputPort("iiwa_state_estimated"),
                            self.controller.GetInputPort("robot_state"))

            builder.Connect(self.controller.get_output_port(),
                            self.station.GetInputPort("iiwa_position"))

        params = MeshcatVisualizerParams()
        params.delete_on_initialization_event = False
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            builder, self.station.GetOutputPort("query_object"), meshcat, params)

        wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
        builder.Connect(wsg_position.get_output_port(),
                        self.station.GetInputPort("wsg_position"))

        self.diagram = builder.Build()

        context = self.CreateDefaultContext()
        self.diagram.Publish(context)

    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, context)
        station_context = self.diagram.GetMutableSubsystemContext(
            self.station, context)

        # set the joint positions of the kuka arm
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
        wsg = self.plant.GetModelInstanceByName("wsg")
        self.plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg, [0, 0])

        return context

    def paint(self):
        context = self.CreateDefaultContext()
        simulator = Simulator(self.diagram, context)
        simulator.set_target_realtime_rate(1.0)

        return simulator


if __name__ == "__main__":
    sqp = SolveSQP(0., -0.035)
    y_locs = sqp.re_plan()
    plt.title("End effector trajectory (horizon T=10)")
    plt.xlabel("time")
    plt.ylabel("end effector position along y axis")
    plt.plot(np.linspace(0, len(sqp.x_graph), len(sqp.x_graph), endpoint=False), sqp.x_graph)
    plt.show()
    trajgen = TrajectorySimulator(y_locs)
    simulator = trajgen.paint()
    i = 0.01
    ready = input("ready to record")
    if ready == "y":
        while i < 100.:
            simulator.AdvanceTo(i)
            i += 0.01
