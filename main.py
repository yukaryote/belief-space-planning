import os
import numpy as np
import pydot
from IPython.display import SVG, display
from manipulation import running_as_notebook, FindResource
from pydrake.all import (BasicVector, ConstantVectorSource, DiagramBuilder,
                         GenerateHtml, Integrator, JacobianWrtVariable,
                         LeafSystem, MathematicalProgram, MeshcatVisualizer,
                         Simulator, SnoptSolver, Solve, StartMeshcat, eq, ge,
                         le, RigidTransform, RenderCameraCore, CameraInfo, ClippingRange, DepthRange)
from directives import robot_directives
from utils.utils import MakeManipulationStationCustom

meshcat = StartMeshcat()


class DifferentialIKSystem(LeafSystem):
    """Wrapper system for Differential IK.
        @param plant MultibodyPlant of the simulated plant.
        @param diffik_fun function object that handles diffik. Must have the signature
               diffik_fun(J_G, V_G_desired, q_now, v_now, X_now)
    """

    def __init__(self, plant, diffik_fun):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._diffik_fun = diffik_fun

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

        v = self._diffik_fun(J_G, V_G_desired, q_now, v_now, p_now)
        output.SetFromVector(v)


def BuildAndSimulate(diffik_fun, V_d, plot_system_diagram=False):
    builder = DiagramBuilder()
    time_step = 4e-3
    station = builder.AddSystem(
        MakeManipulationStationCustom(robot_directives, time_step=time_step))
    plant = station.GetSubsystemByName("plant")

    controller = builder.AddSystem(DifferentialIKSystem(plant, diffik_fun))
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
    if running_as_notebook and plot_system_diagram:
        display(SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg()))

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

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)

    time = 0
    while True:
        time += 1

    return simulator


def DiffIK_Zero(J_G, V_G_desired, q_now, v_now, p_now):
    return np.zeros(7)


V_d = np.zeros(6)
simulator = BuildAndSimulate(DiffIK_Zero, V_d, plot_system_diagram=True)

simulator.AdvanceTo(5.0 if running_as_notebook else 0.1)
