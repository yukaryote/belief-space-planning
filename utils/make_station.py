import os
import sys

import numpy as np
from pydrake.all import (
    AbstractValue, Adder, AddMultibodyPlantSceneGraph, BallRpyJoint, BaseField,
    Box, CameraInfo, ClippingRange, CoulombFriction, Cylinder, Demultiplexer,
    DepthImageToPointCloud, DepthRange, DepthRenderCamera, DiagramBuilder,
    FindResourceOrThrow, GeometryInstance, InverseDynamicsController,
    LeafSystem, LoadModelDirectivesFromString,
    MakeMultibodyStateToWsgStateSystem, MakePhongIllustrationProperties,
    MakeRenderEngineVtk, ModelInstanceIndex, MultibodyPlant, Parser,
    PassThrough, PrismaticJoint, ProcessModelDirectives, RenderCameraCore,
    RenderEngineVtkParams, RevoluteJoint, Rgba, RgbdSensor, RigidTransform,
    RollPitchYaw, RotationMatrix, SchunkWsgPositionController, SpatialInertia,
    Sphere, StateInterpolatorWithDiscreteDerivative, UnitInertia)

from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddWsg

from manipulation.utils import AddPackagePaths, FindResource


def AddLaserSensor(builder,
                   plant,
                   scene_graph,
                   also_add_point_clouds=False,
                   model_instance_prefix="camera",
                   depth_camera=None,
                   renderer=None):
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=2, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                           X_PB=RigidTransform(),
                           depth_camera=depth_camera,
                           show_window=False))
            rgbd.set_name(model_name)
            builder.Connect(scene_graph.get_query_output_port(),
                            rgbd.query_object_input_port())

            # Export the camera outputs
            builder.ExportOutput(rgbd.color_image_output_port(),
                                 f"{model_name}_rgb_image")
            builder.ExportOutput(rgbd.depth_image_32F_output_port(),
                                 f"{model_name}_depth_image")
            builder.ExportOutput(rgbd.label_image_output_port(),
                                 f"{model_name}_label_image")

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(camera_info=rgbd.depth_camera_info(),
                                           fields=BaseField.kXYZs
                                                  | BaseField.kRGBs))
                builder.Connect(rgbd.depth_image_32F_output_port(),
                                to_point_cloud.depth_image_input_port())
                builder.Connect(rgbd.color_image_output_port(),
                                to_point_cloud.color_image_input_port())

                class ExtractBodyPose(LeafSystem):

                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort(
                            "poses",
                            plant.get_body_poses_output_port().Allocate())
                        self.DeclareAbstractOutputPort(
                            "pose",
                            lambda: AbstractValue.Make(RigidTransform()),
                            self.CalcOutput)

                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(pose.rotation(),
                                                       pose.translation())

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(plant.get_body_poses_output_port(),
                                camera_pose.get_input_port())
                builder.Connect(camera_pose.get_output_port(),
                                to_point_cloud.GetInputPort("camera_pose"))

                # Export the point cloud output.
                builder.ExportOutput(to_point_cloud.point_cloud_output_port(),
                                     f"{model_name}_point_cloud")


def MakeManipulationStationCustom(model_directives=None,
                                  filename=None,
                                  time_step=0.002,
                                  iiwa_prefix="iiwa",
                                  wsg_prefix="wsg",
                                  camera_prefix="camera",
                                  prefinalize_callback=None,
                                  package_xmls=[]):
    """
    Creates a manipulation station system, which is a sub-diagram containing:
      - A MultibodyPlant with populated via the Parser from the
        `model_directives` argument AND the `filename` argument.
      - A SceneGraph
      - For each model instance starting with `iiwa_prefix`, we add an
        additional iiwa controller system
      - For each model instance starting with `wsg_prefix`, we add an
        additional schunk controller system
      - For each body starting with `camera_prefix`, we add a RgbdSensor
    Args:
        builder: a DiagramBuilder
        model_directives: a string containing any model directives to be parsed
        filename: a string containing the name of an sdf, urdf, mujoco xml, or
        model directives yaml file.
        time_step: the standard MultibodyPlant time step.
        iiwa_prefix: Any model instances starting with `iiwa_prefix` will get
        an inverse dynamics controller, etc attached
        wsg_prefix: Any model instance starting with `wsg_prefix` will get a
        schunk controller
        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.
        prefinalize_callback: A function, setup(plant), that will be called
        with the multibody plant before calling finalize.  This can be useful
        for e.g. adding additional bodies/models to the simulation.
        package_xmls: A list of filenames to be passed to
        PackageMap.AddPackageXml().  This is useful if you need to add more
        models to your path (e.g. from your current working directory).
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    AddPackagePaths(parser)
    if model_directives:
        directives = LoadModelDirectivesFromString(model_directives)
        ProcessModelDirectives(directives, parser)
    if filename:
        parser.AddAllModelsFromFile(filename)
    if prefinalize_callback:
        prefinalize_callback(plant)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(iiwa_position.get_input_port(),
                                model_instance_name + "_position")
            builder.ExportOutput(iiwa_position.get_output_port(),
                                 model_instance_name + "_position_commanded")

            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
            builder.Connect(plant.get_state_output_port(model_instance),
                            demux.get_input_port())
            builder.ExportOutput(demux.get_output_port(0),
                                 model_instance_name + "_position_measured")
            builder.ExportOutput(demux.get_output_port(1),
                                 model_instance_name + "_velocity_estimated")
            builder.ExportOutput(plant.get_state_output_port(model_instance),
                                 model_instance_name + "_state_estimated")

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            if plant.num_positions(model_instance) == 3:
                controller_iiwa = AddPlanarIiwa(controller_plant)
            else:
                controller_iiwa = AddIiwa(controller_plant)
            AddWsg(controller_plant, controller_iiwa, welded=True)
            controller_plant.Finalize()

            # Add the iiwa controller
            iiwa_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=False))
            iiwa_controller.set_name(model_instance_name + "_controller")
            builder.Connect(plant.get_state_output_port(model_instance),
                            iiwa_controller.get_input_port_estimated_state())

            # Add in the feed-forward torque
            adder = builder.AddSystem(Adder(2, num_iiwa_positions))
            builder.Connect(iiwa_controller.get_output_port_control(),
                            adder.get_input_port(0))
            # Use a PassThrough to make the port optional (it will provide zero
            # values if not connected).
            torque_passthrough = builder.AddSystem(
                PassThrough([0] * num_iiwa_positions))
            builder.Connect(torque_passthrough.get_output_port(),
                            adder.get_input_port(1))
            builder.ExportInput(torque_passthrough.get_input_port(),
                                model_instance_name + "_feedforward_torque")
            builder.Connect(adder.get_output_port(),
                            plant.get_actuation_input_port(model_instance))

            # Add discrete derivative to command velocities.
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    time_step,
                    suppress_initial_transient=True))
            desired_state_from_position.set_name(
                model_instance_name + "_desired_state_from_position")
            builder.Connect(desired_state_from_position.get_output_port(),
                            iiwa_controller.get_input_port_desired_state())
            builder.Connect(iiwa_position.get_output_port(),
                            desired_state_from_position.get_input_port())

            # Export commanded torques.
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_commanded")
            builder.ExportOutput(adder.get_output_port(),
                                 model_instance_name + "_torque_measured")

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(
                    model_instance), model_instance_name + "_torque_external")

        elif model_instance_name.startswith(wsg_prefix):

            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(wsg_controller.get_generalized_force_output_port(),
                            plant.get_actuation_input_port(model_instance))
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_controller.get_state_input_port())
            builder.ExportInput(
                wsg_controller.get_desired_position_input_port(),
                model_instance_name + "_position")
            builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                                model_instance_name + "_force_limit")
            wsg_mbp_state_to_wsg_state = builder.AddSystem(
                MakeMultibodyStateToWsgStateSystem())
            builder.Connect(plant.get_state_output_port(model_instance),
                            wsg_mbp_state_to_wsg_state.get_input_port())
            builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                                 model_instance_name + "_state_measured")
            builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                                 model_instance_name + "_force_measured")

    # Cameras.
    AddLaserSensor(builder,
                   plant,
                   scene_graph,
                   model_instance_prefix=camera_prefix)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram
