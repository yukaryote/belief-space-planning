import numpy as np

from pydrake.all import Box, RigidTransform, SpatialInertia, UnitInertia, CoulombFriction


BOX_SIZE = [0.09, 0.09, 0.15]
WALL_SIZE = [0.05, 0.75, 0.5]


def AddBoxDifferentGeometry(plant, visual_shape, collision_shape, name, mass=1., mu=1., color=None):
    if color is None:
        color = [.5, .5, .9, 1.0]
    instance = plant.AddModelInstance(name)
    inertia = UnitInertia.SolidBox(visual_shape.width(), visual_shape.depth(),
                                   visual_shape.height())

    body = plant.AddRigidBody(
        name, instance,
        SpatialInertia(mass=mass,
                       p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=inertia))
    if plant.geometry_source_is_registered():
        """ register collision geometry"""
        plant.RegisterCollisionGeometry(body, RigidTransform(), collision_shape, name,
                                        CoulombFriction(mu, mu))
        """ register visual geometry"""
        plant.RegisterVisualGeometry(body, RigidTransform(), visual_shape, name, color)
    return


def add_boxes(plant):
    mass = 1.
    mu = 1.
    AddBoxDifferentGeometry(plant, Box(*BOX_SIZE), Box(*[x for x in BOX_SIZE]), "box_1",
                            mass, mu, color=[0.8, 0, 0, 1.0])
    AddBoxDifferentGeometry(plant, Box(*BOX_SIZE), Box(*[x for x in BOX_SIZE]), "box_2",
                            mass, mu, color=[0, 0.8, 0, 1.0])
    AddBoxDifferentGeometry(plant, Box(*WALL_SIZE), Box(*[x for x in WALL_SIZE]), "wall",
                            mass, mu, color=[0, 0, 0.8, 1.0])
    return
