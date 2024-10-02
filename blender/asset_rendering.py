# References: https://github.com/allenai/objaverse-rendering/blob/970731404ae2dd091bb36150e04c4bd6ff59f0a0/scripts/blender_script.py#L151
import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import bpy
from mathutils import Vector
import argparse
import glob


"""
Blender python script for rendering multi-view images of a 3D object.
"""


context = bpy.context
scene = context.scene
render = scene.render


# Stackoverflow: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
        # if obj.type not in {"CAMERA", "LIGHT"}:
        #     bpy.data.objects.remove(obj, do_unlink=True)
        # bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def setup_blender_env():

    reset_scene()

    # Set render engine and parameters
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 64
    scene.cycles.samples = 64  # 32 for testing, 256 or higher 512 for final
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    preferences = context.preferences
    preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    preferences.addons["cycles"].preferences.get_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # Check and delete any existing light object
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
            bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="SUN")
    light2 = bpy.data.lights["Sun"]
    light2.energy = 3
    bpy.data.objects["Sun"].location[2] = 1.2
    bpy.data.objects["Sun"].scale[0] = 100
    bpy.data.objects["Sun"].scale[1] = 100
    bpy.data.objects["Sun"].scale[2] = 100
    # bpy.context.view_layer.update() # Update the scene with the new light object


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> bpy.types.Object:
    """Loads an object asset into the scene."""
    # import the object
    if object_path.endswith(".glb") or object_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path, axis_forward='Y', axis_up='Z')
    elif object_path.endswith(".ply"):
        # bpy.ops.import_mesh.ply(filepath=object_path)                             # only used for snap blender
        bpy.ops.wm.ply_import(filepath=object_path, forward_axis='Y', up_axis='Z')  # used for blender 4.0 & snap blender
    elif object_path.endswith(".obj"):
        # bpy.ops.import_scene.obj(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')  # only used for snap blender
        bpy.ops.wm.obj_import(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')       # used for blender 4.0 & snap blender
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    new_obj = bpy.context.object
    return new_obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bpy.ops.object.select_all(action="DESELECT")
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else object_meshes(single_obj):
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def object_meshes(single_obj):
    for obj in [single_obj] + single_obj.children_recursive:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def normalize_scene(single_obj=None):
    bbox_min, bbox_max = scene_bbox(single_obj)
    scale = 1 / max(bbox_max - bbox_min)
    # if single_obj is None:
    #     # Apply scale to all objects in the scene.
    #     for obj in scene_root_objects():
    #         obj.scale = obj.scale * scale
    single_obj.scale = single_obj.scale * scale
    bpy.context.view_layer.update()             # Ensure the scene is fully updated
    bbox_min, bbox_max = scene_bbox(single_obj)
    offset = -(bbox_min + bbox_max) / 2
    # if single_obj is None:
    #     # Apply offset to all objects in the scene.
    #     for obj in scene_root_objects():
    #         obj.matrix_world.translation += offset
    single_obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    # Find a camera in the scene
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            print("found camera")
            break

    # If no camera is found, create a new one
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    
     # Set the camera as the active camera for the scene
    bpy.context.scene.camera = cam
    
    return cam, cam_constraint


def run_object_render(object_file: str, output_dir: str, num_images: int, camera_dist: int) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(output_dir, exist_ok=True)
    setup_blender_env()
    # load the object
    obj = load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(obj)
    add_lighting()
    cam, cam_constraint = setup_camera()
    bpy.context.view_layer.update()  # Ensure the scene is fully updated
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    for i in range(num_images):
        # set the camera position
        theta = (i / num_images) * math.pi * 2 - math.pi/2
        phi = math.radians(70)
        point = (
            camera_dist * math.sin(phi) * math.cos(theta),
            camera_dist * math.sin(phi) * math.sin(theta),
            camera_dist * math.cos(phi),
        )
        cam.location = point
        # render the image
        render_path = os.path.join(output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--object_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--camera_dist', type=float, default=1.5)
    args = parser.parse_args()
    run_object_render(args.object_file, args.output_dir, args.num_images, args.camera_dist)