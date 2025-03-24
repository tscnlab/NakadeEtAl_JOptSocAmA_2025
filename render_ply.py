import json
import numpy as np
from tqdm import tqdm

from common_params import CAMERA, NUMBERS, DIRECTORIES
from naming import NAMING

import mitsuba as mi

__doc__ = """Render images from the perspective of the right eye of the head model.

For each of the head models exported by 
:py:mod:`export_from_blender.py <export_from_blender>` 
(generic with all shape parameters 0, id +- meshes with individual parameters 
set to 1 or -1 and random faces with the shape parameters uniformly 
distributed), the camera is placed at the center of the pupil of the right eye.
The head is surrounded by a constant emitter with ``radiance = 1`` at all 
wavelengths. The head is completely non-reflective, creating complete contrast 
between the head and the background. This makes it easier to tell where the 
Visual Field (VF) boundary is. The FOV of the camera is 90 degrees.
The images are rendered with the camera pointing in the front, up, down, left, 
and right directions, thus covering the entire hemisphere in front of the eye. 
The images are saved as .npy files in the ``rendered_images_numpy`` directory.
"""

# Set the Mitsuba variant to use according to the priority list
MI_VARIANTS_PRIORITY = ['cuda_ad_spectral', 'cuda_spectral', 'llvm_ad_spectral',
                        'llvm_spectral', 'scalar_ad_spectral']

MI_VARIANTS_AVAILABLE = mi.variants()

MI_VARIANTS_AVAILABLE_PRIORITY = [variant for variant in MI_VARIANTS_PRIORITY
                                  if variant in MI_VARIANTS_AVAILABLE]

if not MI_VARIANTS_AVAILABLE_PRIORITY:
    raise ValueError('None of the available Mitsuba variants are suitable.\n'
                     'Please make sure that at least one of the following '
                     f'variants is available: {MI_VARIANTS_PRIORITY}')

mi.set_variant(MI_VARIANTS_AVAILABLE_PRIORITY[0])
from mitsuba import ScalarTransform4f as Transform

with open(DIRECTORIES.vf / NAMING.eye_centers.right.json, 'r') as f:
    EYE_CENTERS_RIGHT = json.load(f)


FILM = {
    'type': 'specfilm',
    'width': CAMERA.image_size,
    'height': CAMERA.image_size,
    'component_format': 'float32',
    'rfilter': {
        'type': 'box'
    },
    str(NAMING.y_cie): {
        "type": "spectrum",
        "filename": str(DIRECTORIES.output_channels / NAMING.y_cie.spd)
    }
}


def camera_dict(origin_, camera_direction_, up_, fov_, fov_axis_=CAMERA.fov_axis):
    """Create a dictionary with the camera parameters.

    The camera is placed at the center of the right eye of the
    head model (:py:attr:`origin`) and points towards
    :py:attr:`camera_direction`. The field of view is :py:attr:`fov` degrees.

    Parameters
    ----------
    origin_ : numpy.ndarray
        The location of the center of projection of the camera.
    camera_direction_ : numpy.ndarray
        The direction in which the camera is pointing.
    up_ : numpy.ndarray
        The direction that is considered 'up' in the camera's coordinate system.
    fov_ : float
        The field of view of the camera.
    fov_axis_ : str, default CAMERA.fov_axis = 'x'
        The axis along which the field of view is specified.

    Returns
    -------
    dict
        A dictionary with the camera parameters.
    """
    target = origin_ + camera_direction_
    return {
        'type': 'perspective',
        'near_clip': CAMERA.near_clip,
        'fov': fov_,
        'fov_axis': fov_axis_,
        'to_world': Transform.look_at(
            origin=origin_,
            target=target,
            up=up_
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 150
        },
        'film': FILM
    }


def scene_dict(filepath):
    """Create a dictionary with the scene parameters.

    The head model is surrounded by a constant emitter with ``radiance = 1``
    at all wavelengths.
    The head model is completely non-reflective.
    This creates complete contrast between the head model and the background
    and makes it easier to determine the visual field boundary.
    Since we are using the ``Y CIE 1931`` spectrum for the sensor response,
    the expected value of the bright pixels will be ``106.857``, which is the
    integral of the ``Y CIE 1931`` spectrum.

    Parameters
    ----------
    filepath : str
        The path to the PLY file that contains the head model.

    Returns
    -------
    dict
        A dictionary with the scene parameters.
    """
    return {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": -1,
            "rr_depth": 20
        },
        'emitter_constant': {
            'type': 'constant',
            'radiance': {
                'type': 'spectrum',
                'value': 1
            }
        },
        'head': {
            'type': 'ply',
            'filename': filepath,
            'to_world': Transform.scale(NUMBERS.cm_to_mm).rotate([0, 0, 1], 90),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'spectrum',
                    'value': 0
                }
            }
        }
    }


def camera_is_in_front_of_eye(image, center_threshold=100, unique_threshold=10000):
    """Check if the camera is in front of the eye.

    Parameters
    ----------
    image : numpy.ndarray
        The image rendered by the camera.
    center_threshold : float, default 100
        The threshold for the pixel value at the center of the image.
        If the camera is behind the mesh, this value will be much lower.
        The threshold is set to 100 because the expected value of the pixel
        at the center of the image is ``106.857`` (integral of the ``Y CIE 1931``
        spectrum, because the radiance of the constant light source is 1
        for all wavelengths).
    unique_threshold : float, default 10000
        The threshold for the number of unique pixel values in the image.
        If the camera is in front of the mesh, this value will be much lower
        because there is complete contrast between the mesh and the background.

    Returns
    -------
    bool
        ``True`` if the camera is in front of the eye, ``False`` otherwise.
    """
    return (image[CAMERA.image_size // 2, CAMERA.image_size // 2] > center_threshold and
            len(np.unique(image)) < unique_threshold)


def render(file_stem):
    """Render images from the perspective of the right eye of the head model.

    First the camera is placed at the coordinates of the center of the right eye
    obtained from the PLY file. The camera is then moved forward by 1e-4 mm
    until it is in front of the eye. The camera is then moved forward by
    another 1e-4 mm. The camera has a field of view of 90 degrees.
    The images are rendered with the camera pointing in the front, up, down,
    left, and right directions

    Parameters
    ----------
    file_stem : str
        The stem of the PLY file name that contains the head model.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(5, *CAMERA.image_shape)`` containing the
        rendered images.
    """
    scene = mi.load_dict(scene_dict(str(DIRECTORIES.ply / NAMING.make_pathlike(file_stem).ply)))
    # The further shift forward by tan(FOV / 2) * near_clip is to ensure that
    # the near clip plane of the camera pointing in directions other than
    # the front does not intersect the mesh.
    camera_origin = (np.array(EYE_CENTERS_RIGHT[file_stem]) +
                     np.array([np.tan(CAMERA.fov / 2) * CAMERA.near_clip, 0, 0]))
    camera_dictionary = camera_dict(camera_origin, CAMERA.directions[0]['camera_direction'], CAMERA.directions[0]['up'],
                                    CAMERA.fov)
    camera = mi.load_dict(camera_dictionary)
    rendered_image = mi.render(scene, sensor=camera)
    while not camera_is_in_front_of_eye(rendered_image.numpy().reshape(CAMERA.image_shape)):
        camera_origin += np.array([1e-4, 0, 0])
        camera_dictionary = camera_dict(camera_origin, CAMERA.directions[0]['camera_direction'],
                                        CAMERA.directions[0]['up'], CAMERA.fov)
        camera = mi.load_dict(camera_dictionary)
        rendered_image = mi.render(scene, sensor=camera)
    camera_origin += np.array([1e-4, 0, 0])
    images_ = np.zeros((len(CAMERA.directions), *CAMERA.image_shape), dtype=np.float32)
    for k, v in CAMERA.directions.items():
        camera = mi.load_dict(camera_dict(camera_origin, v['camera_direction'], v['up'], CAMERA.fov))
        rendered_image = mi.render(scene, sensor=camera, seed=NUMBERS.mitsuba_seed)
        images_[k] = rendered_image.numpy().reshape(CAMERA.image_shape)
    return images_


def main():
    """Render images from the perspective of the right eye of the head models.

    The images are saved as npy files in the ``rendered_images_numpy`` directory.
    """
    for file_stem in tqdm(EYE_CENTERS_RIGHT, desc='Rendering images'):
        images = render(file_stem)
        np.save(DIRECTORIES.rendered_imgs_np / NAMING.add_suffix(file_stem, 'rendered').npy, images)


if __name__ == '__main__':
    main()
