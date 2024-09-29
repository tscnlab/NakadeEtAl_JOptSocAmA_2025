import json
import numpy as np

from common_params import CAMERA, NUMBERS, DIRECTORIES, NAMING

import mitsuba as mi
from mitsuba import ScalarTransform4f as Transform

mi.set_variant('cuda_ad_spectral')

with open(DIRECTORIES.vf / NAMING.json.eye_centers_right, 'r') as f:
    eye_centers_right = json.load(f)


FILM = {
    'type': 'specfilm',
    'width': CAMERA.image_size,
    'height': CAMERA.image_size,
    'component_format': 'float32',
    'rfilter': {
        'type': 'box'
    },
    NAMING.spd.film_spd_stem: {
        "type": "spectrum",
        "filename": str(DIRECTORIES.output_channels / NAMING.spd.film_spd)
    }
}


def camera_dict(origin, camera_direction, up, fov, fov_axis=CAMERA.fov_axis):
    target = origin + camera_direction
    return {
        'type': 'perspective',
        'near_clip': CAMERA.near_clip,
        'fov': fov,
        'fov_axis': fov_axis,
        'to_world': Transform.look_at(
            origin=origin,
            target=target,
            up=up
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 150
        },
        'film': FILM
    }


def scene_dict(filepath):
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


def camera_is_in_front_of_eye(image):
    return image[CAMERA.image_size // 2, CAMERA.image_size // 2] > 100 and len(np.unique(image)) < 10000


def render(file_stem):
    scene = mi.load_dict(scene_dict(str(DIRECTORIES.ply / (NAMING.ply.add_file_type(file_stem)))))
    camera_origin = np.array(eye_centers_right[file_stem]) + np.array([np.tan(CAMERA.fov / 2) * CAMERA.near_clip, 0, 0])
    camera_dictionary = camera_dict(camera_origin, CAMERA.directions[0]['camera_direction'],
                                    CAMERA.directions[0]['up'], CAMERA.fov)
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
    for file_stem in eye_centers_right:
        images = render(file_stem)
        np.save(DIRECTORIES.rendered_imgs_np / NAMING.npy.add_suffix(file_stem, 'rendered'), images)


if __name__ == '__main__':
    main()
