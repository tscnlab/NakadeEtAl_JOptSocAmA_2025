import numpy as np
from tqdm import tqdm

from hemispherical_interpolator import ImageSetWeights
from common_params import CAMERA, NUMBERS, DIRECTORIES, save_npy_files
from naming import NAMING

IMAGE_SET_WEIGHTS = ImageSetWeights(CAMERA.image_shape, len(CAMERA.directions),
                                    CAMERA.fov, CAMERA.fov_axis,
                                    CAMERA.directions, CAMERA.front)


def get_projected_solid_angle(images, image_set_weights=IMAGE_SET_WEIGHTS):
    """Get the projected solid angle corresponding to the set of images.

    Gives the projected solid angle by summing over the pixels in the images
    after weighting by the solid angle of each pixel and its cosine factor.

    Parameters
    ----------
    images : numpy.ndarray
        The rendered images. The shape is assumed to be (no. of sensors, *shape of each sensor).
    image_set_weights : hemispherical_interpolator.ImageSetWeights
        The weights corresponding to the images.

    Returns
    -------
    float
        The projected solid angle corresponding to the set of images.
    """
    return np.sum(images * image_set_weights.weights)


def get_projected_solid_angles(images_multiple_heads, image_set_weights=IMAGE_SET_WEIGHTS):
    """Get the projected solid angle corresponding to more than one head.

    Parameters
    ----------
    images_multiple_heads : numpy.ndarray
        The rendered images. The shape is assumed to be (no. of heads, no. of sensors, *shape of each sensor).
    image_set_weights : hemispherical_interpolator.ImageSetWeights
        The weights corresponding to the images. Assumed to be the same for all the heads.

    Returns
    -------
    numpy.ndarray
        The projected solid angles corresponding to the heads. Has a shape (no. of heads,)
    """
    return np.sum(images_multiple_heads * image_set_weights.weights[np.newaxis], axis=(1, 2, 3))


def main():
    """Get the projected solid angles of the VFs of all heads.

    Get projected solid angles of the VFs of random, id+- and generic heads.
    """
    projected_solid_angles = np.zeros(NUMBERS.num_total_rand)
    for head_num in tqdm(range(len(projected_solid_angles)), desc='Calculating projected solid angles for random heads'):
        images = np.load(DIRECTORIES.rendered_imgs_np / NAMING.random(head_num).rendered.npy)
        images /= NUMBERS.y_channel_integral
        projected_solid_angles[head_num] = get_projected_solid_angle(images)
    projected_solid_angles_percentages = (projected_solid_angles - np.pi) / np.pi * 100
    projected_solid_angles_ids_pos = np.zeros(NUMBERS.num_ids)
    projected_solid_angles_ids_neg = np.zeros(NUMBERS.num_ids)
    for id_num in tqdm(range(NUMBERS.num_ids), desc='Calculating projected solid angles for ID parameter heads'):
        images_pos = np.load(DIRECTORIES.rendered_imgs_np / NAMING.id(id_num).pos.rendered.npy)
        images_neg = np.load(DIRECTORIES.rendered_imgs_np / NAMING.id(id_num).neg.rendered.npy)
        images_pos /= NUMBERS.y_channel_integral
        images_neg /= NUMBERS.y_channel_integral
        projected_solid_angles_ids_pos[id_num] = get_projected_solid_angle(images_pos)
        projected_solid_angles_ids_neg[id_num] = get_projected_solid_angle(images_neg)
    generic_images = np.load(DIRECTORIES.rendered_imgs_np / NAMING.generic.rendered.npy)
    projected_solid_angle_generic = get_projected_solid_angle(generic_images)
    save_npy_files({
        DIRECTORIES.vf / NAMING.random.projected_solid_angles.npy: projected_solid_angles,
        DIRECTORIES.vf / NAMING.random.projected_solid_angles_percentages.npy: projected_solid_angles_percentages,
        DIRECTORIES.vf / NAMING.id.pos.projected_solid_angles.npy: projected_solid_angles_ids_pos,
        DIRECTORIES.vf / NAMING.id.neg.projected_solid_angles.npy: projected_solid_angles_ids_neg,
        DIRECTORIES.vf / NAMING.generic.projected_solid_angles.npy: [projected_solid_angle_generic],
    })


if __name__ == '__main__':
    main()
