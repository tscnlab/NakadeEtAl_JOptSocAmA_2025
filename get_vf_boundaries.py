import numpy as np

import utils_img
from common_params import CAMERA, NUMBERS, DIRECTORIES, save_npy_files
from naming import NAMING

__doc__ = """Module used to get theta(phi) boundaries from the rendered images.

Using the 5 images rendered from the locations of the right eyes of the 
head models, this module gets the theta(phi) boundaries of the visual fields.
"""


def get_pixel_theta_phi(camera_directions_, image_shape_, fov_):
    """Get the theta and phi values for pixels in all the images.

    Get :py:func:`utils_img.img_shape_to_theta_phi` for all the
    :py:attr:`camera_directions_`.

    Parameters
    ----------
    camera_directions_ : dict
        The camera directions for the images. The keys are integers from 0 to
        (number of images - 1).
    image_shape_ : tuple[int, int]
        ``(height, width)`` of the images.
    fov_ : float
        The field of view of the camera in degrees.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(len(camera_directions_), *image_shape_, 2)``
        containing the theta and phi values for the pixels in all the images.
    """
    thetas_phis_for_img_pixels_ = np.zeros((len(camera_directions_), *image_shape_, 2))
    for i, v in camera_directions_.items():
        theta, phi = utils_img.img_shape_to_theta_phi(v['camera_direction'], v['up'],
                                                      image_shape_=image_shape_, fov_=fov_)
        phi = utils_img.phi_neg_to_pos(phi)
        thetas_phis_for_img_pixels_[i] = np.stack([theta, phi], axis=-1)
    return thetas_phis_for_img_pixels_


def get_theta_phi_boundary(images_, thetas_phis_for_img_pixels_, tol=2e-1):
    """Get the theta and phi values for the boundary pixels in the images.

    Get the theta and phi values for the pixels in the images that are not
    within :py:attr:`tol` of either 0 or the maximum value in the images
    (expected to be ``106.857``).

    Parameters
    ----------
    images_ : numpy.ndarray
        An array of shape (number of images, height, width) containing the
        images.
    thetas_phis_for_img_pixels_ : numpy.ndarray
        An array of shape (number of images, height, width, 2) containing
        the theta and phi values for the pixels in the images.
        To be obtained from :py:func:`get_pixel_theta_phi`.
    tol : float, default 2e-1
        The tolerance around 0 and ``images_.max()`` to determine
        non-boundary pixels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The theta and phi values for the boundary pixels in the images.
        Both returned arrays are 1D.
    """
    images_max = images_.max()
    is_close_0 = np.isclose(images_, 0, atol=tol)
    is_close_max = np.isclose(images_, images_max, atol=tol)
    is_boundary = ~is_close_0 & ~is_close_max
    theta_phi_boundary = thetas_phis_for_img_pixels_[is_boundary]
    theta_array = theta_phi_boundary[..., 0]
    phi_array = theta_phi_boundary[..., 1]
    arg_sort = np.argsort(phi_array)
    return theta_array[arg_sort], phi_array[arg_sort]


def get_bin_mids(minima, maxima, num_bins):
    """Get the midpoints of the bins.

    Divide the range from :py:attr:`minima` to :py:attr:`maxima` into
    :py:attr:`num_bins` bins and return their midpoints.

    Parameters
    ----------
    minima : float
        The minimum value of the range.
    maxima : float
        The maximum value of the range.
    num_bins : int
        The number of bins to divide the range into.

    Returns
    -------
    numpy.ndarray
        An array of shape (num_bins,) containing the midpoints of the bins.
    """
    bin_size = (maxima - minima) / num_bins
    return np.linspace(minima + bin_size / 2, maxima - bin_size / 2, num_bins)


def binned(t_, p_, num_phi_bins=NUMBERS.num_phi_bins):
    """Get theta values corresponding to phi bins.

    Create equally spaced bins for phi values and get the mean of the
    theta values for points in each bin. Empty bins (bins with no points) are
    not included in the output.

    Parameters
    ----------
    t_ : numpy.ndarray
        A 1D array of the theta values.
    p_ : numpy.ndarray
        A 1D array of the corresponding phi values.
    num_phi_bins : int, default NUMBERS.num_phi_bins
        The number of bins to divide the range of phi values (0 to 2*pi) into.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The mean theta value for each non-empty bin,
        the mean phi value for each non-empty bin,
        the midpoints of :py:attr:`num_phi_bins` equally spaced phi bins
        from 0 to 2*pi.
    """
    rad_bin = 2 * np.pi / num_phi_bins
    p_bin_edges = np.arange(0, 2 * np.pi + rad_bin, rad_bin)
    p_bins = []
    t_bins = []
    for ind in range(len(p_bin_edges) - 1):
        bin_indices = np.where((p_ >= p_bin_edges[ind]) & (p_ < p_bin_edges[ind + 1]))
        if len(bin_indices[0]) > 0:
            p_bins.append(np.mean(p_bin_edges[ind:ind + 2]))
            t_bins.append(np.mean(t_[bin_indices]))
    return np.array(t_bins), np.array(p_bins), get_bin_mids(0, 2 * np.pi, num_phi_bins)


def binned_interpolated(t_bins_, p_bins_, p_mids_):
    """Interpolate theta values for phi bins.

    Interpolate the theta values for the midpoints of the equally spaced
    phi bins.

    Parameters
    ----------
    t_bins_ : numpy.ndarray
        1D array with theta values.
    p_bins_ : numpy.ndarray
        1D array with phi values corresponding to :py:attr:`t_bins_`.
    p_mids_: numpy.ndarray
        1D array with the midpoints of the phi bins.

    Returns
    -------
    numpy.ndarray
        1D array with the interpolated theta values for :py:attr:`p_mids_`.
    """
    t_mids_ = np.interp(p_mids_, p_bins_, t_bins_, period=2 * np.pi)
    return t_mids_


def main():
    """Get theta(phi) boundaries from the visual field images.

    Gets the theta(phi) boundaries from the rendered images,
    groups them into the id\\_+, id\\_-, random, and generic categories,
    and saves them to .npy files.
    """
    thetas_phis_for_img_pixels = get_pixel_theta_phi(CAMERA.directions,
                                                     CAMERA.image_shape, CAMERA.fov)
    ls = DIRECTORIES.rendered_imgs_np.glob(str(NAMING.asterisk.rendered.npy))
    p_mids = get_bin_mids(0, 2 * np.pi, NUMBERS.num_phi_bins)
    id_p_t_mids = np.zeros((NUMBERS.num_ids, NUMBERS.num_phi_bins))
    id_m_t_mids = np.zeros((NUMBERS.num_ids, NUMBERS.num_phi_bins))
    rand_t_mids = np.zeros((NUMBERS.num_total_rand, NUMBERS.num_phi_bins))
    generic_t_mids = np.zeros(NUMBERS.num_phi_bins)
    for file_path in ls:
        file_pathlike = NAMING.replace_suffix(file_path.stem, 'rendered', '')
        file_stem = str(file_pathlike)
        images = np.load(file_path)
        t_mids = binned_interpolated(
            *binned(*get_theta_phi_boundary(images, thetas_phis_for_img_pixels)))
        np.save(DIRECTORIES.boundaries / file_pathlike.theta_boundary.npy, t_mids)
        # Group the boundaries into id_+, id_-, random, and generic categories
        if file_stem.startswith(str(NAMING.id_)):
            id_num, pm = NAMING.get_id_num_pm(file_stem)
            if pm == '+':
                id_p_t_mids[id_num] = t_mids
            else:
                id_m_t_mids[id_num] = t_mids
        elif file_stem.startswith(str(NAMING.random_)):
            rand_num = NAMING.get_random_num(file_stem)
            rand_t_mids[rand_num] = t_mids
        elif file_stem.startswith(str(NAMING.generic_)):
            generic_t_mids = t_mids
    save_npy_files({
        DIRECTORIES.vf / NAMING.phis.npy: p_mids,
        DIRECTORIES.vf / NAMING.id.pos.theta_boundary.npy: id_p_t_mids,
        DIRECTORIES.vf / NAMING.id.neg.theta_boundary.npy: id_m_t_mids,
        DIRECTORIES.vf / NAMING.random.theta_boundary.npy: rand_t_mids[:NUMBERS.num_rand],
        DIRECTORIES.vf / NAMING.random.val.theta_boundary.npy: rand_t_mids[NUMBERS.num_rand:],
        DIRECTORIES.vf / NAMING.generic.theta_boundary.npy: generic_t_mids
    })


if __name__ == '__main__':
    main()
