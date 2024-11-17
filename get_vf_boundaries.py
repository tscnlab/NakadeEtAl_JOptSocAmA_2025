import numpy as np

import utils_img
from common_params import CAMERA, NUMBERS, DIRECTORIES, save_npy_files
from naming import NAMING


def get_pixel_theta_phi(camera_directions_, image_shape_, fov_):
    thetas_phis_for_img_pixels_ = np.zeros((len(camera_directions_), *image_shape_, 2))
    for i, v in camera_directions_.items():
        theta, phi = utils_img.img_shape_to_theta_phi(v['camera_direction'], v['up'],
                                                      image_shape_=image_shape_, fov_=fov_)
        phi = utils_img.phi_neg_to_pos(phi)
        thetas_phis_for_img_pixels_[i] = np.stack([theta, phi], axis=-1)
    return thetas_phis_for_img_pixels_


def get_theta_phi_boundary(images_, thetas_phis_for_img_pixels_, tol=2e-1):
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
    bin_size = (maxima - minima) / num_bins
    return np.linspace(minima + bin_size / 2, maxima - bin_size / 2, num_bins)


def binned(t_, p_, num_phi_bins=NUMBERS.num_phi_bins):
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


def binned_interpolated(t_bins, p_bins, p_mids_):
    t_mids_ = np.interp(p_mids_, p_bins, t_bins, period=2 * np.pi)
    return t_mids_


def main():
    thetas_phis_for_img_pixels = get_pixel_theta_phi(CAMERA.directions, CAMERA.image_shape, CAMERA.fov)
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
        if file_stem.startswith(str(NAMING.id_)):
            id_num, pm = NAMING.get_id_pm(file_stem)
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
