import matplotlib.pyplot as plt
import numpy as np

from utils_img import add_transparency
from common_params import DIRECTORIES, NAMING, NUMBERS, COLORS


def add_polar_axes(fig_, axes_coords_):
    ax_polar = fig_.add_axes(axes_coords_, projection='polar')
    ax_polar.set_yticks(np.deg2rad(NUMBERS.polar_y_ticks))
    ax_polar.set_rlim(0, 90)
    ax_polar.set_rticks(NUMBERS.polar_r_ticks)
    ax_polar.yaxis.set_major_formatter('{x}°')
    ax_polar.patch.set_alpha(0)


def add_image(fig_, image_, axes_coords_, cmap='gray', **kwargs):
    ax_img = fig_.add_axes(axes_coords_)
    axes_image = ax_img.imshow(image_, cmap=cmap, **kwargs)
    ax_img.set_axis_off()
    return axes_image


def polar_plot_grayscale_image(image_, output_file_path, **kwargs):
    image_ = add_transparency(image_)
    plt.clf()
    fig = plt.figure()
    _ = add_image(fig, image_, NUMBERS.polar_axes_coords, **kwargs)
    add_polar_axes(fig, NUMBERS.polar_axes_coords)
    plt.savefig(output_file_path, dpi=NUMBERS.dpi)
    plt.close()


def polar_plot_grayscale_images(images_, output_file_paths):
    for image_, output_file_path in zip(images_, output_file_paths):
        polar_plot_grayscale_image(image_, output_file_path)


def add_colorbar(fig_, axes_image_, cax_coords_):
    cax = fig_.add_axes(cax_coords_)
    fig_.colorbar(axes_image_, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')


def polar_plot_color_image(image_, output_file_path, cmap='bwr', **kwargs):
    plt.clf()
    fig = plt.figure(figsize=NUMBERS.figsize_colorbar, constrained_layout=True)
    vmax = np.abs(image_).max()
    axes_image = add_image(fig, image_, NUMBERS.polar_axes_coords_colorbar, cmap=cmap, vmin=-vmax, vmax=vmax, **kwargs)
    add_colorbar(fig, axes_image, NUMBERS.colorbar_axes_coords_colorbar)
    add_polar_axes(fig, NUMBERS.polar_axes_coords_colorbar)
    plt.savefig(output_file_path, dpi=NUMBERS.dpi)
    plt.close()


def get_ticks(y_min, y_max, y_tick_every):
    y_tick_min = (y_min // y_tick_every + 1) * y_tick_every
    y_tick_max = (y_max // y_tick_every) * y_tick_every
    return np.arange(y_tick_min, y_tick_max + 1, y_tick_every)


def plot_vf_xy(plots_dict, output_file_path):
    plt.clf()
    for k, v in plots_dict.items():
        plt.plot(np.rad2deg(v['phis']), np.rad2deg(v['thetas']),
                 label=k, **v['kwargs'])
    plt.legend()
    plt.xticks(np.arange(0, 360 + 1, NUMBERS.x_tick_every))
    y_min, y_max = plt.gca().get_ylim()
    plt.yticks(get_ticks(y_min, y_max, NUMBERS.y_tick_every))
    plt.xlabel(r"$\phi$ [degrees]")
    plt.ylabel(r"$\theta$ [degrees]")
    plt.savefig(output_file_path, dpi=NUMBERS.dpi)
    plt.close()


def plot_rendered_predicted_comparison(rendered_thetas, predicted_thetas, phis, output_file_path):
    plot_vf_xy({
        'rendered': {'thetas': rendered_thetas, 'phis': phis,
                     'kwargs': {'color': COLORS.rendered, 'linewidth': 2}},
        'predicted': {'thetas': predicted_thetas, 'phis': phis,
                      'kwargs': {'color': COLORS.predicted, 'linewidth': 1}}},
        output_file_path)


def plot_rendered_predicted_comparisons(random_predicted, random_rendered, phis):
    for i in range(random_rendered.shape[0]):
        plot_rendered_predicted_comparison(
            random_rendered[i], random_predicted[i], phis,
            DIRECTORIES.comparison_plots / NAMING.png.random_suffix(i, 'comparison')
        )


def get_id_pm_diff(id_num):
    p_img = np.load(DIRECTORIES.boundaries / NAMING.npy.id_pm_suffix(id_num, +1, 'hemispherical_vf'))
    m_img = np.load(DIRECTORIES.boundaries / NAMING.npy.id_pm_suffix(id_num, -1, 'hemispherical_vf'))
    return p_img - m_img


def plot_id_pm_diff(id_num):
    diff = get_id_pm_diff(id_num)
    polar_plot_color_image(diff, DIRECTORIES.comparison_plots / NAMING.png.id_diff(id_num))


def plot_id_pm_diffs(id_nums_arr):
    for id_num in id_nums_arr:
        plot_id_pm_diff(id_num)


def main():
    random_thetas = np.load(DIRECTORIES.vf / NAMING.npy.random_thetas)
    random_val_thetas = np.load(DIRECTORIES.vf / NAMING.npy.random_val_thetas)
    predicted_thetas = np.load(DIRECTORIES.vf / NAMING.npy.predictions)
    phis = np.load(DIRECTORIES.vf / NAMING.npy.phis)
    plot_rendered_predicted_comparisons(
        random_predicted=predicted_thetas,
        random_rendered=np.concatenate([random_thetas, random_val_thetas], axis=0),
        phis=phis
    )
    plot_id_pm_diffs(range(NUMBERS.num_ids))
    ls = sorted(list(DIRECTORIES.boundaries.glob(NAMING.npy.add_suffix('*', 'hemispherical_vf'))), key=str)
    images = [np.load(file) for file in ls]
    polar_plot_grayscale_images(images, [DIRECTORIES.comparison_plots / NAMING.png.add_file_type(file.stem) for file in ls])


if __name__ == '__main__':
    main()
