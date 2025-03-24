import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils_img import add_transparency
from common_params import DIRECTORIES, NUMBERS, COLORS
from naming import NAMING


def add_polar_axes(fig_, axes_coords_):
    """Add polar axes to the figure.

    Adds polar axes to the figure at the specified coordinates and
    sets the ticks and limits for the polar plot.

    Parameters
    ----------
    fig_ : matplotlib.figure.Figure
        The figure to which the polar axes are to be added.
    axes_coords_ : tuple[float, float, float, float]
        The coordinates of the polar axes in the figure.

    Returns
    -------
    None
    """
    ax_polar = fig_.add_axes(axes_coords_, projection='polar')
    ax_polar.set_yticks(np.deg2rad(NUMBERS.polar_y_ticks))
    ax_polar.set_rlim(0, 90)  # 0 to 90 because this represents theta on a hemisphere
    ax_polar.set_rticks(NUMBERS.polar_r_ticks)
    ax_polar.yaxis.set_major_formatter('{x}°')
    ax_polar.patch.set_alpha(0)


def add_image(fig_, image_, axes_coords_, cmap='gray', **kwargs):
    """Add an image to the figure at the specified coordinates.

    Parameters
    ----------
    fig_ : matplotlib.figure.Figure
        The figure to which the image is to be added.
    image_ : numpy.ndarray
        The image to be added.
    axes_coords_ : tuple[float, float, float, float]
        The coordinates of the image in the figure.
    cmap : str, default 'gray'
        The colormap to be used for the image.
    kwargs : dict
        Additional keyword arguments to be passed to
        :py:func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.image.AxesImage
        The image added to the figure.
    """
    ax_img = fig_.add_axes(axes_coords_)
    axes_image = ax_img.imshow(image_, cmap=cmap, **kwargs)
    ax_img.set_axis_off()
    return axes_image


def polar_plot_grayscale_image(image_, output_file_path, **kwargs):
    """Plot a grayscale image and add polar axes with ticks.

    Parameters
    ----------
    image_ : numpy.ndarray
        The grayscale image to be plotted.
    output_file_path : str | pathlib.Path
        The path to the output file.
    kwargs : dict
        Additional keyword arguments to be passed to
        :py:func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    None
    """
    image_ = add_transparency(image_)
    plt.clf()
    fig = plt.figure()
    _ = add_image(fig, image_, NUMBERS.polar_axes_coords, **kwargs)
    add_polar_axes(fig, NUMBERS.polar_axes_coords)
    plt.savefig(output_file_path, dpi=NUMBERS.dpi)
    plt.close()


def polar_plot_grayscale_images(images_, output_file_paths):
    """Plot grayscale images and add polar axes with ticks.

    Calls :py:func:`polar_plot_grayscale_image` for multiple images.

    Parameters
    ----------
    images_ : list[numpy.ndarray, ...]
        The grayscale images to be plotted.
    output_file_paths : list[str | pathlib.Path, ...]
        The paths to the output files.

    Returns
    -------
    None
    """
    for image_, output_file_path in tqdm(zip(images_, output_file_paths), desc='Creating random face VF plots'):
        polar_plot_grayscale_image(image_, output_file_path)


def add_colorbar(fig_, axes_image_, cax_coords_):
    """Add a colorbar to the figure.

    Parameters
    ----------
    fig_ : matplotlib.figure.Figure
        The figure to which the colorbar is to be added.
    axes_image_ : matplotlib.image.AxesImage
        The image for which the colorbar is to be added.
    cax_coords_ : tuple[float, float, float, float]
        The coordinates of the colorbar in the figure.

    Returns
    -------
    None
    """
    cax = fig_.add_axes(cax_coords_)
    fig_.colorbar(axes_image_, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')


def polar_plot_color_image(image_, output_file_path, cmap='bwr', **kwargs):
    """Plot a color image, add polar axes with ticks and add a colorbar.

    Parameters
    ----------
    image_ : numpy.ndarray
        The color image to be plotted.
    output_file_path : str | pathlib.Path
        The path to the output file.
    cmap : str | matplotlib.colors.Colormap, default 'bwr'
        The colormap to be used for the image.
    kwargs : dict
        Additional keyword arguments to be passed to
        :py:func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    None
    """
    plt.clf()
    fig = plt.figure(figsize=NUMBERS.figsize_colorbar, constrained_layout=True)
    vmax = np.abs(image_).max()
    axes_image = add_image(fig, image_, NUMBERS.polar_axes_coords_colorbar, cmap=cmap, vmin=-vmax, vmax=vmax, **kwargs)
    add_colorbar(fig, axes_image, NUMBERS.colorbar_axes_coords_colorbar)
    add_polar_axes(fig, NUMBERS.polar_axes_coords_colorbar)
    plt.savefig(output_file_path, dpi=NUMBERS.dpi)
    plt.close()


def get_ticks(min_, max_, tick_at_every_):
    """Get all the multiples of :py:attr:`tick_at_every_` between
    :py:attr:`min_` and :py:attr:`max_`.

    Parameters
    ----------
    min_ : int
        The minimum value.
    max_ : int
        The maximum value.
    tick_at_every_ : int
        The number for which multiples are to be returned.

    Returns
    -------
    numpy.ndarray
        An array containing all the multiples of :py:attr:`tick_at_every_`
        between :py:attr:`min_` and :py:attr:`max_`.
    """
    tick_min = (min_ // tick_at_every_ + 1) * tick_at_every_
    tick_max = (max_ // tick_at_every_) * tick_at_every_
    return np.arange(tick_min, tick_max + 1, tick_at_every_)


def plot_vf_xy(plots_dict, output_file_path):
    """Plot the Visual Field boundaries with theta on the y-axis and
    phi on the x-axis.

    Parameters
    ----------
    plots_dict : dict
        A dictionary containing the data to be plotted.
        The keys are the labels for the plots and the values are
        dictionaries with the following keys

        - 'thetas' : numpy.ndarray
            The theta values for the boundary pixels.
        - 'phis' : numpy.ndarray
            The phi values for the boundary pixels.
        - 'kwargs' : dict
            Additional keyword arguments to be passed to
            :py:meth:`matplotlib.pyplot.plot`.

    output_file_path : str | pathlib.Path
        The path to the output file

    Returns
    -------
    None
    """
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
    """Plot the rendered and predicted Visual Field boundaries.

    Parameters
    ----------
    rendered_thetas : numpy.ndarray
        The theta values for the rendered boundary.
    predicted_thetas : numpy.ndarray
        The theta values for the predicted boundary.
    phis : numpy.ndarray
        The phi values for the boundary.
    output_file_path : str | pathlib.Path
        The path to the output file.

    Returns
    -------
    None
    """
    plot_vf_xy({
        'rendered': {'thetas': rendered_thetas, 'phis': phis,
                     'kwargs': {'color': COLORS.rendered, 'linewidth': 2}},
        'predicted': {'thetas': predicted_thetas, 'phis': phis,
                      'kwargs': {'color': COLORS.predicted, 'linewidth': 1}}},
               output_file_path)


def plot_rendered_predicted_comparisons(random_predicted, random_rendered, phis):
    """Plot the rendered and predicted Visual Field boundaries for random faces.

    Calls :py:func:`plot_rendered_predicted_comparison` for multiple faces.

    Parameters
    ----------
    random_predicted : numpy.ndarray
        The predicted theta values for the random faces.
    random_rendered : numpy.ndarray
        The rendered theta values for the random faces.
    phis : numpy.ndarray
        The phi values for the boundary.

    Returns
    -------
    None
    """
    for i in tqdm(range(random_rendered.shape[0]), desc='Creating rendered vs predicted comparison plots'):
        plot_rendered_predicted_comparison(
            random_rendered[i], random_predicted[i], phis,
            DIRECTORIES.comparison_plots / NAMING.random(i).comparison.png
        )


def get_id_pm_diff(id_num):
    """Difference in the Visual Field when an id parameter changes from -1 to 1.

    Parameters
    ----------
    id_num : int
        The id parameter number.

    Returns
    -------
    numpy.ndarray
        The difference in the Visual Field.
    """
    p_img = np.load(DIRECTORIES.boundaries / NAMING.id(id_num).pos.hemispherical_vf.npy)
    m_img = np.load(DIRECTORIES.boundaries / NAMING.id(id_num).neg.hemispherical_vf.npy)
    return p_img - m_img


def plot_id_pm_diff(id_num):
    """Plot the difference in the VF when an id parameter changes from -1 to 1

    Saves the plot to a png file.

    Parameters
    ----------
    id_num : int
        The id number.

    Returns
    -------
    None
    """
    diff = get_id_pm_diff(id_num)
    polar_plot_color_image(diff, DIRECTORIES.comparison_plots / NAMING.id(id_num).diff.png)


def plot_id_pm_diffs(id_nums_arr):
    """Plot the differences in the VFs when id parameters change from -1 to 1.

    Calls :py:func:`plot_id_pm_diff` for multiple id numbers.

    Parameters
    ----------
    id_nums_arr : list[int, ...]
        The id numbers.

    Returns
    -------
    None
    """
    for id_num in tqdm(id_nums_arr, desc='Creating VF difference plots for ID parameters'):
        plot_id_pm_diff(id_num)


def main():
    """Plot various Visual Field (VF) boundaries and VF differences.

    Plot the rendered and predicted VF boundaries for random faces,
    the difference in the VF when an id parameter changes from -1 to 1,
    and the hemispherical VF images for all faces.
    """
    random_thetas = np.load(DIRECTORIES.vf / NAMING.random.theta_boundary.npy)
    random_val_thetas = np.load(DIRECTORIES.vf / NAMING.random.val.theta_boundary.npy)
    predicted_thetas = np.load(DIRECTORIES.vf / NAMING.optimization.predictions.npy)
    phis = np.load(DIRECTORIES.vf / NAMING.phis.npy)
    plot_rendered_predicted_comparisons(
        random_predicted=predicted_thetas,
        random_rendered=np.concatenate([random_thetas, random_val_thetas], axis=0),
        phis=phis
    )
    try:
        plot_id_pm_diffs(range(NUMBERS.num_ids))
        ls = sorted(list(DIRECTORIES.boundaries.glob(str(NAMING.asterisk.hemispherical_vf.npy))), key=str)
        images = [np.load(file) for file in ls]
        polar_plot_grayscale_images(images, [DIRECTORIES.comparison_plots / NAMING.make_pathlike(file.stem).png for file in ls])
    except FileNotFoundError:
        print("No hemispherical VF images found, skipping")


if __name__ == '__main__':
    main()
