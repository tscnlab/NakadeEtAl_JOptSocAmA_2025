import numpy as np
from pathlib import Path

__doc__ = """Module containing common parameters for the project."""


def save_npy_files(paths_arrays_dict):
    """Save numpy arrays to disk.

    Parameters
    ----------
    paths_arrays_dict : dict
        Dictionary containing paths as keys and numpy arrays as values.

    Returns
    -------
    None
    """
    for path_, array_ in paths_arrays_dict.items():
        np.save(path_, array_)


class CameraParameters:
    """Class containing the default camera parameters.

    `CAMERA` is an instance of this class.

    Attributes
    ----------
    fov : int
        Field of view (FOV) in degrees. Set to 90.
    fov_axis : str
        Axis along which the FOV is specified. Set to 'x'.
    image_size : int
        Size of the image in pixels. Set to 1024.
    image_shape : tuple[int, int]
        Shape of the image. Set to (1024, 1024).
    near_clip : float
        Distance to the near clipping plane. Set to 1e-6 (mm).
    front : numpy.ndarray
        Vector along which the head is facing. Set to [1, 0, 0].
    directions : dict
        Dictionary containing the camera directions. The keys are integers
        and the values are dictionaries containing the camera direction, the
        up vector, and the name of the direction. The directions are:
        0: front, 1: up, 2: down, 3: left, 4: right.
    """
    def __init__(self):
        self.fov = 90
        self.fov_axis = 'x'
        self.image_size = 1024
        self.image_shape = (self.image_size, self.image_size)
        self.near_clip = 1e-6
        self.front = np.array([1, 0, 0])
        self.directions = {
            0: {
                'camera_direction': np.array([1, 0, 0]),
                'up': np.array([0, 0, 1]),
                'name': 'front'
            },
            1: {
                'camera_direction': np.array([0, 0, 1]),
                'up': np.array([-1, 0, 0]),
                'name': 'up'
            },
            2: {
                'camera_direction': np.array([0, 0, -1]),
                'up': np.array([1, 0, 0]),
                'name': 'down'
            },
            3: {
                'camera_direction': np.array([0, 1, 0]),
                'up': np.array([0, 0, 1]),
                'name': 'left'
            },
            4: {
                'camera_direction': np.array([0, -1, 0]),
                'up': np.array([0, 0, 1]),
                'name': 'right'
            }
        }


CAMERA = CameraParameters()


class Directories:
    """Class containing the directories used in the project.

    `DIRECTORIES` is an instance of this class.

    Attributes
    ----------
    base_dir : pathlib.Path
        Base directory of the project. Set to the directory containing this file.
    output_channels : pathlib.Path
        Directory containing the .spd files for the output channel spectra.
    vf : pathlib.Path
        Directory where the outputs of the project are stored.
        Set to base_dir / 'Visual_Field_PCA'.
    ply : pathlib.Path
        Directory containing the .ply files.
        Set to vf / 'ply_files'.
    rendered_imgs_np : pathlib.Path
        Directory containing the rendered images in numpy format.
        Set to vf / 'rendered_images_numpy'.
    boundaries : pathlib.Path
        Directory containing the boundaries of the visual field.
        Set to vf / 'boundaries'.
    comparison_plots : pathlib.Path
        Directory containing the rendered vs predicted comparison plots.
        Set to vf / 'comparison_plots'.
    """
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).resolve()
        self.output_channels = self.base_dir / 'output_channels'
        self.vf = self.base_dir / 'Visual_Field_PCA'
        self.ply = self.vf / 'ply_files'
        self.rendered_imgs_np = self.vf / 'rendered_images_numpy'
        self.boundaries = self.vf / 'boundaries'
        self.comparison_plots = self.vf / 'comparison_plots'

    def create_directories(self):
        """Create all the required directories if they do not exist.

        Returns
        -------
        None
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                v.mkdir(parents=True, exist_ok=True)


DIRECTORIES = Directories(Path(__file__).resolve().parent)


class Colors:
    """Class containing the colors used in the project.

    `COLORS` is an instance of this class.

    Attributes
    ----------
    rendered : str
        Color used for the rendered plot. Set to '#2ca02c'.
    predicted : str
        Color used for the predicted plot. Set to '#d62728'.
    """
    def __init__(self):
        self.rendered = '#2ca02c'
        self.predicted = '#d62728'


COLORS = Colors()


class Numbers:
    """Class containing the numbers used in the project.

    `NUMBERS` is an instance of this class.

    Attributes
    ----------
    cm_to_mm : int
        Conversion factor from cm to mm. Equal to 10.
    digits_shape_key_ids : int
        Number of digits in the Blender shape key ID names. Equal to 3.
    num_phi_bins : int
        Number of phi bins for the VF boundaries. Set to 36000.
    num_ids : int
        Number of id parameters. Equal to 100.
    digits_num_ids : int
        Number of digits in id file names. Set to 2.
    num_rand : int
        Number of random faces. Set to 200.
    num_val : int
        Number of random validation faces. Set to 20.
    num_total_rand : int
        Total number of random faces. Equal to num_rand + num_val = 220.
    digits_num_rand : int
        Number of digits in the random face file names. Set to 3.
    np_seed : int
        Seed for numpy random number generator. Set to 42.
    mitsuba_seed : int
        Seed for Mitsuba random number generator. Set to 42.
    tf_seed : int
        Seed for TensorFlow random number generator. Set to 42.
    learning_rate : float
        Learning rate for the optimization of the VF boundaries. Set to 1e-6.
    patience : int
        Number of gradient steps after which to stop if there is no improvement
        in `val_loss`. Set to 50.
    training_patience : int
        Number of gradient steps after which to stop if there is no improvement
        in training loss. Set to 5.
    x_tick_every : int
        Frequency of x ticks in the theta vs phi plots of VF boundaries.
        Set to 60.
    y_tick_every : int
        Frequency of y ticks in the theta vs phi plots of VF boundaries.
        Set to 15.
    polar_axes_coords : tuple[float, float, float, float]
        Coordinates of the polar axes in the pyplot figure.
        Set to (0.1, 0.1, 0.8, 0.8).
    polar_r_ticks : numpy.ndarray
        Radial ticks for the polar plot. Set to [30, 60, 90].
    polar_y_ticks : numpy.ndarray
        Angular ticks for the polar plot.
        Set to [0, 45, 90, 135, 180, 225, 270, 315].
    polar_axes_coords_colorbar : tuple[float, float, float, float]
        Coordinates of the polar axes for the plot containing a colorbar.
        Set to (2.5 / 12.5, 1 / 11, 9 / 12.5, 9 / 11).
    figsize_colorbar : tuple[float, float]
        Figure size for the plot containing a colorbar. Set to (12.5/2,11/2).
    colorbar_axes_coords_colorbar : tuple[float, float, float, float]
        Coordinates of the colorbar for the plot containing a colorbar.
        Set to (1/12.5,1/11,.5/12.5,9/11).
    dpi : int
        Dots per inch for the plots. Set to 250.
    """
    def __init__(self):
        self.cm_to_mm = 10
        self.digits_shape_key_ids = 3
        self.num_phi_bins = 36000
        self.num_ids = 100
        self.digits_num_ids = len(str(self.num_ids - 1))
        self.num_rand = 200
        self.num_val = 20
        self.num_total_rand = self.num_rand + self.num_val
        self.digits_num_rand = len(str(self.num_total_rand - 1))
        self.np_seed = 42
        self.mitsuba_seed = 42
        self.tf_seed = 42
        self.learning_rate = 1e-6
        self.patience = 50
        self.training_patience = 5
        self.x_tick_every = 60
        self.y_tick_every = 15
        self.polar_axes_coords = (0.1, 0.1, 0.8, 0.8)
        self.polar_r_ticks = np.linspace(30, 90, 3, endpoint=True, dtype=int)
        self.polar_y_ticks = np.linspace(0, 360, 8, endpoint=False, dtype=int)
        self.polar_axes_coords_colorbar = (2.5 / 12.5, 1 / 11, 9 / 12.5, 9 / 11)
        self.figsize_colorbar = (12.5/2, 11/2)
        self.colorbar_axes_coords_colorbar = (1/12.5, 1/11, 0.5/12.5, 9/11)
        self.dpi = 250


NUMBERS = Numbers()
