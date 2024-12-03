import numpy as np
from pathlib import Path


def save_npy_files(paths_arrays_dict):
    for path_, array_ in paths_arrays_dict.items():
        np.save(path_, array_)


class CameraParameters:
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
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).resolve()
        self.output_channels = self.base_dir / 'output_channels'
        self.vf = self.base_dir / 'Visual_Field_PCA'
        self.ply = self.vf / 'ply_files'
        self.rendered_imgs_np = self.vf / 'rendered_images_numpy'
        self.boundaries = self.vf / 'boundaries'
        self.comparison_plots = self.vf / 'comparison_plots'
    def create_directories(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                v.mkdir(parents=True, exist_ok=True)


DIRECTORIES = Directories(Path(__file__).resolve().parent)


class Colors:
    def __init__(self):
        self.rendered = '#2ca02c'
        self.predicted = '#d62728'


COLORS = Colors()


class Numbers:
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
        self.figsize_colorbar = (12.5/2,11/2)
        self.colorbar_axes_coords_colorbar = (1/12.5,1/11,.5/12.5,9/11)
        self.dpi = 250


NUMBERS = Numbers()
