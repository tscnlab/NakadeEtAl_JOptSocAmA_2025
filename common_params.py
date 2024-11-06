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
        self.num_phi_bins = 36000
        self.num_ids = 100
        self.num_rand = 200
        self.num_val = 20
        self.num_total_rand = self.num_rand + self.num_val
        self.digits_num_rand = len(str(self.num_total_rand - 1))
        self.np_seed = 42
        self.mitsuba_seed = 42
        self.tf_seed = 42
        self.learn_rate = 1e-6
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


class BaseNaming:
    def __init__(self):
        self.generic = 'generic_neutral_mesh'
        self.id_pm_ = 'id_{id_num:02d}_{pm:+}'
        self.random_ = 'random_{:0{digits_num_rand}}'
        self.id_start = 'id'
        self.random_start = 'random'
    def id_pm(self, id_num, pm):
        return self.id_pm_.format(id_num=id_num, pm=pm)
    def random(self, num):
        return self.random_.format(num, digits_num_rand=NUMBERS.digits_num_rand)
    @staticmethod
    def get_id_pm(name):
        return int(name[2:4]), name.split('_')[1][0]
    @staticmethod
    def get_random_num(name):
        return int(name.split('_')[1].split('.')[0])


BASE_NAMING = BaseNaming()


class SubNaming:
    def __init__(self, base):
        self.file_type = ''
        self.suffixes = {
            '': '',
            'ascii': '_ascii',
            'rendered': '_rendered_images',
            'theta_boundary': '_t_mids',
            'hemispherical_vf': '_hemispherical_vf',
            'comparison': '_rendered_vs_predicted',
            'diff': '_diff',
        }
        self.base = base
    def add_file_type(self, name):
        return name + self.file_type
    def add_suffix(self, stem, suffix):
        return self.add_file_type(stem + self.suffixes[suffix])
    def add_suffix_anew(self, name, suffix):
        return self.add_suffix(self.get_stem(name), suffix)
    def generic_suffix(self, suffix_key):
        return self.add_suffix(self.base.generic, suffix_key)
    def id_pm_suffix(self, id_num, pm, suffix_key):
        return self.add_suffix(self.base.id_pm(id_num, pm), suffix_key)
    def random_suffix(self, num, suffix_key):
        return self.add_suffix(self.base.random(num), suffix_key)
    def remove_suffix(self, name, suffix):
        return name.replace(self.suffixes[suffix], self.suffixes[''])
    def remove_all_suffixes(self, name):
        name_temp = name
        for suffix in self.suffixes:
            name_temp = self.remove_suffix(name_temp, suffix)
        return name_temp
    def get_stem(self, name):
        name_temp = self.remove_all_suffixes(name)
        return name_temp.replace(self.file_type, '')


class PlyNaming(SubNaming):
    def __init__(self, base_naming):
        super().__init__(base_naming)
        self.file_type = '.ply'
        self.generic = self.add_suffix(self.base.generic, '')
        self.generic_ascii = self.add_suffix(self.base.generic, 'ascii')


class NpyNaming(SubNaming):
    def __init__(self, base_naming):
        super().__init__(base_naming)
        self.file_type = '.npy'
        self.phis = self.add_file_type('p_mids')
        self.random_params = self.add_file_type('random_params')
        self.random_val_params = self.add_file_type('random_val_params')
        self.id_p_thetas = self.add_file_type('id_+_t_mids')
        self.id_m_thetas = self.add_file_type('id_-_t_mids')
        self.random_thetas = self.add_file_type('random_t_mids')
        self.random_val_thetas = self.add_file_type('random_val_t_mids')
        self.generic_thetas = self.add_file_type('generic_t_mids')
        self.lowest_val_vf = self.add_file_type('optimized_ids_vf')
        self.lowest_val_g = self.add_file_type('optimized_generic_vf')
        self.losses = self.add_file_type('optimization_losses')
        self.losses_val = self.add_file_type('optimization_losses_validation')
        self.predictions = self.add_file_type('predictions')


class PngNaming(SubNaming):
    def __init__(self, base_naming):
        super().__init__(base_naming)
        self.file_type = '.png'
        self.hemispherical_vf = self.add_file_type('hemispherical_vf')
        self.id_diff_ = 'id_{:02d}_diff'
    def id_diff(self, id_num):
        return self.add_file_type(self.id_diff_.format(id_num))


class JsonNaming(SubNaming):
    def __init__(self, base_naming):
        super().__init__(base_naming)
        self.file_type = '.json'
        self.eye_centers_right = self.add_file_type('eye_centers_right')
        self.eye_centers_left = self.add_file_type('eye_centers_left')


class SpdNaming(SubNaming):
    def __init__(self, base_naming):
        super().__init__(base_naming)
        self.file_type = '.spd'
        self.film_spd_stem = 'y_CIE_1931'
        self.film_spd = self.add_file_type(self.film_spd_stem)


class Naming:
    def __init__(self):
        self.shape_key_ = 'identity{id_num:03d}'
        self.base = BASE_NAMING
        self.ply = PlyNaming(self.base)
        self.npy = NpyNaming(self.base)
        self.json = JsonNaming(self.base)
        self.spd = SpdNaming(self.base)
        self.png = PngNaming(self.base)
    def shape_key(self, id_num):
        return self.shape_key_.format(id_num=id_num)


NAMING = Naming()
