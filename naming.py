from common_params import NUMBERS
import os


class BaseNaming(os.PathLike):
    def __fspath__(self):
        return ''


BASE_NAMING = BaseNaming()

FILE_TYPE_SUFFIXES = ['png', 'npy', 'ply', 'json', 'spd']


class NamingPathLike(os.PathLike):
    def __init__(self, start, suffix, separator='_', next_suffixes=None):
        self.start = start
        self.suffix = suffix
        self.separator = separator
        if next_suffixes is None:
            self.next_suffixes = {}
        else:
            self.next_suffixes = next_suffixes

    def __getattr__(self, item):
        if item in FILE_TYPE_SUFFIXES:
            return NamingPathLike(self, item, separator='.')
        elif item in self.next_suffixes:
            return NamingPathLike(self, self.next_suffixes[item])
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def get(self, name, default=None):
        return getattr(self, name, default)

    def remove_suffix(self):
        return self.start

    def add_suffix_anew(self, suffix):
        return NamingPathLike(self.start, suffix)

    def __str__(self):
        return self.__fspath__()

    def __fspath__(self):
        if self.start.__fspath__() != '' and self.suffix != '':
            return self.start.__fspath__() + self.separator + self.suffix
        else:
            return self.start.__fspath__() + self.suffix


ID_NUM_RANDOM_SUFFIXES = {
    '': '',
    'ascii': 'ascii',
    'rendered': 'rendered_images',
    'theta_boundary': 'thetas',
    'hemispherical_vf': 'hemispherical_vf',
    'comparison': 'rendered_vs_predicted',
    'diff': 'diff'
}

ID_PM_SUFFIXES = {
    'pos': '+',
    'neg': '-'
}

ID_NUM_PM_SUFFIXES = {k: v + '1' for k, v in ID_PM_SUFFIXES.items()}

RANDOM_SUFFIXES = {'params', 'thetas'}

VAL_SUFFIX = 'val'

EYE_CENTER_SUFFIXES = {'left', 'right'}

OPTIMIZED_SUFFIXES = {
    'optimized': 'optimized'
}

OPTIMIZATION_SUFFIXES = {
    'losses': 'losses',
    'predictions': 'predictions'
}


class Naming:
    def __init__(self, base):
        self.base = base
        self.id_ = NamingPathLike(self.base, 'id')
        self.random_ = NamingPathLike(self.base, 'random')
        self.generic_ = NamingPathLike(self.base, 'generic_neutral_mesh')
        self.generic = NamingPathLike(self.generic_, '', next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        for k, v in ID_PM_SUFFIXES.items():
            self.id.__dict__[k] = NamingPathLike(self.id_, v, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        for k, v in OPTIMIZED_SUFFIXES.items():
            self.id.__dict__[k] = NamingPathLike(self.id_, v, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
            self.generic.__dict__[k] = NamingPathLike(self.generic_, v, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        self.optimization = NamingPathLike(self.base, 'optimization')
        for k, v in OPTIMIZATION_SUFFIXES.items():
            self.optimization.__dict__[k] = NamingPathLike(self.optimization, v)
            self.optimization.__dict__[k].__dict__[VAL_SUFFIX] = NamingPathLike(self.optimization.__dict__[k], VAL_SUFFIX)
        self.eye_centers = NamingPathLike(self.base, 'eye_centers')
        for eye in EYE_CENTER_SUFFIXES:
            self.eye_centers.__dict__[eye] = NamingPathLike(self.eye_centers, eye)
        self.random.__dict__[VAL_SUFFIX] = NamingPathLike(self.random_, VAL_SUFFIX)
        for suffix in RANDOM_SUFFIXES | set(ID_NUM_RANDOM_SUFFIXES.keys()):
            self.random.__dict__[suffix] = NamingPathLike(
                self.random_, suffix, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
            self.random.__dict__[VAL_SUFFIX].__dict__[suffix] = NamingPathLike(
                self.random.__dict__[VAL_SUFFIX], suffix, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        self.asterisk = NamingPathLike(self.base, '*', next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        self.y_cie = NamingPathLike(self.base, 'y_CIE_1931')
        self.phis = NamingPathLike(self.base, 'phis')

    def id(self, num):
        id_num_pathlike_temp = NamingPathLike(self.id_,
                                              f'{num:0{NUMBERS.digits_num_ids}d}',
                                              next_suffixes=ID_NUM_PM_SUFFIXES | ID_NUM_RANDOM_SUFFIXES)
        for k, v in ID_NUM_PM_SUFFIXES.items():
            id_num_pathlike_temp.__dict__[k] = NamingPathLike(id_num_pathlike_temp, v,
                                                              next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        return id_num_pathlike_temp

    def random(self, num):
        return NamingPathLike(self.random_, f'{num:0{NUMBERS.digits_num_rand}d}', next_suffixes=ID_NUM_RANDOM_SUFFIXES)

    @staticmethod
    def shape_key(num):
        return f'identity{num:0{NUMBERS.digits_shape_key_ids}d}'

    def replace_suffix(self, old_name, old_suffix, new_suffix, suffixes_dict=None, next_suffixes_dict=None):
        if suffixes_dict is None:
            suffixes_dict = ID_NUM_RANDOM_SUFFIXES
        if next_suffixes_dict is None:
            next_suffixes_dict = ID_NUM_RANDOM_SUFFIXES
        if old_suffix != '':
            return NamingPathLike(
                NamingPathLike(self.base, old_name.replace('_' + suffixes_dict[old_suffix], '')),
                suffixes_dict[new_suffix], next_suffixes=next_suffixes_dict)
        else:
            if new_suffix == '':
                return NamingPathLike(self.base, old_name)
            else:
                return NamingPathLike(NamingPathLike(self.base, old_name),
                                      suffixes_dict[new_suffix], next_suffixes=next_suffixes_dict)

    def make_pathlike(self, name, suffixes_dict=None):
        if suffixes_dict is None:
            suffixes_dict = ID_NUM_RANDOM_SUFFIXES
        return NamingPathLike(self.base, name, next_suffixes=suffixes_dict)

    def add_suffix(self, name, suffix, suffixes_dict=None):
        return self.replace_suffix(name, '', suffix, suffixes_dict)

    @staticmethod
    def get_id_num_pm(name):
        name_stem = name.split('.')[0]
        id_num = int(name_stem.split('_')[1])
        pm = name_stem.split('_')[2][0]
        return id_num, pm

    @staticmethod
    def get_random_num(name):
        name_stem = name.split('.')[0]
        return int(name_stem.split('_')[1])


NAMING = Naming(BASE_NAMING)
