from common_params import NUMBERS
import os

__doc__ = """Module for simplifying the naming of files.

In order to avoid hard-coding, this module provides a way to create file names 
in a structured manner. The goal is to have an object :py:obj:`NAMING` that can 
be used as follows::

    NAMING.generic.theta_boundary.npy -> 'generic_neutral_mesh_thetas.npy'
    NAMING.id(1).pos.rendered.npy -> 'id_01_+1_rendered_images.npy'
    NAMING.random(20).comparison.png -> 'random_020_rendered_vs_predicted.png'

The reason for inheriting from :py:class:`os.PathLike` is that such objects 
can be used along with :py:class:`pathlib.Path` objects. For example:

>>> from pathlib import Path
>>> from naming import NAMING
>>> Path('path/to/directory') / NAMING.generic.theta_boundary.npy
Path('path/to/directory/generic_neutral_mesh_thetas.npy')
"""


class BaseNaming(os.PathLike):
    """Base class for naming paths.

    Since the :py:class:`NamingPathLike` class requires the
    :py:attr:`~NamingPathLike.start` attribute to have inherited from
    :py:class:`os.PathLike`, this class is used to create the
    :py:const:`BASE_NAMING` object that can be used as the starting point
    for creating file names.

    The :py:meth:`__fspath__` method returns an empty string.
    """
    def __fspath__(self):
        """Return an empty string.

        This method is required for classes inheriting from 
        :py:class:`os.PathLike`. It's supposed to return the string 
        representation of the path, which in this case should be an 
        empty string.
        """
        return ''


BASE_NAMING = BaseNaming()

FILE_TYPE_SUFFIXES = ['png', 'npy', 'ply', 'json', 'spd']


class NamingPathLike(os.PathLike):
    """Class for creating PathLike objects used by the :py:class:`Naming` class.

    A base class for file names represented as PathLike objects.

    Attributes
    ----------
    start : os.PathLike
        The starting point for the file name.
        Must have inherited from :py:class:`os.PathLike`, i.e., must have an
        :py:meth:`__fspath__` method.
    suffix : str
        The suffix to be added to the file name.
        This can either be an internal suffix or a file type suffix.
        For an internal suffix, the :py:attr:`separator` should preferably
        be '_', but can be changed if necessary.
        For a file type suffix, the :py:attr:`separator` must be '.'.
    separator : str, default '_'
        The separator between the :py:attr:`start` and :py:attr:`suffix`.
    next_suffixes : dict, optional
        A dictionary with the next suffixes that can be added to the file name.
        The keys are the names of the suffixes, and the values are the suffixes.

    Methods
    -------
    __getattr__(item)
        This method is defined to allow the creation of new
        :py:type:`NamingPathLike` objects by accessing attributes that are not
        defined in the class but are present in the :py:attr:`next_suffixes`
        dictionary.
    """
    def __init__(self, start, suffix, separator='_', next_suffixes=None):
        self.start = start
        self.suffix = suffix
        self.separator = separator
        if next_suffixes is None:
            self.next_suffixes = {}
        else:
            self.next_suffixes = next_suffixes

    def __getattr__(self, item):
        """Return new :py:class:`NamingPathLike` objects if the attribute is in
        :py:attr:`next_suffixes`.

        For example:

        >>> id_naming = NamingPathLike(BASE_NAMING, 'id', next_suffixes={'pos': '+'})
        >>> id_naming.__fspath__()
        'id'
        >>> id_naming.pos.__fspath__()
        'id_+'

        Parameters
        ----------
        item : str
            The name of the attribute to be accessed.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the :py:attr:`item`
            suffix. If :py:attr:`item` is in :py:const:`FILE_TYPE_SUFFIXES`,
            the :py:attr:`separator` is ``'.'``, otherwise it is ``'_'``.

        Raises
        ------
        AttributeError
            If the attribute is not in :py:attr:`next_suffixes`.
        """
        if item in FILE_TYPE_SUFFIXES:
            return NamingPathLike(self, item, separator='.')
        elif item in self.next_suffixes:
            return NamingPathLike(self, self.next_suffixes[item], self.separator)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def remove_suffix(self):
        """Remove the suffix from the file name.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the suffix removed.
            Returns the :py:attr:`start` attribute of the object on which this
            is called.
        """
        return self.start

    def add_suffix_anew(self, new_suffix):
        """Replace the suffix in the file name with a new one.

        Parameters
        ----------
        new_suffix : str
            The new suffix to be added to the file name.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the new suffix.
        """
        return NamingPathLike(self.start, new_suffix)

    def __str__(self):
        """Redefined to return :py:meth:`__fspath__`."""
        return self.__fspath__()

    def __fspath__(self):
        """Return the file name as a string.

        Returns ``start.__fspath__() + suffix`` if either are ``''``,
        otherwise returns ``start.__fspath__() + separator + suffix``.
        """
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
    'diff': 'diff',
    'projected_solid_angles': 'projected_solid_angles',
    'projected_solid_angles_percentages': 'projected_solid_angles_percentages',
}

ID_PM_SUFFIXES = {
    'pos': '+',
    'neg': '-'
}

ID_NUM_PM_SUFFIXES = {k: v + '1' for k, v in ID_PM_SUFFIXES.items()}

RANDOM_SUFFIXES = {
    'params': 'params',
    'theta_boundary': 'thetas',
}

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
    """Class for creating file names in a structured manner.

    :py:const:`NAMING` is an instance of this class.
    It is used to create file names as follows:

    >>> str(NAMING.generic.theta_boundary.npy)
    'generic_neutral_mesh_thetas.npy'
    >>> str(NAMING.id(1).pos.rendered.npy)
    'id_01_+1_rendered_images.npy'
    >>> str(NAMING.random(20).comparison.png)
    'random_020_rendered_vs_predicted.png'

    Parameters
    ----------
    base : os.PathLike
        The base name for the file names.
        Must have inherited from :py:class:`os.PathLike`, i.e., must have an
        :py:meth:`__fspath__` method.
        For the :py:const:`NAMING` instance, this is :py:const:`BASE_NAMING`.

    Attributes
    ----------
    base : os.PathLike
        The base name for the file names.
        Must have inherited from :py:class:`os.PathLike`, i.e., must have an
        :py:meth:`__fspath__` method.
    id_ : NamingPathLike
        Base PathLike object for the id file names.
    random_ : NamingPathLike
        Base PathLike object for the random file names.
    generic_ : NamingPathLike
        Base PathLike object for the generic neutral mesh file names.
    generic : NamingPathLike
        Base PathLike object for the generic neutral mesh file names with
        suffixes.
    optimization : NamingPathLike
        Base PathLike object for file names involved in
        Visual Field boundary optimizations.
    eye_centers : NamingPathLike
        Base PathLike object for the json files containing eye centers.
    asterisk : NamingPathLike
        Base PathLike object used to create file names with an asterisk.
        These can be used as regular expressions in the
        :py:meth:`pathlib.Path.glob` method.
    y_cie : NamingPathLike
        Base PathLike object for the y_CIE_1931 spectrum file name.
    phis : NamingPathLike
        Base PathLike object for the file storing regularly spaced phi values.

    Methods
    -------
    id(num)
        Create a new :py:class:`NamingPathLike` object for the id file names.
    random(num)
        Create a new :py:class:`NamingPathLike` object for the random head file
        names.
    shape_key(num)
        Create a shape key name.
    replace_suffix(old_name, old_suffix, new_suffix, suffixes_dict=None, next_suffixes_dict=None)
        Replace the suffix in the file name with a new one.
    make_pathlike(name, suffixes_dict=None)
        Create a new :py:class:`NamingPathLike` object from a file name string.
    add_suffix(name, suffix, suffixes_dict=None)
        Add a suffix to the file name.
    get_id_num_pm(name)
        Get the id number and the sign of the id parameter from the file name.
    get_random_num(name)
        Get the random face number from the file name.
    """
    def __init__(self, base):
        self.base = base
        self.id_ = NamingPathLike(self.base, 'id')
        self.random_ = NamingPathLike(self.base, 'random')
        self.generic_ = NamingPathLike(self.base, 'generic_neutral_mesh')
        self.generic = NamingPathLike(self.generic_, '', next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        # __dict__ needs to be updated to add the attributes dynamically
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
        for suffix_name, suffix in RANDOM_SUFFIXES.items():
            self.random.__dict__[suffix_name] = NamingPathLike(
                self.random_, suffix, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
            self.random.__dict__[VAL_SUFFIX].__dict__[suffix_name] = NamingPathLike(
                self.random.__dict__[VAL_SUFFIX], suffix, next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        self.asterisk = NamingPathLike(self.base, '*', next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        self.y_cie = NamingPathLike(self.base, 'y_CIE_1931')
        self.phis = NamingPathLike(self.base, 'phis')
        self.projected_solid_angles_hist = NamingPathLike(self.base, 'percent_change_projected_solid_angles_histogram')

    def id(self, num):
        """Create a new :py:class:`NamingPathLike` object for the id file names.

        When called with a number (say 4), this method creates a new
        :py:class:`NamingPathLike` object with ``__fspath__() == 'id_04'``.
        When the class is initialized, a few other attributes are added to this
        method, such as :py:attr:`pos`, :py:attr:`neg`, :py:attr:`optimized`,
        etc., so that file names that are related to the id parameters but do
        not depend explicitly on any one id number can be created (see examples).

        Parameters
        ----------
        num : int
            The id parameter number.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the id number.
            Also has attributes :py:attr:`pos` and :py:attr:`neg` for adding
            the ``+1`` and ``-1`` suffixes to the file names.
            :py:attr:`next_suffixes` is the :py:const:`ID_NUM_RANDOM_SUFFIXES`
            dict.

        Examples
        --------

        >>> NAMING.id.pos.theta_boundary.npy.__fspath__()
        'id_+_thetas.npy'
        >>> NAMING.id(20).hemispherical_vf.npy.__fspath__()
        'id_20_hemispherical_vf.npy'
        >>> NAMING.id(20).diff.png.__fspath__()
        'id_20_diff.png'
        """
        id_num_pathlike_temp = NamingPathLike(self.id_,
                                              f'{num:0{NUMBERS.digits_num_ids}d}',
                                              next_suffixes=ID_NUM_PM_SUFFIXES | ID_NUM_RANDOM_SUFFIXES)
        for k, v in ID_NUM_PM_SUFFIXES.items():
            id_num_pathlike_temp.__dict__[k] = NamingPathLike(id_num_pathlike_temp, v,
                                                              next_suffixes=ID_NUM_RANDOM_SUFFIXES)
        return id_num_pathlike_temp

    def random(self, num):
        """Create a new :py:class:`NamingPathLike` object for the random
        parameter files.

        When called with a number (say 4), this method creates a new
        :py:class:`NamingPathLike` object with ``__fspath__() = 'random_004'``.
        When the class is initialized, a few other attributes are added to this
        method, such as :py:attr:`params`, :py:attr:`theta_boundary`, etc.,
        so that file names that are related to the random faces but do not
        depend explicitly on any one of them can be created.

        Parameters
        ----------
        num : int
            The random face number.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the random face number.
            Also has attributes :py:attr:`params` and :py:attr:`theta_boundary`
            for adding the ``params`` and ``thetas`` suffixes to the file names.
            :py:attr:`next_suffixes` is the :py:const:`ID_NUM_RANDOM_SUFFIXES`
            dict.

        Examples
        --------

        >>> NAMING.random.val.theta_boundary.npy.__fspath__()
        'random_val_thetas.npy'
        >>> NAMING.random(20).comparison.png.__fspath__()
        'random_020_rendered_vs_predicted.png'
        >>> NAMING.random(20).hemispherical_vf.npy.__fspath__()
        'random_020_hemispherical_vf.npy'
        """
        return NamingPathLike(self.random_, f'{num:0{NUMBERS.digits_num_rand}d}',
                              next_suffixes=ID_NUM_RANDOM_SUFFIXES)

    @staticmethod
    def shape_key(num):
        """Create a shape key name.

        Used in Blender to get the shape key from the parameter number.

        Parameters
        ----------
        num : int
            The id parameter number.

        Returns
        -------
        str
            The shape key name.

        Examples
        --------

        >>> NAMING.shape_key(4)
        'identity004'
        """
        return f'identity{num:0{NUMBERS.digits_shape_key_ids}d}'

    def replace_suffix(self, old_name, old_suffix, new_suffix,
                       suffixes_dict=None, next_suffixes_dict=None):
        """Replace the suffix in the file name with a new one.

        This takes in an :py:attr:`old_name` string and returns a new
        :py:class:`NamingPathLike` object with the :py:attr:`old_suffix`
        replaced by the :py:attr:`new_suffix`.

        Parameters
        ----------
        old_name : str
            The old file stem or file name.
        old_suffix : str
            The name of the old suffix to be replaced.
        new_suffix : str
            The name of the new suffix to be added.
        suffixes_dict : dict, default ID_NUM_RANDOM_SUFFIXES
            The dictionary with the suffix names and suffixes.
        next_suffixes_dict : dict, default ID_NUM_RANDOM_SUFFIXES
            The dictionary with the next suffix names and suffixes that can
            be added to the file name.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the
            :py:attr:`old_suffix` in :py:attr:`old_name` replaced by the
            :py:attr:`new_suffix`.
        """
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
        """Create a new :py:class:`NamingPathLike` object from a file name
        string.

        Used to convert a file name or file stem string to a
        :py:class:`NamingPathLike` object. This is useful when the file name is
        found by searching a directory using either :py:meth:`os.listdir` or
        :py:meth:`pathlib.Path.glob`.

        Parameters
        ----------
        name : str
            The file name or file stem.
        suffixes_dict : dict, default ID_NUM_RANDOM_SUFFIXES
            The dictionary with the next suffix names and suffixes.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with the given file name or
            file stem.
        """
        if suffixes_dict is None:
            suffixes_dict = ID_NUM_RANDOM_SUFFIXES
        return NamingPathLike(self.base, name, next_suffixes=suffixes_dict)

    def add_suffix(self, name, suffix, suffixes_dict=None):
        """Add a suffix to the file name.

        :py:attr:`name` is a file name or file stem string and not a
        :py:class:`NamingPathLike`
        object.

        Parameters
        ----------
        name : str
            The file name or file stem.
        suffix : str
            The name of the suffix to be added to the file name.
        suffixes_dict : dict, default ID_NUM_RANDOM_SUFFIXES
            The dictionary with the suffix names and suffixes.

        Returns
        -------
        NamingPathLike
            A new :py:class:`NamingPathLike` object with
            ``suffixes_dict[suffix]`` added to the file name.
        """
        return self.replace_suffix(name, '', suffix, suffixes_dict)

    @staticmethod
    def get_id_num_pm(name):
        """Get the id number and the sign of the id parameter from the file name.

        Parameters
        ----------
        name : str
            The file name or file stem.

        Returns
        -------
        tuple[int, str]
            The id number and the sign of the id parameter.

        Examples
        --------

        >>> NAMING.get_id_num_pm('id_01_+1_rendered_images.npy')
        (1, '+')
        """
        name_stem = name.split('.')[0]
        id_num = int(name_stem.split('_')[1])
        pm = name_stem.split('_')[2][0]
        return id_num, pm

    @staticmethod
    def get_random_num(name):
        """Get the random face number from the file name.

        Parameters
        ----------
        name : str
            The file name or file stem.

        Returns
        -------
        int
            The random face number.

        Examples
        --------

        >>> NAMING.get_random_num('random_020_rendered_vs_predicted.png')
        20
        """
        name_stem = name.split('.')[0]
        return int(name_stem.split('_')[1])


NAMING = Naming(BASE_NAMING)
