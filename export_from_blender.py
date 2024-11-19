import bpy
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from common_params import DIRECTORIES, NUMBERS  #, NAMING
from naming import NAMING

__doc__ = """Export the generic neutral, id+-, and random meshes to PLY files."""


def change_shape_key_slider_limits(shape_keys_, shape_key_name_, min_=-1, max_=1):
    """ Change the slider limits for one shape key.

    Refer to `change_all_shape_key_slider_limits` for more details.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    shape_key_name_ : str
        The name of the shape key.
    min_ : float, default -1
        The lower limit.
    max_ : float, default 1
        The upper limit.

    Returns
    -------
    None
        Changes the slider limits in place.
    """
    shape_keys_[shape_key_name_].slider_min = min_
    shape_keys_[shape_key_name_].slider_max = max_


def change_all_shape_key_slider_limits(shape_keys_, min_=-1, max_=1):
    """Change the slider limits of all shape keys.

    The blendshape sliders in Blender are limited to the range [0, 1] by default.
    This function changes the limits to the specified range (-1 to 1 by default).

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    min_ : float, default -1
        The lower limit.
    max_ : float, default 1
        The upper limit.

    Returns
    -------
    None
        Changes the slider limits in place.
    """
    for id_num in range(NUMBERS.num_ids):
        shape_key_name = NAMING.shape_key(id_num)
        change_shape_key_slider_limits(shape_keys_, shape_key_name, min_, max_)


def set_shape_key_value(shape_keys_, shape_key_name_, value_):
    """Set the value of a shape key.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    shape_key_name_ : str
        The name of the shape key.
    value_ : float
        The value to set the shape key to.

    Returns
    -------
    None
        Sets the shape key value in place.
    """
    shape_keys_[shape_key_name_].value = value_


def set_all_shape_key_values(shape_keys_, values_):
    """Set the values of all shape keys.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    values_ : numpy.ndarray
        An array of shape key values.

    Returns
    -------
    None
        Sets the shape key values in place.
    """
    for id_num, value in enumerate(values_):
        shape_key_name = NAMING.shape_key(id_num)
        set_shape_key_value(shape_keys_, shape_key_name, value)


def id_pm_shape_key_values(id_num_, pm1_, num_ids_=NUMBERS.num_ids):
    """Get an array with only one non-zero value for shape keys.

    Parameters
    ----------
    id_num_ : int
        The index of the shape key which is to be non-zero.
    pm1_ : int
        Either 1 or -1.
    num_ids_ : int, default NUMBERS.num_ids = 100
        The total number of shape keys.

    Returns
    -------
    numpy.ndarray
        An array of size `num_ids_` with the `id_num_`th value equal
        to `pm1_` and others 0.
    """
    values_temp = np.zeros(num_ids_)
    values_temp[id_num_] = pm1_
    return values_temp


def random_shape_key_values(rng_, size_, min_=-1, max_=1):
    """Get uniform random values.

    To be used to generate random faces using shape keys.

    Parameters
    ----------
    rng_ : numpy.random.Generator
        A random number generator.
    size_ : int
        The number of shape keys (also the number of random values generated).
    min_ : float, default -1
        The lower limit for the uniform distribution
        (lower limit of the shape key slider).
    max_ : float, default 1
        The upper limit for the uniform distribution
        (upper limit of the shape key slider).

    Returns
    -------
    numpy.ndarray
        An array of uniform random values between `min_` and `max_`.
    """
    return rng_.uniform(min_, max_, size=size_)


def export_ply(filepath_, shape_keys_, shape_key_values_, use_selection_=True, use_ascii_=True):
    """Export the mesh to a PLY file.

    Parameters
    ----------
    filepath_ : pathlib.Path
        The path to the PLY file.
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    shape_key_values_ : numpy.ndarray
        An array of shape key values.
    use_selection_ : bool, default True
        Whether to export only the selected mesh.
    use_ascii_ : bool, default True
        Whether to use ASCII encoding.

    Returns
    -------
    None
        Exports the mesh to the PLY file.
    """
    set_all_shape_key_values(shape_keys_, shape_key_values_)
    bpy.ops.export_mesh.ply(filepath=str(filepath_.resolve()), use_selection=use_selection_,
                            use_ascii=use_ascii_)


def export_generic_neutral_mesh(shape_keys_):
    """Export the mesh with all shape keys 0.

    Two PLY files are exported: one in ASCII format and the other in binary format.
    The ASCII format is required to get the coordinates of the eye centers.
    The binary format is used with mitsuba 3 as it's faster to load.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.

    Returns
    -------
    None
        Exports the mesh to the generic neutral PLY file.
    """
    export_ply(DIRECTORIES.ply / NAMING.generic.ply, shape_keys_,
               np.zeros(NUMBERS.num_ids), use_ascii_=False)
    export_ply(DIRECTORIES.ply / NAMING.generic.ascii.ply, shape_keys_,
               np.zeros(NUMBERS.num_ids), use_ascii_=True)


def export_id_pm_meshes(shape_keys_, num_ids_=NUMBERS.num_ids):
    """Export meshes with individual shape keys set to 1 and -1.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    num_ids_ : int, default NUMBERS.num_ids = 100
        The total number of shape keys.

    Returns
    -------
    None
        Exports the meshes to the PLY files.
    """
    for id_num in range(num_ids_):
        for pm, pos_neg in zip([1, -1], ['pos', 'neg']):
            shape_key_values = id_pm_shape_key_values(id_num, pm)
            export_ply(DIRECTORIES.ply / NAMING.id(id_num).get(pos_neg).ply,
                       shape_keys_, shape_key_values, use_ascii_=False)
            export_ply(DIRECTORIES.ply / NAMING.id(id_num).get(pos_neg).ascii.ply,
                       shape_keys_, shape_key_values, use_ascii_=True)


def export_random_meshes(shape_keys_, rng_, num_random_=NUMBERS.num_rand,
                         num_validation_=NUMBERS.num_val):
    """Export random meshes.

    The random meshes are generated by setting the shape keys to random values.
    The meshes are exported to PLY files and
    the random parameters are saved to numpy files.

    Parameters
    ----------
    shape_keys_ : bpy.types.bpy_prop_collection
        A dict like object containing the shape keys.
    rng_ : numpy.random.Generator
        A random number generator.
    num_random_ : int, default NUMBERS.num_rand = 200
        The number of random meshes to generate.
    num_validation_ : int, default NUMBERS.num_val = 20
        The number of validation meshes to generate.

    Returns
    -------
    None
        Exports the meshes to the PLY files and saves the random parameters
        to numpy files.
    """
    random_params = random_shape_key_values(rng_, (num_random_ + num_validation_, NUMBERS.num_ids), -1, 1)
    for j in range(num_random_ + num_validation_):
        set_all_shape_key_values(shape_keys_, random_params[j])
        export_ply(
            DIRECTORIES.ply / NAMING.random(j).ply,
            shape_keys_, random_params[j], use_ascii_=False)
        export_ply(
            DIRECTORIES.ply / NAMING.random(j).ascii.ply,
            shape_keys_, random_params[j], use_ascii_=True)
    np.save(DIRECTORIES.vf / NAMING.random.params.npy, random_params[:NUMBERS.num_rand])
    np.save(DIRECTORIES.vf / NAMING.random.val.params.npy, random_params[NUMBERS.num_rand:])


def main():
    """Export generic neutral, id+-, and random meshes.

    Changes all shape key limits to -1 to 1.
    Exports the generic neutral mesh to a PLY file.
    Exports the meshes with individual id parameters set to -1 and 1 to PLY files.
    Exports the random meshes to PLY files.
    Saves the random parameters to numpy files.
    """
    DIRECTORIES.create_directories()
    ict_face_model = bpy.data.objects['ICTFaceModel']
    ict_face_model.select_set(True)
    shape_keys = ict_face_model.data.shape_keys.key_blocks
    change_all_shape_key_slider_limits(shape_keys)
    export_generic_neutral_mesh(shape_keys)
    export_id_pm_meshes(shape_keys)
    rng = np.random.default_rng(NUMBERS.np_seed)
    export_random_meshes(shape_keys, rng)


if __name__ == '__main__':
    main()
