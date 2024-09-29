import bpy
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from common_params import DIRECTORIES, NUMBERS, NAMING


def change_shape_key_slider_limits(shape_keys_, shape_key_name_, min_=-1, max_=1):
    shape_keys_[shape_key_name_].slider_min = min_
    shape_keys_[shape_key_name_].slider_max = max_


def change_all_shape_key_slider_limits(shape_keys_, min_=-1, max_=1):
    for id_num in range(NUMBERS.num_ids):
        shape_key_name = NAMING.shape_key(id_num=id_num)
        change_shape_key_slider_limits(shape_keys_, shape_key_name, min_, max_)


def set_shape_key_value(shape_keys_, shape_key_name_, value_):
    shape_keys_[shape_key_name_].value = value_


def set_all_shape_key_values(shape_keys_, values_):
    for id_num, value in enumerate(values_):
        shape_key_name = NAMING.shape_key(id_num=id_num)
        set_shape_key_value(shape_keys_, shape_key_name, value)


def id_pm_shape_key_values(id_num_, pm1_):
    values_temp = np.zeros(NUMBERS.num_ids)
    values_temp[id_num_] = pm1_
    return values_temp


def random_shape_key_values(rng_, size_, min_=-1, max_=1):
    return rng_.uniform(min_, max_, size=size_)


def export_ply(filepath_, shape_keys_, shape_key_values_, use_selection_=True, use_ascii_=True):
    set_all_shape_key_values(shape_keys_, shape_key_values_)
    bpy.ops.export_mesh.ply(filepath=str(filepath_.resolve()), use_selection=use_selection_,
                            use_ascii=use_ascii_)


def export_generic_neutral_mesh(shape_keys_):
    export_ply(DIRECTORIES.ply / NAMING.ply.generic_suffix(''), shape_keys_,
               np.zeros(NUMBERS.num_ids), use_ascii_=False)
    export_ply(DIRECTORIES.ply / NAMING.ply.generic_suffix('ascii'), shape_keys_,
               np.zeros(NUMBERS.num_ids), use_ascii_=True)


def export_id_pm_meshes(shape_keys_, num_ids_=NUMBERS.num_ids):
    for id_num in range(num_ids_):
        for pm in [1, -1]:
            shape_key_values = id_pm_shape_key_values(id_num, pm)
            export_ply(DIRECTORIES.ply / NAMING.ply.id_pm_suffix(id_num=id_num, pm=pm, suffix_key=''),
                       shape_keys_, shape_key_values, use_ascii_=False)
            export_ply(DIRECTORIES.ply / NAMING.ply.id_pm_suffix(id_num=id_num, pm=pm, suffix_key='ascii'),
                       shape_keys_, shape_key_values)


def export_random_meshes(shape_keys_, rng_, num_random_=NUMBERS.num_rand, num_validation_=NUMBERS.num_val):
    random_params = random_shape_key_values(rng_, (num_random_ + num_validation_, NUMBERS.num_ids), -1, 1)
    for j in range(num_random_ + num_validation_):
        set_all_shape_key_values(shape_keys_, random_params[j])
        export_ply(
            DIRECTORIES.ply / NAMING.ply.random_suffix(j, suffix_key=''),
            shape_keys_, random_params[j], use_ascii_=False)
        export_ply(
            DIRECTORIES.ply / NAMING.ply.random_suffix(j, suffix_key='ascii'),
            shape_keys_, random_params[j], use_ascii_=True)
    np.save(DIRECTORIES.vf / NAMING.npy.random_params, random_params[:NUMBERS.num_rand])
    np.save(DIRECTORIES.vf / NAMING.npy.random_val_params, random_params[NUMBERS.num_rand:])


def main():
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
