import os
import json
import numpy as np

from common_params import NUMBERS, DIRECTORIES, NAMING

INDEX_R = 23728
INDEX_L = 23577
LEN_HEADER = 15

ROTATION = np.array([[0, -1, 0],
                     [1, 0, 0],
                     [0, 0, 1]])


def get_ply_vertex_coordinates(index_, lines_):
    vertex = list(map(float, lines_[index_ + LEN_HEADER].split(' ')[:3]))
    vertex_rotated_scaled = NUMBERS.cm_to_mm * ROTATION @ np.array(vertex)
    return vertex_rotated_scaled.tolist()


def get_eye_centers(file_paths):
    eye_centers_right_dict = {}
    eye_centers_left_dict = {}
    for filepath in file_paths:
        with open(filepath.resolve(), 'r') as file:
            lines = file.readlines()
        file_key = NAMING.ply.get_stem(filepath.name)
        eye_centers_right_dict[file_key] = get_ply_vertex_coordinates(INDEX_R, lines)
        eye_centers_left_dict[file_key] = get_ply_vertex_coordinates(INDEX_L, lines)
        os.remove(filepath)
    return eye_centers_right_dict, eye_centers_left_dict


def main():
    ls = DIRECTORIES.ply.glob(NAMING.ply.add_suffix('*', 'ascii'))
    ls = sorted(ls, key=str)
    eye_centers_right_dict, eye_centers_left_dict = get_eye_centers(ls)
    with open(DIRECTORIES.vf / NAMING.json.eye_centers_right, 'w') as file:
        json.dump(eye_centers_right_dict, file, indent=2)
    with open(DIRECTORIES.vf / NAMING.json.eye_centers_left, 'w') as file:
        json.dump(eye_centers_left_dict, file, indent=2)


def main_test(num_plys=2):
    ls = DIRECTORIES.ply.glob(NAMING.ply.add_suffix('*', 'ascii'))
    ls = sorted(ls, key=str)
    eye_centers_right_dict, eye_centers_left_dict = get_eye_centers(ls[:num_plys])
    print(eye_centers_right_dict)
    print(eye_centers_left_dict)
    with open(DIRECTORIES.vf / NAMING.json.eye_centers_right, 'w') as file:
        json.dump(eye_centers_right_dict, file, indent=2)
    with open(DIRECTORIES.vf / NAMING.json.eye_centers_left, 'w') as file:
        json.dump(eye_centers_left_dict, file, indent=2)


if __name__ == '__main__':
    main()
    # main_test(8)
