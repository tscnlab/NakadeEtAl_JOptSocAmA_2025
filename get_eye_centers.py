import os
import json
import numpy as np
from tqdm import tqdm

from common_params import NUMBERS, DIRECTORIES
from naming import NAMING

__doc__ = """Get the eye centers from the PLY files and save them to JSON files."""

INDEX_R = 23728  # Index of the center of the right eye pupil in the PLY file
INDEX_L = 23577  # Index of the center of the left eye pupil in the PLY file
LEN_HEADER = 15  # Number of lines in the header of the PLY file

# Matrix to rotate the mesh in the PLY file to the desired orientation.
# After this, the head will be facing along -z.
# This is helpful because the right and up directions from the perspective
# of the head will be +x and +y directions respectively.
ROTATION = np.array([[0, -1, 0],
                     [1, 0, 0],
                     [0, 0, 1]])


def get_ply_vertex_coordinates(index_, lines_):
    """Get the coordinates of the vertex in the PLY file.

    Rotates the vertex by 90 degrees around ``+z`` and scales it by 10.
    This is done to match the orientation of the mesh in the PLY file with
    the one that Mitsuba will render.

    Parameters
    ----------
    index_ : int
        The index of the vertex in the PLY file.
    lines_ : list[str, ...]
        The lines in the ascii PLY file, obtained by file.readlines().

    Returns
    -------
    list[float, float, float]
        The coordinates of the vertex in the PLY file.
    """
    vertex = list(map(float, lines_[index_ + LEN_HEADER].split(' ')[:3]))
    vertex_rotated_scaled = NUMBERS.cm_to_mm * ROTATION @ np.array(vertex)
    return vertex_rotated_scaled.tolist()


def get_eye_centers(file_paths):
    """Get the eye centers from the PLY files.

    Rotates the eye centers by 90 degrees around ``+z`` and
    scales them by 10.
    This is done to match the orientation of the mesh in the PLY file with
    the one that Mitsuba will render.

    Parameters
    ----------
    file_paths : list[pathlib.Path, ...]
        The paths to the PLY files.

    Returns
    -------
    tuple[dict, dict]
        The eye centers of the right and left eyes.
    """
    eye_centers_right_dict = {}
    eye_centers_left_dict = {}
    for filepath in tqdm(file_paths, desc='Getting eye centers'):
        with open(filepath.resolve(), 'r') as file:
            lines = file.readlines()
        file_key = str(NAMING.replace_suffix(filepath.stem, 'ascii', ''))
        eye_centers_right_dict[file_key] = get_ply_vertex_coordinates(INDEX_R, lines)
        eye_centers_left_dict[file_key] = get_ply_vertex_coordinates(INDEX_L, lines)
        os.remove(filepath)  # PLY files in the ascii format are no longer needed
    return eye_centers_right_dict, eye_centers_left_dict


def main():
    """Get the eye centers from the PLY files and save them to JSON files.

    Searches for the PLY files in the ``Visual_Field_PCA/ply_files`` directory.
    Saves the eye centers to ``Visual_Field_PCA/eye_centers_right.json`` and
    ``Visual_Field_PCA/eye_centers_left.json``.
    """
    ls = DIRECTORIES.ply.glob(str(NAMING.asterisk.ascii.ply))
    ls = sorted(ls, key=str)
    eye_centers_right_dict, eye_centers_left_dict = get_eye_centers(ls)
    with open(DIRECTORIES.vf / NAMING.eye_centers.right.json, 'w') as file:
        json.dump(eye_centers_right_dict, file, indent=2)
    with open(DIRECTORIES.vf / NAMING.eye_centers.left.json, 'w') as file:
        json.dump(eye_centers_left_dict, file, indent=2)


if __name__ == '__main__':
    main()
