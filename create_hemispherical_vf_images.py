import numpy as np

from hemispherical_interpolator import ImageSet
from common_params import DIRECTORIES
from naming import NAMING


def create_hemispherical_vf_image(rendered_images_file_path):
    array = np.load(rendered_images_file_path)
    array /= np.max(array)
    img_set = ImageSet(array)
    out_front = img_set.get_hemispherical_output(weightings=['solid_angles', 'cosine'])
    return out_front


def main():
    rendered_images_file_paths = DIRECTORIES.rendered_imgs_np.glob(
        str(NAMING.asterisk.rendered.npy))
    rendered_images_file_paths = list(filter(lambda x: not x.name.startswith(str(NAMING.random_)),
                                             rendered_images_file_paths))
    for file_path in sorted(rendered_images_file_paths, key=str):
        hemispherical_vf_image = create_hemispherical_vf_image(file_path)
        hemispherical_vf_image = np.clip(hemispherical_vf_image, 0, 1)
        np.save(DIRECTORIES.boundaries / NAMING.replace_suffix(file_path.stem, 'rendered', 'hemispherical_vf').npy,
                hemispherical_vf_image)

if __name__ == '__main__':
    main()
