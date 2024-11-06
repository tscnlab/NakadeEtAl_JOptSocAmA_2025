import numpy as np
from scipy.interpolate import LinearNDInterpolator
import utils_img
from common_params import CAMERA


class ImageSetWeights:
    def __init__(self, image_shape_, num_images_, fov_, fov_axis_, camera_directions_, front_):
        self.num_images = num_images_
        x_min, x_max, x_mid, y_min, y_max, y_mid = utils_img.xy_min_max_mid(image_shape_, fov_, fov_axis_)
        solid_angles_1_image = utils_img.solid_angle_limits_x_y(x_min, x_max, y_min, y_max)
        self.solid_angles = np.repeat(solid_angles_1_image[np.newaxis], self.num_images, axis=0)
        self.pixel_coordinates = np.zeros((self.num_images, *image_shape_, 3))
        pixel_coordinates_1_image = utils_img.normalize(np.stack([x_mid, y_mid, -np.ones(image_shape_)], axis=2))
        for i in range(self.num_images):
            self.pixel_coordinates[i] = utils_img.camera_to_world_coordinates(
                pixel_coordinates_1_image,
                camera_directions_[i]['camera_direction'],
                camera_directions_[i]['up'])
        self.cosine_weights = np.clip(np.tensordot(self.pixel_coordinates, front_, axes=(-1, -1)),
                                      a_min=0, a_max=1)
        self.weights = self.solid_angles * self.cosine_weights


class HemisphericInterpolator:
    def __init__(self, lattice_, values, interpolator=LinearNDInterpolator,
                 mapping_resolution=1024, cos_cutoff=0.1):
        self.lattice = lattice_
        self.values = values
        self.mapping_resolution = mapping_resolution
        self.cos_cutoff = cos_cutoff
        indices_front = np.where(self.lattice[..., 0] > -self.cos_cutoff)
        self.lattice_front = self.lattice[indices_front]
        self.internal_coords_lattice_front = self.map_to_internal_rep(self.lattice_front)
        self.values_front = self.values[indices_front]
        self.interpolator_front = interpolator(
            self.internal_coords_lattice_front,
            self.values_front)
        output_xx, output_yy = utils_img.img_shape_to_xy((mapping_resolution, mapping_resolution))
        self.output_xy = np.stack([output_xx, output_yy], axis=2)
        self.transparency = utils_img.get_transparency(output_xx)
    def map_to_internal_rep(self, vector, front=True):
        vector = utils_img.normalize(vector)
        temp_internal_coords = utils_img.theta_phi_to_graph_coordinates(
            *utils_img.xyz_to_theta_phi(vector),
            graph_res_=self.mapping_resolution,
            front=front)
        return np.stack(temp_internal_coords, axis=-1)
    def output(self):
        output = self.interpolator_front(self.output_xy)
        output = np.where(np.isnan(output), 0, output)
        output = output * self.transparency
        return output


class ImageSet:
    def __init__(self, images, camera=CAMERA):
        self.images = images
        self.weights = ImageSetWeights(images[0].shape, len(images),
                                       camera.fov, camera.fov_axis,
                                       camera.directions, camera.front)
    def get_hemispherical_output(self, weightings=None):
        if weightings is None:
            weightings = []
        images_temp = self.images.copy()
        if 'solid_angle' in weightings:
            images_temp = images_temp * self.weights.solid_angles
        if 'cosine' in weightings:
            images_temp = images_temp * self.weights.cosine_weights
        images_temp = np.concatenate(images_temp, axis=0)
        two_hemispheres_interpolator = HemisphericInterpolator(
            self.weights.pixel_coordinates.reshape((-1, 3)),
            images_temp.reshape(-1))
        return two_hemispheres_interpolator.output()
