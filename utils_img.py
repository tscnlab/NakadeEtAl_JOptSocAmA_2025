import numpy as np
from scipy.interpolate import LinearNDInterpolator

from common_params import CAMERA
from warnings import warn


def normalize(x):
    """
    Normalize a vector, a list of vectors or an array of vectors.
    x must be numpy array.
    :param x: numpy.ndarray
        The last dimension must be 3 (x,y,z).
    :return: numpy.ndarray
        Has the same shape as x.
    """
    if x.shape[-1] != 3:
        warn('normalize(x): last dimension not 3', UserWarning)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def img_shape_to_xy(img_shape: tuple) -> list[np.ndarray]:
    """
    Convert image shape to x, y coordinates.
    Origin is at the center of the image.
    +x is to the right, +y is up.
    :param img_shape: tuple
        img_shape[0] is the height, img_shape[1] is the width
    :return: list[numpy.ndarray]
    """
    x = np.arange(img_shape[1]) - img_shape[1] // 2 + 0.5 * (1 - img_shape[1] % 2)
    y = img_shape[0] // 2 - np.arange(img_shape[0]) - 0.5 * (1 - img_shape[0] % 2)
    return [*np.meshgrid(x, y)]


def img_to_xy(img: np.ndarray) -> list[np.ndarray]:
    """
    Convert image to x, y coordinates.
    Origin is at the center of the image.
    +x is to the right, +y is up.
    :param img: numpy.ndarray
    :return: list[numpy.ndarray]
    """
    return img_shape_to_xy(img.shape)


def xy_to_r_phi(xy: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert x, y coordinates to r, phi coordinates.
    Origin is at the center of the image.
    +x is to the right, +y is up.
    :param xy: list[numpy.ndarray]
    :return: list[numpy.ndarray]
    """
    r = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
    phi = np.arctan2(xy[1], xy[0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return [r, phi]


def img_to_r_phi(img: np.ndarray) -> list[np.ndarray]:
    """
    Convert image to r, phi coordinates.
    Origin is at the center of the image.
    +x is to the right, +y is up.
    :param img: numpy.ndarray
    :return: list[numpy.ndarray]
    """
    return xy_to_r_phi(img_to_xy(img))


def img_shape_to_r_phi(img_shape: tuple) -> list[np.ndarray]:
    """
    Convert image shape to r, phi coordinates.
    Origin is at the center of the image.
    +x is to the right, +y is up.
    :param img_shape: tuple
        img_shape[0] is the height, img_shape[1] is the width
    :return: list[numpy.ndarray]
    """
    return xy_to_r_phi(img_shape_to_xy(img_shape))


def gray_img_to_single_color(img: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """
    Convert gray image to single color image.
    :param img: numpy.ndarray
        Gray image
    :param color: tuple[int, int, int]
        Color to convert to
    :return: numpy.ndarray
    """
    return np.stack([img * color[i] for i in range(3)], axis=-1)


def xyz_to_theta_phi(xyz):
    """
    Convert 3D coordinates to spherical coordinates.
    Theta is measured away from +x.
    Phi is measured away from -y around -x.
    :param xyz: numpy.ndarray
        The last dimension must be 3 (x,y,z).
    :return: tuple[numpy.ndarray]
        theta, phi
    """
    xyz = normalize(xyz)
    theta = np.arccos(xyz[..., 0])
    phi = np.arctan2(xyz[..., 2], -xyz[..., 1])
    return theta, phi


def img_shape_to_3d_coordinates(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    """
    Gives the 3D coordinates of the centers of the pixels in the image.
    The origin is assumed to be at the eye (center of projection) of the perspective camera.
    :param camera_direction: numpy.ndarray
        The direction in which the camera is pointing.
    :param up: numpy.ndarray
        The up direction of the camera. It is assumed to be orthogonal to the camera direction.
    :param image_shape_: tuple
        The shape of the image.
    :param fov_: float
        The field of view of the camera.
    :param fov_axis_: str
        The axis along which the field of view is measured.
    :return: numpy.ndarray
        The 3D coordinates of the centers of the pixels in the image.
    """
    camera_direction = normalize(camera_direction)
    up = normalize(up)
    assert np.isclose(np.dot(camera_direction, up), 0)
    x_array, y_array = img_shape_to_xy(image_shape_)
    fov_ = fov_ * np.pi / 180
    if fov_axis_ == 'x':
        image_size = image_shape_[1]
    else:
        image_size = image_shape_[0]
    distance = image_size / (2 * np.tan(fov_ / 2))
    pixel_mid_coordinates = x_array[:,:,np.newaxis] * np.cross(camera_direction, up) + y_array[:,:,np.newaxis] * up + camera_direction * distance
    return pixel_mid_coordinates


def img_shape_to_theta_phi(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    pixel_mid_coordinates = img_shape_to_3d_coordinates(camera_direction, up,
                                                        image_shape_, fov_, fov_axis_)
    return xyz_to_theta_phi(pixel_mid_coordinates)


def binary_search(func, target, low, high, tol=1e-6):
    while high - low > tol:
        mid = (low + high) / 2
        if func(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def phi_neg_to_pos(phi):
    """
    Change phi from the range (-pi, pi] to the range [0, 2*pi).
    :param phi: numpy.ndarray
    :return: numpy.ndarray
    """
    return np.where(phi < 0, phi + 2 * np.pi, phi)


def solid_angle_integral(x_by_d, y_by_d):
    """
    To calculate the solid angle subtended by a pixel at the eye,
    this gives the indefinite integral.
    The perpendicular distance of the sensor from the eye is assumed to be d.
    :param x_by_d: numpy.ndarray
        The x coordinate in the plane of the sensor divided by d.
    :param y_by_d: numpy.ndarray
        The y coordinate in the plane of the sensor divided by d.
    :return: numpy.ndarray
    """
    return np.arctan(x_by_d * y_by_d / np.sqrt(1 + x_by_d ** 2 + y_by_d ** 2))


def solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d):
    """
    To calculate the solid angle subtended by a pixel at the eye,
    this gives the definite integral along the x-axis and
    the indefinite integral along the y-axis.
    The perpendicular distance of the sensor from the eye is assumed to be d.
    :param x_by_d_min: numpy.ndarray
        The lower limit of the x coordinate in the plane of the sensor divided by d.
    :param x_by_d_max: numpy.ndarray
        The upper limit of the x coordinate in the plane of the sensor divided by d.
    :param y_by_d: numpy.ndarray
        The y coordinate in the plane of the sensor divided by d.
    :return: numpy.ndarray
    """
    return solid_angle_integral(x_by_d_max, y_by_d) - \
        solid_angle_integral(x_by_d_min, y_by_d)


def solid_angle_limits_x_y(x_by_d_min, x_by_d_max, y_by_d_min, y_by_d_max):
    """
    The solid angle subtended by a pixel at the eye.
    The perpendicular distance of the sensor from the eye is assumed to be d.
    :param x_by_d_min: numpy.ndarray
        The lower limit of the x coordinate of the pixel in the plane of the sensor divided by d.
    :param x_by_d_max: numpy.ndarray
        The upper limit of the x coordinate of the pixel in the plane of the sensor divided by d.
    :param y_by_d_min: numpy.ndarray
        The lower limit of the y coordinate of the pixel in the plane of the sensor divided by d.
    :param y_by_d_max: numpy.ndarray
        The upper limit of the y coordinate of the pixel in the plane of the sensor divided by d.
    :return: numpy.ndarray
    """
    return solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d_max) - \
        solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d_min)


def xy_min_max_mid(image_shape_, fov_, fov_axis_='x'):
    """The x and y coordinates of the lower,
    middle and upper limits of the pixels in the image.

    The origin is assumed to be at the center of the image.
    :param image_shape_: tuple
        The shape of the image.
    :param fov_: float
        The field of view of the camera.
    :param fov_axis_: str
        The axis along which the field of view is measured.
    :return: tuple
        x_by_d_min: numpy.ndarray,
            The lower limit of the x coordinate of the pixel in the plane of the sensor divided by d.
        x_by_d_max: numpy.ndarray,
            The upper limit of the x coordinate of the pixel in the plane of the sensor divided by d.
        x_by_d_mid: numpy.ndarray,
            The x coordinates of the centers of the pixels.
        y_by_d_min: numpy.ndarray,
            The lower limit of the y coordinate of the pixel in the plane of the sensor divided by d.
        y_by_d_max: numpy.ndarray,
            The upper limit of the y coordinate of the pixel in the plane of the sensor divided by d.
        y_by_d_mid: numpy.ndarray,
            The y coordinates of the centers of the pixels
    """
    j_array, i_array = np.meshgrid(np.arange(image_shape_[1]), np.arange(image_shape_[0]))
    fov_ = fov_ * np.pi / 180
    if fov_axis_ == 'x':
        image_size = image_shape_[1]
        delta_ = 2 * np.tan(fov_ / 2) / image_size
        x_by_d_0 = -np.tan(fov_ / 2)
        y_by_d_0 = image_shape_[0] * delta_ / 2
    else:
        image_size = image_shape_[0]
        delta_ = 2 * np.tan(fov_ / 2) / image_size
        y_by_d_0 = np.tan(fov_ / 2)
        x_by_d_0 = -image_shape_[1] * delta_ / 2
    x_by_d_min = x_by_d_0 + j_array * delta_
    x_by_d_max = x_by_d_min + delta_
    x_by_d_mid = (x_by_d_min + x_by_d_max) / 2  # the x coordinates of the centers of the pixels
    y_by_d_max = y_by_d_0 - i_array * delta_
    y_by_d_min = y_by_d_max - delta_
    y_by_d_mid = (y_by_d_min + y_by_d_max) / 2
    return x_by_d_min, x_by_d_max, x_by_d_mid, y_by_d_min, y_by_d_max, y_by_d_mid


def transformation_matrix_transpose(camera_direction, up):
    """Gives the matrix needed to transform a point from camera to world coordinates.

    It is assumed that both the camera_direction and up are normalized.
    :param camera_direction: numpy.ndarray
        The direction in which the camera is pointing.
    :param up: numpy.ndarray
        The up direction of the camera. It is assumed to be orthogonal to the camera direction.
    :return: numpy.ndarray
        The transformation matrix.
    """
    return np.array([np.cross(camera_direction, up), up, -camera_direction])


def camera_to_world_coordinates(pixel_coordinates, camera_direction=(1, 0, 0), up=(0, 0, 1)):
    camera_direction = normalize(np.array(camera_direction))
    up = normalize(np.array(up))
    pixel_coordinates = normalize(pixel_coordinates)
    return np.einsum('ijk,kl', pixel_coordinates,
                     transformation_matrix_transpose(camera_direction, up))


def theta_phi_to_graph_coordinates(theta_, phi_, graph_res_, front=True):
    if front:
        graph_r = theta_*graph_res_/np.pi
    else:
        graph_r = (np.pi - theta_)*graph_res_/np.pi
    graph_theta = phi_
    return graph_r * np.cos(graph_theta), graph_r * np.sin(graph_theta)


def get_transparency(img):
    xx, yy = img_shape_to_xy(img.shape)
    r = np.sqrt(xx**2 + yy**2)
    # 0 is transparent, 1 is opaque
    return np.where(r>min(xx.max(), yy.max()), 0, 1)


def add_transparency(img):
    transparency = get_transparency(img)
    img_ = img.copy()
    if len(img_.shape) == 2:
        img_ = np.stack([img_, img_, img_, transparency], axis=-1)
    else:
        img_ = np.stack([img_[..., 0], img_[..., 1], img_[..., 2], transparency], axis=-1)
    return img_


class ImageSetWeights:
    def __init__(self, image_shape_, num_images_, fov_, fov_axis_, camera_directions_, front_):
        self.num_images = num_images_
        x_by_d_min, x_by_d_max, x_by_d_mid, y_by_d_min, y_by_d_max, y_by_d_mid = xy_min_max_mid(
            image_shape_, fov_, fov_axis_)
        solid_angles_1_image = solid_angle_limits_x_y(x_by_d_min,
                                                      x_by_d_max,
                                                      y_by_d_min,
                                                      y_by_d_max)
        self.solid_angles = np.repeat(solid_angles_1_image[np.newaxis], self.num_images, axis=0)
        self.pixel_coordinates = np.zeros((self.num_images, *image_shape_, 3))
        pixel_coordinates_1_image = normalize(
            np.stack([x_by_d_mid, y_by_d_mid, -np.ones(image_shape_)], axis=2))
        for i in range(self.num_images):
            self.pixel_coordinates[i] = camera_to_world_coordinates(
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
        output_xx, output_yy = img_shape_to_xy((mapping_resolution, mapping_resolution))
        self.output_xy = np.stack([output_xx, output_yy], axis=2)
        self.transparency = get_transparency(output_xx)
    def map_to_internal_rep(self, vector, front=True):
        vector = normalize(vector)
        temp_internal_coords = theta_phi_to_graph_coordinates(
            *xyz_to_theta_phi(vector),
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
