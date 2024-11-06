import numpy as np
from scipy.interpolate import LinearNDInterpolator

from common_params import CAMERA

__doc__ = """Library containing utility functions for dealing with images."""


def normalize(xyz_):
    """
    Normalize a vector, a list of vectors or an array of vectors.

    xyz_ must be a numpy array. Returns xyz_ / norm(xyz_, axis=-1).
    :param xyz_: numpy.ndarray
        x, y, (z) coordinates are along the last axis.
    :return: numpy.ndarray
        Has the same shape as xyz_.
    """
    return xyz_ / np.linalg.norm(xyz_, axis=-1, keepdims=True)


def img_shape_to_xy(img_shape: tuple) -> list[np.ndarray]:
    """
    Get the x and y coordinates of the centers of pixels of the image, from image size.

    The x and y axes are in the plane of the sensor.
    Origin is at the center of the image. +x is to the right, +y is up.
    The units are pixels.
    :param img_shape: tuple
        img_shape[0] is the height, img_shape[1] is the width
    :return: list[numpy.ndarray]
        [x: numpy.ndarray, y: numpy.ndarray]
    """
    x = np.arange(img_shape[1]) - img_shape[1] // 2 + 0.5 * (1 - img_shape[1] % 2)
    y = img_shape[0] // 2 - np.arange(img_shape[0]) - 0.5 * (1 - img_shape[0] % 2)
    return [*np.meshgrid(x, y)]


def img_to_xy(img: np.ndarray) -> list[np.ndarray]:
    """
    Get the x and y coordinates of the centers of pixels of the image.

    The x and y axes are in the plane of the sensor.
    Origin is at the center of the image. +x is to the right, +y is up.
    The units are pixels.
    :param img: numpy.ndarray
        The input image
    :return: list[numpy.ndarray]
        [x: numpy.ndarray, y: numpy.ndarray]
    """
    return img_shape_to_xy(img.shape)


def xy_to_r_phi(xy: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert cartesian to polar coordinates in the image plane.

    The x and y axes are in the plane of the sensor.
    Origin is at the center of the image. +x is to the right, +y is up.
    The unit of distance is pixel.
    :param xy: list[numpy.ndarray]
        [x: numpy.ndarray, y: numpy.ndarray]
    :return: list[numpy.ndarray]
        [r: numpy.ndarray, phi: numpy.ndarray]
    """
    r = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
    phi = np.arctan2(xy[1], xy[0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return [r, phi]


def img_to_r_phi(img: np.ndarray) -> list[np.ndarray]:
    """
    Get the polar coordinates of the centers of pixels of the image.

    Origin is at the center of the image. phi is measured anticlockwise away from +x (which points to the right).
    The unit of distance is pixel.
    :param img: numpy.ndarray
        The input image.
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
        [r: numpy.ndarray, phi: numpy.ndarray]
    """
    return xy_to_r_phi(img_shape_to_xy(img_shape))


def gray_img_to_single_color(img: np.ndarray, color: tuple[float, float, float] | tuple[int, int, int]) -> np.ndarray:
    """
    Convert gray image to single color image.

    :param img: numpy.ndarray
        Gray image
    :param color: tuple[float, float, float] or tuple[int, int, int]
        Color to convert to
    :return: numpy.ndarray
        img * color
    """
    if isinstance(color[0], int):
        return np.stack([img * color[i]/255 for i in range(3)], axis=-1).astype(img.dtype)
    return np.stack([img * color[i] for i in range(3)], axis=-1).astype(img.dtype)


def xyz_to_theta_phi(xyz):
    """
    Convert 3D cartesian coordinates to directions in spherical coordinates (only angles).

    Theta is measured away from +x.
    Phi is measured away from -y around -x.
    :param xyz: numpy.ndarray
        numpy array with x, y, z along the last axis.
    :return: tuple[numpy.ndarray]
        (theta: numpy.ndarray, phi: numpy.ndarray)
    """
    xyz = normalize(xyz)
    theta = np.arccos(xyz[..., 0])
    phi = np.arctan2(xyz[..., 2], -xyz[..., 1])
    return theta, phi


def img_shape_to_3d_coordinates(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    """
    Gives the 3D cartesian coordinates of the centers of the pixels of an image with the given shape.

    The origin is assumed to be at the eye (center of projection) of the perspective camera.
    The origin is also directly away from the center of the sensor at a distance of 1 unit.
    :param camera_direction: numpy.ndarray
        The direction in which the camera is pointing.
    :param up: numpy.ndarray
        The up direction of the camera. It is assumed to be orthogonal to camera_direction.
    :param image_shape_: tuple
        The shape of the image.
    :param fov_: float
        The field of view of the camera.
    :param fov_axis_: str
        The axis along which the field of view is measured.
    :return: numpy.ndarray
        The 3D cartesian coordinates of the centers of the pixels in the image.
    :raises: ValueError
        Raises ValueError if up and camera_direction are not orthogonal.
    """
    camera_direction = normalize(camera_direction)
    up = normalize(up)
    if not np.isclose(np.dot(camera_direction, up), 0):
        raise ValueError('camera_direction and up are not orthogonal')
    x_array, y_array = img_shape_to_xy(image_shape_)
    fov_ = fov_ * np.pi / 180
    if fov_axis_ == 'x':
        image_size = image_shape_[1]
    else:
        image_size = image_shape_[0]
    distance = image_size / (2 * np.tan(fov_ / 2))
    pixel_mid_coordinates = (x_array[:,:,np.newaxis] * np.cross(camera_direction, up) +
                             y_array[:,:,np.newaxis] * up +
                             camera_direction * distance)
    return pixel_mid_coordinates


def img_shape_to_theta_phi(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    """
    Gives the 3D polar coordinate angles of the centers of the pixels of an image with the given shape.

    The origin is assumed to be at the eye (center of projection) of the perspective camera.
    The origin is also directly away from the center of the sensor at a distance of 1 unit.
    Theta is measured away from +x.
    Phi is measured away from -y around -x.
    Theta and Phi only constrain the direction to the pixel from the eye.
    :param camera_direction: numpy.ndarray
        The direction in which the camera is pointing.
    :param up: numpy.ndarray
        The up direction of the camera. It must be orthogonal to camera_direction.
    :param image_shape_: tuple
        The shape of the image.
    :param fov_: float
        The field of view of the camera.
    :param fov_axis_: str
        The axis along which the field of view is measured.
    :return: tuple[numpy.ndarray]
        (theta: numpy.ndarray, phi: numpy.ndarray)
    :raises: ValueError
        Raises ValueError if up and camera_direction are not orthogonal.
    """
    pixel_mid_coordinates = img_shape_to_3d_coordinates(camera_direction, up,
                                                        image_shape_, fov_, fov_axis_)
    return xyz_to_theta_phi(pixel_mid_coordinates)


def binary_search(func, target, low, high, tol=1e-6):
    """
    Binary search between low and high for the argument to func that would make it equal to target.
    :param func:
        A callable
    :param target:
        The target value of func
    :param low:
        Lower limit of the search domain
    :param high:
        Higher limit of the search domain
    :param tol: default value 1e-6
        Acceptable tolerance around the argument to func
    :return:
        The argument that would make func(argument +- tol) ~ target
    """
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
        Returns phi if it's positive or phi + 2*pi if it's negative
    """
    return np.where(phi < 0, phi + 2 * np.pi, phi)


def solid_angle_integral(x_by_d, y_by_d):
    """
    The indefinite integral for the solid angle subtended by a pixel at the eye.

    The perpendicular distance of the sensor from the eye is assumed to be d.
    :param x_by_d: numpy.ndarray
        The x coordinates in the plane of the sensor divided by d.
    :param y_by_d: numpy.ndarray
        The y coordinates in the plane of the sensor divided by d.
    :return: numpy.ndarray
        atan((x/d)*(y/d)/sqrt(1+(x/d)^2+(y/d)^2))
    """
    return np.arctan(x_by_d * y_by_d / np.sqrt(1 + x_by_d ** 2 + y_by_d ** 2))


def solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d):
    """
    To calculate the solid angle subtended by a pixel at the eye,
    this gives the definite integral along the x-axis and
    the indefinite integral along the y-axis.
    The perpendicular distance of the sensor from the eye is assumed to be d.
    :param x_by_d_min: numpy.ndarray
        The lower limits of the x coordinates in the plane of the sensor divided by d.
    :param x_by_d_max: numpy.ndarray
        The upper limits of the x coordinates in the plane of the sensor divided by d.
    :param y_by_d: numpy.ndarray
        The y coordinate in the plane of the sensor divided by d.
    :return: numpy.ndarray
        The solid angle integral indefinite along the y-axis and definite along the x-axis
    """
    return solid_angle_integral(x_by_d_max, y_by_d) - \
        solid_angle_integral(x_by_d_min, y_by_d)


def solid_angle_limits_x_y(x_by_d_min, x_by_d_max, y_by_d_min, y_by_d_max):
    """
    The solid angle subtended by a pixel at the eye of the perspective camera.

    The perpendicular distance of the sensor from the eye is assumed to be d.
    All the inputs must have the same shapes: (width, height) of the sensor
    :param x_by_d_min: numpy.ndarray
        The lower limits of the x coordinates of the pixels in the plane of the sensor divided by d.
    :param x_by_d_max: numpy.ndarray
        The upper limits of the x coordinates of the pixels in the plane of the sensor divided by d.
    :param y_by_d_min: numpy.ndarray
        The lower limits of the y coordinates of the pixels in the plane of the sensor divided by d.
    :param y_by_d_max: numpy.ndarray
        The upper limits of the y coordinates of the pixels in the plane of the sensor divided by d.
    :return: numpy.ndarray
        The solid angles of the pixels.
    """
    return solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d_max) - \
        solid_angle_limits_x(x_by_d_min, x_by_d_max, y_by_d_min)


def xy_min_max_mid(image_shape_, fov_, fov_axis_='x'):
    """The x and y coordinates of the lower, middle and upper limits of the pixels in the image.

    The origin is assumed to be at the center of the image. In order to fix the scale of the sensor,
    the distance d from the eye of the camera (center of projection of the perspective camera,
    the point from where the FOV is measured) to the center of the sensor is assumed to be 1.
    :param image_shape_: tuple
        The (height, width) of the image in pixels.
    :param fov_: float
        The field of view of the camera.
    :param fov_axis_: str
        The axis along which the field of view is measured.
    :return: tuple
        x_min: numpy.ndarray,
            The lower limits of the x coordinates (left edges) of the pixels in the plane of the sensor.
        x_max: numpy.ndarray,
            The upper limits of the x coordinates (right edges) of the pixels in the plane of the sensor.
        x_mid: numpy.ndarray,
            The x coordinates of the centers of the pixels.
        y_min: numpy.ndarray,
            The lower limits of the y coordinates (bottom edges) of the pixels in the plane of the sensor.
        y_max: numpy.ndarray,
            The upper limits of the y coordinates (top edges) of the pixels in the plane of the sensor.
        y_mid: numpy.ndarray,
            The y coordinates of the centers of the pixels
    """
    j_array, i_array = np.meshgrid(np.arange(image_shape_[1]), np.arange(image_shape_[0]))
    fov_ = fov_ * np.pi / 180
    if fov_axis_ == 'x':
        image_size = image_shape_[1]
        delta_ = 2 * np.tan(fov_ / 2) / image_size
        x_0 = -np.tan(fov_ / 2)
        y_0 = image_shape_[0] * delta_ / 2
    else:
        image_size = image_shape_[0]
        delta_ = 2 * np.tan(fov_ / 2) / image_size
        y_0 = np.tan(fov_ / 2)
        x_0 = -image_shape_[1] * delta_ / 2
    x_min = x_0 + j_array * delta_
    x_max = x_min + delta_
    x_mid = (x_min + x_max) / 2  # the x coordinates of the centers of the pixels
    y_max = y_0 - i_array * delta_
    y_min = y_max - delta_
    y_mid = (y_min + y_max) / 2
    return x_min, x_max, x_mid, y_min, y_max, y_mid


def transformation_matrix_transpose(camera_direction, up):
    """Gives the matrix needed to transform a point from camera to world coordinates.

    In camera coordinates, the origin is assumed to be at the eye,
    the xy plane is parallel to the sensor and up points towards +y.
    The vector (0,0,-1) joins the origin to the center of the sensor and is also perpendicular to it.
    camera_direction and up must be normalized and orthogonal to each other.
    :param camera_direction: numpy.ndarray
        The direction in which the camera is pointing (in world coordinates).
    :param up: numpy.ndarray
        The up direction of the camera (in world coordinates).
    :return: numpy.ndarray
        The transformation matrix.
    """
    return np.array([np.cross(camera_direction, up), up, -camera_direction])


def camera_to_world_coordinates(pixel_coordinates, camera_direction=(1, 0, 0), up=(0, 0, 1)):
    """
    Function to transform points from camera to world coordinates.

    In camera coordinates, the origin is assumed to be at the eye,
    the xy plane is parallel to the sensor and up points towards +y.
    The vector (0,0,-1) joins the origin to the center of the sensor and is also perpendicular to it.
    camera_direction and up must be normalized and orthogonal to each other.
    :param pixel_coordinates:
        Coordinates of the centers of pixels in camera frame
    :param camera_direction:
        The direction in which the camera is pointing (in world coordinates).
    :param up:
        The up direction of the camera (in world coordinates).
    :return:
        Transformed coordinates
    """
    camera_direction = normalize(np.array(camera_direction))
    up = normalize(np.array(up))
    pixel_coordinates = normalize(pixel_coordinates)
    return np.einsum('ijk,kl', pixel_coordinates,
                     transformation_matrix_transpose(camera_direction, up))


def theta_phi_to_graph_coordinates(theta_, phi_, graph_res_, front=True):
    """
    A function to map theta and phi to planar polar coordinates r and theta

    If we want to map a scalar quantity that varies with the directions covering the hemisphere in front of the eye,
    we can map theta (measured away from the direction directly towards the front of the eye) to 2D planar r
    and phi (measured around the direction directly towards the front of the eye) to the 2D planar theta,
    then map to the pixels of an image as if it were the 2D plane.
    We can then graph the scalar quantity as a grayscale image.
    For this purpose, this function converts from theta, phi to the pixel coordinates of the graph
    :param theta_:
        2D array of theta values
    :param phi_:
        2D array of phi values
    :param graph_res_:
        The resolution of the final graph
    :param front:
        Boolean indicating whether the hemisphere in front of or to the back of the eye is being graphed.
    :return:
        graph_x, graph_y
    """
    if front:
        graph_r = theta_*graph_res_/np.pi
    else:
        graph_r = (np.pi - theta_)*graph_res_/np.pi
    graph_theta = phi_
    return graph_r * np.cos(graph_theta), graph_r * np.sin(graph_theta)


def get_transparency(img):
    """
    Get an edge to edge circular transparency that can be added to the image

    Returns a 2D array such that, if added as the transparency channel to an image,
    will make everything lying outside the largest circle centered at the center of the image transparent.
    :param img:
        The image for which a circular transparency is required.
    :return:
        2D array that's 0 where distance from the center is larger than the distance to the closest edge of the image
        and 1 otherwise. 0 is transparent, 1 is opaque
    """
    xx, yy = img_shape_to_xy(img.shape)
    r = np.sqrt(xx**2 + yy**2)
    return np.where(r>min(xx.max(), yy.max()), 0, 1)


def add_transparency(img):
    """
    Add an edge to edge circular transparency to the image

    Makes everything lying outside the largest circle centered at the center of the image transparent.
    :param img:
        The image.
    :return:
        New image which transparent outside the largest circle fitting in the image
    """
    transparency = get_transparency(img)
    img_ = img.copy()
    if len(img_.shape) == 2:
        img_ = np.stack([img_, img_, img_, transparency], axis=-1)
    else:
        img_ = np.stack([img_[..., 0], img_[..., 1], img_[..., 2], transparency], axis=-1)
    return img_


