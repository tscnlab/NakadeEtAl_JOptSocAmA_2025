import numpy as np

__doc__ = """Library containing utility functions for dealing with images."""


def normalize(xyz_):
    """Normalize a vector, a list of vectors or an array of vectors.

    Parameters
    ----------
    xyz_ : numpy.ndarray
        The vector(s) to be normalized.
        The last axis should contain the x, y, z components.

    Returns
    -------
    numpy.ndarray
        The normalized vector(s). Has the same shape as `xyz_`.
        Returns `xyz_ / norm(xyz_, axis=-1)`.
    """
    return xyz_ / np.linalg.norm(xyz_, axis=-1, keepdims=True)


def img_shape_to_xy(img_shape: tuple) -> list[np.ndarray]:
    """Get (x, y) coordinates of the centers of pixels from image size.

    The x and y axes are in the plane of the sensor.
    Origin is at the center of the image. +x is to the right, +y is up.
    The units are pixels.
    If there are an odd number of pixels in a direction,
    the origin is at the center of a pixel.
    Hence, the centers of pixels are at integer coordinates.
    If there are an even number of pixels in a direction,
    the origin is between two pixels.
    Hence, the centers of pixels are at half-integer coordinates.

    Parameters
    ----------
    img_shape : tuple[int, int]
        (height, width) of the image.

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray]
        [x, y]
        The x and y coordinates of the centers of pixels.
        x and y are 2D arrays with the shape = `img_shape`.

    Examples
    --------
    In case of an odd number of pixels in a dimension,
    the origin is at the center of a pixel.
    >>> img_shape_to_xy((3,4))
    [array([[-1.5, -0.5,  0.5,  1.5],
           [-1.5, -0.5,  0.5,  1.5],
           [-1.5, -0.5,  0.5,  1.5]]),
    array([[ 1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.],
           [-1., -1., -1., -1.]])]
    """
    x = np.arange(img_shape[1]) - img_shape[1] // 2 + 0.5 * (1 - img_shape[1] % 2)
    y = img_shape[0] // 2 - np.arange(img_shape[0]) - 0.5 * (1 - img_shape[0] % 2)
    return [*np.meshgrid(x, y)]


def img_to_xy(img: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """Get (x, y) coordinates of the centers of pixels of an image.

    Please see `img_shape_to_xy` for more details.

    Parameters
    ----------
    img : numpy.ndarray
        The input image

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray]
        [x, y]
        The x and y coordinates of the centers of pixels.
        x and y are 2D arrays with the same shape as `img.shape[:2]`.
    """
    return img_shape_to_xy(img.shape)


def xy_to_r_phi(xy: list[np.ndarray]) -> list[np.ndarray]:
    """Convert cartesian to polar coordinates in the image plane.

    The origin is at the center of the image.
    +x is to the right, +y is up.
    Returns the distance from the origin and
    the angle measured anticlockwise from +x.
    The distance is given in pixels.

    Parameters
    ----------
    xy : list[numpy.ndarray, numpy.ndarray]
        [x, y]
        The x and y coordinates of the centers of pixels.
        x and y are 2D arrays with the same shape as img_shape[:2].

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray]
        [r, phi]
        r is the distance from the origin.
        phi is the angle measured anticlockwise from +x.
    """
    r = np.sqrt(xy[0] ** 2 + xy[1] ** 2)
    phi = np.arctan2(xy[1], xy[0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)  # phi in the range [0, 2*pi)
    return [r, phi]


def img_shape_to_r_phi(img_shape: tuple) -> list[np.ndarray]:
    """Get (r, phi) coordinates of the centers of pixels from image size.

    Refer to `img_shape_to_xy` and `xy_to_r_phi` for more details.

    Parameters
    ----------
    img_shape : tuple[int, int]
        (height, width) of the image.

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray]
        [r, phi]
    """
    return xy_to_r_phi(img_shape_to_xy(img_shape))


def img_to_r_phi(img: np.ndarray) -> list[np.ndarray]:
    """Get (r, phi) coordinates of the centers of pixels of an image.

    Refer to `img_shape_to_xy` and `xy_to_r_phi` for more details.

    Parameters
    ----------
    img : numpy.ndarray
        The image.

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray]
        [r, phi]
    """
    return img_shape_to_r_phi(img.shape[:2])


def gray_img_to_single_color(
        img: np.ndarray,
        color: tuple[float, float, float] | tuple[int, int, int]
) -> np.ndarray:
    """Convert gray image to single color image.

    Parameters
    ----------
    img : numpy.ndarray
        Gray image.
    color : tuple[float, float, float] | tuple[int, int, int]
        Color to convert to.
        If the values are floats, they are assumed to be in the range [0, 1].
        If the values are integers, they are assumed to be in the range [0, 255].

    Returns
    -------
    numpy.ndarray
        img * color
    """
    if isinstance(color[0], int):
        return np.stack([img * color[i] / 255 for i in range(3)], axis=-1).astype(img.dtype)
    return np.stack([img * color[i] for i in range(3)], axis=-1).astype(img.dtype)


def xyz_to_theta_phi(xyz):
    """Convert (x, y, z) to directions in spherical coordinates (theta, phi).

    Theta is measured away from +x.
    Phi is measured away from -y around -x.

    Parameters
    ----------
    xyz : numpy.ndarray
        numpy array with x, y, z along the last axis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (theta, phi)
        Theta is measured away from +x.
        Phi is measured away from -y around -x.
    """
    xyz = normalize(xyz)
    theta = np.arccos(xyz[..., 0])
    phi = np.arctan2(xyz[..., 2], -xyz[..., 1])
    return theta, phi


def img_shape_to_3d_coordinates(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    """Gives (x, y, z) coordinates of the centers of pixels of an image.

    The distance unit is pixel.
    The origin is at the eye (center of projection) of the perspective camera.
    The sensor is parallel to the xy plane with `camera_direction` along -z.
    +y is along the `up` direction.

    Parameters
    ----------
    camera_direction : numpy.ndarray
        The direction in which the camera is pointing.
    up : numpy.ndarray
        The up direction of the camera.
        It must be orthogonal to `camera_direction`.
    image_shape_ : tuple[int, int]
        The shape of the image as a (height, width) tuple.
    fov_ : float
        The field of view of the camera in degrees.
    fov_axis_ : str, default 'x'
        The axis along which the field of view is specified.

    Returns
    -------
    numpy.ndarray
        The 3D cartesian coordinates of the centers of the pixels.
        Shape is `image_shape_ + (3,)`.

    Raises
    ------
    ValueError
        If `up` and `camera_direction` are not orthogonal.
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
    pixel_mid_coordinates = (x_array[:, :, np.newaxis] * np.cross(camera_direction, up) +
                             y_array[:, :, np.newaxis] * up +
                             camera_direction * distance)
    return pixel_mid_coordinates


def img_shape_to_theta_phi(camera_direction, up, image_shape_, fov_, fov_axis_='x'):
    """Gives (theta, phi) coordinates of the centers of pixels from image shape.

    The distance unit is pixel.
    The origin is at the eye (center of projection) of the perspective camera.
    The sensor is parallel to the xy plane with `camera_direction` along -z.
    +y is along the `up` direction.
    Theta is measured away from +x.
    Phi is measured away from -y around -x.

    Parameters
    ----------
    camera_direction : numpy.ndarray
        The direction in which the camera is pointing.
    up : numpy.ndarray
        The up direction of the camera.
        It must be orthogonal to `camera_direction`.
    image_shape_ : tuple[int, int]
        The shape of the image as a (height, width) tuple.
    fov_ : float
        The field of view of the camera in degrees.
    fov_axis_ : str, default 'x'
        The axis along which the field of view is specified.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (theta, phi)
        Theta is measured away from +x.
        Phi is measured away from -y around -x.

    Raises
    ------
    ValueError
        If `up` and `camera_direction` are not orthogonal.
    """
    pixel_mid_coordinates = img_shape_to_3d_coordinates(camera_direction, up,
                                                        image_shape_, fov_, fov_axis_)
    return xyz_to_theta_phi(pixel_mid_coordinates)


def binary_search(func, target, low, high, tol=1e-6):
    """Generalized binary search using a callable.

    Binary search for the argument to `func` that would make it equal to `target`.
    Searches between `low` and `high`.
    `target` must be between `func(low)` and `func(high)`.
    `func(binary_search(func, target, low, high, tol) +- tol)` ~ `target`.

    Parameters
    ----------
    func : callable
        A function that takes a single argument.
    target : float
        The target value of func. Same type as the output of `func`.
    low : float
        Lower limit of the search domain. Same type as the input of `func`.
    high : float
        Higher limit of the search domain. Same type as the input of `func`.
    tol : float, default 1e-6
        Acceptable tolerance around the argument to `func`.

    Returns
    -------
    float
        The argument that would make `func(argument +- tol)` ~ `target`.

    Raises
    ------
    ValueError
        If `target` is not between `func(low)` and `func(high)`.
    """
    if (func(high) - target) * (func(low) - target) > 0:
        raise ValueError(f'target must be between func(low) and func(high).')
    while high - low > tol:
        mid = (low + high) / 2
        if func(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def phi_neg_to_pos(phi):
    """Change the input from the range (-pi, pi] to the range [0, 2*pi).

    Parameters
    ----------
    phi : numpy.ndarray
        The `phi` values in the range (-pi, pi].

    Returns
    -------
    numpy.ndarray
        Returns `phi` if it's positive or `phi` + 2*pi if it's negative
    """
    return np.where(phi < 0, phi + 2 * np.pi, phi)


def solid_angle_integral(x_, y_):
    """The indefinite integral for the solid angle subtended by a pixel at the eye.

    The perpendicular distance of the sensor from the eye is assumed to be 1.
    `x_` and `y_` must have the same shape, given by (height, width) of the image.
    The solid angle subtended by an infinitesimal area dA at the eye is given by
    .. math::
        d\\Omega = \\frac{dA * cos(\theta)}{r^2}
                 = \\frac{dx * dy * (1/sqrt(x^2 + y^2 + 1))}{(x^2 + y^2 + 1)}
                 = \\frac{dx * dy}{(x^2 + y^2 + 1)^{3/2}}
    Hence, the solid angle subtended by a pixel at the eye is given by
    .. math::
        \\int\\int\\frac{dx * dy}{(x^2 + y^2 + 1)^{3/2}}
        = atan((x * y) / sqrt(x^2 + y^2 + 1))

    Parameters
    ----------
    x_ : numpy.ndarray
        The x coordinates of the pixel centers.
    y_ : numpy.ndarray
        The y coordinates of the pixel centers.

    Returns
    -------
    numpy.ndarray
        `atan((x_ * y_) / sqrt(1 + x_^2 + y_^2))`
    """
    return np.arctan(x_ * y_ / np.sqrt(1 + x_ ** 2 + y_ ** 2))


def solid_angle_limits_x(x_min, x_max, y_):
    """`solid_angle_integral` definite along the x-axis.

    The perpendicular distance of the sensor from the eye is assumed to be 1.
    Refer to `solid_angle_integral` for more details.

    Parameters
    ----------
    x_min : numpy.ndarray
        The lower limits of the x coordinates (left edges) of the pixels.
    x_max : numpy.ndarray
        The upper limits of the x coordinates (right edges) of the pixels.
    y_ : numpy.ndarray
        The y coordinates of the pixel centers.

    Returns
    -------
    numpy.ndarray
        The solid angle integral definite along the x-axis.
    """
    return solid_angle_integral(x_max, y_) - \
        solid_angle_integral(x_min, y_)


def solid_angle_limits_x_y(x_min, x_max, y_min, y_max):
    """The solid angles subtended by pixels at the eye of the perspective camera.

    The edges of the pixels are given by `x_min`, `x_max`, `y_min`, `y_max`.
    The perpendicular distance of the sensor from the eye is assumed to be 1.
    Refer to `solid_angle_integral` for more details.

    Parameters
    ----------
    x_min : numpy.ndarray
        The lower limits of the x coordinates (left edges) of the pixels.
    x_max : numpy.ndarray
        The upper limits of the x coordinates (right edges) of the pixels.
    y_min : numpy.ndarray
        The lower limits of the y coordinates (bottom edges) of the pixels.
    y_max : numpy.ndarray
        The upper limits of the y coordinates (upper edges) of the pixels.

    Returns
    -------
    numpy.ndarray
        The solid angles subtended by the pixels.
    """
    return solid_angle_limits_x(x_min, x_max, y_max) - \
        solid_angle_limits_x(x_min, x_max, y_min)


def xy_min_max_mid(image_shape_, fov_, fov_axis_='x'):
    """The x and y coordinates of the lower, middle and upper limits
    of the pixels in the image.

    The origin is assumed to be at the center of the image.
    The return values correspond to the edges and the center of a pixel
    as shown below:

    (x_min, y_max) ._______________________. (x_max, y_max)
                   |                       |
                   |                       |
                   |                       |
                   |           .           |
                   |    (x_mid, y_mid)     |
                   |                       |
                   |                       |
    (x_min, y_min) ._______________________. (x_max, y_min)

    Parameters
    ----------
    image_shape_ : tuple[int, int]
        The shape of the image as a (height, width) tuple.
    fov_ : float
        The field of view of the camera.
    fov_axis_ : {'x', 'y'}
        The axis along which the field of view is specified.

    Returns
    -------
    tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray
    ]
        x_min: numpy.ndarray
            The lower limits of the x coordinates (left edges) of the pixels.
        x_max: numpy.ndarray
            The upper limits of the x coordinates (right edges) of the pixels.
        x_mid: numpy.ndarray
            The x coordinates of the centers of the pixels.
        y_min: numpy.ndarray
            The lower limits of the y coordinates (bottom edges) of the pixels.
        y_max: numpy.ndarray
            The upper limits of the y coordinates (top edges) of the pixels.
        y_mid: numpy.ndarray
            The y coordinates of the centers of the pixels.

    Raises
    ------
    ValueError
        If `fov_axis_` is not 'x' or 'y'.
    """
    if fov_axis_ not in ['x', 'y']:
        raise ValueError('fov_axis_ must be either "x" or "y"')
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

    Parameters
    ----------
    camera_direction : numpy.ndarray
        The direction in which the camera is pointing (in world coordinates).
        shape = (3,)
    up : numpy.ndarray
        The up direction of the camera (in world coordinates).
        shape = (3,)

    Returns
    -------
    numpy.ndarray
        The transformation matrix. shape = (3, 3)
    """
    return np.array([np.cross(camera_direction, up), up, -camera_direction])


def camera_to_world_coordinates(pixel_coordinates,
                                camera_direction=(1, 0, 0), up=(0, 0, 1)):
    """Transforms points from camera to world coordinates.

    In camera coordinates, the origin is assumed to be at the eye,
    the xy plane is parallel to the sensor and `up` points towards +y.
    The vector (0,0,-1) joins the origin to the center of the sensor.
    The origin is common between the world and the camera coordinates.
    `camera_direction` and `up` must be normalized and orthogonal to each other.

    Parameters
    ----------
    pixel_coordinates : numpy.ndarray
        (x, y, z) coordinates of the centers of the pixels in camera frame.
        shape = (height, width, 3)
    camera_direction : numpy.ndarray
        The direction of the camera in the world frame. shape = (3,)
    up : numpy.ndarray
        The up direction of the camera in the world frame. shape = (3,)

    Returns
    -------
    numpy.ndarray
        Transformed coordinates. shape = (height, width, 3)
    """
    camera_direction = normalize(np.array(camera_direction))
    up = normalize(np.array(up))
    pixel_coordinates = normalize(pixel_coordinates)
    return np.einsum('ijk,kl', pixel_coordinates,
                     transformation_matrix_transpose(camera_direction, up))


def theta_phi_to_graph_coordinates(theta_, phi_, graph_res_, front_=True):
    """A function to map theta and phi to planar coordinates x and y

    Converts the `theta_` and `phi_` coordinates covering a hemisphere to
    planar cartesian coordinates. A square with side length `graph_res_`
    is drawn with its center at the origin. `theta_`,
    which covers an interval of pi/2 for a hemisphere,
    is mapped to the r coordinate with pi/2 at the edge of the square.
    `phi_`, which covers an interval of 2*pi for a hemisphere,
    is mapped to the theta coordinate with 0 at the right edge.
    r and theta are converted to x and y and returned.
    `theta_` is measured away from the direction
    directly towards the front of the eye and `phi_` is measured around it,
    anti-clockwise starting from the direction pointing right.
    For the hemisphere in front of the eye, `theta_` varies from 0 to pi/2 and
    `phi_` varies from 0 to 2*pi. For the hemisphere behind the eye,
    `theta_` varies from pi/2 to pi and `phi_` varies from 0 to 2*pi.

    Parameters
    ----------
    theta_ : numpy.ndarray
        The theta coordinates. Must have the same shape as `phi_`.
    phi_ :numpy.ndarray
        The phi coordinates. Must have the same shape as `theta_`.
    graph_res_ : int
        The resolution of the final graph
    front_ : bool, default True
        If True, the hemisphere in front of the eye is considered.
        If False, the hemisphere behind the eye is considered.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (x, y)
        `theta_` and `phi_` converted to graph coordinates.

    Raises
    ------
    ValueError
        If `theta_` and `phi_` do not have the same shape.
    """
    if theta_.shape != phi_.shape:
        raise ValueError('theta_ and phi_ must have the same shape')
    if front_:
        graph_r = theta_ * graph_res_ / np.pi
    else:
        graph_r = (np.pi - theta_) * graph_res_ / np.pi
    graph_theta = phi_
    return graph_r * np.cos(graph_theta), graph_r * np.sin(graph_theta)


def get_transparency(img):
    """Gives a 2D array that's 0 where distance from the center is larger than
    the distance to the closest edge of the image and 1 otherwise.

    This array can be added as the transparency channel to an image to make
    everything lying outside the largest circle centered at the center
    of the image transparent.

    Parameters
    ----------
    img : numpy.ndarray
        The image for which a circular transparency is required.

    Returns
    -------
    numpy.ndarray
        2D array that's 0 where distance from the center is larger than
        the distance to the closest edge of the image and 1 otherwise.
        0 is transparent, 1 is opaque.
    """
    xx, yy = img_shape_to_xy(img.shape)
    r = np.sqrt(xx ** 2 + yy ** 2)
    return np.where(r > min(xx.max(), yy.max()), 0, 1)


def add_transparency(img):
    """Add an edge to edge circular transparency to the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image.

    Returns
    -------
    numpy.ndarray
        New image which is transparent outside the largest circle centered
        at the center of the image and opaque inside it.
    """
    transparency = get_transparency(img)
    img_ = img.copy()
    if len(img_.shape) == 2:
        img_ = np.stack([img_, img_, img_, transparency], axis=-1)
    else:
        img_ = np.stack([img_[..., 0], img_[..., 1], img_[..., 2], transparency], axis=-1)
    return img_
