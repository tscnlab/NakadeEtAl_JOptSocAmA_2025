import numpy as np
from scipy.interpolate import LinearNDInterpolator

import utils_img
from common_params import CAMERA


class ImageSetWeights:
    """Class for storing solid angles, pixel coordinates and cosine weights.

    Attributes
    ----------
    num_images : int
        The number of images.
    solid_angles : numpy.ndarray
        The solid angles of the pixels in the images.
    pixel_coordinates : numpy.ndarray
        The 3D coordinates of the pixels in the images, normalized.
    cosine_weights : numpy.ndarray
        The cosine weights of the pixels in the images.
        This is the dot product of the pixel coordinates and the front vector.
    weights : numpy.ndarray
        The final weights of the pixels in the images.
        This is the product of the solid angles and the cosine weights.
    """
    def __init__(self, image_shape_, num_images_, fov_, fov_axis_, camera_directions_, front_):
        """Initialize the attributes.

        Parameters
        ----------
        image_shape_ : tuple[int, int]
            The shape of each image given as (height, width).
        num_images_ : int
            The number of images.
        fov_ : float
            The field of view of the camera in degrees.
        fov_axis_ : str
            The axis of the field of view.
        camera_directions_ : dict
            The camera directions for the images. The keys are integers from 0 to
            (number of images - 1). The values are dictionaries with the
            following items:
            - 'camera_direction' : numpy.ndarray
                The direction in which the individual camera is pointing.
            - 'up' : numpy.ndarray
                The up vector of the camera.
            - 'name' : str
                The name of the camera direction.
        front_ : numpy.ndarray
            The front vector of the camera. Not necessarily the same as the
            direction in which each sub-camera is pointing.
            There are multiple sub-cameras so that we can cover the entire
            hemisphere in front of the eye and a perspective camera cannot have
            a field of view equal to 180 degrees.
        """
        self.num_images = num_images_
        x_min, x_max, x_mid, y_min, y_max, y_mid = utils_img.xy_min_max_mid(image_shape_,
                                                                            fov_, fov_axis_)
        solid_angles_1_image = utils_img.solid_angle_limits_x_y(x_min, x_max, y_min, y_max)
        self.solid_angles = np.repeat(solid_angles_1_image[np.newaxis], self.num_images, axis=0)
        self.pixel_coordinates = np.zeros((self.num_images, *image_shape_, 3))
        pixel_coordinates_1_image = utils_img.normalize(
            np.stack([x_mid, y_mid, -np.ones(image_shape_)], axis=2))
        for i in range(self.num_images):
            self.pixel_coordinates[i] = utils_img.camera_to_world_coordinates(
                pixel_coordinates_1_image,
                camera_directions_[i]['camera_direction'],
                camera_directions_[i]['up'])
        self.cosine_weights = np.clip(np.tensordot(self.pixel_coordinates, front_, axes=(-1, -1)),
                                      a_min=0, a_max=1)
        self.weights = self.solid_angles * self.cosine_weights


class HemisphericInterpolator:
    """Class for interpolating values on a hemisphere.

    As there are no good interpolators for data that varies as a function of 3D
    coordinates, we first map the coordinates of the pixels on the sphere to
    a 2D grid. This is done by mapping the angle away from the front direction
    to the 2D r coordinate and the angle around the front direction to the 2D
    theta coordinate. We then create a regular xy grid and interpolate the
    values on this grid.

    Attributes
    ----------
    lattice : numpy.ndarray
        The 3D coordinates of the points on the sphere.
    values : numpy.ndarray
        The values at the lattice points.
    interpolator : scipy.interpolate.LinearNDInterpolator
        The interpolator for the lattice points.
    mapping_resolution : int
        The resolution of the internal 2D mapping.
    cos_cutoff : float
        The cosine cutoff for the hemispheres.
        For the interpolator for one of the hemispheres, points outside the
        hemisphere are also included. This ensures that the interpolated values
        at the boundary of the hemisphere do not suffer from edge effects.
        The pixels are cut off where |cosine weight| is less than this value.
    lattice_front : numpy.ndarray
        The lattice points used for interpolation on the front hemisphere.
    internal_coords_lattice_front : numpy.ndarray
        The internal coordinates of the lattice points on the front hemisphere.
    values_front : numpy.ndarray
        The values at the lattice points on the front hemisphere.
    lattice_back : numpy.ndarray
        The lattice points used for interpolation on the back hemisphere.
    internal_coords_lattice_back : numpy.ndarray
        The internal coordinates of the lattice points on the back hemisphere.
    values_back : numpy.ndarray
        The values at the lattice points on the back hemisphere.
    output_xy : numpy.ndarray
        The 2D grid for interpolation.
        Has shape (mapping_resolution, mapping_resolution, 2).
    transparency : numpy.ndarray
        Transparency that makes the largest circle centered at the center of
        the image opaque and the rest transparent.
    """
    def __init__(self, lattice_, values, interpolator=LinearNDInterpolator,
                 mapping_resolution=1024, cos_cutoff=0.1):
        """Initialize the attributes.

        Parameters
        ----------
        lattice_ : numpy.ndarray
            The 3D coordinates of the points on the sphere.
        values : numpy.ndarray
            The values at the lattice points.
        interpolator : scipy.interpolate.interpnd.NDInterpolatorBase
            default scipy.interpolate.LinearNDInterpolator
            The interpolator for the lattice points.
        mapping_resolution : int, default 1024
            The resolution of the internal mapping.
        cos_cutoff : float, default 0.1
            The cosine of the angle at which the hemisphere is cut off.
            We include points outside the hemisphere in the interpolator to
            avoid edge effects at the boundary of the hemisphere.
        """
        self.lattice = lattice_
        self.values = values
        self.interpolator = interpolator
        self.mapping_resolution = mapping_resolution
        self.cos_cutoff = cos_cutoff
        indices_front = np.where(self.lattice[..., 0] > -self.cos_cutoff)
        self.lattice_front = self.lattice[indices_front]
        self.internal_coords_lattice_front = self.map_to_internal_rep(self.lattice_front)
        self.values_front = self.values[indices_front]
        indices_back = np.where(self.lattice[..., 0] < self.cos_cutoff)
        self.lattice_back = self.lattice[indices_back]
        self.internal_coords_lattice_back = self.map_to_internal_rep(self.lattice_back)
        self.values_back = self.values[indices_back]
        output_xx, output_yy = utils_img.img_shape_to_xy((mapping_resolution, mapping_resolution))
        self.output_xy = np.stack([output_xx, output_yy], axis=2)
        self.transparency = utils_img.get_transparency(output_xx)

    def map_to_internal_rep(self, vector, front=True):
        """Map the 3D coordinates to the internal 2D representation.

        Parameters
        ----------
        vector : numpy.ndarray
            The 3D coordinates to be mapped.
        front : bool, default True
            Whether the coordinates are for the front or the back hemisphere.

        Returns
        -------
        numpy.ndarray
            The internal coordinates for the input vector.
        """
        vector = utils_img.normalize(vector)
        temp_internal_coords = utils_img.theta_phi_to_graph_coordinates(
            *utils_img.xyz_to_theta_phi(vector),
            graph_res_=self.mapping_resolution,
            front_=front)
        return np.stack(temp_internal_coords, axis=-1)

    def create_interpolator(self, front=True):
        """Create an interpolator for the front or back hemisphere.

        This function is intended to be used only when the interpolator is
        needed as this is a time-consuming operation.

        Parameters
        ----------
        front : bool, default True
            Whether the interpolator is for the front or the back hemisphere.

        Returns
        -------
        scipy.interpolate.LinearNDInterpolator
            The interpolator for the front or the back hemisphere.
        """
        if front:
            return self.interpolator(
                self.internal_coords_lattice_front,
                self.values_front)
        else:
            return self.interpolator(
                self.internal_coords_lattice_back,
                self.values_back)

    def output(self, front=True):
        """Get the output image.

        Get the result of the interpolation on the 2D grid.

        Parameters
        ----------
        front : bool, default True
            Whether the output is for the front or the back hemisphere.

        Returns
        -------
        numpy.ndarray
            The output image.
        """
        output = self.create_interpolator(front)(self.output_xy)
        output = np.where(np.isnan(output), 0, output)
        output = output * self.transparency
        return output


class ImageSet:
    """Class for storing images and their corresponding weights.

    Attributes
    ----------
    images : numpy.ndarray
        The images.
    weights : ImageSetWeights
        The weights for the images.
        This includes the solid angles, pixel coordinates and cosine weights.
    """
    def __init__(self, images, camera=CAMERA):
        """Initialize the attributes.

        Parameters
        ----------
        images : numpy.ndarray
            The images.
        camera : CameraParameters, default CAMERA
            Parameters of the camera used to render the images.
        """
        self.images = images
        self.weights = ImageSetWeights(images[0].shape, len(images),
                                       camera.fov, camera.fov_axis,
                                       camera.directions, camera.front)

    def get_hemispherical_output(self, weightings=None, front=True):
        """Get the output of the interpolator for a hemisphere.

        Parameters
        ----------
        weightings : list[str, ...], default None
            The weightings to be applied to the images.
            The options are 'solid_angle' and 'cosine'.
            If None, no weightings are applied.
        front : bool, default True
            Whether the output is for the front or the back hemisphere.

        Returns
        -------
        numpy.ndarray
            The output of the interpolator for the hemisphere.
        """
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
        return two_hemispheres_interpolator.output(front=front)
