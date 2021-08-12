import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class GeologyScaler(object):
    def __init__(self, img, lim=[0, 1]):
        if len(img.shape) == 3:
            # print('Apply GeologyScaler to 3 channel data')
            vmax = np.array([img[:, :, i].max() for i in range(3)])
            vmin = np.array([img[:, :, i].min() for i in range(3)])
            self.vmax = vmax.reshape(1, 1, 3)
            self.vmin = vmin.reshape(1, 1, 3)
        else:
            self.vmax = img.max()
            self.vmin = img.min()
        self.lmin, self.lmax = lim

    # From geological scale to reference scale
    def g2r(self, img):
        return (img - self.vmin) * (self.lmax - self.lmin) / (self.vmax - self.vmin) + self.lmin

    # From reference scale to geological scale
    def r2g(self, img):
        return (img - self.lmin) * (self.vmax - self.vmin) / (self.lmax - self.lmin) + self.vmin


def get_reflectivity(h, k=0.85):
    r = -1 + 2 * np.random.random(h)
    mask = np.random.random(h)
    mask[mask < k] = 0
    return  np.multiply(0.25 * r, mask)


def ref2vel(r, v0):
    # Assuming rho=1, then R = (v2 - v1) / (v2 + v1)
    vel = np.zeros(len(r) + 1)
    vel[0] = v0
    for i in range(len(vel) - 1):
        vel[i+1] = vel[i] * (r[i] + 1) / (1 - r[i])
    return vel[:-1]


def get_trend(h, v0, v1):
    idx = np.arange(h)
    return v0 + idx * (v1 - v0) / h


def get_2d_layered_model(h, w, vmin_trend=1, vmax_trend=1, dv0=1, vmin=0., vmax=1.):#, alpha_x, alpha_z, beta_x, beta_z):
    """
    Args:
        more_layers (float): 0..1, makes layers finer
    """
    r = get_reflectivity(h)
    vel = ref2vel(r, dv0)
    trend = get_trend(h, vmin_trend, vmax_trend)
    v = trend + vel
    vrand = np.repeat(np.expand_dims(v,1), w, 1) / np.max(v)
    # vrand = np.repeat(np.expand_dims(vrand, -1), 3, -1)
    # vrand = elastic_transform(vrand, alpha_x, alpha_z, beta_x, beta_z)
    return vrand


def to_3D(img):
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
        return np.concatenate((img, img, img), axis=2)
    else:
        return img


def elastic_transform(image, alpha_x, alpha_z, sigma_x, sigma_z, seed=None, mode='mirror'):
    """Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis", in Proc. of the International Conference on Document Analysis and Recognition, 2003.
    .. Vladimir Kazei, 2019; Oleg Ovcharenko, 2019
        mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
        Default is 'mirror'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    """
    image = to_3D(image)

    if seed:
        random_state_number = int(seed)
    else:
        random_state_number = np.random.randint(1, 1000)

    geo_before = GeologyScaler(image)
    random_state = np.random.RandomState(random_state_number)
    shape = image.shape
    # print(shape)
    # with our velocities dx is vertical shift
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma_x, sigma_z, 1), mode="constant", cval=0) * alpha_x
    # with our velocities dy is horizontal shift
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), (sigma_x, sigma_z, 1), mode="constant", cval=0) * alpha_z
    dz = np.zeros_like(dx)

    dx[..., 1] = dx[..., 0]
    dx[..., 2] = dx[..., 0]
    dy[..., 1] = dy[..., 0]
    dy[..., 2] = dy[..., 0]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode=mode, prefilter=False)
    distorted_image = distorted_image.reshape(image.shape)

    geo = GeologyScaler(distorted_image)
    distorted_image = geo.g2r(distorted_image)
    distorted_image = geo_before.r2g(distorted_image)
    return distorted_image