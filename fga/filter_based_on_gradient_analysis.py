# Vladimir Gudkov, Ilia Moiseev
# South Ural State University, Chelyabinsk, Russia, 2020
# Image smoothing Algorithm Based on Gradient Analysis

import numpy as np


def _grad(x, y, image):
    gradx = image[y][x - 1] - image[y][x + 1]
    grady = image[y + 1][x] - image[y - 1][x]
    return [gradx, grady]


def compute_grads_channel(image, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                grads[y, x] = _grad(x, y, image)


def compute_grads(image, grads):
    for i in range(3):
        compute_grads_channel(image[:, :, i], grads[:, :, :, i])


def compute_modules_channel(image, modules, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                modules[y][x] = np.linalg.norm(grads[y][x])


def compute_modules(image, modules, grads):
    for i in range(3):
        compute_modules_channel(image[:, :, i], modules[:, :, i], grads[:, :, :, i])


def compute_angles_channel(image, angles, grads):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if 0 < x < image.shape[1] - 1 and 0 < y < image.shape[0] - 1:
                g = grads[y, x]
                angle = np.arctan2(g[1], g[0])
                angles[y, x] = angle if not np.isnan(angle) else 0


def compute_angles(image, angles, grads):
    for i in range(3):
        compute_angles_channel(image[:, :, i], angles[:, :, i], grads[:, :, :, i])


def smooth_channel(src, k_size, n=1, grads=None, modules=None, angles=None, dst=None):
    """
    Smooth 1-channel image with filter based on gradient analysis

    :param src: source image with shape (n, m, 3)
    :param k_size: kernel size - odd integer
    :param n: the number of sequential runs
    :param grads: gradient vectors for each pixel with shape (n, m, 2, 3)
    :param modules: gradient modules for each pixel with shape (n, m, 3)
    :param angles: gradient angles for each pixel with shape (n, m, 3)
    :param dst: destination image with the same shape of type np.float64
    :return: only if dst is not None: smoothed image with same shape as src and type np.float64
    """

    src_proxy = np.copy(src)
    if dst is None:
        dst = np.zeros(src.shape, np.float64)
    for i in range(n):
        if i == 0:
            _smooth_channel(src_proxy, k_size, grads=grads, modules=modules, angles=angles, dst=dst)
        else:
            _smooth_channel(src_proxy, k_size, dst=dst)
        src_proxy = dst
    if dst is not None:
        return dst


def _smooth_channel(src, k_size, grads=None, modules=None, angles=None, dst=None):
    if dst is None:
        dst = np.zeros(src.shape, dtype=np.float64)
    if grads is None:
        grads = np.zeros((src.shape[0], src.shape[1], 2))
        compute_grads_channel(src.astype(np.float64), grads)
    if modules is None:
        modules = np.zeros((src.shape[0], src.shape[1]))
        compute_modules_channel(src.astype(np.float64), modules, grads)
    if angles is None:
        angles = np.zeros((src.shape[0], src.shape[1]))
        compute_angles_channel(src.astype(np.float64), angles, grads)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            up = i - k_size // 2
            left = j - k_size // 2
            down = i + k_size // 2 + 1
            right = j + k_size // 2 + 1
            sum_weights = 0
            result = 0
            for s in range(up, down):
                if s < 0 or s >= src.shape[0]:
                    continue
                for t in range(left, right):
                    if t < 0 or t >= src.shape[1]:
                        continue
                    if modules[s][t] == .0:
                        continue

                    if s != i or t != j:
                        alpha = 1. / modules[s][t]
                        beta = 2. * (angles[i][j] - angles[s][t])
                        weight = (np.cos(beta) + 1) * alpha
                    else:
                        # weight of central pixel
                        weight = 1.
                    result += weight * src[s][t]
                    sum_weights += weight
            if sum_weights != 0:
                dst[i][j] = round(result / sum_weights)
            else:
                # pixel remains without changes if sum of weights = 0
                dst[i][j] = src[i][j]
    if dst is not None:
        return dst


def _smooth(src, dst, k_size, grads=None, modules=None, angles=None):
    if grads is None:
        grads = np.zeros((src.shape[0], src.shape[1], 2, 3))
        compute_grads(src.astype(np.float64), grads)
    if modules is None:
        modules = np.zeros((src.shape[0], src.shape[1], 3))
        compute_modules(src.astype(np.float64), modules, grads)
    if angles is None:
        angles = np.zeros((src.shape[0], src.shape[1], 3))
        compute_angles(src.astype(np.float64), angles, grads)

    for i in range(3):
        smooth_channel(src[:, :, i].astype(np.float64),
                       k_size,
                       grads=grads[:, :, :, i],
                       modules=modules[:, :, i],
                       angles=angles[:, :, i],
                       dst=dst[:, :, i])
    return dst


def smooth(src, k_size, n=1, grads=None, modules=None, angles=None):
    """
    Smooth 3-channel image with filter based on gradient analysis

    :param src: source image with shape (n, m, 3)
    :param k_size: kernel size - odd integer
    :param n: the number of sequential runs
    :param grads: gradient vectors for each pixel with shape (n, m, 2, 3)
    :param modules: gradient modules for each pixel with shape (n, m, 3)
    :param angles: gradient angles for each pixel with shape (n, m, 3)
    :return: smoothed image with same shape as src and type np.float64
    """
    if k_size % 2 == 0:
        raise ValueError(f'k_size should be odd, got {k_size} instead')

    src_proxy = np.copy(src)
    dst = np.zeros(src.shape, np.float64)

    if n == 1:
        _smooth(src_proxy, dst, k_size, grads=grads, modules=modules, angles=angles)
    else:
        for i in range(n):
            _smooth(src_proxy, dst, k_size)
            src_proxy = dst
    return dst
