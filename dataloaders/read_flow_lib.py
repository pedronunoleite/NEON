import numpy as np
import cv2

def read_flow(flow_or_path, quantize=False, *args, **kwargs):
    """Read an optical flow map
    Args:
        flow_or_path(ndarray or str): either a flow map or path of a flow
        quantize(bool): whether to read quantized pair, if set to True,
                        remaining args will be passed to :func:`dequantize_flow`
    Returns:
        ndarray: optical flow
    """
    if isinstance(flow_or_path, np.ndarray):
        if (flow_or_path.ndim != 3) or (flow_or_path.shape[-1] != 2):
            raise ValueError(
                'Invalid flow with shape {}'.format(flow_or_path.shape))
        return flow_or_path
    elif not isinstance(flow_or_path, str):
        raise TypeError(
            '"flow_or_path" must be a filename or numpy array, not {}'.format(
                type(flow_or_path)))

    if not quantize:
        with open(flow_or_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except:
                raise IOError('Invalid flow file: {}'.format(flow_or_path))
            else:
                if header != 'PIEH':
                    raise IOError(
                        'Invalid flow file: {}, header does not contain PIEH'.
                        format(flow_or_path))

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
    
    return flow.astype(np.float32)

def make_color_wheel(bins=None):
    """Build a color wheel
    Args:
        bins(list or tuple, optional): specify number of bins for each color
            range, corresponding to six ranges: red -> yellow, yellow -> green,
            green -> cyan, cyan -> blue, blue -> magenta, magenta -> red.
            [15, 6, 4, 11, 13, 6] is used for default (see Middlebury).
    
    Returns:
        ndarray: color wheel of shape (total_bins, 3)
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image
    Args:
        flow(ndarray): optical flow
        color_wheel(ndarray or None): color wheel used to map flow field to RGB
            colorspace. Default color wheel will be used if not specified
        unknown_thr(str): values above this threshold will be marked as unknown
            and thus ignored
    
    Returns:
        ndarray: an RGB image that can be visualized
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |  (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    [h, w] = dx.shape

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img


def rgb2bgr(img):
    """Convert a RGB image to BGR image
    Args:
        img(ndarray or str): either an image or path of an image
    Returns:
        ndarray: the BGR image
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return out_img

def show_img(img, win_name='', wait_time=0):
    """Show an image
    Args:
        img(str or ndarray): the image to be shown
        win_name(str): the window name
        wait_time(int): value of waitKey param
    """
    from PIL import Image
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.show()
    