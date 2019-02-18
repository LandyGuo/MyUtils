#coding=utf-8
import numpy as np
import cv2

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import tensorflow as tf


def transformer(U, theta, out_size, name='SpatialTransformer2dAffine'):
    """Spatial Transformer Layer for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__
    , see :class:`SpatialTransformer2dAffineLayer` class.

    Parameters
    ----------
    U : list of float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the localisation network should be [num_batch, 6], value range should be [0, 1] (via tanh).
    out_size: tuple of int
        The size of the output of the network (height, width)
    name: str
        Optional function name

    Returns
    -------
    Tensor
        The transformed tensor.

    References
    ----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    Notes
    -----
    To initialize the network to the identity transform init.


    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0] # 1
            height = tf.shape(im)[1] # 720
            width = tf.shape(im)[2] # 1280
            channels = tf.shape(im)[3] # 3

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            # clip coordinates to [-1, 1]
            x = tf.clip_by_value(x, -1, 1)
            y = tf.clip_by_value(y, -1, 1)

            # scale coordinates from [-1, 1] to [0, width/height-1]
            x = (x + 1) / 2 * (width_f - 1)
            y = (y + 1) / 2 * (height_f - 1)

            # do sampling
            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1
            x0 = tf.cast(x0_f, 'int32')
            y0 = tf.cast(y0_f, 'int32')
            x1 = tf.cast(tf.minimum(x1_f, width_f - 1), 'int32')
            y1 = tf.cast(tf.minimum(y1_f, height_f - 1), 'int32')


            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)


            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t, y_t = tf.meshgrid(tf.linspace(-1., 1., width), tf.linspace(-1., 1., height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

            output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def get_affine_matrix(input_image_shape, coords):
    """
    :param input_image_shape: origin image shape
    :param coords: (8, ) float points in origin image,
                must be the order:[lt_x1,lt_y1,lb_x1,lb_y2,rb_x2,rb_y2,rt_x2,rt_y1]
    :return:
        affine_matrix: M,  used in spatial transform network
        outsize: target [height, width], used in spatial transform network
    """
    # step1: change all coords into -1,1
    img_h, img_w, _ = input_image_shape
    coord_x, coord_y = coords[::2], coords[1::2]
    coord_x, coord_y = coord_x * 2. / img_w - 1, coord_y * 2. / img_h - 1
    scaled_coord = np.stack([coord_x, coord_y], axis=-1)  # 4x2
    source_coord = counter_clock_wise_coord = scaled_coord[:3]

    # step2: cal outsize
    four_point_coord = coords.reshape(-1, 2)
    side1 = np.sqrt(np.sum(np.square(four_point_coord[0] - four_point_coord[1])))
    side2 = np.sqrt(np.sum(np.square(four_point_coord[1] - four_point_coord[2])))

    long_side, short_side = max(side1, side2), min(side1, side2)
    outsize = (int(short_side), int(long_side)) # [height, width] output
    outsize = (32, int(32.*outsize[1]/outsize[0])) # resize height to 32 for OCR

    # step3: use triangle define affine matrix, all transform are defined in image coord-system:[-1,1]
    if side1 < side2: # horizental box
        dest_coord = np.float32([[-1, -1],
                                 [-1, 1],
                                 [1, 1]])

    else: # vertical box
        dest_coord = np.float32([[-1, 1],
                                  [1, 1],
                                  [1, -1]])

    # Note: here we need dst->src inverse transform
    affine_matrix = cv2.getAffineTransform(dest_coord, source_coord)

    # Note: opencv apply affine matrix only work in original image space, not in [-1,1]. try yourself
    # warped = cv2.warpAffine(input_image, affine_matrix, (out_size[1], out_size[0]), cv2.INTER_LINEAR)
    # cv2.imwrite('warpped_image.jpg', warped)

    return affine_matrix, outsize

# TODO3: extend to perspective transform

if __name__=='__main__':
    input_image = cv2.imread("samples/00067e3bd541346bb58e8bd073dba73c_f5e82200-5f18-4b34-8958-699bc3740a46.jpg")

    # coords in counter_clock-wise, lty, ltx, lby, lbx,...
    coords = np.float32([139.84, 115.20, 133.44, 149.76, 485.44, 205.44, 490.56, 161.92])

    from rotate_utils.rotate_coords_transform import RotateCoord
    # from rotate_utils.vis import mold_vertext_on_image
    # imgnp = mold_vertext_on_image(input_image, coords.reshape(-1, 2), 'VINCODE', 1.0)
    cv2.imwrite('input.jpg', input_image)

    affine_matrix, out_size = get_affine_matrix(input_image.shape, coords)

    theta = tfe.Variable(initial_value=affine_matrix.flatten())
    output = transformer(tf.expand_dims(input_image, 0), theta, out_size=out_size).numpy()[0]

    print("output shape:", output.shape)

    cv2.imwrite('out_final6.jpg', output)

