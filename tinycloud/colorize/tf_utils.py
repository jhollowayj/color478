import tensorflow as tf
import numpy as np

def _get_image(batch, index):
  shape = batch.get_shape().as_list()
  i = tf.slice(batch, [index, 0, 0, 0], [1, -1, -1, -1])
  return tf.reshape(i, shape[1:])

def assert_unit_length(t):
  between = tf.logical_and(tf.greater_equal(t, 0.0), tf.less_equal(t, 1.0))
  return tf.Assert(tf.reduce_all(between), [tf.reduce_max(t), tf.reduce_min(t)])

def build_summary_image(grayscale, inferred_rgb, rgb, show_input=False):
  shape = rgb.get_shape().as_list()
  batch_size = shape[0]
  h = shape[1]
  w = shape[2]

  if show_input:
    left   = tf.pad(grayscale,    [[0, 0], [0, 0], [0, 2*w], [0, 0]])
    middle = tf.pad(inferred_rgb, [[0, 0], [0, 0], [w,   w], [0, 0]])
    right  = tf.pad(rgb,          [[0, 0], [0, 0], [2*w, 0], [0, 0]])
    return left + middle + right

  else:
    left = tf.pad(inferred_rgb, [[0, 0], [0, 0], [0, w], [0, 0]])
    right  = tf.pad(rgb,        [[0, 0], [0, 0], [w, 0], [0, 0]])
    return left + right


def differentiable_mod(a, b):
  # Assume b is an integer
  b_factors = tf.floordiv(a, b)
  x = a - b * b_factors
  return tf.select(x < 0, x + b, x)

# https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
# using the JPEG ersion of YCbCr 
def rgb2yuv(rgb): 
  rgb *= 255

  # Y'  =   0 + (0.299     * R) + (0.587     * G) + (0.114     * B)
  # Cb = 128 - (0.168736  * R) - (0.331264  * G) + (0.5       * B)
  # Cr = 128 + (0.5       * R) - (0.418688  * G) - (0.081312  * B)

  w = [[[
    [ .299, -0.168736, 0.5], 
    [ .587, -0.331264, -0.418688], 
    [ .114, 0.5, -0.081312],
  ]]]

  b = [ 0, 128, 128 ]   

  filter = tf.constant(w, dtype="float", name="rgb2yuv_filter")
  bias = tf.constant(b, dtype="float", name="rgb2yuv_bias")
  yuv = tf.nn.conv2d(rgb, filter, [1, 1, 1, 1], "SAME")
  yuv = tf.nn.bias_add(yuv, bias) / 255
  return tf.clip_by_value(yuv, 0.0, 1.0)

def yuv2rgb(yuv, name=None):
  yuv *= 255

  #  R  = Y + 1.402    * (Cr - 128) 
  #     = Y + 1.402 * Cr - 179.456
  #  G  = Y - 0.34414  * (Cb - 128)  - 0.71414  * (Cr - 128)
  #     = Y - 0.34414 * Cb  + 128 * 0.34414 - 0.71414 * Cr + 128 * 0.71414
  #     = Y - 0.34414 * Cb  - 0.71414 * Cr  + 135.459839
  #  B  = Y + 1.772    * (Cb - 128)
  #     = Y + 1.772 * Cb - 226.816
  w = [[[
    [ 1,     1,         1     ],
    [ 0,     -0.34414 , 1.772 ],
    [ 1.402, -0.71414,  0     ],
  ]]]
  b = [ -179.456, 135.459839, -226.816 ]

  filter = tf.constant(w, dtype="float32", name="yuv2rgb_filter")
  bias = tf.constant(b, dtype="float32", name="yuv2rgb_bias")
  rgb = tf.nn.conv2d(yuv, filter, [1, 1, 1, 1], padding="SAME")
  rgb =  tf.nn.bias_add(rgb, bias) / 255
  return tf.clip_by_value(rgb, 0.0, 1.0, name=name)

# assume rgb values [0, 1] and shaped [ batch, height, width, 3 ]
# http://www.rapidtables.com/convert/color/rgb-to-hsl.htm
def rgb2hsl(rgb):
  red, green, blue = tf.split(3, 3, rgb)

  Cmax = tf.reduce_max(rgb, reduction_indices=[3], keep_dims=True)
  Cmin = tf.reduce_min(rgb, reduction_indices=[3], keep_dims=True)
  delta = Cmax - Cmin

  zero = tf.zeros_like(Cmax)

  epsilon = 0.0001

  delta_is_zero = tf.less(delta, epsilon)
  max_is_red = tf.less(Cmax - red, epsilon)
  max_is_green = tf.less(Cmax - green, epsilon)
  max_is_blue = tf.less(Cmax - blue, epsilon)

  hue = tf.select(delta_is_zero, zero,
    tf.select(max_is_red, 60 * differentiable_mod((green - blue) / delta, 6),
    tf.select(max_is_green, 60 * ( (blue - red) / delta + 2 ),
    60 * (( red - green) / delta + 4)
  )))

  L = (Cmax + Cmin) / 2

  sat = tf.select(delta_is_zero, zero, delta / (1 - tf.abs(2*L - 1)))

  hsl = tf.concat(3, [ hue / 360, sat, L ])

  return hsl

# Where 0 <= H < 1, 0 <= S <= 1 and 0 <= L <= 1
# http://www.rapidtables.com/convert/color/hsl-to-rgb.htm
def hsl2rgb(hsl):
  H, S, L = tf.split(3, 3, hsl)

  H = H * 360

  zero = tf.zeros_like(L)

  C = (1 - tf.abs(2 * L - 1)) * S
  X = C * (1 - tf.abs(differentiable_mod(H / 60, 2) - 1))
  m = L - C/2

  H_less_60 = tf.less(H, 60.0)
  H_less_120 = tf.less(H, 120.0)
  H_less_180 = tf.less(H, 180.0)
  H_less_240 = tf.less(H, 240.0)
  H_less_300 = tf.less(H, 300.0)

  red = tf.select(H_less_60, C,
    tf.select(H_less_120, X,
    tf.select(H_less_180, zero,
    tf.select(H_less_240, zero,
    tf.select(H_less_300, X, C)
  ))))

  green = tf.select(H_less_60, X,
    tf.select(H_less_120, C,
    tf.select(H_less_180, C,
    tf.select(H_less_240, X,
    tf.select(H_less_300, zero, zero)
  ))))

  blue = tf.select(H_less_60, zero,
    tf.select(H_less_120, zero,
    tf.select(H_less_180, X,
    tf.select(H_less_240, C,
    tf.select(H_less_300, C, X)
  ))))
        
  rgb = tf.concat(3, [red, green, blue]) + m

  return rgb 


def assert_almost_eq(a, b, name=None, decimal=3):
  e = 10 ** -decimal
  dist = tf.sqrt(tf.reduce_sum(tf.square(a - b)))
  return tf.Assert(dist < e, [dist], name=name)


def gaussianKernel(size, fwhm = 3):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def blur(img, n=5):
    #    return tf.nn.avg_pool(img, [1,2,2,1], [1,1,1,1], padding='SAME')
    depth = img.get_shape().as_list()[-1]
    k = np.zeros((n, n, depth, depth))
    for i in range(0, depth):
        k[:,:,i,i] = gaussianKernel(n) ## i suck at numpy
    filt = tf.constant(k, dtype=tf.float32)
    return tf.nn.conv2d(img, filt, [1,1,1,1], padding="SAME")
