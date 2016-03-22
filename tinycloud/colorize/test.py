import colorize
import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf
import tf_utils

def rgb_equal(a, expected):
  assert_array_almost_equal(a, np.array(expected) / 255.0, decimal=1)

def assert_equal(a, b):
  assert_array_almost_equal(a, b, decimal=2)

def test_yuv():
  white = [1.0, 1.0, 1.0]
  black = [0.0, 0.0, 0.0]
  olive = [0.5, 0.5, 0.0]
  green = [ 36.0 / 255, 85.0 / 255, 40.0 / 255 ]

  i = [ [ white, black ],
        [ olive, green ] ]

  rgb_batch = np.asarray(i, dtype="float32").reshape((1, 2, 2, 3)) 

  yuv = tf_utils.rgb2yuv(rgb_batch).eval()
  print yuv

  yuv_white = [1, 0.5, 0.5]
  yuv_black = [0, 0.5, 0.5]
  yuv_olive = [0.4447, 0.2509, 0.5427]
  yuv_green = [0.2557, 0.4461, 0.4202]
  
  assert_equal(yuv[0][0][0], yuv_white)
  assert_equal(yuv[0][0][1], yuv_black)
  assert_equal(yuv[0][1][0], yuv_olive)
  assert_equal(yuv[0][1][1], yuv_green)


  rgb_again = tf_utils.yuv2rgb(yuv).eval()

  assert_equal(rgb_again[0][0][0], white)
  assert_equal(rgb_again[0][0][1], black)
  assert_equal(rgb_again[0][1][0], olive)
  assert_equal(rgb_again[0][1][1], green)

  print "yuv ok!"
 
def test_hsl():
  white = [1.0, 1.0, 1.0]
  black = [0.0, 0.0, 0.0]
  olive = [0.5, 0.5, 0.0]
  green = [ 36.0 / 255, 85.0 / 255, 40.0 / 255 ]

  i = [ [ white, black ],
        [ olive, green ] ]

  rgb_batch = np.array(i).reshape((1, 2, 2, 3)) 

  hsl = tf_utils.rgb2hsl(rgb_batch).eval()
  
  white_hsl = hsl[0][0][0]
  black_hsl = hsl[0][0][1]
  olive_hsl = hsl[0][1][0]
  green_hsl = hsl[0][1][1]

  print "white_hsl", white_hsl
  assert_equal(white_hsl, [0, 0, 1])

  print "black_hsl", black_hsl
  assert_equal(black_hsl, [0, 0, 0])

  print "olive_hsl", olive_hsl
  assert_equal(olive_hsl, [60 / 360.0, 1.0, 0.25])

  print "green_hsl", green_hsl
  assert_equal(green_hsl,
      [124.897959 / 360.0,    0.404959,    0.237255])

  # check that we can reverse what we just did.
  rgb = tf_utils.hsl2rgb(hsl).eval()

  assert_equal(rgb[0][0][0], white)
  assert_equal(rgb[0][0][1], black)
  assert_equal(rgb[0][1][0], olive)
  assert_equal(rgb[0][1][1], green)

  assert_equal(rgb, rgb_batch)


  a = [ 77 / 360.0, 0.45, 0.79 ] 
  a_rgb = [ 211, 225, 177 ]

  b = [ 290 / 360.0, 0.96, 0.96 ]
  b_rgb = [ 251, 235, 254 ] 

  c = [ 0 / 360.0, 0.24, 0.96 ]
  c_rgb = [ 247, 242, 242 ]

  d = [ 60 / 360.0, 0.0, 1.0 ]
  d_rgb = [ 255, 255, 255 ] 

  hsl = [ [ a, b ], [ c, d ] ]
  hsl_batch = np.array(hsl).reshape((1, 2, 2, 3))

  rgb = tf_utils.hsl2rgb(hsl_batch).eval()

  rgb_equal(rgb[0][0][0], a_rgb)
  rgb_equal(rgb[0][0][1], b_rgb)
  rgb_equal(rgb[0][1][0], c_rgb)
  rgb_equal(rgb[0][1][1], d_rgb)

def test_differentiable_mod():
  a = [ 13, 13.5, -13, -11.5 ] 
  r = tf_utils.differentiable_mod(a, 6).eval()
  assert_equal(r, [ 1, 1.5, 5, .5 ])


if __name__ == '__main__':
  sess = tf.Session()
  with sess.as_default():
    test_hsl()
    test_yuv()
    test_differentiable_mod()
