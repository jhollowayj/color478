import tensorflow as tf
import numpy as np
import colorize

def tf_format(i):
  # The way I write it is channel, row, cols.
  #                       0        1    2
  i = np.asarray(i).transpose((1, 2, 0)) # rows, cols, channel
                                         # 1       2      0
  return i[np.newaxis, ...].astype("float32") # add batch axis

class UpsampleTest(tf.test.TestCase):
  def testUpsample(self):
    with self.test_session():
      i = np.asarray(
          [ [ [ 1, 2, 3 ],
              [ 4, 5, 6 ] ],
            [ [ 7, 8, 9 ],
              [ 10, 11, 12 ] ] ])
      self.assertAllEqual(i.shape, (2,2,3)) # channel, rows, cols
      i = tf_format(i)
      self.assertAllEqual(i.shape, (1,2,3,2))
      batch, rows, cols, depth = i.shape

      result = colorize.upsample(i, repeat=2).eval()

      expected = tf_format([
          [
            [ 1, 1, 2, 2, 3, 3 ],
            [ 1, 1, 2, 2, 3, 3 ],
            [ 4, 4, 5, 5, 6, 6 ],
            [ 4, 4, 5, 5, 6, 6 ],
          ],
          [
            [ 7,   7,  8,  8,  9,  9 ],
            [ 7,   7,  8,  8,  9,  9 ],
            [ 10, 10, 11, 11, 12, 12 ],
            [ 10, 10, 11, 11, 12, 12 ],
          ]
      ]) 
      self.assertAllEqual(expected.shape, (batch, 2 * rows, 2 * cols, depth))
      print result
      self.assertAllEqual(expected, result)


if __name__ == "__main__":
  tf.test.main()
