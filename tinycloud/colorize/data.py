import os
import numpy as np
import skimage.io
import skimage.transform
from os.path import isfile, join

def save_image(path, a):
    skimage.io.imsave(path, a)

# Returns a numpy array of shape [height, width, 3]
def load_image(path, allow_bw=False):
    # load image
    img = skimage.io.imread(path)

    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    #print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 224, 224

    img = skimage.transform.resize(crop_img, (224, 224))

    # if it's a black and white photo, we need to change it to 3 channel
    # or raise an error if we're not allowing b&w (which we do during training)
    if len(img.shape) == 2:
        if allow_bw:
            img = np.stack([img, img, img], axis=-1)
        else:
            raise ValueError("load_image doesn't handle black and white photos")

    assert img.shape == (224, 224, 3)

    return img

class DataSet(object):
    def __init__(self, dir_txt):
        self.data_dir = os.path.splitext(dir_txt)[0]
        self._filenames = []
        with open(dir_txt, 'r') as f:
            for line in f:
                if line[0] == '.': continue
                line = line.rstrip()
                fn = os.path.join(self.data_dir, line)
                self._filenames.append(fn)

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._shuffle_filenames()

    def _shuffle_filenames(self):
        np.random.shuffle(self._filenames)

    def length(self):
        return len(self._filenames)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def epoch_complete(self):
        # Finished epoch
        self._epochs_completed += 1
        # Shuffle the data
        self._shuffle_filenames()
        # Start next epoch
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= len(self._filenames)

        imgs = []
        while len(imgs) < batch_size:
            try:
                fn = self._filenames[self._index_in_epoch]
                img = load_image(fn)
                imgs.append(img)
                self._index_in_epoch += 1 
                if self._index_in_epoch >= len(self._filenames):
                    self.epoch_complete()
            except ValueError:
                print "error loading", fn
                del self._filenames[self._index_in_epoch]

        batch = np.stack(imgs)
        assert batch.shape == (batch_size, 224, 224, 3)
        
        return batch

