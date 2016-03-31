import sys
import cv2
import re
from os import listdir
from os.path import isfile, join
import operator

if len(sys.argv) < 2:
    print "Usage: video.py <image_directory>"
    sys.exit(1)

# Determine directory
try:
    image_dir = sys.argv[1]
except:
    print "Unable to open the specified file"
    sys.exit(1)

image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
indexes = {}
for f in image_files:
    indexes[int(re.findall(r'\d+',f)[0])] = f
indexes = sorted(indexes.items(), key=operator.itemgetter(0))
files = []
for k, f in indexes:
    files.append(image_dir + "/" + f)
height, width, layers = cv2.imread(files[0]).shape
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
video = cv2.VideoWriter('video.avi', fourcc, 20.0, (width,height))
for f in files:
    video.write(cv2.imread(f))
cv2.destroyAllWindows()
video.release()
