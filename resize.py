import sys
from PIL import Image

if len(sys.argv) < 4:
    print "Usage: python resize.py <file_name> <width> <height>"
    sys.exit(1)

try:
    image = Image.open(sys.argv[1])
except:
    print "Unable to open the specified file"
    sys.exit(1)

image.resize([int(sys.argv[2]),int(sys.argv[3])]).save("resized_"+sys.argv[1])
