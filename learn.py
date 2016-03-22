import sys
from PIL import Image

if len(sys.argv) < 3:
    print "Usage: python learn.py <training_file> <test_file>"
    sys.exit(1)

# Open image
try:
    image = Image.open(sys.argv[1])
    image2 = Image.open(sys.argv[2])
except:
    print "Unable to open the specified file"
    sys.exit(1)

# Learn
data = {}
for p in image.getdata():
    av = (p[0] + p[1] + p[2]) / 3
    if not av in data:
        data[av] = [(p[0], p[1], p[2])]
    else:
        data[av].append((p[0], p[1], p[2]))

# Predict
pixels = []
for p in image2.getdata():
    if p[0] in data:
        pixels.append(data[p[0]][0])
    else:
        val = p[0]
        diff = 0
        while True:
            if val == 0:
                pixels.append((0, 0, 0))
                break
            else:
                val -= 1
            if val in data:
                pixels.append(data[val][0])
                break
image.putdata(pixels)

file_name = "col_"+sys.argv[2]
image.save(file_name)
