import sys
import random
from PIL import Image

def get_mean(pixels):
    r = 0
    g = 0
    b = 0
    for p in pixels:
        r += p[0]
        g += p[1]
        b += p[2]
    return (r / len(pixels), g / len(pixels), b / len(pixels))

def get_random(pixels):
    index = int(random.random() * len(pixels))
    return pixels[index]

def get_median(pixels):
    r = []
    g = []
    b = []
    for p in pixels:
        r.append(p[0])
        g.append(p[1])
        b.append(p[2])
    r.sort()
    g.sort()
    b.sort()
    return (r[len(pixels)/2], g[len(pixels)/2], b[len(pixels)/2])

def get_mode(pixels):
    r = {}
    g = {}
    b = {}
    for p in pixels:
        try:
            r[p[0]] += 1
        except:
            r[p[0]] = 1
        try:
            g[p[1]] += 1
        except:
            g[p[1]] = 1
        try:
            b[p[2]] += 1
        except:
            b[p[2]] = 1
    m = 0
    red = 0
    green = 0
    blue = 0
    for k, v in r.items():
        if v > m:
            red = k
            m = v
    m = 0
    for k, v in g.items():
        if v > m:
            green = k
            m = v
    m = 0
    for k, v in b.items():
        if v > m:
            blue = k
            m = v
    return (red, green, blue)

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
        #pixels.append(data[p[0]][0])
        #pixels.append(get_mean(data[p[0]]))
        #pixels.append(get_median(data[p[0]]))
        #pixels.append(get_mode(data[p[0]]))
        pixels.append(get_random(data[p[0]]))
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
                #pixels.append(data[val][0])
                #pixels.append(get_mean(data[val]))
                #pixels.append(get_median(data[val]))
                #pixels.append(get_mode(data[val]))
                pixels.append(get_random(data[val]))
                break
image2.putdata(pixels)

file_name = "col_"+sys.argv[2]
image2.save(file_name)
