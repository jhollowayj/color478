import sys
from PIL import Image

class Grayscaler:
    image = ""
    def __init__(self, image=""):
        self.image = image
    def setImage(image):
        self.image = image
    def save(self, name):
        self.image.save(name)

    def average(self):
        '''Average the red, green, and blue values'''
        pixels = []
        for p in self.image.getdata():
            val = (p[0] + p[1] + p[2]) / 3
            pixels.append((val, val, val))
        self.image.putdata(pixels)

    def desaturate(self):
        '''Perform desaturation to grayscale an image'''
        pixels = []
        for p in self.image.getdata():
            gray = (int)(max(p[0], p[1], p[2]) + min(p[0], p[1], p[2])) / 2
            pixels.append((gray, gray, gray))
        self.image.putdata(pixels)
        
    def luminosity(self):
        '''Use the luminosity approach to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            val = ((int)(p[0]*.3) + (int)(p[1]*.59) + (int)(p[2]*0.11))
            pixels.append((val, val, val))
        self.image.putdata(pixels)

    def min(self):
        '''Use the min of the RGB values to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            gray = min(p[0], p[1], p[2])
            pixels.append((gray, gray, gray))
        self.image.putdata(pixels)

    def max(self):
        '''Use the max of the RGB values to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            gray = max(p[0], p[1], p[2])
            pixels.append((gray, gray, gray))
        self.image.putdata(pixels)

    def red(self):
        '''Use the red value to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            pixels.append((p[0], p[0], p[0]))
        self.image.putdata(pixels)

    def green(self):
        '''Use the green value to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            pixels.append((p[1], p[1], p[1]))
        self.image.putdata(pixels)

    def blue(self):
        '''Use the blue value to grayscale the image'''
        pixels = []
        for p in self.image.getdata():
            pixels.append((p[2], p[2], p[2]))
        self.image.putdata(pixels)

if len(sys.argv) < 3:
    print "Usage: python gray.py <file_name> <method> <saved_file_name>"
    print "Available methods:\n","  average\n","  luminosity\n"
    sys.exit(1)

# Open image
try:
    image = Image.open(sys.argv[1])
except:
    print "Unable to open the specified file"
    sys.exit(1)

# See http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
# for details on the algorithms used
g = Grayscaler(image)
if sys.argv[2].lower() == "average":
    g.average()
elif sys.argv[2].lower() == "luminosity":
    g.luminosity()
elif sys.argv[2].lower() == "desaturate":
    g.desaturate()
elif sys.argv[2].lower() == "min":
    g.min()
elif sys.argv[2].lower() == "max":
    g.max()
elif sys.argv[2].lower() == "red":
    g.red()
elif sys.argv[2].lower() == "green":
    g.green()
elif sys.argv[2].lower() == "blue":
    g.blue()
else:
    print "Invalid method specified, aborting."
    sys.exit(1)

file_name = "bw_"+sys.argv[1]
if len(sys.argv) > 3:
    file_name = sys.argv[3]

g.save(file_name)
