import cv2
import optparse

# Parse arguments
parser = optparse.OptionParser(usage="%prog [options]")
parser.add_option('-f','--file', dest='file_name',
                   default='', type="str",
                   help='The video file to split into images')
parser.add_option('-n','--number', dest='num_images',
                   default=1, type="int",
                   help='The number of images to capture')
parser.add_option('-i','--interval', dest='interval',
                   default=24, type="int",
                   help='The sampling interval, e.g. create picture every 24th frame')
(options,args) = parser.parse_args()

# Arguments
interval = options.interval
num_images = options.num_images
video_file = options.file_name

count = 0
images = 0
success = True
vidcap = cv2.VideoCapture(video_file)
while success and images < num_images:
    success,image = vidcap.read()
    if success and count % interval == 0:
        images += 1
        cv2.imwrite("frame%d.jpg" % count, image)
    elif count == 0 and not success:
        print "Unable to parse video file %s\n" % video_file
    count += 1
