# USAGE
# python ..\real_time_faciel_landmarks_webcam.py --shape-predictor shape_predictor_68_face_landmarks.dat --input ..\video_xx.mp4 --output ..\video_xx_out.mp4

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-f", "--output_fps", type=int, default=25,
	help="FPS of output video")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["input"]).start()
fileStream = True
time.sleep(1.0)

frame_number = 1

# loop over the frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	
	# grab the frame from the threaded video stream, resize it and convert it to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	if frame_number == 1:
		# get frame dimentions
		height, width = frame.shape[:2]
		# define the codec and create VideoWriter object
		out = cv2.VideoWriter(args["output"],cv2.VideoWriter_fourcc('m','p','4','v'), args["output_fps"], (width,height))
	
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# display the resulting frame and save it to video
	cv2.imshow("Frame", frame)
	out.write(frame)
	
	# press ESC on keyboard to break
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
		
	frame_number = frame_number+1

out.release()

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
