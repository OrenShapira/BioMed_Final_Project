# USAGE
# python ..\file_video_stream_read_slow.py --video ..\video_xx.mp4 --output ..\video_xx_out.mp4

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-f", "--output_fps", type=int, default=25,
	help="FPS of output video")
args = vars(ap.parse_args())

# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture(args["video"])
# start the FPS timer
fps = FPS().start()
frame_number = 1

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video file stream
	(grabbed, frame) = stream.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break
		
	# resize the frame and convert it to grayscale (while still retaining 3 channels)
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])

	if frame_number == 1:
		# get frame dimentions
		height, width = frame.shape[:2]
		# define the codec and create VideoWriter object
		out = cv2.VideoWriter(args["output"],cv2.VideoWriter_fourcc('m','p','4','v'), args["output_fps"], (width,height))
	
	# display a piece of text to the frame (so we can benchmark fairly against the fast method)
	cv2.putText(frame, "Slow Method", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)	
	# display the frame number
	cv2.putText(frame, "Frame idx: {}".format(frame_number), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
	# display the resulting frame and save it to video
	cv2.imshow("Frame", frame)
	out.write(frame)
	
	# press ESC on keyboard to break
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break
	
	# update the FPS counter
	fps.update()
	frame_number = frame_number+1

# stop the timer and display FPS information
out.release()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()