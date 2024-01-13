import cv2
from settings import set



def adjust_brightness(frame):
	"""
	Function to adjust the brightness of te input webcam feed.
	This function was written by ChatGPT but have been tested
	and does the work.
	"""
	alpha = (set['filter']['brightness'] + 50) / 50.0
	adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
	return adjusted_frame



def adjust_contrast(frame):
	"""
	Function to adjust the cotrast of te input webcam feed.
	This function was written by ChatGPT but have been tested
	and does the work.
	"""
	alpha = (set['filter']['contrast'] + 50) / 50.0
	beta = 128 * (1 - alpha)
	adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
	return adjusted_frame



def apply_filters(frame):

	"""
	The apply_filter function simply appllies different filters and modifies the input so that
	edge detection can be done with a higher accuracy. Some of the action done are required,
	such as grayscaling, while others are choosen by the user via different inputs to work best
	in that specific case (such as brightness and contrast).
	"""
			
	# Convert the frame to grayscale for better contour detection.
	filtered_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Invert bitwise.
	filtered_frame = cv2.bitwise_not(filtered_frame)
			
	# Adjust brightness.
	filtered_frame = adjust_brightness(filtered_frame)
			
	# Adjust contrast.
	filtered_frame = adjust_contrast(filtered_frame)
		
	# Add threshold
	retval, filtered_frame = cv2.threshold(filtered_frame, set['filter']["threshold"], 255, cv2.THRESH_BINARY_INV) # Note that retval isn't used but much be captured.

	# Apply a blur to the frame to reduce noise.
	if set['filter']['blur'] > 1:
		filtered_frame = cv2.GaussianBlur(filtered_frame, (set['filter']['blur'], set['filter']['blur']), 0)
	
	# Return results.
	return filtered_frame
	
	