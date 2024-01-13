"""
This file should contain all logic related to toolbar inputs.
These functions are used for modifying, validating and saving
the inputed data.

Notes:
1. Try to strick to the convention of "getters" and "setters".
2. Add "update_settings()" when deadling with settings that might be hard to find.
"""

from logic import feed_logic
from settings import set, update_settings



"""
Function called upon changing LOWER threshold input.
This is used for preventing invalid input values as the lower
threshold must remain smaller than the upper threshold.

Comment: A stop should be added so that this function stops the
inputs from being changed too much (a physical stop for the input).
"""
def set_edge_detection_lower_threshold(value):
	t = float(value)
	if set["filter"]["edge_detection_upper_threshold"] > t:
		set["filter"]["edge_detection_lower_threshold"] = t
	update_settings()
	feed_logic.clear_squares()



"""
Function called upon changing UPPER thredhold input.
This is used for preventing invalid input values as the upper
threshhold must remain larger than the lower threshold.

Comment: A stop should be added so that this function stops the
inputs from being changed too much (a physical stop for the input).
"""
def set_edge_detection_upper_threshold(value):
	t = float(value)
	if set["filter"]["edge_detection_lower_threshold"] < t:
		set["filter"]["edge_detection_upper_threshold"] = t
	update_settings()
	feed_logic.clear_squares()



"""
Function to adjust the min circumference tollerance.
This is used to remove false rectangles detected.
"""
def set_min_circumference(value):
	set["filter"]["min_circumference"] = float(value)
	update_settings()
	feed_logic.clear_squares()



"""
This function is used to update the BRIGHTNESS value based on user input.
"""
def set_brightness(value):
	set["filter"]["brightness"] = int(value)
	update_settings()
	feed_logic.clear_squares()



"""
This function is used to update the CONTRAST value based on user input.
"""
def set_contrast(value):
	set["filter"]["contrast"] = int(value)
	update_settings()
	feed_logic.clear_squares()



"""
This function is used to update the THRESHOLD value based on user input.
"""
def set_threshold(value):
	set["filter"]["threshold"] = int(value)
	update_settings()
	feed_logic.clear_squares()
 


"""
This function is sued for updating the GAUSIAN BLUR used for removing
articats from the feed. Note that this value has to be odd.

Comment: Add a potential check to force it to be odd as an even value
will lead to progrma crach.
"""
def set_blur(value):
	set["filter"]["blur"] = int(value)
	update_settings()
	feed_logic.clear_squares()
 


"""
This function is used for setting MIN_CENTER_POINT_DISTANCE which is
used for comparing square between frames.
"""
def set_min_center_point_distance(value):
	set["filter"]["min_center_point_distance"] = int(value)
	update_settings()
	feed_logic.clear_squares()
 
 