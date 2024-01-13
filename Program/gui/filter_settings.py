
import tkinter as tk
from logic import filter_settings_logic as tbl
from tkinter import Scale
from settings import set



def add(master):

	"""
	The following section adds user inputs for settings used for
	optimizing the edge detection. It can contain stuff such as
	filters and cropping.
	"""

	# Create a slider for min circumference.
	# By increasing this, we can a form of additional requirement
	# and can thus figure out more false rectangles found.
	min_circumference_slider = Scale(
		master,
		label="Min. circumference",
		from_=50,
		to=2000,
		orient=tk.HORIZONTAL,
		resolution=1,
		length=200,
		command=tbl.set_min_circumference,)
	min_circumference_slider.pack()
	min_circumference_slider.set(set["filter"]["min_circumference"])

	# Create a slider for lower threshold for edge detection.
	edge_detection_lower_threshold_slider = Scale(
		master,
		label="Lowed threshold",
		from_=10,
		to=250,
		orient=tk.HORIZONTAL,
		resolution=1,
		length=200,
		command=tbl.set_edge_detection_lower_threshold)
	edge_detection_lower_threshold_slider.pack()
	edge_detection_lower_threshold_slider.set(set["filter"]["edge_detection_lower_threshold"])

	# Create a slider for upper threshold for edge detection.
	edge_detection_upper_threshold_slider = Scale(
		master,
		label="Upper threshold",
		from_=10,
		to=500,
		orient=tk.HORIZONTAL,
		resolution=1,
		length=200,
		command=tbl.set_edge_detection_upper_threshold)
	edge_detection_upper_threshold_slider.pack()
	edge_detection_upper_threshold_slider.set(set["filter"]["edge_detection_upper_threshold"])

	# Create sliders for adjusting brightness.
	brightness_slider = Scale(
		master, from_=0, to=10, orient=tk.HORIZONTAL, label="Brightness",
		length=200, command=tbl.set_brightness
	)
	brightness_slider.set(set["filter"]["brightness"])
	brightness_slider.pack()

	# Create sliders for adjusting contrast.
	contrast_slider = Scale(
		master, from_=0, to=100, orient=tk.HORIZONTAL, label="Contrast",
		length=200, command=tbl.set_contrast
	)
	contrast_slider.set(set["filter"]["contrast"])
	contrast_slider.pack()

	# Create sliders for threshold.
	threshold_slider = Scale(
		master, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold",
		length=200, command=tbl.set_threshold
	)
	threshold_slider.set(set["filter"]["threshold"])
	threshold_slider.pack()

	# Create sliders for blur.
	blur_slider = Scale(
		master, from_=1, to=11, resolution=2, orient=tk.HORIZONTAL, label="Blur",
		length=200, command=tbl.set_blur
	)
	blur_slider.set(set["filter"]["blur"])
	blur_slider.pack()
 
	# Create sliders for min distance between center points.
	center_point_distance_slider = Scale(
		master, from_=0, to=1000, resolution=1, orient=tk.HORIZONTAL, label="Min. center point distance",
		length=200, command=tbl.set_min_center_point_distance
	)
	center_point_distance_slider.set(set["filter"]["min_center_point_distance"])
	center_point_distance_slider.pack()
 
 