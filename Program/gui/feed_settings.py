
import tkinter as tk
from logic import feed_settings_logic as fbl
from settings import set
from tkinter import Scale




def add(master):

	"""
	The following section adds user inputs for settings used for
	optimizing the edge detection. It can contain stuff such as
	filters and cropping.
	"""

	# Checkbox to set edge detection in feed.
	show_edge_detection = tk.BooleanVar(value = set['feed']['show_edge_detection'])
	checkbox_show_edge_detection = tk.Checkbutton(
		master,
		text="Show edge detection",
		variable=show_edge_detection,
		command=lambda: fbl.set_show_edge_detection(show_edge_detection.get())
	)
	checkbox_show_edge_detection.pack()

	# Checkbox to set between filtered and unfiltered feed.
	show_filtered_feed = tk.BooleanVar(value = set['feed']['show_filtered_feed'])
	checkbox_show_squares = tk.Checkbutton(
		master,
		text="Show filtered feed",
		variable=show_filtered_feed,
		command=lambda: fbl.set_show_filtered_feed(show_filtered_feed.get())
	)
	checkbox_show_squares.pack()

	# Create sliders for contour thickness.
	contour_thickness_slider = Scale(
		master, from_=1, to=15, orient=tk.HORIZONTAL, label="Contour thickness",
		length=200, command=fbl.set_contour_thickness
	)
	contour_thickness_slider.set(set["feed"]["contour_thickness"])
	contour_thickness_slider.pack()

