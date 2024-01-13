import tkinter as tk
from logic.feed_logic import clear_squares



def add(frame):

	"""
	Put action buttons here.
	"""

	# Create a button and associate it with a function to be called when clicked
	button = tk.Button(frame, text="Clear frames", command=clear_squares)

	# Pack the button into the window
	button.pack()
  
  