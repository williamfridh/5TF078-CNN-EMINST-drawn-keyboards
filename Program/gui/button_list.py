from logic import feed_logic
import tkinter as tk
import extra



def print_list(frame):

	"""
	Prints a list of all the detected buttons (squares) onto the given frame.
	"""

	# Clear frame.
	extra.clear_frame(frame)

	# For each square/button found.
	for i in range(len(feed_logic.squares)):

		# Select square and convert color into hex.
		square = feed_logic.squares[i]
		hex_color = "#{:02x}{:02x}{:02x}".format(*square['color'])

		# Create a cell for the object.
		list_obj = tk.Frame(frame, bd=1, relief="solid")
		list_obj.grid(padx=5, pady=5, row=int(i/4), column=i%4)
  
		# Draw the thumbnail.
		image_el = tk.Canvas(list_obj, height=28, width=28)
		image_el.create_image(0, 0, anchor=tk.NW, image=square['thumbnail'])
		image_el.pack(side=tk.LEFT)
  
		# Draw the color block.
		color_el = tk.Canvas(list_obj, width=5, height=5, bg=hex_color)
		color_el.pack(side=tk.LEFT)

		# Print the text.
		text_el = tk.Label(list_obj, text=f"{i}: {square['character']} ({square['certainty']}%)")
		text_el.pack(side=tk.LEFT)
  
  