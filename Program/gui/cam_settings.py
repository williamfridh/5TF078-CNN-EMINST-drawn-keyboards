import tkinter as tk
from logic import cam_settings_logic as stl
from settings import set



def add(master):
	
	# Create label.
	label = tk.Label(master, text="Resolution")
	label.pack()
 
	# Generate options.
	cam_res_options = []
	for res in set["camera"]["resolutions"]:
		cam_res_options.append(f"{res[0]}x{res[1]}")

	# Create a Tkinter StringVar to store the selected option
	cam_res_selected_option = tk.StringVar()
	cam_res_selected_option.set(cam_res_options[set["camera"]["resolution_index"]])
	cam_res_option_menu = tk.OptionMenu(master, cam_res_selected_option, *cam_res_options, command=stl.set_cam_res)
	cam_res_option_menu.pack()
 
