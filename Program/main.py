"""
Set TF_ENABLE_ONEDNN_OPTS to 0, then import the rest.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from gui import filter_settings
from gui import cam_settings
from gui import feed_settings
from gui import action_bar
from logic import feed_logic
import tensorflow as tf



"""
Set global variables and load settings.
"""
root = tk.Tk()
width = 880
height = 600
root.geometry(f"{width}x{height}")
root.resizable(False, True)
root.title("Webcam Input")
char_classification = tf.keras.models.load_model('models/character_classification_model_101')



""""
The following section sets up the base layout.
Additional frames can be added, but make sure to adjust
the grid settings if you do so.
"""

# Camera setttings frame.
left_column = tk.Frame(root)
left_column.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

# Setup feed frame (where the video shows).
feed_frame = tk.Canvas(root, width=640, height=360)
feed_frame.grid(row=0, column=1, padx=10, pady=10)

# Setup action bar beteath frame.
action_bar_frame = tk.Frame(root)
action_bar_frame.grid(row=1, column=1, padx=0, pady=0)

# Setup button list.
button_list_frame = tk.Frame(root)
button_list_frame.grid(row=2, column=1, padx=0, pady=0)

# Add elements.
cam_settings.add(left_column)
filter_settings.add(left_column)
feed_settings.add(left_column)
action_bar.add(action_bar_frame)

"""
Start the loop.
"""
feed_logic.update_feed(root, feed_frame, button_list_frame, char_classification)
root.mainloop()

