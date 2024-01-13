from settings import set, update_settings
from cam_setup import update_cam_config
from logic import feed_logic



"""
Function to save cam resolution.
"""
def set_cam_res(value):
    arr = value.split("x")
    width = int(arr[0])
    for i in range(len(set["camera"]["resolutions"])):
        if set["camera"]["resolutions"][i][0] == width:
            set["camera"]["resolution_index"] = i
    update_settings()
    update_cam_config()
    feed_logic.clear_squares()
    
    