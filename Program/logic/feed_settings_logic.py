"""
This file should contain all logic related to toolbar inputs.
These functions are used for modifying, validating and saving
the inputed data.

Notes:
1. Try to strick to the convention of "getters" and "setters".
2. Add "settings.update_settings()" when deadling with settings that might be hard to find.
"""

import settings

set = settings.set



def set_show_filtered_feed(val):
    """
    Set SHOW_FILTER_FEED setting.
    """
    set['feed']['show_filtered_feed'] = val
    settings.update_settings()
    


def set_show_edge_detection(val):
    """
    Set SHOW_EDGE_DETECTION setting.
    """
    set['feed']['show_edge_detection'] = val
    settings.update_settings()
    
    

def set_contour_thickness(val):
    """
    Set the contour thickness. This is the thickness of the drawn contours on the canvas.
    Note that this has no effect on the result of the tracking nor classification.
    """
    val = float(val)
    set["feed"]["contour_thickness"] = val
    settings.update_settings()

