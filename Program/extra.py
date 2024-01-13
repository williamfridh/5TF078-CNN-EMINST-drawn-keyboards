"""
This file contains simple extra/helper functions that makes the code somwhat
shorter and easier to read elsewhere. Feel free to add more functions here
that doesn't fit elsehwere.
"""

import math
import numpy as np
from scipy.spatial import distance as dist



# Destroy all widgets inside the frame
def clear_frame(frame):
    """
    This function is used for clearing frames which can come in handy
    when one wishes to refresh the contents of the frame. Clear it,
    then print the new content.
    """
    for widget in frame.winfo_children():
        widget.destroy()
        
        

def calculate_distance(point1, point2):
    """
    A simple method for calculating the distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return distance



def order_points(pts):
    """
    This function is uesd for sorting the outputet points form
    cv2.approxPolyDP which is required for a proper perspective
    transformation to take place.
    
    Source: https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

