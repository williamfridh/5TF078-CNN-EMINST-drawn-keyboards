import cv2
import feed_filter
import cam_setup
from logic import image_editing_logic
from settings import set
import tkinter as tk
import math
import random
from gui import button_list
from PIL import Image, ImageTk
import numpy as np
from converter import ascii_codes
from extra import calculate_distance, order_points



# Global variables.
squares = []    # Store all existing squres.
frame_num = 0   # Used for limiting computations.



"""
This function is recursive and is the main loop used by the program.
One iteration means one frame.
It handles the feed, filters, tracking and so on.

Note that this loop is started futher down in this file.
"""
def update_feed(root, feed_frame, button_list_frame, char_classification):
    
    global frame_num
    
    # Get a frame from the webcam.
    ret, frame = cam_setup.data['device'].read()

    # Make sure the frame was read.
    if ret:
        
        # Apply filters.
        filtered_frame = feed_filter.apply_filters(frame)
        
        # Limit the square detection to every second.
        if frame_num == cam_setup.data['fps']:
            
            # Use Canny edge detection to find edges in the frame.
            # This loads threshold limits based on user input.
            edges = cv2.Canny(filtered_frame, set["filter"]["edge_detection_lower_threshold"], set["filter"]["edge_detection_upper_threshold"])

            # Find contours in the edge-detected image.
            # Multiple modes and approximation methods were tested, but the current ones performed the best.
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
            
            # Find all NEW SQUARES in the array of contours.
            new_squares = []
            cropped_image_arr = []
            cropped_image_arr_thumbnail = []
            for i in range(len(contours)): # Iterate trough countours.
                contour = contours[i]
                # Calculate 4% of the contour length.
                epsilon = 0.04 * cv2.arcLength(contour, True)
                # Approximate the contour using the alforithm "Douglas-Peucker" (mainly a fun fact).
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # Check that there are 4 vertices with all right angels.
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    # Check that the circumference is large enough. This is used for as an additional filter.
                    if cv2.arcLength(contour, True) >= set["filter"]["min_circumference"]:
                        # If the contour has no parent, it's a potential square. This will give us a cleaner result will no overlapping squares.
                        if hierarchy[0][i][3] == -1:
                            
                            """
                            Perspective transformation is important so that the true region of interest
                            can be fetched without including too much drawn borders.
                            """
                            
                            # Order the approximated points.
                            approx_ordered = order_points(approx.reshape(-1, 2))
                            
                            # Convert to integers
                            approx_ordered = approx_ordered.astype(int)

                            # Reshape into the desired format
                            approx_ordered = np.reshape(approx_ordered, (4, 1, 2))
                            
                            # Calculate new dimensions.
                            new_w = calculate_distance(approx_ordered[0][0], approx_ordered[1][0])
                            new_h = int(calculate_distance(approx_ordered[0][0], approx_ordered[3][0]) * (1 + math.cos(math.pi / 6))) # Increase the the height due to camera angle.
                            
                            # Perspective transformation.
                            old_points = np.float32([approx_ordered[0][0],approx_ordered[1][0],approx_ordered[2][0],approx_ordered[3][0]])
                            target_points = np.float32([[0,0],[new_w,0],[new_w,new_h],[0,new_h]])
                            
                            M = cv2.getPerspectiveTransform(old_points, target_points)
                            new_img = cv2.warpPerspective(filtered_frame, M, (new_w, new_h))
                            
                            # Check function used for keeping track of image dimensitons.
                            continue_outer = False
                            def check():
                                global continue_outer
                                #print(cropped_image_data.shape)
                                if new_img.shape[0] < 20 or new_img.shape[1] < 20:
                                    continue_outer = True
                                    return True
                                else:
                                    return False
                                
                            # Aggreisve cropping from all sides to remove border.
                            while np.min(new_img[0, :]) <= 50 or  np.min(new_img[-1, :]) <= 50 or  np.min(new_img[:, 0]) <= 50 or  np.min(new_img[0, -1]) <= 50 :
                                new_img = new_img[1:-1, 1:-1]
                                if check():
                                    break
                            
                            if continue_outer:
                                continue
                            
                            # Cut out the symbol at the center.        
                            while np.min(new_img[:, 0]) > 50:
                                if check():
                                    break
                                new_img = new_img[:, 1:]
                            
                            while np.min(new_img[:, -1]) > 50 and not continue_outer:
                                if check():
                                    break
                                new_img = new_img[:, :-1]
                            
                            while np.min(new_img[0, :]) > 50 and not continue_outer:
                                if check():
                                    break
                                new_img = new_img[1:, :]
                            
                            while np.min(new_img[-1, :]) > 50 and not continue_outer:
                                if check():
                                    break
                                new_img = new_img[:-1, :]
                            
                            if continue_outer or check():
                                continue
                                
                            # Generate a new empty image.
                            t = int(np.max(new_img.shape) * 1.4) # Note the 40% width increase. Required to make the data similar to EMNIST.
                            tt = np.full((t, t), 255, dtype=np.uint8)
                            
                            # Paste the cut out symbol into the center.
                            row_start = int((tt.shape[0] - new_img.shape[0]) / 2)
                            row_end = row_start + new_img.shape[0]
                            col_start = int((tt.shape[1] - new_img.shape[1]) / 2)
                            col_end = col_start + new_img.shape[1]
                            tt[row_start:row_end, col_start:col_end] = new_img
                            
                            new_img = tt
                            
                            # Turn into image adjust.
                            new_img = Image.fromarray(new_img, "L")
                            rot_new_img = new_img.rotate(180)
                            scl_rot_new_img = image_editing_logic.scale_image(rot_new_img, (112, 112))
                            
                            # Save data to be sent to CNN-model.
                            np_scl_rot_new_img = np.array(scl_rot_new_img) # Convert PIL Image to numpy array.
                            exp_np_scl_rot_new_img = np.expand_dims(np_scl_rot_new_img, axis=0) # Add a batch dimension.
                            cropped_image_arr.append(exp_np_scl_rot_new_img)
                            
                            # Save data as thumbnailt o be displayed in list.
                            rot_new_img_thumb = image_editing_logic.scale_image(rot_new_img, (28, 28))
                            scl_rot_new_img_thumb = ImageTk.PhotoImage(rot_new_img_thumb) # Convert the resized image to PhotoImage
                            cropped_image_arr_thumbnail.append(scl_rot_new_img_thumb)
                            
                            # Save the approximation.
                            new_squares.append(approx)
                        
            compare_squares(new_squares, button_list_frame, cropped_image_arr, cropped_image_arr_thumbnail, char_classification)
            
            # Reset frame number.
            frame_num = 0
        
        # Increment.
        frame_num += 1

        # Draw squares.
        display_frame = None
        if set["feed"]["show_filtered_feed"]:
            display_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
        else:
            display_frame = frame
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
        if set["feed"]["show_edge_detection"]:
            for i in range(len(squares)):
                square = squares[i]
                x, y, w, h = cv2.boundingRect(square["approx"])
                cv2.putText(display_frame, f'{i}: {square["character"]} ({square["certainty"]}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, square["color"], 2, cv2.LINE_AA) # Note that -1 means "draw all".
                cv2.drawContours(display_frame, [square["approx"]], -1, square["color"], int(set["feed"]["contour_thickness"]))
                
                for point in square["approx"]:
                    #print(point[0])
                    cv2.circle(display_frame, point[0], int(set["feed"]["contour_thickness"]) * 2, square["color"], -1)

        # Convert the frame to ImageTk format to be displayed in the window.???
        img = Image.fromarray(display_frame)

        # Update the canvas with the new frame.
        img_scaled = image_editing_logic.scale_image(img) # Scale image to be displayed.
        img_scaled = ImageTk.PhotoImage(img_scaled)
        feed_frame.img_tk = img_scaled  # Keep a reference to avoid garbage collection.
        feed_frame.create_image(0, 0, anchor=tk.NW, image=img_scaled)

    # Schedule the update method after a based on the webcam fps.
    root.after(int(1000/cam_setup.data['fps']), lambda: update_feed(root, feed_frame, button_list_frame, char_classification))
    
   
    
"""
This function compares the existing squares with the new squares to prevent
copies, overlapping, and missing squares as the tracking isn't perfect and
the camera might lose vision of a certain field from time to time.
"""
def compare_squares(new_squares, button_list_frame, cropped_image_arr, cropped_image_arr_thumbnail, char_classification):
    
    for i in range(len(new_squares)): # Iterate over each new square.
        
        new_square = new_squares[i]
        
        # Calculate the moments of the first contour.
        moments = cv2.moments(new_square)

        # Calculate the centroid (center) of the contour.
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        new_square_found = True
        
        for square in squares: # Iterate over each existing square.
            
            # Calculate distance between center points.
            distance = math.sqrt((square['center'][0] - cx)**2 + (square['center'][1] - cy)**2)
            
            if distance < set["filter"]["min_center_point_distance"]: # Same square.
                new_square_found = False
                
        if new_square_found:

            # Perform prediction and sort out the data.
            pred = char_classification(cropped_image_arr[i] / 255)
            pred = pred.numpy()
            ascii = int(ascii_codes[np.argmax(pred)])
            character = chr(ascii)

            # Save as a successfull square detection.
            squares.append({
                "approx": new_square,
                "center": [cx, cy],
                "color": (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
                "ascii": ascii,
                "character": character,
                "certainty": int(np.max(pred)*100),
                "image": None,
                "thumbnail": cropped_image_arr_thumbnail[i]
            })
            button_list.print_list(button_list_frame)



def clear_squares():
    """
    Simple method to clear the squares detected.
    """
    global squares
    squares = []
    
    