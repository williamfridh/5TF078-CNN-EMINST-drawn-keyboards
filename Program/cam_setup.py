import cv2
from settings import set



# Global variable.
data = {
	'device': None,
	'fps': None
}



def update_cam_config():
    """
    This function updated the camera configuration object.
    Make sure to call upon this when changing camera settings such as resolution or FPS.
    """
    data['device'] = cv2.VideoCapture(set["camera"]["device_index"])      		# Select webcam feed. (Should be choosen via an input.)
    cam_res = set["camera"]["resolutions"][set["camera"]["resolution_index"]]	# Select resolution.
    data['device'].set(cv2.CAP_PROP_FRAME_WIDTH, cam_res[0])					# Set resolution width.
    data['device'].set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res[1])					# Ser resolution height.
    data['fps'] = data['device'].get(cv2.CAP_PROP_FPS)  						# Store webcam fpst
    


# Default run.
update_cam_config()

