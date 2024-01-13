import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from PIL import Image
import numpy as np
from converter import ascii_codes

char_classification = tf.keras.models.load_model('models/character_classification_model_101')

#char_classification.summary()

def jpg_to_tensor(file_path):
	# Open the image using PIL
	img = Image.open(file_path)

	# Convert the image to a numpy array
	img_array = np.array(img)
	
	

	# Convert the numpy array to a tensor
	img_tensor = np.array(img_array)
	
	#img_tensor = img_tensor.reshape((1, 112, 112, 1))
	
	#img_tensor = np.expand_dims(img_tensor, axis=-1)
	
	img_tensor = np.expand_dims(img_tensor, axis=0)  # Adding a batch dimension
	img_tensor = np.expand_dims(img_tensor, axis=-1)  # Adding a channel dimension
 
	img_tensor = img_tensor / 255

	return img_tensor

# Example usage
file_path = "tmp/[763 780]_[611 788]_[573 936]_[740 919].jpg"
tensor_data = jpg_to_tensor(file_path)

#print(type(tensor_data[0][0]))

#print(tensor_data.shape)
print(tensor_data)

pred = char_classification(tensor_data)
pred = pred.numpy()
ascii = int(ascii_codes[np.argmax(pred)])
character = chr(ascii)

print(pred)
print(f"{character} ({np.max(pred)})")