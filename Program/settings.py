"""
This file contains all functionaloties needed for working with the file settings.json.
The settings.json file should contain anything one might wish to save for a better
user experience and faster debugging.

Note that a great amount of the code in this file was generated used ChatGPT!
"""

import json
import shutil

settings_file_name = "data/settings.json"
default_settings_file_name = "data/settings.default.json"



def load_settings():
    """
    Load the settings file and return either an object contining the data
    or throw and expetion based on the error: file not found,
    or deconding error.
    """
    try:
        with open(settings_file_name, "r") as file:
            settings = json.load(file)
        return settings
    except FileNotFoundError:
        print(f"Error: settings file '{settings_file_name}' not found.")	# Print error.
        reset_settings_file()													# Reset the settings file.
        load_settings()														    # Call upon itself.
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON syntax in '{settings_file_name}'.")
        return None
    
    
    
def reset_settings_file():
    """
    Copies the default setting form the file [default_settings_file_name]
    into a new file called [settings_file_name]. This solution is used for first
    time usage and resetting the settings file.
    """
    try:
        shutil.copy(default_settings_file_name, settings_file_name)
        print(f"File copied from '{default_settings_file_name}' to '{settings_file_name}' successfully.")
    except FileNotFoundError:
        pass
        print(f"Error: Source file '{default_settings_file_name}' not found.")
    except PermissionError:
        pass
        print(f"Error: Permission denied while copying '{default_settings_file_name}' to '{settings_file_name}'.")
    except Exception as e:
        pass
        print(f"An error occurred: {e}")



def update_settings():
    """
    Update the settings file with the new data.

    Comment: Adjust this so that it only saves the settings every now and then.
    """
    try:
        with open(settings_file_name, "w") as file:
            json.dump(set, file, indent=2)
        #print(f"JSON object saved to '{settings_file_name}' successfully.")
    except Exception as e:
        print(f"Error: {e}")



"""
Perform default actions.
"""
set = load_settings()

