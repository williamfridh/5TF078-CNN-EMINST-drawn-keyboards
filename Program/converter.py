import numpy as np



def importAscii(filepath, up_to_char):

    """
    Thie function is used for reading the ASCII-code file which is required
    for the trained model to be able to connect the results with the
    correct ASCII code. The ASCII-code is then sent to the Python function chr().
    """

    # Prepare variables.
    ascii_codes = None

    # Read from file.
    try:
        with open(filepath, 'r') as file:
            # Read the entire content of the file and store.
            ascii_codes = file.read()
    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Convert into a numpy array.
    ascii_codes = np.array(ascii_codes.splitlines())

    # Filter.
    ascii_codes = ascii_codes[:up_to_char]

    # Remove the indices.
    for i in range(len(ascii_codes)):
        ascii_codes[i] = ascii_codes[i].split()[1]

    # Return the results.
    return ascii_codes


# Perform default run.
ascii_codes = importAscii("data/emnist-balanced-mapping.txt", 36)

