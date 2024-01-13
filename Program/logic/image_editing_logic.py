def scale_image(original_image, target_resolution = (640, 360)):
    
    """
    At this stage, this function makes no sense to keep. But it's already
    integrated into the program and thus will be kept and maybe be modified
    in the future to cover more use-cases.
    """

    # Resize the image to the target resolution using subsample
    resized_image = original_image.resize(target_resolution)

    return resized_image

