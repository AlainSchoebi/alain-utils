# Typing
from typing import List

# Python
import numpy as np
from pathlib import Path
try:
    from tqdm import tqdm
except:
    pass

# Utils
from .decorators import requires_package

# OpenCV
try:
    import cv2
except ImportError:
    pass

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

@requires_package('cv2', 'tqdm')
def generate_video(
    images: List[Path] | List[str],
    video_filename: str | Path,
    frame_rate: int = 5
    ) -> None:

    if len(images) == 0:
        logger.warning("The images list contains zero images. " +
                       "Not generating any video.")
        return

    # Get the image dimensions
    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_filename), fourcc, frame_rate,
                            (width, height))

    # Loop through the images and add them to the video
    for image in tqdm(images, total=len(images), desc="Generating video"):

        # Read image and add the frame to the video
        frame = cv2.imread(str(image))
        video.write(frame)

    # Close the video file
    video.release()
    logger.info(f"Successfully generated video with {len(images)} frames to " +
                f"'{video_filename}'.")


@requires_package('cv2')
def generate_4_tile_video(
    images_corners: List[List[Path]] | List[List[str]],
    video_filename: str | Path,
    frame_rate: int = 5
    ) -> None:

    # Checks
    if len(images_corners) != 4:
        logger.error(f"Expected length of images_corners to be '4', found " +
                     f"'{len(images_corners)}'.")
        raise ValueError(f"Expected length of images_corners to be '4', " +
                         f" + '{len(images_corners)}'.")

    if len(np.unique([len(images) for images in images_corners])) != 1:
        logger.error(f"The four images list don't contain the same number of " +
                     f"images.")
        raise ValueError(f"The four images list don't contain the same " +
                         f"number of images.")

    if len(images_corners[0]) == 0:
        logger.warning("Each images corner list contains zero images. " +
                       "Not generating any video.")
        return

    # Get image dimensions
    ref_frame = cv2.imread(str(images_corners[0][0]))
    height, width, layers = ref_frame.shape

    for images in images_corners:
        for img in images:
            frame = cv2.imread(str(img))
            if frame.shape != ref_frame.shape:
                logger.error(f"All the images don't have the same dimensions.")
                raise ValueError(f"All the images don't have the same " +
                                 f"dimensions.")

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_filename), fourcc, frame_rate,
                            (2 * width, 2 * height))

    combined_frame = np.zeros((2 * height, 2 * width, layers), dtype=np.uint8)

    # Loop through the images and add them to the video
    for i in tqdm(range(len(images_corners[0])), desc="Generating video"):

        # Read frames
        frame_top_left = cv2.imread(str(images_corners[0][i]))
        frame_top_right = cv2.imread(str(images_corners[1][i]))
        frame_bottom_left = cv2.imread(str(images_corners[2][i]))
        frame_bottom_right = cv2.imread(str(images_corners[3][i]))

        # Construct the combined frame
        combined_frame[:height, :width] = frame_top_left
        combined_frame[:height, width:] = frame_top_right
        combined_frame[height:, :width] = frame_bottom_left
        combined_frame[height::, width:] = frame_bottom_right

        # Read image and add the frame to the video
        video.write(combined_frame)


    # Close the video file
    video.release()
    logger.info(f"Successfully generated video with {len(images_corners[0])} " +
                f"frames to '{video_filename}'.")