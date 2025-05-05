# Typing
from typing import List, Optional

# NumPy
import numpy as np

# Python
import math
from pathlib import Path
try:
    from tqdm import tqdm
except:
    pass

# ImageIO
try:
    import imageio
except:
    pass

# Utils
from .decorators import requires_package
from .color import Color

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

@requires_package('imageio', 'tqdm')
def generate_video(
    images: List[Path] | List[str],
    video_filename: str | Path,
    fps: Optional[int] = None,
    duration: Optional[float] = None,
    padding_color: Color | None = None,
    no_resize_warning: bool = False,
    ) -> None:
    """
    Generate a video from a list of images using imageio.

    Inputs
    - images: `List[Path | str] list of images to be included in the video.
    - video_filename: `str | Path` name of the output video file.

    Partially Required Inputs
    - fps: `int` frame rate per seconds of the video.
    - duration: `float` duration of the video in seconds.

    Optional Inputs
    - padding_color: `Color` color to be used for padding the images. If None,
                     the images will be padded with the edge color of the image.
    - no_resize_warning: `bool` if True, no warning will be raised if the image
                         dimensions are not divisible by the macro block size.

    Note: ImageIO automatically resizes the images such that the dimensions are
          divisible by the `macro_block_size`. This is done to ensure that the
          videos are properly encoded and compatible with most video players.
    """

    if len(images) == 0:
        logger.warning("The images list contains zero images. " +
                       "Not generating any video.")
        return

    # Get the image dimensions
    frame = imageio.imread(str(images[0]))
    if not frame.ndim == 3 or not frame.shape[2] in (3, 4):
        logger.error("The images must be RGB or RGBA images.")
        raise ValueError("The images must be RGB or RGBA images.")
    H, W, C = frame.shape

    if padding_color is not None and \
       not padding_color.has_alpha == (frame.shape[2] == 4):
        logger.error("The padding color must have the same number of " +
                     "channels as the images.")
        raise ValueError("The padding color must have the same number of " +
                         "channels as the images.")

    # Handle macro_block_size via padding if necessary
    macro_block_size = 16
    def pad_fct(frame): return frame
    if not W % macro_block_size == 0 or not H % macro_block_size == 0:
        W_new = math.ceil(W / macro_block_size) * macro_block_size
        H_new = math.ceil(H / macro_block_size) * macro_block_size
        if not no_resize_warning:
            logger.warning(
                f"The image dimensions ({W} x {H}) are not divisible by the " +
                f"macro block size ({macro_block_size}). The images will be " +
                f"resized to ({W_new} x {H_new}) to ensure proper encoding " +
                f"compatibility."
            )
        def pad_fct(frame):
            return np.pad(
                frame,
                ((0, H_new - H), (0, W_new - W), (0, 0)),
                **({"mode": "edge"} if padding_color is None else \
                   {"constant_values": padding_color})
            )

    # Frame rate
    if fps is None and duration is None:
        logger.error("Either `fps` or `duration` must be provided.")
        raise ValueError("Either `fps` or `duration` must be provided.")
    if fps is None:
        fps = len(images) / duration

    # Generate video
    with imageio.get_writer(
            str(video_filename), fps=fps, ffmpeg_log_level="error",
            ffmpeg_params=["-probesize", "5000000"],
            macro_block_size=macro_block_size,
        ) as writer:

        for image in images:
            frame = imageio.imread(str(image))

            if not frame.shape == (H, W, C):
                logger.error(f"Image `{image}` has different dimensions than " +
                             f"the first image.")
                raise ValueError(f"Image `{image}` has different dimensions " +
                                 f"than the first image.")
            writer.append_data(pad_fct(frame))

    # Log success
    logger.info(f"Successfully generated video with {len(images)} frames to " +
                f"`{video_filename}`.")
