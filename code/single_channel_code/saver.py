import os
import imageio
import numpy as np


def save_hyperspectral_images(images, output_folder, file_format='jpg'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    if len(images.shape) == 3:
        images = images[None, ...]
    # Iterate over each frame in the hyperspectral image sequence
    for frame_idx, frame in enumerate(images):
        # Create a subfolder for each frame
        frame_folder = os.path.join(output_folder, f"frame_{frame_idx}")
        os.makedirs(frame_folder, exist_ok=True)

        # Iterate over each channel in the frame
        for channel_idx, channel in enumerate(frame.transpose(2, 0, 1)):
            # Convert channel to grayscale (average of color channels)
            # grayscale_channel = np.mean(channel, axis=0)
            # grayscale_channel = channel / np.max(channel) * 255
            grayscale_channel = channel * 255
            grayscale_channel_uint8 = grayscale_channel.astype(np.uint8)
            # Normalize channel values to 0-255 range
            # normalized_channel = (grayscale_channel - np.min(grayscale_channel)) / (
            #         np.max(grayscale_channel) - np.min(grayscale_channel)) * 255

            # Convert channel to unsigned 8-bit integer
            # channel_uint8 = normalized_channel.astype(np.uint8)

            # Define the output filename
            output_filename = os.path.join(frame_folder, f"channel_{channel_idx}.{file_format}")

            # Save the grayscale image using imageio
            imageio.imwrite(output_filename, grayscale_channel_uint8)



