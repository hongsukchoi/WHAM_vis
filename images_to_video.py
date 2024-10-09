# read images and save as video to the video_path

import cv2
import os
from pathlib import Path
import imageio
import tyro


def main(image_folder: str = './posetrack_val/024159_mpii_test/', video_save_path: str = './024159_mpii_test.mp4'):
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')])

    if not image_files:
        print(f"No image files found in {image_folder}")
        exit()

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, layers = first_image.shape


    writer = imageio.get_writer(
        video_save_path, 
        fps=30, mode='I', format='FFMPEG', macro_block_size=1
    )

    # Iterate through images and write to video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)

    # Release the VideoWriter
    writer.close()

    print(f"Video saved to {video_save_path}")



if __name__ == '__main__':
    tyro.cli(main)
    # video_save_path = './024159_mpii_test.mp4'
    # image_folder = './posetrack_val/024159_mpii_test/'




