import os 
from shutil import copyfile 
from main import resized_animations_directory
import cv2 
import numpy as np

#Arrange the frames in order
def get_frame_index(filename):
    try:
        # Assuming filenames are in the format 'frame_001.png', 'frame_002.png', etc.
        return int(filename.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return float('inf') 

def rearrange_frames_in_directory(directory_path):
    frame_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=get_frame_index
    )
    
    # Create a temporary directory to store correctly ordered frames
    temp_directory = os.path.join(directory_path, 'temp')
    os.makedirs(temp_directory, exist_ok=True)

    # Copy and rename files in correct order
    for i, filename in enumerate(frame_files):
        src_path = os.path.join(directory_path, filename)
        dst_path = os.path.join(temp_directory, f"frame_{i:03d}.png")
        copyfile(src_path, dst_path)
    
    # Move files back to the original directory
    for filename in frame_files:
        os.remove(os.path.join(directory_path, filename))
    
    for filename in os.listdir(temp_directory):
        src_path = os.path.join(temp_directory, filename)
        dst_path = os.path.join(directory_path, filename)
        os.rename(src_path, dst_path)
    
    # Remove the temporary directory
    os.rmdir(temp_directory) 

def read_frames_from_directory(directory_path):
    frames = []
    for filename in sorted(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath) and filename.endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(filepath)
            if frame is not None:
                frames.append((frame, filename))  # Keep track of filename
            else:
                print(f"Error reading frame from file: {filepath}")
    return frames

def resize_animation(directory_path, target_frame_count=16):
    frames_with_filenames = read_frames_from_directory(directory_path)
    total_frames = len(frames_with_filenames)
    if total_frames == 0:
        print(f"No frames found in directory: {directory_path}")
        return
    
    if total_frames >= target_frame_count:
        # Select evenly spaced frames
        indices = np.linspace(0, total_frames - 1, num=target_frame_count, dtype=int)
        selected_frames = [frames_with_filenames[i] for i in indices]
    else:
        # Duplicate frames to reach target_frame_count
        selected_frames = frames_with_filenames * (target_frame_count // total_frames)
        remaining_frames = target_frame_count % total_frames
        selected_frames += frames_with_filenames[:remaining_frames]

    # Sort selected frames based on extracted index
    selected_frames.sort(key=lambda x: get_frame_index(x[1]))

    # Create a new directory to store resized animation
    output_directory = os.path.join(resized_animations_directory, os.path.basename(directory_path))
    os.makedirs(output_directory, exist_ok=True)

    # Save selected frames to the new directory
    for i, (frame, _) in enumerate(selected_frames):
        output_filepath = os.path.join(output_directory, f"frame_{i:03d}.png")
        cv2.imwrite(output_filepath, frame)
    
    print(f"Resized animation saved to: {output_directory}")