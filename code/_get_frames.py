from moviepy.editor import VideoFileClip

import os
import cv2

def extract_frames_moviepy(video_path, output_folder):
    # Load the video clip
    video_clip = VideoFileClip(filename=video_path)
    # video_clip = VideoFileClip(filename=video_path, audio=True)

    fps = video_clip.fps
    # duration = video_clip.duration
    total_frames = int(video_clip.duration * fps)
    
    nbr_frames = 10  #to customize later 


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frames at the specified interval
    j = 0
    for frame_num in range(0, total_frames, nbr_frames):
        frame = video_clip.get_frame(frame_num / video_clip.fps)
        frame_path = os.path.join(output_folder, f"frame_{frame_num}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
        print(f"iteration {j} ,  Frame {frame_num} saved")
        j += 1
        
    print("########## Frames extraction completed ##########")
