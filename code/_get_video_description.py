from moviepy.editor import VideoFileClip

import os
import cv2

global output_folder 
output_folder = ''

def extract_frames_moviepy(video_path):
    global output_folder
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    name = video_path.replace('.mp4', '')
    output_folder = name + "_" + "frames"

    fps = int(video_clip.fps)
    duration = video_clip.duration
    total_frames = int(duration * fps)
    
    # print(f"This video has : {fps} fps, i.e., with duration of : {duration} sec,\
    # it makes it {total_frames} frames in Total")
    
    # nbr_frames = int(input("enter the rate of frames extraction (e.g., 2 : means : 0, 2, 4, 6, ...) : \n"))
    
    nbr_frames = fps  #means 1 frame for each second, you can change it later 


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
        
    print("Frames extraction completed")
 

# ******************************* claude *******************************
import anthropic

# new one : 
key = "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"


client = anthropic.Anthropic(api_key= key)

def describe_frame(base64_image, prompt="What's in this image", 
             model_name="claude-3-sonnet-20240229", media_type = "image/jpeg"):

    message = client.messages.create(
        model= model_name,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text

# --------------------------------------------------------------
import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def encode_all_imgs(frames):
    all_imgs = []

    for img in frames:
        image_path = output_folder + '/' + img
        base64_image = encode_image(image_path)
        all_imgs.append(base64_image)

    return all_imgs

# --------------------------------------------------------------

import os
def get_frames():
    global output_folder
    frames = os.listdir(output_folder)
    return frames

# --------------------------------------------------------------
from _get_gpt_answer import answer
from _get_nbr_of_tokens import num_tokens_from_string

# --------------------------------------------------------------
# Main fct
# --------------------------------------------------------------

def describe_video(video_path, p):
    extract_frames_moviepy(video_path)
    frames = get_frames()
    all_imgs = encode_all_imgs(frames)
    all_res = []
    for i, img in enumerate(all_imgs):
        res = describe_frame(img, prompt=p)
        all_res.append(res)
        print(f"frame {i} described")

    all_res_str = '\n------------------------------------\n'.join(all_res)
    summary_prompt = "the following are seperate descriptions of successive frames of a video" + \
    "I want you to take all of these descriptions and return a reasonable summary for the corresponding video." + \
    "\n Here are the seperate descriptions : \n " + all_res_str

    # if nbr of tokens is > 1024, crop it
    ntok = num_tokens_from_string(summary_prompt, "cl100k_base")
    while ntok > 1024:
        crop = int(len(summary_prompt)/2)
        summary_prompt = summary_prompt[0:crop]
        ntok = num_tokens_from_string(summary_prompt, "cl100k_base")

    summary = answer(summary_prompt)
    return summary








