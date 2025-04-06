from _get_frames import extract_frames_moviepy
from _extract_audio import extract_audio, transcribe_audio
from _get_answer import answer
from _get_caption import describe_frame_with_gtp, describe_frame_with_gemini, describe_frame_with_claude
from PIL import Image
import os, io, base64
import shutil


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img_bytes = io.BytesIO(img_data)
    img = Image.open(img_bytes)
    return img

def describe_video(video_path, 
                   audio=False,
                   model="gpt", 
                   output_frames_path="video_frames", 
                   output_audio_path = "audio.wav",
                   interval=3, 
                   prompt="Describe the scene you see in the image."):

    all_res = []

    # Extract frames in a video
    extract_frames_moviepy(video_path, output_frames_path)

    # Get all frames extracted
    frames = os.listdir(output_frames_path)
    sorted_frames = sorted(frames, key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(sorted_frames[:6])

    print(f"There are {len(sorted_frames)} frames in this video.")

    # Extract audio from the video
    if audio:
        extract_audio(video_path=video_path)

        # Transcribe the audio
        audio_transcription = transcribe_audio(audio_path=output_audio_path)

    # Describe all frames extracted in the video
    print('************* Describe All Frames *************')
    if model == "gpt" or model == "claude" :
        # Encode all frames in a video
        for img in sorted_frames[0:-1:interval]:
            image_path = output_frames_path + '/' + img
            base64_image = encode_image(image_path)
            if model == "gpt":
                res = describe_frame_with_gtp(img_base64=base64_image, prompt=prompt)
            else:
                res = describe_frame_with_claude(base64_image=base64_image, prompt=prompt)
            all_res.append(res)
        all_res_str = '\n------------------------------------\n'.join(all_res)
    elif model == "gemini":
        for img in sorted_frames[0:-1:interval]:
            image_path = output_frames_path + '/' + img
            res = describe_frame_with_gemini(img_path=image_path, prompt=prompt)
            if res != "":
                all_res.append(res)
        all_res_str = '\n------------------------------------\n'.join(all_res)
    else:
        print("You should choose one model from this list: [gpt, claude, gemini].")
        return None


    # LLM-generated summary of all results using GPT
    if audio:
        summary_prompt = "the following are seperate descriptions of successive frames of a video and audio transcription" + \
        "I want you to take all of these descriptions and return a reasonable summary for the corresponding video." + \
        "\n Here are the seperate descriptions : \n " + all_res_str  + \
        "\n Here is the audio transcription : \n " + audio_transcription
    else :
        summary_prompt = "the following are seperate descriptions of successive frames of a video" + \
        "I want you to take all of these descriptions and return a reasonable summary for the corresponding video." + \
        "\n Here are the seperate descriptions : \n " + all_res_str

    # print('************* Generate a summary for all results *************')
    if model == "gpt":
        summary = answer(model_name=model, prompt=summary_prompt, engine="gpt-4o-mini")
    elif model == "claude":
        summary = answer(model_name=model, prompt=summary_prompt, engine="claude-3-5-haiku-20241022")
    else:
        summary = answer(model_name=model, prompt=summary_prompt, engine="gemini-1.5-flash")

    shutil.rmtree(output_frames_path)
    print("The frames folder has been deleted.")
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
        print("The audio file has been deleted.")

    return summary