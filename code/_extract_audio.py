import subprocess
import whisper

def extract_audio(video_path, output_audio_path="audio.wav"):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",            # Highest audio quality
        "-map", "a",            # Extract audio only
        output_audio_path       # Output audio file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print("Error during audio extraction:", e)


def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]
