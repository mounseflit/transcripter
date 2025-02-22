import streamlit as st
import openai
import ffmpeg
from io import BytesIO
from pydub import AudioSegment
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use the OpenAI API key from the environment
openai.api_key = os.getenv('OPENAI_API_KEY')

def transcribe_audio(audio_file):
    try:
        audio_data = audio_file.getvalue()
        response = openai.Audio.transcribe(
            model="whisper-1",  # Whisper model
            file=BytesIO(audio_data),
            language='en'  # You can change to another language if needed
        )
        return response['text']
    except Exception as e:
        return f"Error: {e}"

# Extract audio from video using ffmpeg
def extract_audio_from_video(video_file):
    try:
        video_path = os.path.join("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())

        audio_path = "temp_audio.wav"
        # Use ffmpeg to extract audio from video file
        ffmpeg.input(video_path).output(audio_path).run()

        audio_segment = AudioSegment.from_wav(audio_path)
        os.remove(video_path)  # Clean up video file
        return audio_segment
    except Exception as e:
        return f"Error extracting audio: {e}"

# Split audio if too long
def split_audio(audio_segment, chunk_length_ms=300000):  # 5 minutes default
    chunks = []
    for start_ms in range(0, len(audio_segment), chunk_length_ms):
        chunk = audio_segment[start_ms:start_ms + chunk_length_ms]
        chunks.append(chunk)
    return chunks

# Streamlit UI
st.title("Audio and Video Transcription with OpenAI Whisper")

# File uploader
audio_file = st.file_uploader("Upload Audio or Video File", type=["mp3", "mp4", "wav", "m4a"])

if audio_file is not None:
    st.write(f"Transcribing: {audio_file.name}")

    # Handle video files (extract audio)
    if audio_file.type in ["mp4", "m4a", "mov"]:
        audio_segment = extract_audio_from_video(audio_file)
        if isinstance(audio_segment, AudioSegment):
            st.write("Audio extracted from video.")
    else:
        # Handle audio files directly
        audio_segment = AudioSegment.from_file(audio_file)

    # Split long audio files into chunks
    chunks = split_audio(audio_segment)

    transcription = ""
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Processing chunk {i + 1}/{len(chunks)}..."):
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            with open(chunk_path, "rb") as f:
                chunk_file = BytesIO(f.read())
                transcription += transcribe_audio(chunk_file) + "\n\n"
            
            os.remove(chunk_path)  # Clean up temporary chunk file

    # Display transcription
    st.subheader("Transcription Output")
    st.text_area("Transcription", transcription, height=500)
