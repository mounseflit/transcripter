import streamlit as st
import moviepy.editor as mp
import tempfile
import os
import numpy as np
import openai
import wave

# Retrieve API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit UI
st.title('Video Transcription with Whisper API')
st.write('Upload a video file and get the transcript')

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

def chunk_audio(audio_path, max_chunk_size=25 * 1024 * 1024):
    """
    Splits the audio into chunks of max_chunk_size bytes (default 25MB).
    """
    # Load the audio file
    audio_clip = mp.AudioFileClip(audio_path)
    audio_array = audio_clip.to_soundarray(fps=16000)  # Whisper works best with 16 kHz sample rate

    # Calculate the number of samples per chunk
    chunk_length = max_chunk_size // (audio_array.itemsize * audio_array.shape[1])

    # Split the audio array into chunks
    chunks = []
    for start in range(0, len(audio_array), chunk_length):
        chunk = audio_array[start:start + chunk_length]
        chunks.append(chunk)

    return chunks

def save_chunk_as_wav(chunk, file_path):
    """
    Save the audio chunk as a WAV file.
    """
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)  # Sample rate of Whisper
        wf.writeframes(chunk.tobytes())

def transcribe_with_whisper_api(audio_chunk_path):
    """
    Send the audio chunk to the Whisper API for transcription.
    """
    with open(audio_chunk_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            language="en"  # Change this to the required language, e.g., 'fr' for French
        )
        return response['text']

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Extract audio from the video using moviepy
    video = mp.VideoFileClip(tmp_file_path)
    audio_path = tmp_file_path + "_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Split audio into chunks to avoid exceeding Whisper API limits
    audio_chunks = chunk_audio(audio_path)

    # Transcribe each chunk and concatenate the results
    st.write("Transcribing video... Please wait.")
    full_transcript = ""
    
    for idx, chunk in enumerate(audio_chunks):
        chunk_audio_path = f"{audio_path}_chunk_{idx}.wav"
        
        # Save the chunk to a temporary WAV file
        save_chunk_as_wav(chunk, chunk_audio_path)

        # Transcribe the chunk using Whisper API
        transcript_chunk = transcribe_with_whisper_api(chunk_audio_path)
        full_transcript += transcript_chunk + "\n"

        # Clean up chunk file
        os.remove(chunk_audio_path)

    # Create a text document with the transcript
    transcript = full_transcript

    # Display the transcript
    st.subheader("Transcript")
    st.text_area("Full Transcript", transcript, height=400)

    # Save the transcript to a text file
    transcript_file = tmp_file_path + "_transcript.txt"
    with open(transcript_file, "w") as f:
        f.write(transcript)

    # Provide a download link for the transcript
    st.download_button(
        label="Download Transcript",
        data=open(transcript_file, "rb").read(),
        file_name="transcript.txt",
        mime="text/plain"
    )

    # Clean up temporary files
    os.remove(tmp_file_path)
    os.remove(audio_path)
    os.remove(transcript_file)
