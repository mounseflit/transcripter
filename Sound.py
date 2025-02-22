import streamlit as st
import os
import time
import requests
from pydub import AudioSegment

# Streamlit app
st.title("Video Transcription with Whisper API")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

def split_audio(file_path, chunk_size=25*1024*1024):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start = end
    return chunks

def transcribe_with_whisper_api(audio_path):
    api_url = "https://api.whisper.com/v1/transcribe"
    api_key = st.secrets["WHISPER_API_KEY"]  # Ensure the secret is set in Streamlit
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav"
    }
    with open(audio_path, "rb") as audio_file:
        response = requests.post(api_url, headers=headers, data=audio_file)
    response.raise_for_status()
    return response.json()["text"]

if uploaded_file is not None:
    # Save the uploaded file
    temp_dir = "tempDir"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully")

    # Split the audio into chunks
    st.write("Splitting audio into chunks...")
    audio_chunks = split_audio(file_path)
    
    # Estimate time for transcription
    st.write("Estimating transcription time...")
    start_time = time.time()
    chunk_path = os.path.join(temp_dir, "sample_chunk.wav")
    audio_chunks[0].export(chunk_path, format="wav")
    transcribe_with_whisper_api(chunk_path)
    end_time = time.time()
    estimated_time_per_chunk = end_time - start_time
    estimated_total_time = estimated_time_per_chunk * len(audio_chunks)
    st.write(f"Estimated time to complete transcription: {estimated_total_time:.2f} seconds")

    # Transcribe each chunk
    transcription = ""
    for i, chunk in enumerate(audio_chunks):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        st.write(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
        result = transcribe_with_whisper_api(chunk_path)
        transcription += result + " "
    
    # Display the transcription
    st.write(transcription.strip())
    
    # Save the transcription to a text file
    with open(os.path.join(temp_dir, uploaded_file.name + ".txt"), "w") as f:
        f.write(transcription.strip())
    
    st.success("Transcription saved to text file")
