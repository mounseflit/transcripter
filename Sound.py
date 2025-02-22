import openai
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file (for secure API key storage)
load_dotenv()

# Set OpenAI API Key (using the .env file)
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(audio_file):
    try:
        # Use OpenAI's Whisper API for transcription with the modern API
        response = openai.Audio.create(
            model="whisper-1",  # Whisper model for transcription
            file=audio_file,
            response_format="text"  # Get plain text transcription
        )
        
        # Return the transcribed text
        return response['text']
    
    except Exception as e:
        return f"Error during transcription: {e}"

# Streamlit UI
st.title("Audio Transcription with OpenAI Whisper")

# File uploader widget to upload audio files
audio_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    # Display file name
    st.write(f"Transcribing: {audio_file.name}")

    # Display a spinner while the file is being processed
    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(audio_file)
        st.subheader("Transcription Output")
        st.text_area("Transcription", transcription, height=500)
