import streamlit as st
import openai
from io import BytesIO
from pydub import AudioSegment
import os
from dotenv import load_dotenv

# Set up your OpenAI API key (you can use .env file for security)
load_dotenv()  # Load the .env file for sensitive data like API keys
openai.api_key = os.getenv('OPENAI_API_KEY')  # Make sure you set up your key in a .env file

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

# Streamlit UI
st.title("Audio Transcription with OpenAI Whisper")

# File uploader widget
audio_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.write(f"Transcribing: {audio_file.name}")

    # Load the audio file directly using pydub
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        st.write("Audio file loaded successfully.")
        
        # Call the transcription function
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_file)
            st.subheader("Transcription Output")
            st.text_area("Transcription", transcription, height=500)
    
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
