import streamlit as st 
from deepspeech import Model
from Speech_pounction import Audio_to_text
from punctuationmodel import PunctuationModel
import time
st.header("Speech_To_Text")
audio_file=st.sidebar.file_uploader("Select audio file")
st.sidebar.audio(audio_file)

obj = Audio_to_text(Model)
model=obj.model_prepare()
stream = model.createStream()
audio_path=audio_file

with st.spinner('Wait for it...'):
    time.sleep(10)
      

if audio_path:
    res=obj.real_time_transcription(audio_path,stream)
    model1 = PunctuationModel()
    result = model1.restore_punctuation(res)
    st.write(result)