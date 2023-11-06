from deepspeech import Model
import numpy as np
import os
import wave
from IPython.display import Audio

from punctuationmodel import PunctuationModel

class Audio_to_text:
    def __init__(self,Model):
        self.model_file_path ='deepspeech-0.8.2-models.pbmm'
        self.lm_file_path='deepspeech-0.8.2-models.scorer'
        self.beam_width=100
        self.lm_alpha = 0.93
        self.lm_beta = 1.18
        self.Model=Model
    def model_prepare(self):
        model=self.Model(self.model_file_path)    
        model.enableExternalScorer(self.lm_file_path)
        model.setScorerAlphaBeta(self.lm_alpha,self.lm_beta)
        model.setBeamWidth(self.beam_width)
        return model
        
    def read_audio_file(self,filename):
        with wave.open(filename, 'rb') as w:
                rate = w.getframerate()
                frames = w.getnframes()
                buffer = w.readframes(frames)

        return buffer, rate
    
    def real_time_transcription(self,audio_file,stream):
        buffer, rate = self.read_audio_file(audio_file)
        offset=0
        batch_size=8196
        text=''

        while offset < len(buffer):
            end_offset=offset+batch_size
            chunk=buffer[offset:end_offset]
            data16 = np.frombuffer(chunk, dtype=np.int16)

            stream.feedAudioContent(data16)
            text=stream.intermediateDecode()
            # print(text)
            offset=end_offset
         
        return text
# if __name__ == "__main__":    
#     obj = Audio_to_text(Model)
#     model=obj.model_prepare()
#     stream = model.createStream()
#     audio_path="man1_wb.wav"
#     res=obj.real_time_transcription(audio_path,stream)
#     print(res)

#     model1 = PunctuationModel()
#     result = model1.restore_punctuation(res)
#     print(result)

    # text = "welcome to pakistan hahahaha  you is my favorate no no no hi"
    # # restore add missing punctuation
    # result = model.restore_punctuation(text)
    # print(result)

    # clean_text = model.preprocess(text)
    # labled_words = model.predict(clean_text)
    # print(labled_words)  

    
# pip install transformers[sentencepiece]
