from transformers import pipeline
from model import llm_model_vi

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
output = transcriber('trichdocnguyentuan.mp3')

response = llm_model_vi(output["text"])
print(response)