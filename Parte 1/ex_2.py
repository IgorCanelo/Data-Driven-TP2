from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

app = FastAPI()

class TextInput(BaseModel):
    texto: str

@app.post("/traduzir/")
def traduzir_texto(input_data: TextInput):

    inputs = tokenizer(input_data.texto, return_tensors="pt")
    outputs = model.generate(**inputs)
    texto_traduzido = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"traducao": texto_traduzido}