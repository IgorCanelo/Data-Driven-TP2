from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

app = FastAPI()

class TextInput(BaseModel):
    texto: str

@app.post("/gerar")
def gerar_texto(input_data: TextInput):
    inputs = tokenizer.encode(input_data.texto, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"resultado": texto_gerado}