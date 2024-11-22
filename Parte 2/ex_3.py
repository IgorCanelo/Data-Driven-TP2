from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

class TranslationInput(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(input: TranslationInput):
    try:
        inputs = tokenizer(input.text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"original": input.text, "translated": translation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))