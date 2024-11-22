from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


app = FastAPI()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyDUm58IAr5Ufp6kTw-HWRKnIoU0hBBI-qc")

class TranslationInput(BaseModel):
    text: str

@app.post("/translate")
async def translate_text(input: TranslationInput):
    try:
        prompt = f"Translate the following text to French: {input.text}"
        translation = llm.invoke(prompt).content
        return {"original": input.text, "translated": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))