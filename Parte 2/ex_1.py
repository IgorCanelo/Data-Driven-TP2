from fastapi import FastAPI, HTTPException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


class ChatInput(BaseModel):
    message: str

# Fake LLM com respostas pré-definidas
fake_responses = [
    "Olá! Como posso ajudar hoje?",
    "Infelizmente, não entendi sua pergunta.",
    "Estou aprendendo e posso responder perguntas simples.",
    "Que interessante! Pode me contar mais?",
    "Desculpe, não tenho informações suficientes para responder."
]

fake_llm = FakeListLLM(responses=fake_responses)
prompt_template = PromptTemplate.from_template(
    "Responda de forma simples e direta: {input}"
)
chain = prompt_template | fake_llm | StrOutputParser()


app = FastAPI(title="Chatbot Simulado")

@app.post("/chat")
async def chat_endpoint(input: ChatInput):
    try:
        response = chain.invoke({"input": input.message})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))