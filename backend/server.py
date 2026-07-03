from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama

from rag import search_knowledge

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    text: str


@app.post("/chat")
def chat(message: Message):

    user_question = message.text

    # Search knowledge base
    context = search_knowledge(user_question)

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
You are Britney AI.

Use the provided knowledge base as your primary source of truth.

When responding:
- Follow the knowledge base consistently.
- Never contradict it or invent new concepts.
- Do not predict outcomes or give false certainty.
- Guide the user back to identity rather than circumstances.
- Keep responses calm, clear, and conversational.
"""
            },
            {
                "role": "user",
                "content": f"""
Knowledge Base:
{context}

User Question:
{user_question}
"""
            }
        ]
    )

    return {
        "reply": response["message"]["content"]
    }