from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import uvicorn

from predict import Prediction

import json
from typing import List, Optional

app = FastAPI()

prediction = Prediction()

class Item(BaseModel):
    id: str
    document: Optional[str] = None

class Prompt(BaseModel):
    prompt: str
    documents: List[Item]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.post("/prediction")
async def send_embedding(prompt: Prompt): 
    similarities = prediction.get_similarities(str(prompt.prompt), list(prompt.documents), "cos")
    json_str = json.dumps(similarities, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.post("/chatroom")
async def send_message(prompt: Prompt):
    answer = "목업데이터"
    return answer

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")