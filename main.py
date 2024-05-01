from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from predict import Prediction

app = FastAPI()

prediction = Prediction()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class Prompt(BaseModel):
    prompt: str

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
    item = prediction.get_embedding(str(prompt.prompt))
    data = {"prompt": str(prompt), 
            "embedding": str(item)}
    print(data)
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7979)