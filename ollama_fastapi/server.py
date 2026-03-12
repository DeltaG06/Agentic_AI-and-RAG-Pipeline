from fastapi import FastAPI ,Body 
from ollama import Client

app = FastAPI()
client = Client(
    host = "http://localhost:11434/",
)



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/contact")
def read_root():
    return {"email": "parkergaurav@gmail.com"}

@app.post("/chat")
def chat(
        message: str = Body(...,description = "The message")

):
    response = client.chat(model = "gemma:2b", messages = [
        {"role" : "user", "content" : message}
        ])
    return{"response": response.message.content}