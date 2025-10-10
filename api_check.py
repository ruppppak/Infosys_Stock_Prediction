from fastapi import FastAPI
app=FastAPI()

@app.get("/")
def Welcome():
    return {"message":"Welcome to FastAPI"}