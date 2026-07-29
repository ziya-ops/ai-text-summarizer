from fastapi import FastAPI
from app.routers import summarize

app = FastAPI(title="Production LLM API")

app.include_router(summarize.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
