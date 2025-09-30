import uvicorn
from fastapi import FastAPI
from app.endpoints.mcqa_endpoints import ROUTER as MCQA_ROUTER

app = FastAPI(title="MCQA API")
app.include_router(MCQA_ROUTER)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )
