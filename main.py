from fastapi import FastAPI
from routers.codegen_router import router as codegen_router

app = FastAPI(title="Two-Agent LangChain - LM Studio")

app.include_router(codegen_router, prefix="/api/codegen", tags=["CodeGen"])
