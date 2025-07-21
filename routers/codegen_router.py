from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from functions.agent_logic import generate_code_from_user_input

router = APIRouter()

class CodeRequest(BaseModel):
    prompt: str

@router.post("/")
async def generate_code(request: CodeRequest):
    try:
        code = generate_code_from_user_input(request.prompt)
        return {"generated_code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
