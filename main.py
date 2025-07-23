from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import os
import uuid
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi Agent Architecture Task 21/07")

class UserRequest(BaseModel):
    query: str
    thread_id: str | None = None

os.environ["TAVILY_API_KEY"] = "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"  

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = "meta-llama-3.1-8b-instruct"

def validate_lm_studio():
    try:
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        response.raise_for_status()
        logger.info("LM Studio API is accessible.")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to connect to LM Studio: {str(e)}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_model():
    try:
        if not validate_lm_studio():
            raise Exception("LM Studio API is not running or inaccessible.")
        model = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="not-needed",
            model=MODEL_NAME
        )
       
        test_response = model.invoke("Hello")
        logger.info("Model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

try:
    model = initialize_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise Exception(f"Model initialization failed: {str(e)}")

# Define tools
@tool
def decide_if_web_search_needed(query: str) -> str:
    """Decide if a web search is needed to fulfill the user's request."""
    complex_keywords = ["example", "tutorial", "best practice", "library", "framework"]
    if any(keyword in query.lower() for keyword in complex_keywords):
        return "Web search is recommended to gather additional context or examples."
    return "No web search needed; the request can be handled directly."

try:
    tavily_search = TavilySearchAPIWrapper()
    search_tool = TavilySearchResults(api_wrapper=tavily_search, max_results=3)
except Exception as e:
    logger.error(f"Failed to initialize Tavily search: {str(e)}")
    raise Exception(f"Tavily initialization failed: {str(e)}")

messenger = create_react_agent(
    model,
    tools=[
        decide_if_web_search_needed,
        create_handoff_tool(agent_name="Coder", description="Handoff to Coder for code generation"),
        create_handoff_tool(agent_name="WebSearch", description="Handoff to WebSearch for external information")
    ],
    prompt="You are Messenger, a helpful assistant that interacts with users, evaluates their requests, and delegates tasks. Use the decide_if_web_search_needed tool to determine if a web search is required. If yes, hand off to WebSearch; if no, hand off to Coder. Always return the final response to the user after receiving results.",
    name="Messenger"
)

coder = create_react_agent(
    model,
    tools=[create_handoff_tool(agent_name="Messenger", description="Return results to Messenger")],
    prompt="You are Coder, an expert in generating working code. Write clean, functional code based on the user's request and return it to Messenger.",
    name="Coder"
)

web_search = create_react_agent(
    model,
    tools=[
        search_tool,
        create_handoff_tool(agent_name="Messenger", description="Return search results to Messenger")
    ],
    prompt="You are WebSearch, an expert in finding relevant information online. Use the Tavily search tool to gather information based on the user's request, summarize the findings, and return them to Messenger.",
    name="WebSearch"
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    agents=[messenger, coder, web_search],
    default_active_agent="Messenger"
)
swarm_app = workflow.compile(checkpointer=checkpointer)

def get_swarm_app():
    return swarm_app

@app.post("/generate")
async def generate_code(request: UserRequest, swarm=Depends(get_swarm_app)):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Processing request: {request.query} with thread_id: {thread_id}")
        
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def invoke_swarm():
            return swarm.invoke(
                {"messages": [{"role": "user", "content": request.query}]},
                config
            )
        
        response = invoke_swarm()
        
        for message in response["messages"][::-1]:
            if isinstance(message, AIMessage):
                logger.info(f"Returning response: {message.content}")
                return {
                    "response": message.content,
                    "thread_id": thread_id
                }
        
        logger.error("No valid AIMessage found in response.")
        raise HTTPException(status_code=500, detail="No valid response generated.")
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    if validate_lm_studio():
        return {"status": "healthy", "lm_studio": "accessible", "model": MODEL_NAME}
    raise HTTPException(status_code=503, detail="LM Studio API is inaccessible.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)