from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

# Agent 2: The Coder Agent setup
def get_coder_executor() -> AgentExecutor:
    llm = ChatOpenAI(
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="lm-studio",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0
    )

    coder_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant. Only return valid Python code based on the user's request. Do not explain."),
        MessagesPlaceholder("input"),
    ])

    coder_agent = create_react_agent(llm, coder_prompt)
    coder_executor = AgentExecutor(agent=coder_agent, tools=[], verbose=True)
    return coder_executor


# Tool wrapper class to allow Agent 1 to call Agent 2
class CoderTool:
    def __init__(self, coder_executor: AgentExecutor):
        self.coder_executor = coder_executor

    def send_to_coder(self, message: str) -> str:
        """Send user request to the Coder agent and return code."""
        result = self.coder_executor.invoke({"input": message})
        return result["output"]


# Agent 1: The Messenger Agent
def generate_code_from_user_input(user_prompt: str) -> str:
    try:
        print("Agent 1 (Messenger) received:", user_prompt)

        # Step 1: Create Coder agent executor
        coder_executor = get_coder_executor()
        coder_tool = CoderTool(coder_executor)

        # Step 2: Create LangChain tool from CoderTool method
        send_to_coder_tool = Tool(
            name="send_to_coder",
            func=coder_tool.send_to_coder,
            description="Use this to get Python code from the Coder agent based on user prompt."
        )

        # Step 3: Create Messenger agent
        llm = ChatOpenAI(
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="lm-studio",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0
        )

        messenger_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Messenger agent. Your job is to collect user coding requests and pass them to the Coder agent using the tool."),
            MessagesPlaceholder("input"),
        ])

        messenger_agent = create_react_agent(llm, messenger_prompt)
        messenger_executor = AgentExecutor(agent=messenger_agent, tools=[send_to_coder_tool], verbose=True)

        # Step 4: Invoke Messenger with user input
        response = messenger_executor.invoke({"input": user_prompt})
        return response["output"]

    except Exception as e:
        print("Agent 1 (Messenger) failed:", str(e))
        return f"Error: {str(e)}"
