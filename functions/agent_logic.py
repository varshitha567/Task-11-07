from langchain_openai import ChatOpenAI

def generate_code_from_user_input(user_prompt: str) -> str:
    try:
        print("Running Agent 1 with prompt:", user_prompt)
        
        llm = ChatOpenAI(
            openai_api_base="http://localhost:1234/v1", 
            openai_api_key="lm-studio", 
            model="mistralai/Mistral-7B-Instruct-v0.2",  
            temperature=0
        )

        response = llm.invoke(user_prompt)
        return response.content

    except Exception as e:
        print("Agent 1 failed:", str(e))
        return f"Error: {str(e)}"
