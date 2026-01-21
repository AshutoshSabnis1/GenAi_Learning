from dotenv import load_dotenv            
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def get_response(query: str) -> str:
    response = llm.invoke(query)
    return response.content