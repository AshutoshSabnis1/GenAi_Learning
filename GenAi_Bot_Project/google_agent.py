from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm=ChatGroq(model="llama-3.3-70b-versatile")

search = GoogleSerperAPIWrapper()
agent=create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant that can search the web for information."
    
)

print("Welcome ! Type 'exit' to quit the agent.")
while True:
    query=input("User: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the agent. Goodbye!")
        break
    response=agent.invoke({"messages": [{"role": "user", "content": query}]},
                          {"configurable":{"thread_id": "1"}}
                          )
    print("Assistant:", response["messages"][-1].content)