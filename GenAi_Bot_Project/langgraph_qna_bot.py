# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import BaseModel to define the graph state structure
from pydantic import BaseModel

# Import Groq chat model
from langchain_groq import ChatGroq

# Import LangGraph core classes
from langgraph.graph import StateGraph, START, END

# add_messages helps LangGraph automatically append new messages
from langgraph.graph.message import add_messages

# InMemorySaver stores chat history in memory using thread_id
from langgraph.checkpoint.memory import InMemorySaver

# Annotated is used to attach add_messages behavior to the messages field
from typing import Annotated

# HumanMessage is used to send user input in proper message format
from langchain_core.messages import HumanMessage


# Define the state of the graph
class ChatState(BaseModel):
    # messages will store the whole conversation
    # add_messages tells LangGraph to append messages instead of replacing them
    messages: Annotated[list, add_messages]


# Create the LLM object
llm = ChatGroq(model="llama-3.3-70b-versatile")


# Define the chatbot node
def chatBotNode(state: ChatState):
    # Send the current conversation history to the model
    res = llm.invoke(state.messages)

    # Return only the new AI response
    # add_messages will automatically append it to existing messages
    return {"messages": [res]}


# Create memory object to store conversation history
memory = InMemorySaver()

# Create the graph using ChatState as schema
graph = StateGraph(ChatState)

# Add chatbot node
graph.add_node("chatBot", chatBotNode)

# Define graph flow: START -> chatBot
graph.add_edge(START, "chatBot")

# Define graph flow: chatBot -> END
graph.add_edge("chatBot", END)

# Compile the graph with memory support
graph = graph.compile(checkpointer=memory)


# Optional: show graph diagram in Jupyter/IPython
# from IPython.display import Image
# Image(graph.get_graph().draw_mermaid_png())


# Config with thread_id
# Same thread_id means same conversation memory
config = {"configurable": {"thread_id": "bot1"}}


# Infinite chat loop
while True:
    # Take input from user
    query = input("User: ")

    # Stop the chat if user types exit or quit
    if query.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break

    # Send the ACTUAL user input to the graph
    res = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    # Get the last message from graph output
    ans = res["messages"][-1].content

    # Print AI response
    print("AI:", ans)