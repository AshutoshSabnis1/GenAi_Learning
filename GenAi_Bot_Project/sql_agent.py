# short_task_agent.py — minimal agent + in-memory memory (langgraph)
import os
from dotenv import load_dotenv
load_dotenv()                         # load GROQ_API_KEY or other vars from .env if present

import streamlit as st
from uuid import uuid4

# LLM + agent imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# langgraph in-memory checkpointer (stores conversation memory in RAM)
from langgraph.checkpoint.memory import InMemorySaver

# -------------------------
# Small config
# -------------------------
SYSTEM_PROMPT = (
    "You are a concise task-management assistant. Respond clearly. "
    "When the user asks, create, list, update, or delete tasks (or say you will)."
)

# instantiate model (ensure GROQ credentials available)
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# -------------------------
# Build and cache the agent with in-memory checkpointer
# -------------------------
@st.cache_resource
def get_agent():
    # create_agent accepts model, optional tools (here none), checkpointer, and system_prompt
    agent = create_agent(
        model=model,
        tools=[],                         # no external tools in this minimal example
        checkpointer=InMemorySaver(),     # langgraph in-memory saver (keeps memory during runtime)
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

agent = get_agent()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Minimal Agent (with memory)", layout="wide")
st.title("Minimal Agent — system prompt + in-memory memory")

# create a stable thread id per session so InMemorySaver can store/retrieve memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid4().hex}"

# single-line user input
user_input = st.chat_input("Type something (e.g., 'add task Buy milk', or 'show tasks')")

if user_input:
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # invoke the agent and pass the required configurable thread_id so the checkpointer works
                response = agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )

                # extract assistant reply (shape may vary by library/version)
                assistant_text = response["messages"][-1].content

            except Exception as e:
                assistant_text = f"⚠️ Error: {type(e).__name__}: {e}"

            st.markdown(assistant_text)
            # Optionally keep a per-session chat history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({"user": user_input, "assistant": assistant_text})

# quick history view
if "history" in st.session_state and st.session_state.history:
    st.markdown("---")
    st.subheader("Session history (cached in-memory)")
    for pair in st.session_state.history[-10:]:
        st.markdown(f"**User:** {pair['user']}")
        st.markdown(f"**Assistant:** {pair['assistant']}")