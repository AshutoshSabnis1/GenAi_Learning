# streamlit_groq_agent.py
# A Streamlit front-end demonstrating: LLM (Groq) + Google Serper search tool + Agent + Memory + streaming responses.

from dotenv import load_dotenv                      # load environment variables from a .env file into os.environ
load_dotenv()                                       # actually read .env now so API keys are available

# LLM & tools imports — these packages must be installed and configured in your environment.
from langchain_groq import ChatGroq                  # ChatGroq: wrapper to talk to Groq LLMs (assumes this integration exists)
from langchain_community.utilities import GoogleSerperAPIWrapper
                                                    # GoogleSerperAPIWrapper: convenience wrapper to call Google Serper API
from langchain.agents import create_agent            # create_agent: helper to build an agent that can use tools
from langgraph.checkpoint.memory import MemorySaver  # MemorySaver: class to persist agent memory / checkpoints
import streamlit as st                               # Streamlit UI library

# -----------------------
# LLM + Tool initialization
# -----------------------

# Instantiate the LLM with streaming enabled so we can stream partial tokens to the UI.
# model="llama-3.3-70b-versatile" is your chosen model identifier; change if you want a different model.
llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

# Instantiate the Google Serper wrapper (expects your SERPER API key in environment variables).
# The wrapper exposes convenient methods such as `.run(query)` for synchronous usage.
search = GoogleSerperAPIWrapper()

# Create a simple callable to be passed to the agent as a tool.
# We store the callable in a variable named google_search_tool for clarity.
# create_agent often expects a list of 'tools' where each tool is either a Tool object or a callable.
google_search_tool = search.run

# -----------------------
# Streamlit session-state: memory & conversation history
# -----------------------

# initialize memory and conversation history in Streamlit session state if not already present
if "memory" not in st.session_state:
    # MemorySaver used as a checkpointer to persist agent state across interactions
    st.session_state.memory = MemorySaver()

# Keep a chat history list; each entry is a dict matching typical chat message shape
# This makes it easy to re-render the chat UI on every rerun.
if "history" not in st.session_state:
    st.session_state.history = []

    

# -----------------------
# Create the agent
# -----------------------

# create_agent wraps an LLM and tools into an "agent" that can decide to call tools and produce responses.
# - model=llm: the LLM instance to use
# - tools=[google_search_tool]: list of tools the agent can call (here we pass the search wrapper's run method)
# - checkpointer=st.session_state.memory: memory persistence for the agent
# - system_prompt: gives the agent a system-level instruction
agent = create_agent(
    model=llm,
    tools=[google_search_tool],                       # pass a list of available tools (here only one)
    checkpointer=st.session_state.memory,
    system_prompt="You are a helpful assistant that can search the web for information."
)

# -----------------------
# Streamlit UI layout & rendering of history
# -----------------------

st.subheader("Google Search Agent with Groq LLM")

# Render the existing chat history (stored in session_state). Each message is shown using Streamlit's chat UI.
for message in st.session_state.history:
    role = message["role"]      # either "user" or "assistant"
    content = message["content"]
    # st.chat_message(role) creates a message bubble with the given role. .markdown() writes content inside.
    st.chat_message(role).markdown(content)

# A simple input box for the user to type a new query.
query = st.chat_input("Type your query here...")

# When the user submits a query, stream the agent's response and append everything to session_state.history
if query:
    # show the user's message instantly in the UI
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    # Call the agent in streaming mode.
    # Many agent implementations expose a `.stream()` method which yields partial tokens/chunks.
    # Here we call it with:
    #  - {"messages":[{"role":"user", "content": query}]} as the input message structure
    #  - {"configurable":{"thread_id":"1"}} as an example metadata/config input (your agent might use different keys)
    #  - stream_mode="messages" to indicate we want message-based chunks (depends on the implementation)
    try:
        response_stream = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="messages"
        )
    except Exception as e:
        # If the call fails immediately, show the error and append an assistant message explaining the failure.
        st.chat_message("assistant").markdown(f"Error starting stream: {e}")
        st.session_state.history.append({"role": "assistant", "content": f"Error starting stream: {e}"})
    else:
        # Create an assistant chat container in Streamlit where we'll stream partial content.
        ai_container = st.chat_message("assistant")
        with ai_container:
            # `space` is a placeholder we can update repeatedly as we receive chunks.
            space = st.empty()
            message = ""     # accumulate chunks here
            try:
                # Iterate over the streaming generator provided by the agent
                for chunk in response_stream:
                    # The exact shape of `chunk` depends on your agent implementation.
                    # In your original code you used `chunk[0].content` — here we try to be defensive:
                    # - if chunk is a list/tuple and first element has `content`, use that
                    # - else if chunk is a mapping with 'content', use that
                    # - otherwise convert chunk to string
                    part = ""
                    try:
                        # common case: chunk is a sequence whose first element is an object with `.content`
                        if isinstance(chunk, (list, tuple)) and len(chunk) > 0:
                            first = chunk[0]
                            # if first is an object with .content attribute:
                            if hasattr(first, "content"):
                                part = getattr(first, "content") or ""
                            # or if it's a dict with key 'content'
                            elif isinstance(first, dict) and "content" in first:
                                part = first["content"] or ""
                            else:
                                part = str(first)
                        elif isinstance(chunk, dict) and "content" in chunk:
                            part = chunk["content"] or ""
                        else:
                            part = str(chunk)
                    except Exception:
                        # fall back to string conversion if anything goes wrong above
                        part = str(chunk)

                    # Append partial text and update the displayed message
                    message += part
                    space.write(message)
            except Exception as stream_err:
                # If streaming raised an error midway, show a small error note and continue.
                space.write(f"\n\n[streaming error] {stream_err}")
                message += f"\n\n[streaming error] {stream_err}"

            # After streaming completes (or errors), append the final assistant message to history
            st.session_state.history.append({"role": "assistant", "content": message})