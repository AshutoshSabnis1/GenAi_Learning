import streamlit as st 
from bot_backend import get_response
st.title("🤖 Chatbot with Gemini")
st.markdown("This is a simple chatbot application using Google Gemini model.")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask me anything !")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res=get_response(query)
    st.chat_message("assistant").markdown(res)
    st.session_state.messages.append({"role":"assistant","content":res})
    print(query) 