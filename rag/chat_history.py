
# To store and manage messages exchanged in the conversation
from langchain_community.chat_message_histories import ChatMessageHistory
# Defines an interface for storing and retrieving chat messages
from langchain_core.chat_history import BaseChatMessageHistory
# For streamlit reference
import streamlit as st

# Function to return base chat message history based on session ID
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store: #Checks for id in store defined in line 75
        st.session_state.store[session_id] = ChatMessageHistory() #stores hist by mapping it to the corresponding ID 
    return st.session_state.store[session_id]