from chatbot_Backend_basic_persistent_mem import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

CONFIG = {"configurable": {"thread_id": "1"}}

st.title("Agentic-Chatbot-LangGraph")

# Store displayed chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom
if prompt := st.chat_input("Type your message..."):

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke chatbot
    response = chatbot.invoke(
        {"message": [HumanMessage(content=prompt)]},
        config=CONFIG
    )

    # Extract only the latest AI message
    ai_message = ""

    if "message" in response:
        messages = response["message"]

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break

    # Fallback
    if not ai_message:
        ai_message = "No response generated."

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_message}
    )

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_message)