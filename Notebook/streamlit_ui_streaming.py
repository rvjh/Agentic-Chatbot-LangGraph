import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_Backend_basic_persistent_mem import chatbot

st.set_page_config(
    page_title="Agentic Chatbot",
    layout="wide"
)

st.title("Agentic Chatbot - LangGraph + Groq")

CONFIG = {
    "configurable": {
        "thread_id": "1"
    }
}

if "messages_history" not in st.session_state:
    st.session_state.messages_history = []

# Display previous messages
for msg in st.session_state.messages_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type Here...")

if user_input:

    # Store and display user message
    st.session_state.messages_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        def response_generator():

            inside_think = False

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                # Skip non-message chunks
                if not hasattr(chunk, "content"):
                    continue

                text = chunk.content

                if not text:
                    continue

                # Handle think tags that may span chunks
                if "<think>" in text:
                    inside_think = True
                    text = text.split("<think>")[0]

                if "</think>" in text:
                    inside_think = False
                    text = text.split("</think>")[-1]

                if inside_think:
                    continue

                if text:
                    yield text

        ai_message = st.write_stream(response_generator)

    st.session_state.messages_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )