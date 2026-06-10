import re
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


def clean_response(text: str) -> str:
    """
    Remove <think>...</think> blocks if model outputs them.
    """
    if not text:
        return ""

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()


if "messages_history" not in st.session_state:
    st.session_state.messages_history = []


# Display chat history
for msg in st.session_state.messages_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Type Here...")


if user_input:

    # Show user message
    st.session_state.messages_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):

        response = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG
        )

        ai_message = response["messages"][-1].content

        # Remove think tags if present
        ai_message = clean_response(ai_message)

        st.markdown(ai_message)

    st.session_state.messages_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )