from chatbot_Backend_basic_persistent_mem import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

## invoke -> wait till all tokens are produced ##
# response = chatbot.invoke(
#         {"messages": [HumanMessage(content="What is OOPS?")]},
#         config=CONFIG
#     )
# print(response["messages"][-1].content)


for message_chunk, metadata in chatbot.stream(
    {"messages": [HumanMessage(content="What is OOPS?")]},
    config=CONFIG,
    stream_mode='messages'):

    if message_chunk.content:
        print(message_chunk.content, end='', flush=True)