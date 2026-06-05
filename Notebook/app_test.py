from chatbot_Backend_basic_persistent_mem import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

thread_id = '2'
config = {'configurable': {'thread_id' : thread_id}}

response = chatbot.invoke({
    'message': [HumanMessage(content='What is OOPS?')]}, config=config)
print(response['message'][-1].content)

