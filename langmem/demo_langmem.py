import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore


USER_ID = "youtube_demo_user"
MEMORY_NAMESPACE = ("learning_coach_memory", USER_ID)

FIRST_USER_MESSAGE = (
    "Im a backend engineer learning Agentic AI. "
    "I prefer practical examples. "
    "Im preparing for AI interviews."
)

SECOND_USER_MESSAGE = "How should I study this week?"


class LearningCoachMemory(BaseModel):
    memory_type: Literal[
        "role",
        "learning_goal",
        "preference",
        "current_focus",
    ] = Field(
        ...,
        description="The kind of information being remembered."
    )

    value: str = Field(
        ...,
        description="The concise thing to remember."
    )

    reason: str = Field(
        ...,
        description="Why this memory can help personalize future coaching."
    )


class MemoryList(BaseModel):
    memories: list[LearningCoachMemory] = Field(
        ...,
        description="List of extracted user memories."
    )


def print_section(title: str):
    print("\n" + "=" * 30)
    print(title)
    print()


def setup_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


def setup_memory_store():
    return InMemoryStore()


def extract_and_store_memory(
    user_message: str,
    llm: ChatGroq,
    store: InMemoryStore,
):
    structured_llm = llm.with_structured_output(MemoryList)

    result = structured_llm.invoke(
        f"""
Extract durable memories from the user.

Allowed memory types:
- role
- learning_goal
- preference
- current_focus

Create one memory for each applicable fact.

User:
{user_message}
"""
    )

    stored = []

    for memory in result.memories:
        memory_id = str(uuid.uuid4())

        store.put(
            MEMORY_NAMESPACE,
            key=memory_id,
            value={"content": memory.model_dump()},
        )

        stored.append(memory)

    return stored


def recall_memories(store: InMemoryStore):
    items = store.search(
        MEMORY_NAMESPACE,
        query=None,
        limit=20,
    )

    memories = []

    for item in items:
        memories.append(
            LearningCoachMemory.model_validate(
                item.value["content"]
            )
        )

    return memories


def format_memories(memories):
    if not memories:
        return "* No memories found."

    return "\n".join(
        f"* {m.memory_type}: {m.value}"
        for m in memories
    )


def generate_personalized_response(
    user_message: str,
    memories,
    llm,
):
    memory_text = format_memories(memories)

    response = llm.invoke(
        [
            SystemMessage(
                content="""
You are an AI learning coach.

Use recalled memories to personalize advice.

Be practical and concise.
"""
            ),
            HumanMessage(
                content=f"""
Recalled memories:

{memory_text}

User question:
{user_message}

Create a 1-week study plan.

Explicitly explain how the memories influenced the advice.
"""
            ),
        ]
    )

    return response.content


def main():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found")

    llm = setup_llm()
    store = setup_memory_store()

    print_section("CONVERSATION 1")

    print("User:")
    print(FIRST_USER_MESSAGE)

    stored_memories = extract_and_store_memory(
        FIRST_USER_MESSAGE,
        llm,
        store,
    )

    print_section("MEMORIES STORED")

    print(format_memories(stored_memories))

    print_section("CONVERSATION 2")

    print("User:")
    print(SECOND_USER_MESSAGE)

    recalled = recall_memories(store)

    print_section("MEMORIES RECALLED")

    print(format_memories(recalled))

    answer = generate_personalized_response(
        SECOND_USER_MESSAGE,
        recalled,
        llm,
    )

    print_section("PERSONALIZED RESPONSE")

    print(answer)


if __name__ == "__main__":
    main()