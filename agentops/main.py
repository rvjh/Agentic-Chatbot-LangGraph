"""
Groq + LangChain + AgentOps Customer Support Agent Demo
"""

import os
import re

import agentops
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


# ============================================================
# TOOLS
# ============================================================

@tool
def get_order_status(order_id: str) -> str:
    """
    Get the status of an order.
    """

    orders = {
        "ORD123": "Shipped and arriving tomorrow.",
        "ORD456": "Processing in warehouse.",
        "ORD789": "Delivered yesterday.",
    }

    return orders.get(
        order_id,
        f"Order {order_id} was not found."
    )


@tool
def get_refund_policy() -> str:
    """
    Return the refund policy.
    """

    return (
        "Items can be returned within 30 days of delivery. "
        "Damaged products qualify for a refund or replacement."
    )


@tool
def create_support_ticket(issue: str) -> str:
    """
    Create a support ticket.
    """

    return (
        f"Support ticket created successfully.\n"
        f"Issue: {issue}\n"
        f"A human support representative will review it."
    )


# ============================================================
# CONFIG
# ============================================================

DEMO_QUERIES = [
    "Where is my order ORD123?",
    "What is your refund policy?",
    "My product arrived damaged. Can you create a ticket?",
]

SUPPORT_AGENT_INSTRUCTIONS = """
You are a helpful customer support AI assistant.

Rules:
- Use tools whenever appropriate.
- Never invent order status.
- If an order ID is missing, ask for it.
- For refund questions, use the refund policy tool.
- For damaged products or escalation requests, use the ticket tool.
- Keep responses concise.
"""


# ============================================================
# ENVIRONMENT
# ============================================================

def check_required_environment() -> None:

    missing = []

    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")

    if not os.getenv("AGENTOPS_API_KEY"):
        missing.append("AGENTOPS_API_KEY")

    if missing:
        print("Missing required environment variables:")

        for key in missing:
            print(f" - {key}")

        raise ValueError(
            "Please add the required API keys to your .env file."
        )


# ============================================================
# AGENTOPS
# ============================================================

def initialize_agentops() -> None:
    """
    Start AgentOps tracing.
    """

    agentops.init(
        api_key=os.getenv("AGENTOPS_API_KEY"),
        tags=["groq", "support-agent-demo"],
    )


# ============================================================
# LLM
# ============================================================

def build_agent():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    tools = [
        get_order_status,
        get_refund_policy,
        create_support_ticket,
    ]

    return llm.bind_tools(tools)


# ============================================================
# HELPERS
# ============================================================

def extract_order_id(text: str):

    match = re.search(
        r"ORD\d+",
        text.upper()
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# AGENT EXECUTION
# ============================================================

def run_agent(agent, query: str):

    query_lower = query.lower()

    # ----------------------------------------
    # ORDER STATUS
    # ----------------------------------------

    if "order" in query_lower:

        order_id = extract_order_id(query)

        if not order_id:
            return "Please provide your order ID."

        return get_order_status.invoke(
            {"order_id": order_id}
        )

    # ----------------------------------------
    # REFUND POLICY
    # ----------------------------------------

    if "refund" in query_lower:

        return get_refund_policy.invoke({})

    # ----------------------------------------
    # SUPPORT TICKET
    # ----------------------------------------

    if any(
        keyword in query_lower
        for keyword in [
            "damaged",
            "broken",
            "ticket",
            "crushed",
            "escalate",
        ]
    ):

        return create_support_ticket.invoke(
            {"issue": query}
        )

    # ----------------------------------------
    # GENERAL LLM RESPONSE
    # ----------------------------------------

    response = agent.invoke(
        [
            SystemMessage(
                content=SUPPORT_AGENT_INSTRUCTIONS
            ),
            HumanMessage(
                content=query
            ),
        ]
    )

    return response.content


# ============================================================
# DEMO
# ============================================================

def run_demo_query(agent, query: str):

    print("\n" + "=" * 72)
    print(f"User Query: {query}")
    print("-" * 72)

    result = run_agent(agent, query)

    print("Final Response:")
    print(result)

    print()
    print(
        "Trace note: check the AgentOps dashboard "
        "for LLM and tool traces."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    load_dotenv()

    check_required_environment()

    initialize_agentops()

    try:

        agent = build_agent()

        for query in DEMO_QUERIES:
            run_demo_query(agent, query)

        agentops.end_session("Success")

        print(
            "\nDemo complete. "
            "Open AgentOps to inspect traces."
        )

    except Exception as exc:

        agentops.end_session("Failure")

        raise exc


if __name__ == "__main__":
    main()