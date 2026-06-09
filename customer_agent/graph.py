"""Customer Agent LangGraph definition.

Uses create_react_agent with a `delegate_to_legal_agent` tool that:
1. Discovers the Law Agent via the registry
2. Sends the question to it via A2A
3. Returns the comprehensive legal response to the user

The tool accepts context propagation data (trace_id, context_id, depth)
via a closure — these are bound per-request in agent_executor.py.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

CUSTOMER_SYSTEM_PROMPT = """You are a helpful legal assistant at the front desk of a multi-agent
legal services platform. Your job is to:

1. Understand the user's legal question
2. Determine if it needs specialist legal analysis (contract issues, tax law,
   regulatory compliance, corporate liability, etc.)
3. If so, use the `delegate_to_legal_agent` tool to send it to the Law Agent,
   which will coordinate specialist sub-agents (Tax and Compliance) as needed
4. Present the comprehensive response clearly to the user

Always use the `delegate_to_legal_agent` tool for any substantive legal question.
Do not attempt to answer complex legal questions from your own knowledge alone.

Be professional, clear, and make the specialist response accessible to the user.
"""


class CustomerState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(trace_id: str, context_id: str, depth: int) -> Any:
    """Build a minimal "delegate-and-return" graph.

    LATENCY OPTIMISATION: the previous version used create_react_agent, which
    needed TWO sequential LLM calls per request (one to decide to delegate, one
    to re-present the answer). Since the Law Agent already returns a complete,
    well-structured response, both of those LLM calls are pure overhead on the
    critical path. This version delegates directly and returns the Law Agent's
    answer verbatim — removing two slow LLM calls per request.

    Args:
        trace_id: UUID generated at this request's entry point.
        context_id: A2A context_id for this conversation.
        depth: Delegation depth (0 at customer agent).

    Returns:
        A compiled LangGraph graph exposing the {"messages": [...]} interface.
    """

    async def delegate_node(state: CustomerState) -> dict:
        from common.a2a_client import delegate
        from common.registry_client import discover

        question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        logger.info(
            "Customer delegate (direct) | trace=%s context=%s depth=%d",
            trace_id, context_id, depth,
        )

        try:
            endpoint = await discover("legal_question")
            result = await delegate(
                endpoint=endpoint,
                question=question,
                context_id=context_id,
                trace_id=trace_id,
                depth=depth + 1,
            )
            if not result:
                result = "The Law Agent returned an empty response. Please try again."
        except Exception as exc:
            logger.exception("delegate_to_legal_agent failed: %s", exc)
            result = f"Could not reach the Law Agent: {exc}"

        return {"messages": [AIMessage(content=result)]}

    graph = StateGraph(CustomerState)
    graph.add_node("delegate", delegate_node)
    graph.add_edge(START, "delegate")
    graph.add_edge("delegate", END)
    return graph.compile()