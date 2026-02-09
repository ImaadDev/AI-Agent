# llm_wrapper.py
from event_log import log_event

async def call_llm(
    llm,
    messages,
    *,
    business_id: str,
    thread_id: str,
    turn_id: str,
    agent_node: str,
):
    resp = await llm.ainvoke(messages)
    usage = (
        resp.response_metadata.get("token_usage")
        or resp.usage_metadata
        or {}
    )
    log_event({
        "event": "llm_usage",
        "business_id": business_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "agent_node": agent_node,
        "model": llm.model_name,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    })

    return resp