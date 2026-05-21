import logging

from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage

from app.langgraph_agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50
MAX_SAME_FAILURE = 3
EVAL_TIMEOUT = 8.0


async def evaluation_node(state: AgentState) -> dict:
    writer = None
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except RuntimeError:
        pass

    last_message = state["messages"][-1]
    assistant_content = getattr(last_message, "content", "") or ""
    iteration = state.get("iteration_count", 0)

    if not assistant_content.strip():
        if writer:
            writer({
                "type": "step",
                "name": "loop_agent",
                "status": "error",
                "label": "Loop Agent",
                "detail": f"迭代 {iteration}: LLM 返回空内容",
            })
        return {"final_content": "", "continue_loop": False}

    is_ok, reason = await _evaluate_quality(
        state["user_message"],
        assistant_content,
        state["api_key"],
        state["base_url"],
        state["model_name"],
    )

    if writer:
        writer({
            "type": "step",
            "name": "loop_agent",
            "status": "completed" if is_ok else "running",
            "label": "Loop Agent",
            "detail": (
                f"迭代 {iteration}/{MAX_ITERATIONS}: "
                f"{'通过' if is_ok else '未通过 — ' + reason[:50]}"
            ),
        })

    if is_ok:
        return {"final_content": assistant_content, "continue_loop": False}

    # Track repeated failures
    last_reason = state.get("last_failure_reason", "")
    same_count = state.get("same_failure_count", 0)
    if reason == last_reason:
        same_count += 1
    else:
        same_count = 1

    if same_count >= MAX_SAME_FAILURE:
        if writer:
            writer({
                "type": "step",
                "name": "loop_agent",
                "status": "error",
                "label": "Loop Agent",
                "detail": f"连续 {MAX_SAME_FAILURE} 次相同失败，停止迭代",
            })
        return {
            "final_content": assistant_content,
            "continue_loop": False,
            "same_failure_count": same_count,
            "last_failure_reason": reason,
        }

    if iteration >= MAX_ITERATIONS:
        if writer:
            writer({
                "type": "step",
                "name": "loop_agent",
                "status": "error",
                "label": "Loop Agent",
                "detail": f"达到最大迭代次数 {MAX_ITERATIONS}，请用户介入",
            })
        return {
            "final_content": assistant_content,
            "continue_loop": False,
            "same_failure_count": same_count,
            "last_failure_reason": reason,
        }

    # Append correction and retry
    correction = _build_correction(reason)
    return {
        "messages": [SystemMessage(content=correction)],
        "continue_loop": True,
        "same_failure_count": same_count,
        "last_failure_reason": reason,
    }


async def _evaluate_quality(
    user_message: str,
    assistant_response: str,
    api_key: str,
    base_url: str,
    model_name: str,
) -> tuple[bool, str]:
    prompt = (
        f'用户问题是："{user_message}"\n\n'
        f'以下是助手的回答：\n{assistant_response[:800]}\n\n'
        f'请判断助手的回答是否直接、正确地回应了用户的问题。'
        f'特别注意以下不合格情形：\n'
        f'1. 回答与用户问题无关（例如用户问哲学理论却返回了汉字字典解释或拼音页）\n'
        f'2. 回答基于错误的搜索关键词（如将"拉康"拆成"拉"导致搜索到字典页面）\n'
        f'3. 回答内容明显是某个单字的释义而非用户询问的专有名词概念\n\n'
        f'如果存在上述问题，请回复 FAIL: <简短原因>。'
        f'如果回答基本正确回应了用户问题，请回复 OK。'
        f'只回复 OK 或 FAIL: <原因>，不要回复其他内容。'
    )

    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=80,
            streaming=False,
        )
        resp = await model.ainvoke([SystemMessage(content=prompt)])
        text = (getattr(resp, "content", "") or "").strip()
        logger.debug("Evaluate result: %s", text)

        if text.upper().startswith("OK"):
            return True, ""
        reason = text.split("FAIL:", 1)[-1].strip() if "FAIL:" in text else text[:80]
        return False, reason
    except Exception:
        logger.exception("Evaluation call failed")
        return True, ""


def _build_correction(reason: str) -> str:
    return (
        f"上一轮你的回答被判定为不合格。原因：{reason}\n\n"
        f"请修正后重新回答。特别注意：\n"
        f"1. 搜索时必须使用完整的人名、地名或专有名词，禁止拆成单字。"
        f"错误：'拉的理论' → 正确：'拉康 精神分析 理论'\n"
        f"2. 搜索关键词应包含 2-4 个相关词，用空格分隔，不要太宽泛。"
        f"错误：'拉康' → 正确：'拉康 精神分析 镜像阶段'\n"
        f"3. 如果之前的搜索结果包含字典页、拼音页等无关内容，"
        f"请用更精确的关键词重新调用 web_search_tool。\n"
        f"4. 确保回答内容与用户问题直接相关。"
    )
