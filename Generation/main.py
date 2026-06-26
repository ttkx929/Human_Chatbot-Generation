from typing import Annotated, Callable
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from models import get_model
from prompts import GET_PROMPT
from prompts_gpqa import GET_GPQA_PROMPT
from tqdm import tqdm
import argparse
import jsonlines
from langchain_community.callbacks.manager import get_openai_callback
from utils import UniversalTokenCounter
from data.conversation_utils import CLOSING_SUFFIX, ensure_conversation_ends_with_bot

parser = argparse.ArgumentParser()
parser.add_argument("--inquirer_model", type=str, default="DeepSeek")
parser.add_argument("--responder_model", type=str, default="DeepSeek")
parser.add_argument("--data", type=str, default="oasst1_en")
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="results/")
parser.add_argument("--max_turns", type=int, default=12)
parser.add_argument("--limit", type=int, default=0, help="Process only first N items (0 = all)")
args = parser.parse_args()

DATA_PATHS = {
    "oasst1_en": "data/oasst1_en_min_6_turns_summary.jsonl",
    "arena": "data/arena_model_a_summaries.jsonl",
    "gpqa_diamond": "data/gpqa_diamond_seed.jsonl",
}

PROMPT_GETTERS: dict[str, Callable[..., tuple[str, str, str]]] = {
    "oasst1_en": GET_PROMPT,
    "arena": GET_PROMPT,
    "gpqa_diamond": GET_GPQA_PROMPT,
}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    inquirer_system_prompt: str
    inquirer_prompt: str
    responder_system_prompt: str
    task_summary: str
    pedagogy_mode: str


def count_assistant_turns(messages: list) -> int:
    return sum(1 for message in messages if isinstance(message, AIMessage))


def build_prompts(task_summary: str, *, assistant_turns: int = 0, pedagogy_mode: str = "") -> tuple[str, str, str]:
    getter = PROMPT_GETTERS.get(args.data, GET_PROMPT)
    if args.data == "gpqa_diamond":
        return getter(
            task_summary,
            assistant_turns=assistant_turns,
            pedagogy_mode=pedagogy_mode or None,
        )
    return getter(task_summary)


def inquirer(state: State):
    inquirer_system_prompt = SystemMessage(content=state["inquirer_system_prompt"])
    inquirer_prompt = HumanMessage(content=state["inquirer_prompt"])
    inquirer_message = [inquirer_system_prompt] + state["messages"] + [inquirer_prompt]
    inquirer_response = inquirer_llm.invoke(inquirer_message)
    return {
        "messages": [HumanMessage(
            content=inquirer_response.content,
            additional_kwargs={"source": "generated"},
        )],
        "turns": state["turns"] + 1,
    }


def responder(state: State):
    assistant_turns = count_assistant_turns(state["messages"])
    _, _, system_prompt = build_prompts(
        state["task_summary"],
        assistant_turns=assistant_turns,
        pedagogy_mode=state.get("pedagogy_mode", ""),
    )
    last_human = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human = message.content
            break
    if "<EOD>" in last_human:
        system_prompt += CLOSING_SUFFIX
    responder_prompt = SystemMessage(content=system_prompt)
    responder_message = [responder_prompt] + state["messages"]
    response = responder_llm.invoke(responder_message)
    return {
        "messages": [AIMessage(
            content=response.content,
            additional_kwargs={"source": "generated"},
        )],
        "turns": state["turns"] + 1,
    }


def content_condition(state: State):
    return "responder"


def after_responder_condition(state: State):
    last_human = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human = message.content
            break
    if "<EOD>" in last_human:
        return END
    if state["turns"] >= args.max_turns:
        return END
    return "inquirer"


graph_builder = StateGraph(State)
graph_builder.add_node("inquirer", inquirer)
graph_builder.add_node("responder", responder)
graph_builder.add_edge(START, "inquirer")
graph_builder.add_conditional_edges("responder", after_responder_condition)
graph_builder.add_conditional_edges("inquirer", content_condition)
graph = graph_builder.compile()

inquirer_llm = get_model(args.inquirer_model)
responder_llm = get_model(args.responder_model)


def graph_update(qa_history: list, task_summary: str, metadata: dict | None = None):
    messages = []
    for detail in qa_history:
        if detail["role"] == "human":
            messages.append(HumanMessage(content=detail["content"], additional_kwargs={"source": "qa_history"}))
        elif detail["role"] == "bot":
            messages.append(AIMessage(content=detail["content"], additional_kwargs={"source": "qa_history"}))
        else:
            raise ValueError("Invalid role")

    metadata = metadata or {}
    pedagogy_mode = metadata.get("pedagogy_mode", "")
    assistant_turns = sum(1 for detail in qa_history if detail["role"] == "bot")
    inquirer_system_prompt, inquirer_prompt, responder_system_prompt = build_prompts(
        task_summary,
        assistant_turns=assistant_turns,
        pedagogy_mode=pedagogy_mode,
    )
    initial_state = {
        "messages": messages,
        "turns": len(messages),
        "inquirer_system_prompt": inquirer_system_prompt,
        "inquirer_prompt": inquirer_prompt,
        "responder_system_prompt": responder_system_prompt,
        "task_summary": task_summary,
        "pedagogy_mode": pedagogy_mode,
    }

    try:
        with get_openai_callback() as cb:
            final_state = graph.invoke(initial_state)
            token_usage = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
            }
    except Exception as e:
        print(f"\nOpenAI callback not available, using UniversalTokenCounter: {str(e)}")
        token_counter = UniversalTokenCounter()
        inquirer_llm.callbacks = [token_counter]
        responder_llm.callbacks = [token_counter]
        final_state = graph.invoke(initial_state)
        token_usage = token_counter.get_stats()

    final_state = _ensure_state_ends_with_bot(final_state)
    return final_state, token_usage


def _ensure_state_ends_with_bot(state: dict) -> dict:
    messages = state["messages"]
    if not messages or isinstance(messages[-1], AIMessage):
        return state

    conversation = []
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation.append({"role": "human", "content": message.content})
        elif isinstance(message, AIMessage):
            conversation.append({"role": "bot", "content": message.content})

    assistant_turns = count_assistant_turns(messages)
    _, _, responder_system_prompt = build_prompts(
        state["task_summary"],
        assistant_turns=max(assistant_turns, 3),
        pedagogy_mode=state.get("pedagogy_mode", ""),
    )
    fixed = ensure_conversation_ends_with_bot(
        conversation,
        responder_system_prompt,
        responder_llm,
    )
    if len(fixed) == len(conversation):
        return state

    closing = fixed[-1]
    return {
        **state,
        "messages": messages + [
            AIMessage(content=closing["content"], additional_kwargs={"source": "generated"})
        ],
        "turns": state["turns"] + 1,
    }


def state_to_generated_conversation(messages: list) -> list[dict]:
    generated = []
    for message in messages:
        if message.additional_kwargs.get("source") != "generated":
            continue
        if isinstance(message, HumanMessage):
            content = message.content.replace("<EOD>", "").strip()
            if content:
                generated.append({"role": "human", "content": content})
        elif isinstance(message, AIMessage):
            generated.append({"role": "bot", "content": message.content})
    return generated


def resolve_data_path() -> str:
    if args.input_path:
        return args.input_path
    if args.data not in DATA_PATHS:
        raise ValueError(f"Invalid data: {args.data}. Choose from {list(DATA_PATHS)} or pass --input_path.")
    return DATA_PATHS[args.data]


if __name__ == "__main__":
    data_path = resolve_data_path()
    data = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            data.append(obj)
    if args.limit > 0:
        data = data[: args.limit]

    output_path = (
        f"{args.output_path}{args.data}_{args.inquirer_model}_"
        f"{args.responder_model}_{args.max_turns}.jsonl"
    )

    with jsonlines.open(output_path, mode="w") as writer:
        for dialogue in tqdm(data, total=len(data)):
            seed = dialogue["conversation_id"][:2]
            task_summary = dialogue["task_summary"]
            seed_conversation = dialogue["conversation"][:2]
            metadata = dialogue.get("metadata", {})
            final_state, token_usage = graph_update(seed_conversation, task_summary, metadata)

            generated_conversation = state_to_generated_conversation(final_state["messages"])
            full_conversation = seed_conversation + generated_conversation

            generated_dialogue = {
                "conversation_id": (
                    seed + [""] * (len(final_state["messages"]) - len(seed))
                    if args.data == "oasst1_en"
                    else dialogue.get("conversation_id", seed)
                ),
                "conversation": full_conversation,
                "turns": len(full_conversation),
                "task_summary": task_summary,
                "inquirer_model": args.inquirer_model,
                "responder_model": args.responder_model,
                "token_usage": token_usage,
            }
            if metadata:
                generated_dialogue["metadata"] = metadata

            writer.write(generated_dialogue)

    print(f"Saved dialogues to {output_path}")
