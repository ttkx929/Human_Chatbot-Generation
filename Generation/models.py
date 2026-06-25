from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml

api_path = "api.yaml"
api_keys = yaml.safe_load(open(api_path, "r"))

setting_path = "setting.yaml"
setting = yaml.safe_load(open(setting_path, "r"))


def _valid_key(key) -> bool:
    if key is None:
        return False
    key = str(key).strip()
    return key not in ("", "###", "None", "null")


def _require_key(provider: str) -> str:
    key = api_keys.get(provider)
    if not _valid_key(key):
        raise ValueError(
            f"Missing API key for '{provider}' in api.yaml. "
            f"Set a valid key before using models that depend on it."
        )
    return key


def _openai_chat(name: str):
    cfg = setting[name]
    return ChatOpenAI(
        api_key=_require_key("openai"),
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_retries=cfg["max_retries"],
    )


def _hf_openai_chat(name: str):
    cfg = setting[name]
    return ChatOpenAI(
        base_url="https://router.huggingface.co/novita/v3/openai",
        api_key=_require_key("huggingface"),
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_retries=cfg["max_retries"],
    )


def _endpoint_openai_chat(name: str):
    cfg = setting[name]
    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=_require_key("huggingface"),
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_retries=cfg["max_retries"],
    )


MODEL_BUILDERS = {
    "GPT4o": lambda: _openai_chat("gpt4o"),
    "GPT4oMini": lambda: _openai_chat("gpt4o-mini"),
    "DeepSeek": lambda: ChatOpenAI(
        base_url=setting["deepseek"].get("base_url", "https://api.deepseek.com"),
        api_key=_require_key("deepseek"),
        model=setting["deepseek"]["model"],
        temperature=setting["deepseek"]["temperature"],
        max_tokens=setting["deepseek"]["max_tokens"],
        max_retries=setting["deepseek"]["max_retries"],
    ),
    "Claude3_7": lambda: ChatAnthropic(
        api_key=_require_key("anthropic"),
        model=setting["claude3_7"]["model"],
        temperature=setting["claude3_7"]["temperature"],
        max_tokens=setting["claude3_7"]["max_tokens"],
        max_retries=setting["claude3_7"]["max_retries"],
    ),
    "GeminiFlash": lambda: ChatGoogleGenerativeAI(
        google_api_key=_require_key("google"),
        model=setting["GeminiFlash"]["model"],
        temperature=setting["GeminiFlash"]["temperature"],
        max_tokens=setting["GeminiFlash"]["max_tokens"],
        max_retries=setting["GeminiFlash"]["max_retries"],
    ),
    "llama_1B": lambda: _hf_openai_chat("llama3_2_1B"),
    "llama_3B": lambda: _hf_openai_chat("llama3_2_3B"),
    "llama_8B": lambda: _hf_openai_chat("llama3_1_8B"),
    "llama_70B": lambda: _hf_openai_chat("llama3_3_70B"),
    "mistral_7B": lambda: _hf_openai_chat("mistral_7B"),
    "qwen_14B": lambda: _hf_openai_chat("qwen_14B"),
    "gemma_27b": lambda: _hf_openai_chat("gemma_27b"),
    "llama_3b_v1": lambda: _endpoint_openai_chat("llama_3b_v1"),
    "llama_3b_v2": lambda: _endpoint_openai_chat("llama_3b_v2"),
    "llama_8b_v1": lambda: _endpoint_openai_chat("llama_8b_v1"),
    "llama_8b_v2": lambda: _endpoint_openai_chat("llama_8b_v2"),
    "mistral_v1": lambda: _endpoint_openai_chat("mistral_v1"),
    "mistral_v2": lambda: _endpoint_openai_chat("mistral_v2"),
}

_MODEL_CACHE = {}


def get_model(name: str):
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_BUILDERS)}")
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = MODEL_BUILDERS[name]()
    return _MODEL_CACHE[name]


def __getattr__(name: str):
    if name in MODEL_BUILDERS:
        return get_model(name)
    raise AttributeError(f"module 'models' has no attribute '{name}'")


__all__ = ["get_model", *sorted(MODEL_BUILDERS)]
