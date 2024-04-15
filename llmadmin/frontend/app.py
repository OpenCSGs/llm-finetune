import logging
import random
import re
import uuid
from typing import Any, Dict, List
import gradio as gr
import ray
import requests

from llmadmin.common.backend import get_llmadmin_backend
from llmadmin.common.constants import (
    AVIARY_DESC,
    CSS,
    EXAMPLES_IF,
    EXAMPLES_QA,
    EXAMPLES_ST,
    HEADER,
    LOGO_ANYSCALE,
    LOGO_GITHUB,
    LOGO_RAY,
    LOGO_RAY_TYPEFACE,
    MODEL_DESCRIPTION_FORMAT,
    MODEL_DESCRIPTIONS_HEADER,
    MODELS,
    NUM_LLM_OPTIONS,
    PROJECT_NAME,
    SELECTION_DICT,
    SUB_HEADER,
)
from llmadmin.frontend.javascript_loader import JavaScriptLoader
from llmadmin.frontend.leaderboard import DummyLeaderboard, Leaderboard
from llmadmin.frontend.mongo_secrets import get_mongo_secret_url
from llmadmin.frontend.utils import (
    DEFAULT_STATS,
    LOGGER,
    THEME,
    blank,
    deactivate_buttons,
    gen_stats,
    log_flags,
    paused_logger,
    select_button,
    unset_buttons,
)

std_logger = logging.getLogger("ray.logger")

@ray.remote(num_cpus=0)
def completions(bakend, prompt, llm, index):
    try:
        out = bakend.completions(prompt=prompt, llm=llm)
    except Exception as e:
        if isinstance(e, requests.ReadTimeout) or (
            hasattr(e, "response")
            and ("timeout" in e.response or e.response.status_code in (408, 504))
        ):
            out = (
                "[LLM-ADMIN] The request timed out. This usually means the server "
                "is experiencing a higher than usual load. "
                "Please try again in a few minutes."
            )
        elif hasattr(e, "response"):
            out = (
                f"[LLM-ADMIN] Backend returned an error. "
                f"Status code: {e.response.status_code}"
                f"\nResponse: {e.response.text.split('raise ')[-1]}"
            ).replace("\n", " ")
        else:
            out = f"[LLM-ADMIN] An error occurred. Please try again.\nError: {e}"
        out = {"error": out}
    return out, index