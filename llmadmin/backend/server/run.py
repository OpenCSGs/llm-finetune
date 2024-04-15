import sys
from typing import List, Union
import ray
from llmadmin.backend.server.models import FTApp
from llmadmin.backend.server.utils import parse_args, parse_args_ft
import uuid
import os
from llmadmin.backend.llm.ft import TransformersFT
from llmadmin.backend.logger import get_logger

# ray.init(address="auto")
logger = get_logger(__name__)

def run_ft(ft: Union[FTApp, str]):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/model.yaml") # run one model in the model directory
       run(FTApp)         # run a single LLMApp
    """

    ft = parse_args_ft(ft)
    if not ft:
        raise RuntimeError("No valiabled fine tune defination were found.")
    
    if isinstance(ft, FTApp):
        logger.info(f"Initialized a Finetune instance of FTApp {ft.json(indent=2)}")
    else:
        raise RuntimeError("Not a Finetune App were found.")
    
    ray._private.usage.usage_lib.record_library_usage("llmadmin")

    runner = TransformersFT(ft)
    runner.train()

if __name__ == "__main__":
    run_ft(*sys.argv[1:])


