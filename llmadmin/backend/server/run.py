# import sys
from typing import Dict, List, Union
import ray
from llmadmin.backend.server.app import ApiServer
from llmadmin.backend.server.config import SERVE_RUN_HOST
from llmadmin.backend.server.models import FTApp
from llmadmin.backend.server.utils import parse_args, parse_args_ft
# import uuid
# import os
from llmadmin.backend.llm.ft import TransformersFT
from llmadmin.backend.llm.ft import RayTrain
from llmadmin.backend.logger import get_logger
from ray.serve._private.constants import DEFAULT_HTTP_PORT
from llmadmin.backend.server.utils import get_serve_port
from ray import serve

# ray.init(address="auto")
logger = get_logger(__name__)

def run_ray_ft(ft: Union[FTApp, str]):
    """Run the LLM Train on the local Ray Cluster

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
    
    # ray._private.usage.usage_lib.record_library_usage("llmadmin")

    runner = RayTrain(ft)
    runner.train()

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

def start_apiserver(port: int = DEFAULT_HTTP_PORT, resource_config: str = None, scale_config: str = None):
    """Run the API Server on the local Ray Cluster

    Args:
        *host: The host ip to run.
        *port: The port to run.     

    """
    scale_dict = dict()
    try:
        scale_dict = toDict(scale_config)
    except:
        raise ValueError(f"Invalid value of scale config '{scale_config}'")
    resource_dict = None
    try:
        resource_dict = toDict(resource_config)
    except:
        raise ValueError(f"Invalid value of resource config '{resource_config}'")
    
    # ray._private.usage.usage_lib.record_library_usage("llmfinetune")
    # ray.init(address="auto")
    serve_start_port = get_serve_start_port(port)
    app = ApiServer.options(autoscaling_config=scale_dict, ray_actor_options=resource_dict).bind()
    serve.start(http_options={"host": SERVE_RUN_HOST, "port": serve_start_port})
    logger.info(f"Serve 'apiserver' is running at {SERVE_RUN_HOST}/{serve_start_port}")
    logger.info(f"Serve 'apiserver' run with resource: {resource_dict} , scale: {scale_dict}")
    serve.run(app, name="apiserver", route_prefix="/api")

# parse k1=v1,k2=v2 to dict
def toDict(kv: str) -> Dict:
    if kv:
        s = kv.replace(' ', ', ')
        return eval(f"dict({s})")
    else:
        return dict()

def get_serve_start_port(port: int):
    serve_start_port = port
    serve_runtime_port = get_serve_port()
    if serve_runtime_port > -1:
        logger.info(
            f"Serve is already running at {SERVE_RUN_HOST}:{serve_runtime_port}")
        serve_start_port = serve_runtime_port
    return serve_start_port

# if __name__ == "__main__":
#     run_ft(*sys.argv[1:])


