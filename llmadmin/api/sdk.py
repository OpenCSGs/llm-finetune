# from typing import Any, Dict, List
from llmadmin.api.env import assert_has_backend
from ray.serve._private.constants import DEFAULT_HTTP_PORT
from llmadmin.backend.server import run


# __all__ = ["models", "metadata", "run"]

def start_apiserver(port: int = DEFAULT_HTTP_PORT, resource_config: str = None, scale_config: str = None) -> None:
    """Run Api server on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    run.start_apiserver(port=port, resource_config=resource_config, scale_config=scale_config)

def run_ft(ft: str) -> None:
    """Run LLMAdmin on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    run.run_ft(ft)
    
def run_ray_ft(ft: str) -> None:
    """Run LLMAdmin on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    run.run_ray_ft(ft)

# def models() -> List[str]:
#     """List available models"""
#     from llmadmin.common.backend import get_llmadmin_backend

#     backend = get_llmadmin_backend()
#     return backend.models()

# def _is_llmadmin_model(model: str) -> bool:
#     """
#     Determine if this is an llmadmin model. LLMAdmin
#     models do not have a '://' in them.
#     """
#     return "://" not in model

# def _supports_batching(model: str) -> bool:
#     provider, _ = model.split("://", 1)
#     return provider != "openai"

# def _convert_to_llmadmin_format(model: str, llm_result):
#     generation = llm_result.generations
#     result_list = [{"generated_text": x.text} for x in generation[0]]
#     return result_list

# def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
#     """Get model metadata"""
#     from llmadmin.common.backend import get_llmadmin_backend

#     backend = get_llmadmin_backend()
#     return backend.metadata(model_id)

# def run(*model: str) -> None:
#     """Run LLMAdmin on the local ray cluster

#     NOTE: This only works if you are running this command
#     on the Ray or Anyscale cluster directly. It does not
#     work from a general machine which only has the url and token
#     for a model.
#     """
#     assert_has_backend()
#     from llmadmin.backend.server.run import run
#     run(*model)

# def run_experimental(*model: str) -> None:
#     """Run LLMAdmin on the local ray cluster

#     NOTE: This only works if you are running this command
#     on the Ray or Anyscale cluster directly. It does not
#     work from a general machine which only has the url and token
#     for a model.
#     """
#     assert_has_backend()
#     from llmadmin.backend.server.run import run_experimental

#     run_experimental(*model)

# def del_experimental(app_name: str) -> None:
#     """Delete ray serve on the local ray cluster

#     NOTE: This only works if you are running this command
#     on the Ray or Anyscale cluster directly. It does not
#     work from a general machine which only has the url and token
#     for a model.
#     """
#     assert_has_backend()
#     from llmadmin.backend.server.run import del_experimental

#     del_experimental(app_name)
    
# def run_application(flow: dict) -> None:
#     """Run LLMAdmin on the local ray cluster

#     NOTE: This only works if you are running this command
#     on the Ray or Anyscale cluster directly. It does not
#     work from a general machine which only has the url and token
#     for a model.
#     """
#     assert_has_backend()
#     from llmadmin.backend.server.run import run_application

#     run_application(flow)


# def run_comparation() -> None:
#     """Run LLMAdmin on the local ray cluster

#     NOTE: This only works if you are running this command
#     on the Ray or Anyscale cluster directly. It does not
#     work from a general machine which only has the url and token
#     for a model.
#     """
#     assert_has_backend()
#     from llmadmin.backend.server.run import run_comparation

#     run_comparation()

    
