import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import requests

from llmadmin.common.constants import TIMEOUT


class BackendError(RuntimeError):
    def __init__(self, *args: object, **kwargs) -> None:
        self.response = kwargs.pop("response", None)
        super().__init__(*args)

logger = logging.getLogger("ray.logger")

def get_llmadmin_backend(url: str = "http://127.0.0.1:8000/cmp_models_default"):
    """
    Establishes a connection to the LLMAdmin backed after establishing
    the information using environmental variables.
    If the AVIARY_MOCK environmental variable is set, then a mock backend is used.

    For direct connection to the llmadmin backend (e.g. running on the same cluster),
    no AVIARY_TOKEN is required. Otherwise, the AVIARY_URL and AVIARY_TOKEN environmental variables
    are required.

    Returns:
        backend: An instance of the Backend class.
    """
    mock_backend = os.getenv("AVIARY_MOCK", False)
    if mock_backend:
        backend = MockBackend()
        return backend
    print(os.getenv("AVIARY_URL"))
    llmadmin_url = url 
    assert llmadmin_url is not None, "AVIARY_URL must be set"
    backend_token = os.getenv("AVIARY_TOKEN")
    bearer = f"Bearer {backend_token}" if backend_token is not None else ""
    if not llmadmin_url.endswith("/"):
        llmadmin_url += "/"
    print("Connecting to LLMAdmin backend at: ", llmadmin_url)
    backend = LLMAdminBackend(llmadmin_url, bearer)
    return backend


class Backend(ABC):
    """Abstract interface for talking to LLMAdmin."""

    @abstractmethod
    def models(self) -> List[str]:
        pass

    @abstractmethod
    def metadata(self, llm: str) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def completions(self, prompt: str, llm: str) -> Dict[str, Union[str, float, int]]:
        pass

    @abstractmethod
    def batch_completions(
        self, prompts: List[str], llm: str
    ) -> List[Dict[str, Union[str, float, int]]]:
        pass


class LLMAdminBackend(Backend):
    """Interface for talking to LLMAdmin.
    Deliberately designed to be similar to OpenAI's
    Completions interface.

    https://platform.openai.com/docs/api-reference/completions?lang=python
    """

    def __init__(self, backend_url: str, bearer: str):
        assert "::param" not in backend_url, "backend_url not set correctly"
        assert "::param" not in bearer, "bearer not set correctly"

        self.backend_url = backend_url
        self.bearer = bearer
        self.header = {"Authorization": self.bearer}

    def models(self) -> List[str]:
        url = self.backend_url + "models"
        print("Connecting backend to get models at: ", url)
        response = requests.get(url, headers=self.header, timeout=TIMEOUT)
        try:
            result = response.json()
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
        return result

    def metadata(self, llm: str) -> Dict[str, Dict[str, Any]]:
        url = self.backend_url + "metadata/" + llm.replace("/", "--")
        response = requests.get(url, headers=self.header, timeout=TIMEOUT)
        try:
            result = response.json()
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
        return result

    def completions(self, prompt: str, llm: str) -> Dict[str, Union[str, float, int]]:
        url = self.backend_url + "query/" + llm.replace("/", "--")
        response = requests.post(
            url,
            headers=self.header,
            json={"prompt": prompt},
            timeout=TIMEOUT,
        )
        try:
            return response.json()[llm]
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e

    def batch_completions(
        self, prompts: List[str], llm: str
    ) -> List[Dict[str, Union[str, float, int]]]:
        url = self.backend_url + "query/batch/" + llm.replace("/", "--")
        response = requests.post(
            url,
            headers=self.header,
            json=[{"prompt": prompt} for prompt in prompts],
            timeout=TIMEOUT,
        )
        try:
            return response.json()[llm]
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e


class MockBackend(Backend):
    """Mock backend for testing"""

    def __init__(self):
        pass

    def models(self) -> List[str]:
        return ["A", "B", "C"]

    def metadata(self, llm: str) -> Dict[str, Dict[str, Any]]:
        return {
            "metadata": {
                "model_config": {
                    "model_id": llm,
                    "model_url": f"https://huggingface.co/org/{llm}",
                    "model_description": f"This is a model description for model {llm}",
                }
            }
        }

    def completions(self, prompt: str, llm: str) -> Dict[str, Union[str, float, int]]:
        return {
            "generated_text": prompt,
            "total_time": 99,
            "num_total_tokens": 42.3,
        }

    def batch_completions(
        self, prompts: List[str], llm: str
    ) -> List[Dict[str, Union[str, float, int]]]:
        return [
            {
                "generated_text": prompt,
                "total_time": 99,
                "num_total_tokens": 42.3,
            }
            for prompt in prompts
        ]
