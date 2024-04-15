from typing import TYPE_CHECKING, List, Optional, Union

import torch
from transformers import Pipeline as TransformersPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from llmadmin.backend.logger import get_logger
from llmadmin.backend.server.models import Prompt, Response

from ._base import BasePipeline
from .utils import construct_prompts, construct_prompts_experimental
from llmadmin.backend.server.utils import render_gradio_params
from .default_pipeline import DefaultPipeline

try:
    import transformers
    from transformers import pipelines
except ImportError as ie:
    raise ImportError(
        "transformers not installed. Please try `pip install transformers`"
    ) from ie

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class DefaultTransformersPipeline(BasePipeline):
    """Text generation pipeline using Transformers Pipeline.

    May not support all features.

    Args:
        model (PreTrainedModel): Hugging Face model.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        prompt_format (Optional[str], optional): Prompt format. Defaults to None.
        device (Optional[Union[str, int, torch.device]], optional): Device to place model on. Defaults to model's
            device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        task: str = None,
    ) -> None:
        if not hasattr(model, "generate"):
            raise ValueError("Model must have a generate method.")
        super().__init__(model, tokenizer, prompt_format, device)

        self.pipeline = None
        self.preprocess = None
        self.postprocess = None

    def _get_transformers_pipeline(self, **kwargs) -> TransformersPipeline:
        default_kwargs = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=None,
        )
        transformers_pipe = pipeline(**{**default_kwargs, **kwargs})
        transformers_pipe.device = self.device
        return transformers_pipe

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        if not self.pipeline:
            self.pipeline = self._get_transformers_pipeline()

        logger.info(f"input from pipeline: ****** {inputs}")
        inputs = construct_prompts_experimental(
            inputs, prompt_format=self.prompt_format)
        
        logger.info(f"input from pipeline: ****** {inputs}")

        if self.preprocess:
            data = self.preprocess(inputs)

        logger.info(data)
        kwargs.pop("stopping_sequences", None)
        kwargs.pop("timeout_s", None)
        kwargs.pop("start_timestamp", None)
        # special cases that needs to be handled differently
        if isinstance(
            self.pipeline,
            (
                pipelines.text_classification.TextClassificationPipeline,
                pipelines.text2text_generation.Text2TextGenerationPipeline,
                pipelines.text2text_generation.TranslationPipeline,
            ),
        ):
            data = self.pipeline(*data, **kwargs)
        else:
            data = self.pipeline(**data, **kwargs)

        logger.info(f"output from pipeline: ****** {data}")
        if self.postprocess:
            output = self.postprocess(data)

        return output

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        stopping_sequences: List[Union[int, str]] = None,
        **kwargs,
    ) -> "DefaultTransformersPipeline":
        model_from_pretrained_kwargs = initializer.get_model_from_pretrained_kwargs()
        default_kwargs = dict(
            model=model_id,
            **kwargs,
            **model_from_pretrained_kwargs
        )

        transformers_pipe = pipeline(
            **default_kwargs,
            model_kwargs=initializer.get_model_init_kwargs(),
        )
        # transformers_pipe.model = initializer.postprocess_model(transformers_pipe.model)
        pipe = cls(
            model=transformers_pipe.model,
            tokenizer=transformers_pipe.tokenizer,
            prompt_format=prompt_format,
            device=device,
            # stopping_sequences=stopping_sequences,
            **kwargs,
        )
        pipe.pipeline = transformers_pipe
        transformers_pipe.device = pipe.device

        if "task" in kwargs:
            pipeline_info = render_gradio_params(kwargs["task"])
            pipe.preprocess = pipeline_info["preprocess"]
            pipe.postprocess = pipeline_info["postprocess"]

        return pipe

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
