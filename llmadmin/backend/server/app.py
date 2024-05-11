# import asyncio
# import time
# import traceback
from typing import Any, Dict, List, Optional, Union

# import async_timeout
# import ray
# import ray.util
from ray import serve
from fastapi import FastAPI, Body
# from ray import serve
# from ray.exceptions import RayActorError

# from llmadmin.backend.llm.predictor import LLMPredictor
from llmadmin.backend.logger import get_logger
# from llmadmin.backend.server._batch import QueuePriority, _PriorityBatchQueue, batch
# from llmadmin.backend.server.exceptions import PromptTooLongError
from llmadmin.backend.server.models import Scaling_Config_Simple
from llmadmin.backend.server.utils import parse_args
# from llmadmin.common.constants import GATEWAY_TIMEOUT_S
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# logger = get_logger(__name__)
# logger = get_logger("ray.logger")
logger = get_logger("ray.serve")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelConfig(BaseModel):
    model_id: str = None
    model_task: str = None
    is_oob: bool = False
    scaling_config: Scaling_Config_Simple = None

def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 5,
        "target_ongoing_requests": 3, # the average number of ongoing requests per replica that the Serve autoscaler tries to ensure
    },
    max_ongoing_requests=5,  # the maximum number of ongoing requests allowed for a replica
)
@serve.ingress(app)
class ApiServer:
    def __init__(self) -> None:
        self.deployments = {}
        self.model_configs = {}
        self.compare_models = []
        self.compare_deployments = {}
        self.compare_model_configs = {}
        self.newload_model = []
        self.support_models = parse_args("./models")
    
    @app.get("/oob_models")
    async def list_oob_models(self)-> Dict[str, Any]:
        text ,sum, image2text, trans, qa = [],[],[],[],[]
        for model in self.support_models:
            if model.model_config.model_task == "text-generation":
                text.append(model.model_config.model_id)
            if model.model_config.model_task == "translation":
                trans.append(model.model_config.model_id)
            if model.model_config.model_task == "summarization":
                sum.append(model.model_config.model_id)
            if model.model_config.model_task == "question-answering":
                qa.append(model.model_config.model_id)
            if model.model_config.model_task == "image-to-text":
                image2text.append(model.model_config.model_id) 
        return {
            "text-generation":text,
            "translation":trans,
            "summarization":sum,
            "question-answering":qa,
            "image-to-text":image2text,
        }
    
    @app.get("/metadata/{model}")
    async def metadata(self, model: str) -> Dict[str, Dict[str, Any]]:
        model = _replace_prefix(model)
        metadata = self.model_configs[model].dict(
            exclude={
                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
            }
        )
        logger.info(metadata)
        return {"metadata": metadata}
    
    @app.get("/model/{model}")
    async def get_model(self, model: str) -> Dict[str, Any]:
        return {"model_config": self.model_configs.get(model)}

    @app.post("/train")
    async def do_train(self, model: ModelConfig = Body(..., description="model id or all")) -> Dict[str, Any]:
        logger.info(f"train: {model}")
        return {"result": "start train success"}

# @serve.deployment(
#     autoscaling_config={
#         "min_replicas": 1,
#         "initial_replicas": 2,
#         "max_replicas": 8,
#     },
#     max_concurrent_queries=2,  # Maximum backlog for a single replica
#     health_check_period_s=10,
#     health_check_timeout_s=30,
# )
# class LLMDeployment(LLMPredictor):
#     def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
#         logger.info('LLM Deployment initialize')
#         self.args = None
#         super().__init__()

#     def _should_reinit_worker_group(self, new_args: Args) -> bool:
#         old_args = self.args

#         if not old_args:
#             return True

#         old_scaling_config = self.args.air_scaling_config
#         new_scaling_config = new_args.scaling_config.as_air_scaling_config()

#         if not self.base_worker_group:
#             return True

#         if old_scaling_config != new_scaling_config:
#             return True

#         if not old_args:
#             return True

#         if old_args.model_config.initialization != new_args.model_config.initialization:
#             return True

#         if (
#             old_args.model_config.generation.max_batch_size
#             != new_args.model_config.generation.max_batch_size
#             and isinstance(new_args.model_config.initialization.initializer, DeepSpeed)
#         ):
#             return True

#         # TODO: Allow this
#         if (
#             old_args.model_config.generation.prompt_format
#             != new_args.model_config.generation.prompt_format
#         ):
#             return True

#         return False

#     async def reconfigure(
#         self,
#         config: Union[Dict[str, Any], Args],
#         force: bool = False,
#     ) -> None:
#         logger.info("LLM Deployment Reconfiguring...")
#         if not isinstance(config, Args):
#             new_args: Args = Args.parse_obj(config)
#         else:
#             new_args: Args = config

#         should_reinit_worker_group = force or self._should_reinit_worker_group(new_args)

#         self.args = new_args
#         if should_reinit_worker_group:
#             await self.rollover(
#                 self.args.air_scaling_config,
#                 pg_timeout_s=self.args.scaling_config.pg_timeout_s,
#             )
#         self.update_batch_params(self.get_max_batch_size() ,self.get_batch_wait_timeout_s())
#         logger.info("LLM Deployment Reconfigured.")

#     @property
#     def max_batch_size(self):
#         return (self.args.model_config.generation.max_batch_size if self.args.model_config.generation else 1)
#         # return 1

#     @property
#     def batch_wait_timeout_s(self):
#         return (self.args.model_config.generation.batch_wait_timeout_s if self.args.model_config.generation else 10)

#     def get_max_batch_size(self):
#         return self.max_batch_size

#     def get_batch_wait_timeout_s(self):
#         return self.batch_wait_timeout_s

#     async def validate_prompt(self, prompt: Prompt) -> None:
#         if len(prompt.prompt.split()) > self.args.model_config.max_input_words:
#             raise PromptTooLongError(
#                 f"Prompt exceeds max input words of "
#                 f"{self.args.model_config.max_input_words}. "
#                 "Please make the prompt shorter."
#             )

#     @app.get("/metadata", include_in_schema=False)
#     async def metadata(self) -> dict:
#         return self.args.dict(
#             exclude={
#                 "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
#             }
#         )

#     @app.post("/", include_in_schema=False)
#     async def generate_text(self, prompt: Prompt):
#         await self.validate_prompt(prompt)
#         time.time()
#         with async_timeout.timeout(GATEWAY_TIMEOUT_S):
#             text = await self.generate_text_batch(
#                 prompt,
#                 #[prompt],
#                 # priority=QueuePriority.GENERATE_TEXT,
#                 # start_timestamp=start_timestamp,
#             )
#             # return text[0]
#             return text
    
#     # no need anymore, will be delete soon
#     async def generate(self, prompt: Prompt):
#         time.time()
#         logger.info(prompt)
#         logger.info(self.get_max_batch_size())
#         logger.info(self.get_batch_wait_timeout_s())
#         with async_timeout.timeout(GATEWAY_TIMEOUT_S):
#             text = await self.generate_text_batch(
#                 prompt,
#                 #[prompt],
#                 # priority=QueuePriority.GENERATE_TEXT,
#                 # start_timestamp=start_timestamp,
#             )
#         return text
#         #return text[0]

#     @app.post("/batch", include_in_schema=False)
#     async def batch_generate_text(self, prompts: List[Prompt]):
#         for prompt in prompts:
#             await self.validate_prompt(prompt)
#         time.time()
#         with async_timeout.timeout(GATEWAY_TIMEOUT_S):
#             texts = await asyncio.gather(
#                 *[
#                     self.generate_text_batch(
#                         prompt,
#                         # priority=QueuePriority.BATCH_GENERATE_TEXT,
#                         # start_timestamp=start_timestamp,
#                     )
#                     for prompt in prompts
#                 ]
#             )
#             return texts
        
#     def update_batch_params(self, new_max_batch_size:int, new_batch_wait_timeout_s:float):
#         self.generate_text_batch.set_max_batch_size(new_max_batch_size)
#         self.generate_text_batch.set_batch_wait_timeout_s(new_batch_wait_timeout_s)
#         logger.info(f"new_max_batch_size is {new_max_batch_size}")
#         logger.info(f"new_batch_wait_timeout_s is {new_batch_wait_timeout_s}")

#     # @serve.batch(
#     #      max_batch_size=18,
#     #      batch_wait_timeout_s=1,
#     #  )
#     async def generate_text_batch(
#         self,
#         prompts: List[Prompt],
#         *,
#         start_timestamp: Optional[Union[float, List[float]]] = None,
#         timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
#     ):
#         """Generate text from the given prompts in batch.

#         Args:
#             prompts (List[Prompt]): Batch of prompts to generate text from.
#             start_timestamp (Optional[float], optional): Timestamp of when the
#                 batch was created. Defaults to None. If set, will early stop
#                 the generation.
#             timeout_s (float, optional): Timeout for the generation. Defaults
#                 to GATEWAY_TIMEOUT_S-10. Ignored if start_timestamp is None.
#         """
#         if not prompts or prompts[0] is None:
#             return prompts

#         if isinstance(start_timestamp, list) and start_timestamp[0]:
#             start_timestamp = min(start_timestamp)
#         elif isinstance(start_timestamp, list):
#             start_timestamp = start_timestamp[0]
#         if isinstance(timeout_s, list) and timeout_s[0]:
#             timeout_s = min(timeout_s)
#         elif isinstance(timeout_s, list):
#             timeout_s = timeout_s[0]

#         logger.info(
#             f"Received {len(prompts)} prompts {prompts}. start_timestamp {start_timestamp} timeout_s {timeout_s}"
#         )

#         data_ref = ray.put(prompts)

#         while not self.base_worker_group:
#             logger.info("Waiting for worker group to be initialized...")
#             await asyncio.sleep(1)

#         try:
#             prediction = await self._predict_async(
#                 data_ref, timeout_s=timeout_s, start_timestamp=start_timestamp
#             )
#         except RayActorError as e:
#             raise RuntimeError(
#                 f"Prediction failed due to RayActorError. "
#                 "This usually means that one or all prediction workers are dead. "
#                 "Try again in a few minutes. "
#                 f"Traceback:\n{traceback.print_exc()}"
#             ) from e

#         logger.info(f"Predictions {prediction}")
#         if not isinstance(prediction, list):
#             return [prediction]
#         return prediction[: len(prompts)]

#     # Called by Serve to check the replica's health.
#     async def check_health(self):
#         if self._new_worker_group_lock.locked():
#             logger.info("Rollover in progress, skipping health check")
#             return
#         if self.pg and self.base_worker_group:
#             dead_actors = []
#             for actor in self.base_worker_group:
#                 actor_state = ray.state.actors(actor._ray_actor_id.hex())
#                 if actor_state["State"] == "DEAD":
#                     dead_actors.append(actor)
#             if dead_actors:
#                 raise RuntimeError(
#                     f"At least one prediction worker is dead. Dead workers: {dead_actors}. "
#                     "Reinitializing worker group."
#                 )

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}:{self.args.model_config.model_id}"

# @serve.deployment(
#     # TODO make this configurable in llmadmin run
#     autoscaling_config={
#         "min_replicas": 1,
#         "initial_replicas": 1,
#         "max_replicas": 2,
#     },
#     ray_actor_options={
#         "num_cpus": 0.1 
#     },
#     max_concurrent_queries=50,  # Maximum backlog for a single replica
# )
# @serve.ingress(app)
# class RouterDeployment:
#     def __init__(
#         self, models: Dict[str, ClassNode], model_configurations: Dict[str, Args]
#     ) -> None:
#         self._models = models
#         # TODO: Remove this once it is possible to reconfigure models on the fly
#         self._model_configurations = model_configurations

#     @app.post("/query/{model}")
#     async def query(self, model: str, prompt: Prompt) -> Dict[str, Dict[str, Any]]:
#         model = _replace_prefix(model)
#         results = await asyncio.gather(
#             *(await asyncio.gather(*[self._models[model].generate_text.remote(prompt)]))
#         )

#         results = results[0]
#         logger.info(results)
#         return {model: results}

#     @app.post("/query/batch/{model}")
#     async def batch_query(
#         self, model: str, prompts: List[Prompt]
#     ) -> Dict[str, List[Dict[str, Any]]]:
#         model = _replace_prefix(model)
#         results = await asyncio.gather(
#             *(
#                 await asyncio.gather(
#                     *[self._models[model].batch_generate_text.remote(prompts)]
#                 )
#             )
#         )

#         results = results[0]
#         return {model: results}

#     @app.get("/metadata/{model}")
#     async def metadata(self, model) -> Dict[str, Dict[str, Any]]:
#         model = _replace_prefix(model)
#         # This is what we want to do eventually, but it looks like reconfigure is blocking
#         # when called on replica init
#         # metadata = await asyncio.gather(
#         #     *(await asyncio.gather(*[self._models[model].metadata.remote()]))
#         # )
#         # metadata = metadata[0]
#         metadata = self._model_configurations[model].dict(
#             exclude={
#                 "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
#             }
#         )
#         logger.info(metadata)
#         return {"metadata": metadata}

#     @app.get("/models")
#     async def models(self) -> List[str]:
#         return list(self._models.keys())


# @serve.deployment(
#     # TODO make this configurable in llmadmin run
#     autoscaling_config={
#         "min_replicas": 1,
#         "initial_replicas": 1,
#         "max_replicas": 2,
#     },
#     ray_actor_options={
#         "num_cpus": 0.1 
#     },
#     max_concurrent_queries=50,  # Maximum backlog for a single replica
# )
# class ExperimentalDeployment(GradioIngress):
#     def __init__(
#         self, model: ClassNode, model_configuration: Args
#     ) -> None:
#         logger.info('Experiment Deployment Initialize')
#         self._model = model
#         # TODO: Remove this once it is possible to reconfigure models on the fly
#         self._model_configuration = model_configuration
#         hg_task = self._model_configuration.model_config.model_task
#         pipeline_info = render_gradio_params(hg_task)
        
#         self.pipeline_info = pipeline_info
#         super().__init__(self._chose_ui())

#     async def query(self, *args) -> Dict[str, Dict[str, Any]]:
#         logger.info('Experimental Deployment query', str(args))

#         if len(args) > 1:
#             prompts = args
#         else:
#             prompts = args[0]
#         logger.info(prompts)
#         results = await asyncio.gather(
#             *(await asyncio.gather(*[self._model.generate_text.remote(Prompt(prompt=prompts, use_prompt_format=False))]))
#         )
        
#         results = results[0]
#         logger.info(results)
#         return results
    
#     def _chose_ui(self) -> Callable:
#         logger.info(f'Experiment Deployment chose ui for {self._model_configuration.model_config.model_id}')

#         gr_params = self.pipeline_info
#         del gr_params["preprocess"]
#         del gr_params["postprocess"]

#         return lambda: gr.Interface(self.query, **gr_params, title = self._model_configuration.model_config.model_id)

