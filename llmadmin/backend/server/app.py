import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional, Union

import async_timeout
import ray
import ray.util
from fastapi import FastAPI, Body
# from ray import serve
from ray.exceptions import RayActorError


from llmadmin.backend.llm.predictor import LLMPredictor
from llmadmin.backend.logger import get_logger
# from llmadmin.backend.server._batch import QueuePriority, _PriorityBatchQueue, batch
from llmadmin.backend.server.exceptions import PromptTooLongError
from llmadmin.backend.server.models import (
    Args,
    DeepSpeed,
    Prompt,
    Scaling_Config_Simple,
)
from llmadmin.backend.server.utils import parse_args,render_gradio_params
from llmadmin.common.constants import GATEWAY_TIMEOUT_S
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware

#logger = get_logger(__name__)
logger = get_logger("ray.logger")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel

class ModelConfig(BaseModel):
    model_id: str
    model_task: str
    is_oob: bool
    #initialization: InitializationConfig
    scaling_config: Scaling_Config_Simple

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
class LLMDeployment(LLMPredictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.info('LLM Deployment initialize')
        self.args = None
        super().__init__()

    def _should_reinit_worker_group(self, new_args: Args) -> bool:
        old_args = self.args

        if not old_args:
            return True

        old_scaling_config = self.args.air_scaling_config
        new_scaling_config = new_args.scaling_config.as_air_scaling_config()

        if not self.base_worker_group:
            return True

        if old_scaling_config != new_scaling_config:
            return True

        if not old_args:
            return True

        if old_args.model_config.initialization != new_args.model_config.initialization:
            return True

        if (
            old_args.model_config.generation.max_batch_size
            != new_args.model_config.generation.max_batch_size
            and isinstance(new_args.model_config.initialization.initializer, DeepSpeed)
        ):
            return True

        # TODO: Allow this
        if (
            old_args.model_config.generation.prompt_format
            != new_args.model_config.generation.prompt_format
        ):
            return True

        return False

    async def reconfigure(
        self,
        config: Union[Dict[str, Any], Args],
        force: bool = False,
    ) -> None:
        logger.info("LLM Deployment Reconfiguring...")
        if not isinstance(config, Args):
            new_args: Args = Args.parse_obj(config)
        else:
            new_args: Args = config

        should_reinit_worker_group = force or self._should_reinit_worker_group(new_args)

        self.args = new_args
        if should_reinit_worker_group:
            await self.rollover(
                self.args.air_scaling_config,
                pg_timeout_s=self.args.scaling_config.pg_timeout_s,
            )
        self.update_batch_params(self.get_max_batch_size() ,self.get_batch_wait_timeout_s())
        logger.info("LLM Deployment Reconfigured.")

    @property
    def max_batch_size(self):
        return (self.args.model_config.generation.max_batch_size if self.args.model_config.generation else 1)
        # return 1

    @property
    def batch_wait_timeout_s(self):
        return (self.args.model_config.generation.batch_wait_timeout_s if self.args.model_config.generation else 10)

    def get_max_batch_size(self):
        return self.max_batch_size

    def get_batch_wait_timeout_s(self):
        return self.batch_wait_timeout_s

    async def validate_prompt(self, prompt: Prompt) -> None:
        if len(prompt.prompt.split()) > self.args.model_config.max_input_words:
            raise PromptTooLongError(
                f"Prompt exceeds max input words of "
                f"{self.args.model_config.max_input_words}. "
                "Please make the prompt shorter."
            )

    @app.get("/metadata", include_in_schema=False)
    async def metadata(self) -> dict:
        return self.args.dict(
            exclude={
                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
            }
        )

    @app.post("/", include_in_schema=False)
    async def generate_text(self, prompt: Prompt):
        await self.validate_prompt(prompt)
        time.time()
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            text = await self.generate_text_batch(
                prompt,
                #[prompt],
                # priority=QueuePriority.GENERATE_TEXT,
                # start_timestamp=start_timestamp,
            )
            # return text[0]
            return text
    
    # no need anymore, will be delete soon
    async def generate(self, prompt: Prompt):
        time.time()
        logger.info(prompt)
        logger.info(self.get_max_batch_size())
        logger.info(self.get_batch_wait_timeout_s())
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            text = await self.generate_text_batch(
                prompt,
                #[prompt],
                # priority=QueuePriority.GENERATE_TEXT,
                # start_timestamp=start_timestamp,
            )
        return text
        #return text[0]

    @app.post("/batch", include_in_schema=False)
    async def batch_generate_text(self, prompts: List[Prompt]):
        for prompt in prompts:
            await self.validate_prompt(prompt)
        time.time()
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            texts = await asyncio.gather(
                *[
                    self.generate_text_batch(
                        prompt,
                        # priority=QueuePriority.BATCH_GENERATE_TEXT,
                        # start_timestamp=start_timestamp,
                    )
                    for prompt in prompts
                ]
            )
            return texts
        
    def update_batch_params(self, new_max_batch_size:int, new_batch_wait_timeout_s:float):
        self.generate_text_batch.set_max_batch_size(new_max_batch_size)
        self.generate_text_batch.set_batch_wait_timeout_s(new_batch_wait_timeout_s)
        logger.info(f"new_max_batch_size is {new_max_batch_size}")
        logger.info(f"new_batch_wait_timeout_s is {new_batch_wait_timeout_s}")

    # @serve.batch(
    #      max_batch_size=18,
    #      batch_wait_timeout_s=1,
    #  )
    async def generate_text_batch(
        self,
        prompts: List[Prompt],
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
    ):
        """Generate text from the given prompts in batch.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to GATEWAY_TIMEOUT_S-10. Ignored if start_timestamp is None.
        """
        if not prompts or prompts[0] is None:
            return prompts

        if isinstance(start_timestamp, list) and start_timestamp[0]:
            start_timestamp = min(start_timestamp)
        elif isinstance(start_timestamp, list):
            start_timestamp = start_timestamp[0]
        if isinstance(timeout_s, list) and timeout_s[0]:
            timeout_s = min(timeout_s)
        elif isinstance(timeout_s, list):
            timeout_s = timeout_s[0]

        logger.info(
            f"Received {len(prompts)} prompts {prompts}. start_timestamp {start_timestamp} timeout_s {timeout_s}"
        )

        data_ref = ray.put(prompts)

        while not self.base_worker_group:
            logger.info("Waiting for worker group to be initialized...")
            await asyncio.sleep(1)

        try:
            prediction = await self._predict_async(
                data_ref, timeout_s=timeout_s, start_timestamp=start_timestamp
            )
        except RayActorError as e:
            raise RuntimeError(
                f"Prediction failed due to RayActorError. "
                "This usually means that one or all prediction workers are dead. "
                "Try again in a few minutes. "
                f"Traceback:\n{traceback.print_exc()}"
            ) from e

        logger.info(f"Predictions {prediction}")
        if not isinstance(prediction, list):
            return [prediction]
        return prediction[: len(prompts)]

    # Called by Serve to check the replica's health.
    async def check_health(self):
        if self._new_worker_group_lock.locked():
            logger.info("Rollover in progress, skipping health check")
            return
        if self.pg and self.base_worker_group:
            dead_actors = []
            for actor in self.base_worker_group:
                actor_state = ray.state.actors(actor._ray_actor_id.hex())
                if actor_state["State"] == "DEAD":
                    dead_actors.append(actor)
            if dead_actors:
                raise RuntimeError(
                    f"At least one prediction worker is dead. Dead workers: {dead_actors}. "
                    "Reinitializing worker group."
                )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.args.model_config.model_id}"


def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")


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
# class ApiServer:
#     def __init__(self) -> None:
#         self.deployments = {}
#         self.model_configs = {}
#         self.compare_models = []
#         self.compare_deployments = {}
#         self.compare_model_configs = {}
#         self.newload_model = []
#         self.support_models = parse_args("./models")

#     def list_deployment_from_ray(self, experimetal: bool) -> List[Any]:
#         serve_details = ServeInstanceDetails(
#         **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
#         deployments = []
#         if experimetal:
#             for key, value in serve_details.applications.items():
#                 if "apiserver" in key or "cmp_" in key:
#                     continue
#                 apps = value.dict()
#                 #filtered_deployments= apps.get("deployments").copy()
#                 filtered_deployments = {}
#                 deploy_time = apps.get("last_deployed_time_s")
#                 name = apps.get("name")
#                 model_id = name.replace("--", "/").replace("_", ".")
#                 for k,v in apps.get("deployments").items():
#                     if "ExperimentalDeployment"  in k:
#                         continue
#                     v["last_deployed_time_s"] = deploy_time
#                     v["id"] = model_id
#                     filtered_deployments.update(v.copy()) 
#                 deployments.append(filtered_deployments)         
#         else:
#             for key, value in serve_details.applications.items():
#                 if "cmp_models" not in key:
#                     continue
#                 apps = value.dict()       
#                 deploy_time = apps.get("last_deployed_time_s")
#                 filtered_deployments = {}
#                 cmp_models = []
#                 for k,v in apps.get("deployments").items():
#                     if "RouterDeployment"  in k:
#                         continue  
#                     model_id = v.get("deployment_config").get("user_config").get("model_config").get("model_id")
#                     v["last_deployed_time_s"] = deploy_time
#                     v["id"] = model_id
#                     cmp_models.append(v.copy())
     
#                 prefix = apps.get("name").split('_', 2)
#                 filtered_deployments["url"] = CONFIG.URL + prefix[0] + "_" + prefix[2]
#                 filtered_deployments["id"] = prefix[2]
#                 filtered_deployments["models"] = cmp_models
#                 deployments.append(filtered_deployments) 
#         return deployments
    
#     def load_model(self,models: Union[List[str], List[ModelConfig], List[LLMApp]],comparation: bool):
#         self.newload_model = []
#         self.compare_deployments = {}
#         self.compare_model_configs = {}
#         mds = parse_args(models)
#         if not mds:
#             raise RuntimeError("No enabled models were found.")
#         for model in mds:
#             if model.model_config.model_id in self.model_configs.keys():
#                 continue
#             print("Initializing LLM app", model.json(indent=2))
#             user_config = model.dict()
#             deployment_config = model.deployment_config.dict()
#             deployment_config = deployment_config.copy()
#             max_concurrent_queries = deployment_config.pop(
#                 "max_concurrent_queries", None
#             ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
#             name = model.model_config.model_id.replace("/", "--").replace(".", "_")
#             deployment = LLMDeployment.options(
#                 name=name,
#                 max_concurrent_queries=max_concurrent_queries,
#                 user_config=user_config,
#                 **deployment_config,
#             ).bind()
#             if not comparation:
#                 self.model_configs[model.model_config.model_id] = model
#                 self.deployments[model.model_config.model_id]= deployment
#             else:         
#                 self.compare_model_configs[model.model_config.model_id] = model
#                 self.compare_deployments[model.model_config.model_id]= deployment
#             self.newload_model.append(model.model_config.model_id)

               
#         return 

#     @app.post("/load_model")
#     async def run_model(self,models: Union[ModelConfig, str] = Body(..., embed=True)) -> Dict[str, Any]:
#         if isinstance(models, ModelConfig) and not models.is_oob:
#             return self.load_model_args(models)
#         else:
#             mods = models
#             if isinstance(models, ModelConfig):
#                 mods = models.model_id
#             err = self.load_model(mods,False)
#             if self.newload_model == []:
#                 return {"response": "No models to load, model is already exsit. "}
#             for model in self.newload_model:
#                 serve_conf = {
#                     "name": model.replace("/", "--").replace(".", "_"),
#                 }

#                 app = ExperimentalDeployment.bind(self.deployments.get(model), self.model_configs.get(model))
#                 ray._private.usage.usage_lib.record_library_usage("llmadmin")
#                 serve.run(app, host="0.0.0.0", name = serve_conf["name"], route_prefix = "/" + serve_conf["name"], _blocking = False)
#         return {"url": CONFIG.URL + serve_conf["name"], "models": self.model_configs}
    
   
#     def load_model_args(self,args: ModelConfig) -> Dict[str, Any]:
#         if args.model_id in self.model_configs.keys():
#             model = self.model_configs.get(args.model_id)
#         else:
#             model = CONFIG.EXPERIMENTAL_LLMTEMPLATE
#         if args.scaling_config:
#             for key,value in args.scaling_config.__dict__.items():
#                 setattr(model.scaling_config,key,value)
#         #if args.initialization.initializer:
#         #    for key,value in args.initialization.initializer.__dict__.items():
#         #        setattr(model.model_config.initialization.initializer,key,value)  
#         #if args.initialization.pipeline:
#         #    model.model_config.initialization.pipeline =  args.initialization.pipeline   
#         model.model_config.model_id = args.model_id
#         user_config = model.dict()
#         if args.is_oob :
#             deployment_config = model.deployment_config.dict()
#         else:
#             deployment_config = model.deployment_config
#         deployment_config = deployment_config.copy()
#         max_concurrent_queries = deployment_config.pop(
#             "max_concurrent_queries", None
#         ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
            
#         deployment = LLMDeployment.options(
#                 name=args.model_id.replace("/", "--").replace(".", "_"),
#                 max_concurrent_queries=max_concurrent_queries,
#                 user_config=user_config,
#                 **deployment_config,
#             ).bind()
#         serve_conf = {
#             "name": args.model_id.replace("/", "--").replace(".", "_"),
#         }

#         app = ExperimentalDeployment.bind(deployment, model)
#         ray._private.usage.usage_lib.record_library_usage("llmadmin")
#         serve.run(app, host="0.0.0.0", name = serve_conf["name"], route_prefix = "/" + serve_conf["name"], _blocking = False)
#         self.model_configs[args.model_id] = model
#         self.deployments[args.model_id]= deployment
#         return {"url": CONFIG.URL + serve_conf["name"], "models": self.model_configs}
    
#     @app.post("/delete_model")
#     async def delete_model(self,models: List[str] = Body(..., description="app_name", embed=True)) -> Dict[str, Any]:
#         for mod in models:
#             app = mod.replace("/", "--").replace(".", "_")
#             serve.delete(app, _blocking = True)
#             if self.model_configs.get(app):
#                 self.model_configs.pop(app)
#                 self.deployments.pop(app)
#         return {"delete_models":models,"status": "Successful"}
    
#     @app.get("/list_models")
#     async def list_models(self)-> List[Any]:
#         deployments = self.list_deployment_from_ray(True) 
            
#         return deployments
    
#     @app.get("/list_apps")
#     async def list_apps(self)-> Dict[str,Any]:
#         serve_details = ServeInstanceDetails(
#         **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
            
#         return serve_details.applications
    
#     @app.get("/oob_models")
#     async def list_oob_models(self)-> Dict[str, Any]:
#         text ,sum, image2text, trans, qa = [],[],[],[],[]
#         for model in self.support_models:
#             if model.model_config.model_task == "text-generation":
#                 text.append(model.model_config.model_id)
#             if model.model_config.model_task == "translation":
#                 trans.append(model.model_config.model_id)
#             if model.model_config.model_task == "summarization":
#                 sum.append(model.model_config.model_id)
#             if model.model_config.model_task == "question-answering":
#                 qa.append(model.model_config.model_id)
#             if model.model_config.model_task == "image-to-text":
#                 image2text.append(model.model_config.model_id) 
#         return {
#             "text-generation":text,
#             "translation":trans,
#             "summarization":sum,
#             "question-answering":qa,
#             "image-to-text":image2text,
#         }
    
#     @app.get("/metadata/{model}")
#     async def metadata(self, model) -> Dict[str, Dict[str, Any]]:
#         model = _replace_prefix(model)
#         metadata = self.model_configs[model].dict(
#             exclude={
#                 "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
#             }
#         )
#         logger.info(metadata)
#         print(metadata)
#         return {"metadata": metadata}
    
#     @app.get("/model/{model}")
#     async def get_model(self, model) -> Dict[str, Any]:
#         return {"model_config": self.model_configs.get(model)}
    
#     @app.post("/update_model")
#     async def update_model(self,model: ModelConfig = Body(..., embed=True)) -> Dict[str, Any]:
#         models = self.list_deployment_from_ray(True)
#         serve_conf = {
#             "name": model.model_id.replace("/", "--").replace(".", "_"),
#         }
#         for mod in models:
#             if model.model_id != mod.get("id"):
#                 continue
#             md = mod.get("deployment_config").get("user_config")
#             md = LLMApp(scaling_config=md.get("scaling_config"),model_config=md.get("model_config"),deployment_config= md.get("deployment_config"))
#             if model.scaling_config:
#                 for key,value in model.scaling_config.__dict__.items():
#                     setattr(md.scaling_config,key,value)
                
#                 user_config = md.dict()
#                 deployment_config = md.deployment_config.dict()
#                 deployment_config = deployment_config.copy()
#                 max_concurrent_queries = deployment_config.pop(
#                     "max_concurrent_queries", None
#                 ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
                
#                 deployment = LLMDeployment.options(
#                     name=serve_conf["name"],
#                     max_concurrent_queries=max_concurrent_queries,
#                     user_config=user_config,
#                     **deployment_config,
#                 ).bind()
#                 app = ExperimentalDeployment.bind(md, deployment)
#                 ray._private.usage.usage_lib.record_library_usage("llmadmin")
                
#                 serve.run(app, host="0.0.0.0", name = serve_conf["name"], route_prefix = "/" + serve_conf["name"], _blocking = False)
#         return {"url": CONFIG.URL + serve_conf["name"], "model": md }

#     def  load_model_for_comparation(self, models: List[Union[ModelConfig, str]]):
#         mods = []
#         self.compare_deployments = {}
#         self.compare_model_configs = {}

#         for model in models:
#             logger.info(model)
#             parsed_models =[]
#             template = []
#             if isinstance(model, str):
#                 parsed_models = parse_args(model)
#             else:
#                 if model.is_oob:
#                     parsed_models = parse_args(model.model_id)
#                 else:
#                     template = CONFIG.COMPARATION_LLMTEMPLATE
#                     parsed_model= copy.deepcopy(template)
#                     if model.scaling_config:
#                         for key,value in model.scaling_config.__dict__.items():
#                             setattr(parsed_model.scaling_config,key,value)
#                     parsed_model.model_config.model_id = model.model_id
#                     parsed_models.append(parsed_model)
#             for md in parsed_models:
#                 user_config = md.dict()
#                 if model.is_oob:
#                     deployment_config = md.deployment_config.dict()
#                 else:
#                     deployment_config = md.deployment_config
#                 deployment_config = deployment_config.copy()
#                 max_concurrent_queries = deployment_config.pop(
#                     "max_concurrent_queries", None
#                 ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
#                 name = md.model_config.model_id.replace("/", "--").replace(".", "_")
#                 deployment = LLMDeployment.options(
#                     name=name,
#                     max_concurrent_queries=max_concurrent_queries,
#                     user_config=user_config,
#                     **deployment_config,
#                 ).bind()
            
#                 self.compare_model_configs[md.model_config.model_id] = md
#                 self.compare_deployments[md.model_config.model_id]= deployment
#         return 

#     def run_frontend(self,prefix,compare_prefix):
#         logger.info("startting LLMAdminFrontend") 
#         from llmadmin.frontend.app import LLMAdminFrontend
#         ray._private.usage.usage_lib.record_library_usage("llmadmin")
#         run_duration = 10 * 60
#         start_time = time.time()
#         while True:
#             serve_details = ServeInstanceDetails(
#             **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
#             app = {}
#             for key, value in serve_details.applications.items():
#                 if compare_prefix not in key:
#                     continue
#                 app = value.dict()    
#             ##logger.info(app)
#             if app.get("status") == "RUNNING":
#                 break
#             current_time = time.time()
#             elapsed_time = current_time - start_time
#             if elapsed_time >= run_duration:
#                 break
#             time.sleep(5)
#         logger.info(app)     
     
#         comparationApp = LLMAdminFrontend.options(ray_actor_options={"num_cpus": 1}, name="LLMAdminFrontend").bind(CONFIG.URL + compare_prefix)     
#         serve.run(comparationApp, host="0.0.0.0", name = prefix, route_prefix = "/" + prefix, _blocking = False)

#     @app.post("/launch_comparation")
#     async def launch_comparation(self, models: List[ModelConfig], user: str = Body(..., embed=True)) -> Dict[str, Any]:
#         self.load_model_for_comparation(models)
#         app = RouterDeployment.bind(self.compare_deployments, self.compare_model_configs)
#         logger.info(self.compare_model_configs)
#         ray._private.usage.usage_lib.record_library_usage("llmadmin")
#         prefix = "cmp_models"
#         prefix_cmp = "cmp"
#         uuid_s = str(uuid.uuid4())

#         if user:
#             prefix = prefix + "_" + user + "_" + uuid_s[:6]
#             prefix_cmp = prefix_cmp + "_" + user + "_" + uuid_s[:6]
#         serve.run(app, host="0.0.0.0", name = prefix, route_prefix="/" + prefix, _blocking = False)
     
#         thread = threading.Thread(target=self.run_frontend,args=(prefix_cmp,prefix))
#         thread.daemon = True 
#         thread.start()
#         #await self.run_frontend(prefix_cmp, prefix)
#         return {"url": CONFIG.URL + prefix_cmp , "models" : self.compare_model_configs, "ids" :[prefix, prefix_cmp]}
    
#     @app.post("/update_comparation")
#     async def update_comparation(self, models: List[ModelConfig], name: str = Body(..., embed=True)) -> Dict[str, Any]:
#         self.load_model_for_comparation(models)
#         app = RouterDeployment.bind(self.compare_deployments, self.compare_model_configs)
#         logger.info(self.compare_model_configs)
#         ray._private.usage.usage_lib.record_library_usage("llmadmin")
#         prefix = "cmp_models"
#         prefix_cmp = "cmp"
#         if name:
#             prefix = prefix + "_" + name
#             prefix_cmp = prefix_cmp + "_" + name
#         serve.run(app, host="0.0.0.0", name = prefix, route_prefix="/" + prefix, _blocking = False)
        
#         thread = threading.Thread(target=self.run_frontend,args=(prefix_cmp,prefix))
#         thread.daemon = True 
#         thread.start()
#         # await self.run_frontend(prefix_cmp, prefix)
#         return {"url": CONFIG.URL + prefix_cmp , "models" : self.compare_model_configs, "ids" :[prefix, prefix_cmp]}
    
    
#     @app.get("/models_comparation")
#     async def models_comparation(self)-> Dict[str, Any]:
#         text = []
        
#         for model in self.support_models:
#             if model.model_config.model_task == "text-generation":
#                 text.append(model.model_config.model_id)
#         return {
#             "text-generation":text,
#         }
    
#     @app.get("/list_comparation")
#     async def list_comparation(self)-> List[Any]:     
#         deployments = self.list_deployment_from_ray(False)       
                      
#         return deployments
    
#     @app.post("/delete_comparation")
#     async def delete_app(self, names: List[str] = Body(..., description="model id or all", embed=True)) -> Dict[str, Any]:
#         for name in names:
#             if "all" in name or "All" in names:
#                 serve_details = ServeInstanceDetails(**ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
#                 for key, value in serve_details.applications.items():
#                     if "cmp_" in key:
#                         serve.delete( key, _blocking = False)
#             else:
#                 serve.delete("cmp_models_" + name, _blocking = False)
#                 serve.delete("cmp_" + name, _blocking = False)      
#         return {"comparation":"Delete" + name + "Successful"}