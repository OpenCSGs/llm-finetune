# import ast
# import json
from typing import Annotated, Optional

import typer
# from rich import print as rp
# from rich.console import Console
# from rich.progress import Progress, SpinnerColumn, TextColumn
# from rich.table import Table
from ray.serve._private.constants import DEFAULT_HTTP_PORT
from llmadmin.api import sdk

app = typer.Typer()

model_type = typer.Option(
    default=..., help="The model to use. You can specify multiple models."
)

ft_define = typer.Option(
    default=..., help="the fine tune yaml file"
)
    
app_name = typer.Option(
    default=..., help="The name of ray serve application."
)
host = typer.Option(
    default=..., help="The host ip address of ray api server."
)
port = typer.Option(
    default=...,help="The port of api server."
)
prompt_type = typer.Option(help="Prompt to query")
stats_type = typer.Option(help="Whether to print generated statistics")
prompt_file_type = typer.Option(
    default=..., help="File containing prompts. A simple text file"
)
separator_type = typer.Option(help="Separator used in prompt files")
results_type = typer.Option(help="Where to save the results")
file_type = typer.Option(default=..., help="The flow graph")
port_type = typer.Option(default=..., help="The port of service.")
apiserver_scale_type = typer.Option(default=..., help="A string of dict for scaling service. for example: --scale-config=min_replicas=1,max_replicas=5")
apiserver_resource_type = typer.Option(default=..., help="A string of dict for resource requirement. for example: --resource-config=num_cpus=1")

@app.command()
def start_apiserver(
    port: Annotated[Optional[int], port_type] = DEFAULT_HTTP_PORT,
    resource_config: Annotated[str, apiserver_resource_type] = None,
    scale_config: Annotated[str, apiserver_scale_type] = None
):
    """Start a api server, it will provide apis.
    Args:
        *host: The host ip to run.
        *port: The port to run.
    """
    sdk.start_apiserver(port=port, resource_config=resource_config, scale_config=scale_config)

@app.command()
def run_ft(ft: Annotated[str, ft_define]):
    """Start a fine tune process.

    Args:
        *model: The model to run.
    """
    sdk.run_ft(ft)

@app.command()
def ray_ft(model: Annotated[str, ft_define]):
    """Start a fine tune ray process.

    Args:
        *model: The model to run.
    """
    sdk.run_ray_ft(model)

# @app.command()
# def list_models(metadata: Annotated[bool, "Whether to print metadata"] = False):
#     """Get a list of the available models"""
#     result = sdk.models()
#     if metadata:
#         for model in result:
#             rp(f"[bold]{model}:[/]")
#             rp(sdk.metadata(model))
#     else:
#         print("\n".join(result))


# def _print_result(result, model, print_stats):
#     rp(f"[bold]{model}:[/]")
#     if print_stats:
#         rp("[bold]Stats:[/]")
#         rp(result)
#     else:
#         rp(result)


# def progress_spinner():
#     return Progress(
#         SpinnerColumn(),
#         TextColumn("[progress.description]{task.description}"),
#         transient=True,
#     )


# @app.command()
# def query(
#     model: Annotated[List[str], model_type],
#     prompt: Annotated[Optional[List[str]], prompt_type] = None,
#     prompt_file: Annotated[Optional[str], prompt_file_type] = None,
#     separator: Annotated[str, separator_type] = "----",
#     output_file: Annotated[str, results_type] = "llmadmin-output.json",
#     print_stats: Annotated[bool, stats_type] = False,
# ):
#     """Query one or several models with one or multiple prompts,
#     optionally read from file, and save the results to a file."""
#     with progress_spinner() as progress:
#         if prompt_file:
#             with open(prompt_file, "r") as f:
#                 prompt = f.read().split(separator)

#         results = {p: [] for p in prompt}

#         for m in model:
#             progress.add_task(
#                 description=f"Processing all prompts against model: {m}.",
#                 total=None,
#             )
#             query_results = sdk.batch_query(m, prompt)
#             for result in query_results:
#                 _print_result(result, m, print_stats)

#             for i, p in enumerate(prompt):
#                 result = query_results[i]
#                 text = result
#                 # del result["generated_text"]
#                 results[p].append({"model": m, "result": text, "stats": result})

#         progress.add_task(description="Writing output file.", total=None)
#         with open(output_file, "w") as f:
#             f.write(json.dumps(results, indent=2))


# @app.command(deprecated=True, name="batch_query")
# def batch_query(
#     model: Annotated[List[str], model_type],
#     prompt: Annotated[List[str], prompt_type],
#     print_stats: Annotated[bool, stats_type] = False,
# ):
#     """Query a model with a batch of prompts."""
#     with progress_spinner() as progress:
#         for m in model:
#             progress.add_task(
#                 description=f"Processing prompt against {m}...", total=None
#             )
#             results = sdk.batch_query(m, prompt)
#             for result in results:
#                 _print_result(result, m, print_stats)


# @app.command(deprecated=True, name="multi_query")
# def multi_query(
#     model: Annotated[List[str], model_type],
#     prompt_file: Annotated[str, prompt_file_type],
#     separator: Annotated[str, separator_type] = "----",
#     output_file: Annotated[str, results_type] = "llmadmin-output.json",
# ):
#     """Query one or multiple models with a batch of prompts taken from a file."""

#     with progress_spinner() as progress:
#         progress.add_task(
#             description=f"Loading your prompts from {prompt_file}.", total=None
#         )
#         with open(prompt_file, "r") as f:
#             prompts = f.read().split(separator)
#         results = {prompt: [] for prompt in prompts}

#         for m in model:
#             progress.add_task(
#                 description=f"Processing all prompts against model: {model}.",
#                 total=None,
#             )
#             query_results = sdk.batch_query(m, prompts)
#             for i, prompt in enumerate(prompts):
#                 result = query_results[i]
#                 text = result["generated_text"]
#                 del result["generated_text"]
#                 results[prompt].append({"model": m, "result": text, "stats": result})

#         progress.add_task(description="Writing output file.", total=None)
#         with open(output_file, "w") as f:
#             f.write(json.dumps(results, indent=2))


# evaluator_type = typer.Option(help="Which LLM to use for evaluation")


# @app.command()
# def run(model: Annotated[List[str], model_type]):
#     """Start a model.

#     Args:
#         *model: The model to run.
#     """
#     sdk.run(*model)

# @app.command()
# def run_experimental(model: Annotated[List[str], model_type]):
#     """Start a model for experimental, it will do inference by transformer pipeline.

#     Args:
#         *model: The model to run.
#     """
#     sdk.run_experimental(*model)

# @app.command()
# def del_serve(appname: Annotated[str, app_name]):
#     """Remove a ray serve.

#     Args:
#         *model: The model to run.
#     """
#     sdk.del_experimental(appname)

# @app.command()
# def run_application(file: Annotated[str, file_type]):
#     """Start a model in LLMAdmin for experimental.

#     Args:
#         *model: The model to run.
#     """
#     from pathlib import Path
#     # If input is a file path, load JSON from the file
#     if isinstance(file, (str, Path)):
#         with open(file, "r", encoding="utf-8") as f:
#             flow_graph = json.load(f)
#     else:
#         raise TypeError(
#             "Input must be a file path (str)"
#         )
#     sdk.run_application(flow_graph)



# @app.command()
# def run_comparation():
#     """Start frontend for model comparation.

#     Args:
#         *model: The model to run.
#     """
#     sdk.run_comparation()

# @app.command()
# def evaluate(
#     input_file: Annotated[str, results_type] = "llmadmin-output.json",
#     evaluation_file: Annotated[str, results_type] = "evaluation-output.json",
#     evaluator: Annotated[str, evaluator_type] = "gpt-4",
# ):
#     """Evaluate and summarize the results of a multi_query run with a strong
#     'evaluator' LLM like GPT-4.
#     The results of the ranking are stored to file and displayed in a table.
#     """
#     with progress_spinner() as progress:
#         progress.add_task(description="Loading the evaluator LLM.", total=None)
#         if evaluator == "gpt-4":
#             from llmadmin.common.evaluation import GPT

#             eval_model = GPT()
#         else:
#             raise NotImplementedError(f"No evaluator for {evaluator}")

#         with open(input_file, "r") as f:
#             results = json.load(f)

#         for prompt, result_list in results.items():
#             progress.add_task(
#                 description=f"Evaluating results for prompt: {prompt}.", total=None
#             )
#             evaluation = eval_model.evaluate_results(prompt, result_list)
#             try:
#                 # GPT-4 returns a string with a Python dictionary, hopefully!
#                 evaluation = ast.literal_eval(evaluation)
#             except Exception:
#                 print(f"Could not parse evaluation: {evaluation}")

#             for i, _res in enumerate(results[prompt]):
#                 results[prompt][i]["rank"] = evaluation[i]["rank"]

#         progress.add_task(description="Storing evaluations.", total=None)
#         with open(evaluation_file, "w") as f:
#             f.write(json.dumps(results, indent=2))

#     for prompt in results.keys():
#         table = Table(title="Evaluation results (higher ranks are better)")

#         table.add_column("Model", justify="left", style="cyan", no_wrap=True)
#         table.add_column("Rank", style="magenta")
#         table.add_column("Response", justify="right", style="green")

#         for i, _res in enumerate(results[prompt]):
#             model = results[prompt][i]["model"]
#             response = results[prompt][i]["result"]
#             rank = results[prompt][i]["rank"]
#             table.add_row(model, str(rank), response)

#         console = Console()
#         console.print(table)


if __name__ == "__main__":
    app()
