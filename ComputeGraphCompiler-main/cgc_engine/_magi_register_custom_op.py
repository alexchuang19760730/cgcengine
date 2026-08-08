# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, get_args, get_origin

import torch

from .config import get_compile_config


def _get_num_outputs_from_return_annotation(fn: Callable) -> int:
    """
    Get the number of outputs from the function's return type annotation.

    Returns:
    - 1 if the return type is a single Tensor
    - N if the return type is tuple[Tensor, Tensor, ...] with N elements
    - 1 if no annotation or unrecognized annotation (default to single output)
    """
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation

    if return_annotation is inspect.Parameter.empty:
        return 1

    # Check if it's a tuple type (e.g., tuple[Tensor, Tensor])
    origin = get_origin(return_annotation)
    if origin is tuple:
        args = get_args(return_annotation)
        # Filter out ellipsis (for variable-length tuples like tuple[Tensor, ...])
        if args and args[-1] is not ...:
            return len(args)
        return 1

    return 1


def _generate_op_name(fn: Callable) -> str:
    """
    Generate a unique operator name from function's name and source file.

    Format: {filename_stem}::{function_name}
    Example: my_module.py with function `my_op` -> "my_module::my_op"

    Falls back to "magi_custom::{function_name}" if source file cannot be determined.
    """
    import re
    from pathlib import Path

    func_name = fn.__name__

    # Get the source file path
    try:
        source_file = inspect.getfile(fn)
        # Extract the file stem (without extension) as namespace
        namespace = Path(source_file).stem
        # Clean up namespace: replace invalid characters with underscores
        namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
    except (TypeError, OSError):
        # If we can't get the source file, use a default namespace
        namespace = "magi_custom"

    return f"{namespace}::{func_name}"


def _create_identity_meta_fn(fn: Callable) -> Callable:
    """
    Create a default identity meta function for the given function.

    The generated meta function:
    - Determines number of outputs from return type annotation
    - Uses first N tensor inputs to infer output metadata
    - Returns torch.empty_like() tensors with matching shape/dtype/device

    Raises ValueError if not enough tensor inputs are provided.
    """
    num_outputs = _get_num_outputs_from_return_annotation(fn)
    sig = inspect.signature(fn)
    # Get parameter names, excluding 'self' if present
    param_names = [name for name in sig.parameters.keys() if name != "self"]

    def identity_meta_fn(*args, **kwargs):
        # Bind arguments to get a mapping of param_name -> value
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Collect the first `num_outputs` tensor arguments
        tensor_args = []
        for name in param_names:
            arg = bound.arguments.get(name)
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
                if len(tensor_args) >= num_outputs:
                    break

        if len(tensor_args) < num_outputs:
            raise ValueError(
                f"identity_meta_fn requires at least {num_outputs} tensor inputs to match "
                f"{num_outputs} outputs, but only found {len(tensor_args)} tensor inputs. "
                f"Please provide a custom infer_output_meta_fn."
            )

        # Return outputs with same metadata as the first N inputs
        if num_outputs == 1:
            return torch.empty_like(tensor_args[0])
        return tuple(torch.empty_like(t) for t in tensor_args[:num_outputs])

    return identity_meta_fn


def _create_meta_fn_from_param_names(fn: Callable, param_names: list[str]) -> Callable:
    """
    Create a meta function that returns torch.empty_like() for each specified parameter.

    Args:
        fn: Target function to inspect
        param_names: List of parameter names to use as output templates

    Returns:
        Meta function that maps specified input params to output tensors

    Raises:
        ValueError: If parameter name doesn't exist or isn't a Tensor
    """
    sig = inspect.signature(fn)

    def meta_fn(*args, **kwargs):
        # Bind arguments to get a mapping of param_name -> value
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Collect tensors for each specified parameter name
        tensor_outputs = []
        for name in param_names:
            if name not in bound.arguments:
                raise ValueError(
                    f"Parameter '{name}' not found in function signature. "
                    f"Available parameters: {list(bound.arguments.keys())}"
                )
            arg = bound.arguments[name]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Parameter '{name}' is not a Tensor (got {type(arg).__name__}). "
                    f"infer_output_meta_fn list should only contain tensor parameter names."
                )
            tensor_outputs.append(torch.empty_like(arg))

        # Return single tensor or tuple based on number of outputs
        if len(tensor_outputs) == 1:
            return tensor_outputs[0]
        return tuple(tensor_outputs)

    return meta_fn


def _magi_register_custom_op_impl(
    name: str | None = None,
    mutates_args: tuple[str, ...] = (),
    infer_output_meta_fn: Callable | list[str] | None = None,
    setup_context_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    is_compute_sensitive: bool = False,
    is_subgraph_boundary: bool = False,
):
    def decorator(fn: Callable) -> Callable:
        # Auto-generate name if not provided
        op_name = name if name is not None else _generate_op_name(fn)
        if is_compute_sensitive:
            get_compile_config().recompute_config.custom_compute_sensitive_ops.append(op_name)
        if is_subgraph_boundary:
            get_compile_config().splitting_ops.append(op_name)

        # Step 1: Register the custom op with torch.library.custom_op
        registered_op = torch.library.custom_op(op_name, mutates_args=mutates_args)(fn)

        # Step 2: Register the output meta inference function
        # Determine meta_fn based on the type of infer_output_meta_fn
        if infer_output_meta_fn is None:
            meta_fn = _create_identity_meta_fn(fn)
        elif isinstance(infer_output_meta_fn, list):
            meta_fn = _create_meta_fn_from_param_names(fn, infer_output_meta_fn)
        else:
            meta_fn = infer_output_meta_fn
        torch.library.register_fake(op_name)(meta_fn)

        # Step 3: Register autograd if backward_fn is provided
        if backward_fn is not None:
            registered_op.register_autograd(backward_fn, setup_context=setup_context_fn)

        return registered_op

    return decorator
