# Copyright (c) 2026 SandAI. All Rights Reserved.
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

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch.fx as fx

from .core import emit_event
from .formatters import graph_files
from .naming import lifecycle_event_name, sanitize_event_fragment, scope_attributes
from .registry import get_attrs_resolver

AttrsResolver = Callable[[str, tuple[Any, ...], dict[str, Any], Any | None, Exception | None], dict[str, Any] | None]


def _split_attrs_and_files(attrs: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if attrs is None:
        return None, None
    resolved = dict(attrs)
    extra_files = resolved.pop("__files__", None)
    if not isinstance(extra_files, dict):
        extra_files = None
    return (resolved or None), extra_files


def emit_before_lifecycle_run(
    lifecycle_name: str,
    runtime_shape: Any = None,
    subgraph_index: int | None = None,
    graph: Any | None = None,
    extra_attributes: dict[str, Any] | None = None,
    extra_files: dict[str, str] | None = None,
):
    attributes = scope_attributes(subgraph_index)
    attributes["lifecycle_name"] = lifecycle_name
    attributes["runtime_shape"] = runtime_shape
    if extra_attributes:
        attributes.update(extra_attributes)

    files = graph_files(f"before_{sanitize_event_fragment(lifecycle_name)}", graph)
    if extra_files:
        files.update(extra_files)

    emit_event(name=lifecycle_event_name("before", lifecycle_name, subgraph_index), attributes=attributes, files=files)


def emit_after_lifecycle_run(
    lifecycle_name: str,
    duration_ms: float,
    runtime_shape: Any = None,
    subgraph_index: int | None = None,
    graph: Any | None = None,
    extra_attributes: dict[str, Any] | None = None,
    extra_files: dict[str, str] | None = None,
):
    attributes = scope_attributes(subgraph_index)
    attributes["lifecycle_name"] = lifecycle_name
    attributes["runtime_shape"] = runtime_shape
    attributes["duration_ms"] = duration_ms
    if extra_attributes:
        attributes.update(extra_attributes)

    files = graph_files("after_lifecycle", graph)
    if extra_files:
        files.update(extra_files)

    emit_event(name=lifecycle_event_name("after", lifecycle_name, subgraph_index), attributes=attributes, files=files)


def emit_skip_lifecycle_run(
    lifecycle_name: str,
    runtime_shape: Any = None,
    subgraph_index: int | None = None,
    extra_attributes: dict[str, Any] | None = None,
):
    attributes = scope_attributes(subgraph_index)
    attributes["lifecycle_name"] = lifecycle_name
    attributes["runtime_shape"] = runtime_shape
    if extra_attributes:
        attributes.update(extra_attributes)
    emit_event(name=lifecycle_event_name("skip", lifecycle_name, subgraph_index), attributes=attributes)


def emit_lifecycle_run_failed(
    lifecycle_name: str,
    exception_type: str,
    exception_message: str,
    runtime_shape: Any = None,
    subgraph_index: int | None = None,
    graph: Any | None = None,
    extra_attributes: dict[str, Any] | None = None,
    extra_files: dict[str, str] | None = None,
):
    attributes = scope_attributes(subgraph_index)
    attributes["lifecycle_name"] = lifecycle_name
    attributes["runtime_shape"] = runtime_shape
    attributes["exception_type"] = exception_type
    attributes["exception_message"] = exception_message
    if extra_attributes:
        attributes.update(extra_attributes)

    files = graph_files("failed_lifecycle", graph)
    if extra_files:
        files.update(extra_files)

    emit_event(name=lifecycle_event_name("failed", lifecycle_name, subgraph_index), attributes=attributes, files=files)


def emit_pass_lifecycle(call_fn: Callable[..., Any]) -> Callable[..., bool]:
    """Decorator for pass __call__ methods to unify pass lifecycle emitting."""

    def resolve_default_pass_stage(instance: Any | None) -> str:
        module_name = getattr(instance.__class__, "__module__", "") if instance is not None else ""
        if ".passes.full_graph." in module_name:
            return "full_graph"
        return "post_grad"

    def resolve_pass_lifecycle_name(call_args: tuple[Any, ...], _call_kwargs: dict[str, Any]) -> str:
        instance = call_args[0] if call_args else None
        return instance.__class__.__name__ if instance is not None else "unknown_pass"

    def resolve_pass_attrs(
        _phase: str, call_args: tuple[Any, ...], call_kwargs: dict[str, Any], _result: Any | None, _error: Exception | None
    ) -> dict[str, Any] | None:
        stage = call_kwargs.get("__pass_stage")
        if stage is None:
            stage = resolve_default_pass_stage(call_args[0] if call_args else None)
        return {"pass_stage": stage}

    def resolve_pass_context(
        _phase: str, call_args: tuple[Any, ...], call_kwargs: dict[str, Any], _result: Any | None, _error: Exception | None
    ) -> dict[str, Any]:
        graph = call_args[1] if len(call_args) > 1 else None

        runtime_shape = call_kwargs.get("__pass_runtime_shape")
        subgraph_index = call_kwargs.get("__pass_subgraph_index")
        if runtime_shape is None and subgraph_index is None:
            try:
                from ...passes.pass_base import get_pass_context

                ctx = get_pass_context()
                runtime_shape = ctx.runtime_shape
                subgraph_index = ctx.subgraph_index
            except Exception:
                runtime_shape = None
                subgraph_index = None

        return {
            "runtime_shape": runtime_shape,
            "subgraph_index": subgraph_index,
            "graph": call_kwargs.get("__pass_graph_payload", graph),
        }

    @observe_lifecycle(resolve_pass_lifecycle_name, attrs_resolver=resolve_pass_attrs, context_resolver=resolve_pass_context)
    def observed_pass_call(*call_args, **_call_kwargs):
        instance = call_args[0]
        graph = call_args[1]
        pass_args = call_args[2:]
        call_fn(instance, graph, *pass_args)

    @functools.wraps(call_fn)
    def wrapped(self, graph: fx.Graph, *args, **kwargs) -> bool:
        stage = kwargs.pop("stage", None)
        runtime_shape = kwargs.pop("runtime_shape", None)
        subgraph_index = kwargs.pop("subgraph_index", None)
        graph_payload = kwargs.pop("graph_payload", graph)
        emit_enabled = kwargs.pop("emit", True)
        default_stage = resolve_default_pass_stage(self)
        if kwargs:
            raise TypeError(f"Unexpected kwargs for pass lifecycle wrapper: {sorted(kwargs.keys())}")

        if runtime_shape is None and subgraph_index is None:
            try:
                from ...passes.pass_base import get_pass_context

                ctx = get_pass_context()
                runtime_shape = ctx.runtime_shape
                subgraph_index = ctx.subgraph_index
            except Exception:
                runtime_shape = None
                subgraph_index = None

        if emit_enabled and hasattr(self, "is_applicable") and not self.is_applicable(graph, runtime_shape):
            emit_skip_lifecycle_run(
                self.__class__.__name__,
                runtime_shape=runtime_shape,
                subgraph_index=subgraph_index,
                extra_attributes={"pass_stage": stage or default_stage},
            )
            return False

        if not emit_enabled:
            call_fn(self, graph, *args)
            return True

        observed_pass_call(
            self,
            graph,
            *args,
            __pass_stage=stage or default_stage,
            __pass_runtime_shape=runtime_shape,
            __pass_subgraph_index=subgraph_index,
            __pass_graph_payload=graph_payload,
        )
        return True

    setattr(wrapped, "_magi_emit_lifecycle_wrapped", True)
    return wrapped


def observe_lifecycle(
    lifecycle_name: str | Callable[[tuple[Any, ...], dict[str, Any]], str],
    *,
    attrs_resolver: AttrsResolver | None = None,
    context_resolver: Callable[[str, tuple[Any, ...], dict[str, Any], Any | None, Exception | None], dict[str, Any]]
    | None = None,
):
    """Decorator for generic lifecycle emission with optional attribute and context resolvers."""

    resolved_attrs_resolver = attrs_resolver if attrs_resolver is not None else None

    def decorator(call_fn: Callable[..., Any]):
        @functools.wraps(call_fn)
        def wrapped(*args, **kwargs):
            def resolve_lifecycle_name() -> str:
                if callable(lifecycle_name):
                    return lifecycle_name(args, kwargs)
                return lifecycle_name

            current_lifecycle_name = resolve_lifecycle_name()
            default_attrs_resolver = get_attrs_resolver(current_lifecycle_name)
            effective_attrs_resolver = resolved_attrs_resolver or default_attrs_resolver

            def resolve_graph(result: Any | None = None):
                if isinstance(result, fx.GraphModule):
                    return result
                if isinstance(result, tuple) and result and isinstance(result[0], fx.GraphModule):
                    return result[0]

                for arg in args:
                    if isinstance(arg, fx.GraphModule):
                        return arg
                for arg in kwargs.values():
                    if isinstance(arg, fx.GraphModule):
                        return arg
                return None

            def resolve_context(phase: str, result: Any | None = None, error: Exception | None = None) -> dict[str, Any]:
                if context_resolver is None:
                    return {"runtime_shape": None, "subgraph_index": None, "graph": resolve_graph(result)}
                context = context_resolver(phase, args, kwargs, result, error)
                return {
                    "runtime_shape": context.get("runtime_shape"),
                    "subgraph_index": context.get("subgraph_index"),
                    "graph": context.get("graph", resolve_graph(result)),
                }

            def resolve_attrs(phase: str, result: Any | None = None, error: Exception | None = None):
                if effective_attrs_resolver is None:
                    return None, None
                attrs = effective_attrs_resolver(phase, args, kwargs, result, error)
                return _split_attrs_and_files(attrs)

            before_ctx = resolve_context("before")
            before_attrs, before_files = resolve_attrs("before")
            emit_before_lifecycle_run(
                current_lifecycle_name,
                runtime_shape=before_ctx["runtime_shape"],
                subgraph_index=before_ctx["subgraph_index"],
                graph=before_ctx["graph"],
                extra_attributes=before_attrs,
                extra_files=before_files,
            )

            start_time = time.perf_counter()
            try:
                result = call_fn(*args, **kwargs)
            except Exception as e:
                failed_ctx = resolve_context("failed", error=e)
                failed_attrs, failed_files = resolve_attrs("failed", error=e)
                emit_lifecycle_run_failed(
                    current_lifecycle_name,
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                    runtime_shape=failed_ctx["runtime_shape"],
                    subgraph_index=failed_ctx["subgraph_index"],
                    graph=failed_ctx["graph"],
                    extra_attributes=failed_attrs,
                    extra_files=failed_files,
                )
                raise

            duration_ms = (time.perf_counter() - start_time) * 1000
            after_ctx = resolve_context("after", result=result)
            after_attrs, after_files = resolve_attrs("after", result=result)
            emit_after_lifecycle_run(
                current_lifecycle_name,
                duration_ms,
                runtime_shape=after_ctx["runtime_shape"],
                subgraph_index=after_ctx["subgraph_index"],
                graph=after_ctx["graph"],
                extra_attributes=after_attrs,
                extra_files=after_files,
            )
            return result

        return wrapped

    return decorator


@contextmanager
def observe_lifecycle_context(
    lifecycle_name: str,
    *,
    runtime_shape: Any = None,
    subgraph_index: int | None = None,
    graph: Any | None = None,
    extra_attributes: dict[str, Any] | None = None,
):
    emit_before_lifecycle_run(
        lifecycle_name,
        runtime_shape=runtime_shape,
        subgraph_index=subgraph_index,
        graph=graph,
        extra_attributes=extra_attributes,
    )
    start_time = time.perf_counter()
    try:
        yield
    except Exception as e:
        emit_lifecycle_run_failed(
            lifecycle_name,
            exception_type=type(e).__name__,
            exception_message=str(e),
            runtime_shape=runtime_shape,
            subgraph_index=subgraph_index,
            graph=graph,
            extra_attributes=extra_attributes,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_after_lifecycle_run(
            lifecycle_name,
            duration_ms,
            runtime_shape=runtime_shape,
            subgraph_index=subgraph_index,
            graph=graph,
            extra_attributes=extra_attributes,
        )
