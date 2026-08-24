"""Backward-compatible imports for the function-call task module."""

from .function_call_task import InPlaceFunctionCallTask, ThreadedPoolFunctionCallTask

__all__ = ["InPlaceFunctionCallTask", "ThreadedPoolFunctionCallTask"]
