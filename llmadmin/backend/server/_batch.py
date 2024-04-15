import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Type

# TODO: Upstream to Serve.


def extract_self_if_method_call(args: List[Any], func: Callable) -> Optional[object]:
    """Check if this is a method rather than a function.

    Does this by checking to see if `func` is the attribute of the first
    (`self`) argument under `func.__name__`. Unfortunately, this is the most
    robust solution to this I was able to find. It would also be preferable
    to do this check when the decorator runs, rather than when the method is.

    Returns the `self` object if it's a method call, else None.

    Arguments:
        args: arguments to the function/method call.
        func: the unbound function that was called.
    """
    if len(args) > 0:
        method = getattr(args[0], func.__name__, False)
        if method:
            wrapped = getattr(method, "__wrapped__", False)
            if wrapped and wrapped == func:
                return args[0]

    return None


class QueuePriority(IntEnum):
    """Lower value = higher priority"""

    GENERATE_TEXT = 0
    BATCH_GENERATE_TEXT = 1


@dataclass(order=True)
class _PriorityWrapper:
    """Wrapper allowing for priority queueing of arbitrary objects."""

    obj: Any = field(compare=False)
    priority: int = field(compare=True)


class PriorityQueueWithUnwrap(asyncio.PriorityQueue):
    def get_nowait(self) -> Any:
        # Get just the obj from _PriorityWrapper
        ret: _PriorityWrapper = super().get_nowait()
        return ret.obj


def _validate_max_batch_size(max_batch_size):
    if not isinstance(max_batch_size, int):
        if isinstance(max_batch_size, float) and max_batch_size.is_integer():
            max_batch_size = int(max_batch_size)
        else:
            raise TypeError("max_batch_size must be integer >= 1")

    if max_batch_size < 1:
        raise ValueError("max_batch_size must be an integer >= 1")


def _validate_batch_wait_timeout_s(batch_wait_timeout_s):
    if not isinstance(batch_wait_timeout_s, (float, int)):
        raise TypeError("batch_wait_timeout_s must be a float >= 0")

    if batch_wait_timeout_s < 0:
        raise ValueError("batch_wait_timeout_s must be a float >= 0")


