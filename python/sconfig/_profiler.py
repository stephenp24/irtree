__all__ = ["profiler", "profile"]
from functools import wraps

# ignore LineProfiler due to no python stubs and mypy will complain
from line_profiler import LineProfiler  # type: ignore

# Singleton profiler
profiler = LineProfiler()


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler.add_function(func)
        # FIXME: Uncommenting this somehow obfuscate pytest coverage result
        #        See TODO.md for more details
        # profiler.enable_by_count()
        return func(*args, **kwargs)

    return wrapper
