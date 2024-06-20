# Python
import functools
import importlib.util

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

available_pkgs_cache = []

def requires_package(*pkgs: str):
    """
    Decorator to check if the required packages are installed before running a
    function. If a package is not installed, an ImportError is raised.

    Inputs:
    - pkgs: one or multiple strings, where each string is the name of a package
            that is required for running the function.
    """
    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for pkg in pkgs:

                if pkg in available_pkgs_cache:
                    continue

                if not importlib.util.find_spec(pkg):
                    logger.error(
                        f"The package '{pkg}' is required for running " +
                        f"function '{function.__qualname__}' and could not " +
                        f"be found."
                    )
                    raise ImportError(
                        f"The package '{pkg}' is required for running " +
                        f"function '{function.__qualname__}' and could not " +
                        f"be found."
                    )

                available_pkgs_cache.append(pkg)
            return function(*args, **kwargs)

        return wrapper

    return decorator