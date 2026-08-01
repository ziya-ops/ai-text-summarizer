import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = str(e)

                    if "401" in error_message or "invalid" in error_message.lower():
                        logger.error(f"Authentication error in {func.__name__}: {error_message}")
                        raise

                    if "400" in error_message:
                        logger.error(f"Bad request in {func.__name__}: {error_message}")
                        raise

                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {error_message}")
                        raise

                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after {delay}s: {error_message}")
                    time.sleep(delay)

            return None
        return wrapper
    return decorator
