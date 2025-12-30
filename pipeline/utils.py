"""Utility classes and helpers for GPSO pipeline."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from pipeline.logger import get_logger

logger = get_logger("utils")

T = TypeVar('T')
R = TypeVar('R')


class RateLimiter:
    """Simple thread-safe rate limiter (requests per minute)."""

    def __init__(self, max_requests_per_minute: int = 10) -> None:
        """Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum number of requests allowed per minute.
        """
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60
        self.requests: deque[float] = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if necessary to stay within rate limit.

        Blocks the current thread if the rate limit would be exceeded.
        """
        with self.lock:
            now = time.time()
            # Remove requests outside the time window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # If at rate limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.window_seconds - now + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.window_seconds:
                        self.requests.popleft()

            self.requests.append(time.time())

    def get_current_rate(self) -> int:
        """Get current number of requests in the time window.

        Returns:
            int: Number of requests made in the last minute.
        """
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            return len(self.requests)


def process_in_parallel(
    items: Iterable[T],
    task_fn: Callable[[T], R],
    max_workers: int = 2,
    progress_name: str = "items",
    progress_interval: int = 10
) -> list[R]:
    """Process items in parallel with progress logging.

    Args:
        items: Iterable of items to process.
        task_fn: Function to apply to each item. Should return None to skip the result.
        max_workers: Number of parallel workers.
        progress_name: Name to use in progress logs (e.g., "entities", "content items").
        progress_interval: Log progress every N items.

    Returns:
        list[R]: List of non-None results from task_fn.
    """
    items_list = list(items)
    total_items = len(items_list)
    completed_items = 0
    results_dict = {}  # Track results by index to preserve order

    logger.info(f"Processing {total_items} {progress_name} in parallel (max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks and track their indices
        future_to_index = {executor.submit(task_fn, item): i for i, item in enumerate(items_list)}

        for future in as_completed(future_to_index):
            try:
                result = future.result()
                if result is not None:
                    index = future_to_index[future]
                    results_dict[index] = result
            except Exception as e:
                logger.warning(f"Task failed with error: {str(e)[:100]}")

            completed_items += 1
            if completed_items % progress_interval == 0 or completed_items == total_items:
                logger.info(f"Progress: {completed_items}/{total_items} {progress_name} processed ({completed_items*100//total_items}%)")

    # Return results in original order
    return [results_dict[i] for i in sorted(results_dict.keys())]
