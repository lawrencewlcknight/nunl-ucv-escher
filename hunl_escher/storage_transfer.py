"""Bounded retries for idempotent checkpoint copies, never game/API actions."""
import math
import re
import subprocess
import sys
import time


def transient_error(error):
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    message = getattr(error, 'stderr', '') or ''
    if isinstance(message, bytes):
        message = message.decode('utf-8', errors='replace')
    # Match actual HTTP error/status lines, not incidental numbers in filenames.
    return bool(re.search(r'(?:HTTPError|HTTP(?:\s+Error)?|status(?:\s+code)?)\s*[:=]?\s*(?:408|429|500|502|503|504)\b',
                          message, re.IGNORECASE))


def run_transfer(command, *, timeout, attempts=4):
    """Retry cp/rsync of the same immutable data within one total time budget.

    Permission/configuration errors fail immediately. Publishers must still call
    this synchronously before moving pointers, policy markers or pruning state.
    """
    if command[:2] != ['gcloud', 'storage'] or command[2] not in ('cp', 'rsync'):
        raise ValueError('Retries are restricted to idempotent storage copies')
    if attempts < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid transfer retry limits')
    deadline = time.monotonic() + timeout
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        try:
            result = subprocess.run(command, check=True, timeout=remaining,
                                    stderr=subprocess.PIPE, text=True)
            if result is not None and result.stderr:
                print(result.stderr, file=sys.stderr, end='')
            return result
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, 'stderr', None)
            if detail:
                if isinstance(detail, bytes):
                    detail = detail.decode('utf-8', errors='replace')
                print(detail, file=sys.stderr, end='')
            delay = (5, 15, 30)[min(attempt, 2)]
            if (attempt + 1 == attempts or not transient_error(error) or
                    deadline - time.monotonic() <= delay):
                raise
            print(f'Transient checkpoint transfer failure; retry {attempt+2}/{attempts} in {delay}s.', file=sys.stderr)
            time.sleep(delay)
