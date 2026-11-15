"""Bounded child jobs and atomic artifact publication."""
import os
from pathlib import Path
import signal
import subprocess
import time

from .train import write_json


def run_child(command, log, *, timeout):
    """Each child owns a process group, including any local Ray actors."""
    import psutil
    log = Path(log)
    started = time.monotonic()
    peak_rss = peak_host_used = 0
    monitoring_available = True
    with log.open("x") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            while child.poll() is None:
                try:
                    parent = psutil.Process(child.pid)
                    processes = [parent, *parent.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                    peak_rss = max(peak_rss, rss)
                    memory = psutil.virtual_memory()
                    peak_host_used = max(peak_host_used, memory.total-memory.available)
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    monitoring_available = False
                if time.monotonic()-started >= timeout:
                    timed_out = True
                    break
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        report = dict(returncode=child.returncode, timed_out=timed_out,
                      seconds=time.monotonic()-started, peak_process_tree_rss_bytes=peak_rss,
                      peak_host_unavailable_bytes=peak_host_used,
                      monitoring_available=monitoring_available,
                      memory_note="RSS sums can double-count shared Ray memory; host usage includes other processes")
        write_json(log.with_suffix(".resources.json"), report)
        return report
