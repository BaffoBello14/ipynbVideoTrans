"""
simple_runnable_qt.py – headless stub (no PySide6).
Replaces QRunnable / QThreadPool with standard Python threading.
"""
import threading


def run_in_threadpool(func, *args, **kwargs):
    """Run *func* in a daemon thread, mirroring QThreadPool behaviour."""
    t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
