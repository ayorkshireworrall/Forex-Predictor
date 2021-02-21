import time
import functools

def timed(func):
    """adds the time taken to the return list of a function

    Args:
        func (func): the wrapped / decorated function

    Returns:
        func: the wrapped function with exact same return type but 
    """
    @functools.wraps(func)
    def wrapped_timed(*args, **kwargs):
        func_timer = Timer()
        func_timer.start()
        value = func(*args, **kwargs)
        elapsed_time = func_timer.stop()
        return value, elapsed_time
    return wrapped_timed

class Timer:
    """A simple timer class
    """
    def __init__(self):
        self._start_time = None
        self._context_timed = None
    def start(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Stop the context manager timer, if manually interrupted via keyboard will only exit the current context 
        therefore should not be used within an iterator, instead look to extract each iteration into its own function 
        and instead used the @timed decorator on that function"""
        if exc_type is KeyboardInterrupt:
            self._context_timed = self.stop()
            print('Timed process has been manually stopped')
            return True
        self._context_timed = self.stop()