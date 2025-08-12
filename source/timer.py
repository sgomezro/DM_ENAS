# timer.py

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self.timer_list = [0]

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self,verbose=True):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if verbose:
            print(f"Elapsed time: {elapsed_time/60:0.4f} minutes")
        return elapsed_time/60

    def get_time_minutes(self):
        """ get time as a value"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        return elapsed_time/60     

    
    def pause_time(self):
        """Pause the timer and save the elapsed time since start_timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self.timer_list += [elapsed_time]
        self._start_time = None

    def continue_timer(self):
        """Continue timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()


    def end_timer(self):
        """Stop the timer, and report the elapsed time"""
        print(f'Total elapsed time on timer {sum(self.timer_list)}')
        print(f'times collected {self.timer_list}')
