from PyQt6.QtCore import QRunnable, pyqtSignal, QObject
import traceback


class WorkerSignals(QObject):
    """
    Defines custom signals for communication from a worker thread to the main GUI thread

    This class inherits from QObject to leverage Qt's signal/slot mechanism,
    ensuring thread-safe communication
    """

    # emitted when the worker task is completed or terminated
    finished = pyqtSignal()
    # emitted when an uncaught exception occurs in the worker task
    error = pyqtSignal(tuple)
    # emitted to pass the result of the worker task back to the main thread
    result = pyqtSignal(object)
    # signal for progress updates: current_step, total_steps, percentage, elapsed_sec, remaining_sec
    # emitted periodically by the worker task to report progress
    progress_update = pyqtSignal(int, int, float, float, float)


class Worker(QRunnable):
    """
    A generic QRunnable worker for executing a function in a separate thread

    This allows long-running tasks (like image approximation) to be performed
    without freezing the GUI
    """

    def __init__(self, fn, *args, **kwargs):
        """
        Initializes the Worker instance

        :param fn: The function to execute in the worker thread
        :type fn: callable
        :param args: Positional arguments to pass to the function `fn`
        :param kwargs: Keyword arguments to pass to the function `fn`
                       A special keyword argument `progress_emitter` can be
                       included by the caller; if present, it will be passed to `fn`
        """

        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # add the progress_update signal emitter to the kwargs passed to the target function
        # this allows the target function (fn) to emit progress signals
        self.kwargs["progress_emitter"] = self.signals.progress_update

    def run(self):
        """
        Executes the target function `self.fn` with its arguments in the current thread
        (which is a separate thread managed by QThreadPool)

        Emits signals for results, errors, or completion
        """
        try:
            # execute the function, progress_emitter is now available in self.kwargs
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            # if an unhandled exception occurs, print it and emit an error signal
            print(f"Error in worker thread: {e}")
            traceback.print_exc()
            tb_str_list = traceback.format_exception(type(e), e, e.__traceback__)
            self.signals.error.emit((type(e), e, tb_str_list))
        else:
            # if the function completes successfully
            # emit the result signal only if a result is actually returned
            if result is not None:
                self.signals.result.emit(result)
            else:
                # if the function returned None (which might be valid or indicate an issue handled within fn)
                print("Worker finished, but the target function returned None")
        finally:
            # the finished signal should always be emitted, regardless of success or failure
            self.signals.finished.emit()
