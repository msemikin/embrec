from tensorflow.python.training import basic_session_run_hooks, training_util, session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.contrib.estimator import read_eval_metrics


class ValidationMetricHook(SessionRunHook):
    def __init__(self, estimator, run_fn, run_every_steps=100):
        self._run_fn = run_fn
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=run_every_steps)
        self._global_step_tensor = None
        self._stop_var = None
        self._stop_op = None
        self._estimator = estimator
        self._last_global_step = None

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()

    def before_run(self, run_context):
        del run_context
        return session_run_hook.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            metrics = read_eval_metrics(self._estimator.eval_dir())
            global_step = next(reversed(metrics))
            last_metrics = metrics[global_step]
            if global_step != self._last_global_step:
                self._last_global_step = global_step
                self._run_fn(global_step, last_metrics)


