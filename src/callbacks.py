from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience_steps: int = 300, min_steps: int = 1000, improvement_threshold: float = 0.001):
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.improvement_threshold = improvement_threshold
        self.best_accuracy = -float('inf')
        self.best_step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or 'eval_accuracy' not in metrics or state.global_step < self.min_steps:
            return

        current_accuracy = metrics['eval_accuracy']
        improvement = current_accuracy - self.best_accuracy

        if improvement > self.improvement_threshold:
            self.best_accuracy = current_accuracy
            self.best_step = state.global_step
            print(f"Step {state.global_step}: accuracy improved by {improvement:.4f}, acc: {current_accuracy:.4f}")

        steps_since_improvement = state.global_step - self.best_step
        if steps_since_improvement >= self.patience_steps:
            print(
                f"Early stopping: {steps_since_improvement} steps without accuracy improvement ≥ {self.improvement_threshold}")
            print(f"Best: Accuracy {self.best_accuracy:.4f} at step {self.best_step}")
            control.should_training_stop = True
