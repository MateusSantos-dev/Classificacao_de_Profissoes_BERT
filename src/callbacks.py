from transformers import TrainerCallback


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self,
                 patience_steps: int = 300,
                 min_steps: int = 1000,
                 improvement_threshold: float = 0.001):
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.improvement_threshold = improvement_threshold
        self.best_loss = float('inf')
        self.best_step = 0
        self.should_stop = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or 'loss' not in logs or state.global_step < self.min_steps:
            return

        current_loss = logs['loss']
        current_step = state.global_step

        improvement = self.best_loss - current_loss
        if improvement > self.improvement_threshold:
            self.best_loss = current_loss
            self.best_step = current_step
            print(f"No step {current_step} teve uma melhora de {improvement:.4f}, loss: {current_loss:.4f}")

        steps_since_improvement = current_step - self.best_step
        if steps_since_improvement >= self.patience_steps:
            print(f"PARANDO TREINAMENTO: {steps_since_improvement} steps sem melhora de pelo menos {self.improvement_threshold}")
            print(f"   Melhor: Loss {self.best_loss:.4f} no step {self.best_step}")
            self.should_stop = True
            control.should_training_stop = True

    def on_step_begin(self, args, state, control, **kwargs):
        if self.should_stop:
            control.should_training_stop = True
        return control
