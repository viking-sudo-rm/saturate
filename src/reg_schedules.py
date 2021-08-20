"""Implementation for some saturation regularization schedules."""

def control(step: int, n_steps: int) -> float:
    return 0.

def phased(step: int, n_steps: int) -> float:
    return 0. if step < n_steps / 2 else 1.

def linear(step: int, n_steps: int) -> float:
    return step / n_steps

def ramp(step: int, n_steps: int) -> float:
    if step < n_steps / 2:
        return 0.
    return 2 * step / n_steps - 1

reg_schedules = {
    "control": control,
    "phased": phased,
    "linear": linear,
    "ramp": ramp,
}
