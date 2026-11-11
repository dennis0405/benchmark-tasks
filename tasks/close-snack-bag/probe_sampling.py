"""Sampling schedule for simulation probes (zero-based physics steps)."""


def sample_steps(total, every=0):
    if total <= 0 or every < 0:
        raise ValueError("total must be positive and every must be nonnegative")
    steps = {0, 59, 95, 119, 143, 191, 239, 479, total - 1}
    if every:
        steps.update(range(every - 1, total, every))
    return {step for step in steps if step < total}
