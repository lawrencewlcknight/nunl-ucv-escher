"""Shared orchestration limits; independent of the archived learning algorithm."""


def continuation_labels(baseline_hours=8, additional_hours=24):
    if type(baseline_hours) is not int or baseline_hours < 8 or (baseline_hours - 8) % 24:
        raise ValueError('Baseline must be P8 plus a whole number of 24-hour continuations')
    if type(additional_hours) is not int or additional_hours not in (24, 72):
        raise ValueError('Additional active hours must be 24 or 72')
    return tuple(f'P{baseline_hours + 6*i}' for i in range(additional_hours // 6 + 1))


def continuation_schedule(baseline_hours=8, additional_hours=24):
    labels = continuation_labels(baseline_hours, additional_hours)
    scale = additional_hours // 24
    training, evaluation = 30 * scale, 4 * scale
    return dict(labels=labels, additional_active_hours=additional_hours, stage_hours=6,
                training_wall_hours=training, evaluation_wall_hours=evaluation,
                worker_wall_hours=training + evaluation + 1,
                max_job_hours=training + evaluation + 2,
                secondary_alpha=.04 / (len(labels) - 1))
