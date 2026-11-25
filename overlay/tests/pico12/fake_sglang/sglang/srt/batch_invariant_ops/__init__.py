"""Stand-in of sglang/srt/batch_invariant_ops."""
ENABLED = [False]


def is_batch_invariant_mode_enabled():
    return ENABLED[0]
