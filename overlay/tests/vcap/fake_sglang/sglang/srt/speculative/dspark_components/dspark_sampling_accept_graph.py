"""CPU stand-in: imports accept_draft_tokens by name, like the engine module."""

from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens


def sampling_accept_commit_lens(io, **kwargs):
    return accept_draft_tokens(candidates=io.candidates, target_logits=io.logits)
