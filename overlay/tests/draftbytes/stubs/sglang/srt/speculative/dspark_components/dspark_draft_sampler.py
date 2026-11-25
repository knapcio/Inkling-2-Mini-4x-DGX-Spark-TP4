"""CPU stand-in: imports resolve_draft_params by name, like the image."""

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftParams,
    resolve_draft_params,
)


class DsparkDraftSampler:
    def __init__(self, max_bs):
        self.params = DraftParams(
            temperatures=torch.ones(max_bs),
            greedy_mask=torch.ones(max_bs, dtype=torch.bool),
            step_temperatures=torch.zeros(max_bs),
        )

    def stage_sampling_params(self, *, bs, sampling_info):
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)
