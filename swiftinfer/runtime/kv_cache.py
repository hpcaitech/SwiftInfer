# MIT License

# Copyright (c) 2023 MIT HAN Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file modified from
# https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/kv_cache.py


from typing import Dict, Optional

import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        seq_dim=3,
    ):
        # kv cache shape: [B, 2, H, S, D]
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.seq_dim = seq_dim
        self.slice_fn = DIM_TO_SLICE[seq_dim]

    def evict_for_space(self, past_key_values: Optional[Dict[int, torch.Tensor]], num_coming: int):
        if past_key_values is None:
            return None
        seq_len = next(iter(past_key_values.values())).size(self.seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return {
            k: torch.cat(
                [
                    self.slice_fn(v, 0, self.start_size),
                    self.slice_fn(v, self.start_size + num_coming, seq_len),
                ],
                dim=self.seq_dim,
            )
            for k, v in past_key_values.items()
        }
