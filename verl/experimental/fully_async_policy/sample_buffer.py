# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample buffer with heuristic-based selection for fully async training.

Instead of consuming samples from the message queue in strict FIFO order,
the SampleBuffer accumulates samples and selects training batches based on
a configurable scoring heuristic. Unselected samples remain in the buffer
for future training steps.

The default heuristic scores samples based on which tools were called during
rollout, allowing the trainer to prioritize samples that exercised specific
tool usage patterns.
"""

import logging
from collections.abc import Callable
from typing import Any

from verl.experimental.fully_async_policy.detach_utils import RolloutSample

logger = logging.getLogger(__name__)


def default_tool_score_fn(sample: RolloutSample) -> float:
    """Default scoring: all samples scored equally (preserves FIFO behavior).

    Override this with a function that inspects
    ``sample.full_batch.non_tensor_batch["tool_call_names"]``
    to prioritize based on which tools were invoked.
    """
    return 0.0


class SampleBuffer:
    """Buffer that accumulates rollout samples and selects training batches
    via a pluggable heuristic.

    Args:
        buffer_size: Maximum number of samples to hold. When exceeded, the
            lowest-scored samples are evicted.
        score_fn: Callable that takes a ``RolloutSample`` and returns a float
            score. Higher scores are selected first. Receives the deserialized
            sample so it can inspect ``non_tensor_batch["tool_call_names"]``
            and any other metadata.
    """

    def __init__(
        self,
        buffer_size: int,
        score_fn: Callable[[RolloutSample], float] | None = None,
    ):
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.buffer_size = buffer_size
        self.score_fn = score_fn or default_tool_score_fn
        self._buffer: list[RolloutSample] = []
        self._total_added = 0
        self._total_selected = 0
        self._total_evicted = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, sample: RolloutSample) -> None:
        """Add a deserialized sample to the buffer.

        If the buffer is full, the lowest-scored sample is evicted to make
        room.
        """
        self._buffer.append(sample)
        self._total_added += 1

        if len(self._buffer) > self.buffer_size:
            self._evict_lowest()

    def add_many(self, samples: list[RolloutSample]) -> None:
        """Convenience: add multiple samples."""
        for s in samples:
            self.add(s)

    def select(self, n: int) -> list[RolloutSample]:
        """Select the top *n* samples by score, removing them from the buffer.

        If the buffer has fewer than *n* samples, returns all available
        samples (caller should check length).
        """
        if n <= 0:
            return []

        n = min(n, len(self._buffer))

        if self.score_fn is default_tool_score_fn:
            # Fast path: no scoring needed, take the first n (FIFO)
            selected = self._buffer[:n]
            self._buffer = self._buffer[n:]
        else:
            scored = [(self.score_fn(s), idx, s) for idx, s in enumerate(self._buffer)]
            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [s for _, _, s in scored[:n]]
            selected_indices = {idx for _, idx, _ in scored[:n]}
            self._buffer = [s for idx, s in enumerate(self._buffer) if idx not in selected_indices]

        self._total_selected += len(selected)
        return selected

    @property
    def size(self) -> int:
        return len(self._buffer)

    def get_tool_call_names(self, sample: RolloutSample) -> list[str]:
        """Extract the list of tool call names from a sample.

        Utility for use inside custom ``score_fn`` implementations.
        """
        ntb = getattr(sample.full_batch, "non_tensor_batch", {})
        names_arr = ntb.get("tool_call_names")
        if names_arr is None:
            return []
        # non_tensor_batch values are numpy arrays of object dtype;
        # each element is a list[str] (one per trajectory in the sample).
        result: list[str] = []
        for per_traj in names_arr:
            if per_traj is not None:
                result.extend(per_traj)
        return result

    def get_statistics(self) -> dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_size,
            "total_added": self._total_added,
            "total_selected": self._total_selected,
            "total_evicted": self._total_evicted,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lowest(self) -> None:
        """Remove the single lowest-scored sample from the buffer."""
        if not self._buffer:
            return

        if self.score_fn is default_tool_score_fn:
            # FIFO eviction: drop the oldest
            self._buffer.pop(0)
        else:
            worst_idx = 0
            worst_score = self.score_fn(self._buffer[0])
            for idx in range(1, len(self._buffer)):
                s = self.score_fn(self._buffer[idx])
                if s < worst_score:
                    worst_score = s
                    worst_idx = idx
            self._buffer.pop(worst_idx)

        self._total_evicted += 1
