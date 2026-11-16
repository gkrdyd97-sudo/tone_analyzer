# SPDX-License-Identifier: MIT
# Copyright (c) 2026 정학용 (gkrdyd97-sudo)

from dataclasses import dataclass


DEFAULT_LUT_SIZE = 33
DEFAULT_MAX_SAMPLE_COUNT = 100_000
DEFAULT_KNN_NEIGHBORS = 8
DEFAULT_TONE_POINT_COUNT = 101
DEFAULT_RANDOM_SEED = 42


@dataclass
class ToneMatchingConfig:
    lut_size: int = DEFAULT_LUT_SIZE
    max_sample_count: int = DEFAULT_MAX_SAMPLE_COUNT
    n_neighbors: int = DEFAULT_KNN_NEIGHBORS
    tone_point_count: int = DEFAULT_TONE_POINT_COUNT
    random_seed: int = DEFAULT_RANDOM_SEED

    def validate(self):
        if self.lut_size < 2:
            raise ValueError(
                "lut_size는 2 이상이어야 합니다."
            )

        if self.max_sample_count < 1:
            raise ValueError(
                "max_sample_count는 1 이상이어야 합니다."
            )

        if self.n_neighbors < 1:
            raise ValueError(
                "n_neighbors는 1 이상이어야 합니다."
            )

        if self.tone_point_count < 2:
            raise ValueError(
                "tone_point_count는 2 이상이어야 합니다."
            )

        return self