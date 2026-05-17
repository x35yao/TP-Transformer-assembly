"""CNEP model implementation (mirrored from upstream `cnep-master/models/cnep_g.py`).

Image-conditioned variant (`_g` for "global feature"): accepts a 1280-d
feature vector alongside (time, value) observations, linearly projects it
to 256-d, and concatenates it to every encoder/decoder input.

For the assembly experiment the image feature is a pre-extracted MobileNetV2
vector stored in `../../data/baseline_dataset.pickle`.
"""

from .cnep_g import CNEP

__all__ = ["CNEP"]
