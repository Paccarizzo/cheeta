"""Design-vector helpers shared by the env, the asset builder and the co-adaptation
scripts. Dependency-free (no Isaac Sim needed).

A design is a list of six scale factors xi in [0.8, 2.0] on the base leg segment
lengths (bthigh, bshin, bfoot, fthigh, fshin, ffoot), see Luck et al. (CoRL 2019).
"""

import random
from collections.abc import Sequence

DESIGN_DIM = 6
DESIGN_BOUNDS = (0.8, 2.0)  # paper Sec. 7.1
DESIGN_NAMES = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
IDENTITY_DESIGN = [1.0] * DESIGN_DIM


def random_design(rng: random.Random, decimals: int = 4) -> list[float]:
    """Uniform sample from the design box."""
    lo, hi = DESIGN_BOUNDS
    return [round(rng.uniform(lo, hi), decimals) for _ in range(DESIGN_DIM)]


def sample_designs(n: int, seed: int, include_identity: bool = False) -> list[list[float]]:
    """``n`` uniformly random designs (optionally the first one is the unscaled cheetah)."""
    rng = random.Random(seed)
    designs = [list(IDENTITY_DESIGN)] if include_identity and n > 0 else []
    while len(designs) < n:
        designs.append(random_design(rng))
    return designs


def mutate(design: Sequence[float], sigma: float, rng: random.Random, decimals: int = 4) -> list[float]:
    """Gaussian mutation of every scale factor, clipped to the design box."""
    lo, hi = DESIGN_BOUNDS
    return [round(min(hi, max(lo, xi + rng.gauss(0.0, sigma))), decimals) for xi in design]


def clip_design(design: Sequence[float]) -> list[float]:
    lo, hi = DESIGN_BOUNDS
    return [float(min(hi, max(lo, xi))) for xi in design]


def normalize_design(design: Sequence[float]) -> list[float]:
    """Map the design box [lo, hi]^6 to [-1, 1]^6 (used for the policy observation)."""
    lo, hi = DESIGN_BOUNDS
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    return [(xi - mid) / half for xi in design]


def design_key(design: Sequence[float]) -> str:
    """Short, filesystem-safe identifier of a design (used for asset caching)."""
    return "_".join(f"{v:.4g}" for v in design)


def validate_design(design: Sequence[float]) -> list[float]:
    if len(design) != DESIGN_DIM:
        raise ValueError(f"design vector must have {DESIGN_DIM} entries, got {len(design)}")
    return [float(x) for x in design]
