"""Turning logits into a token.

Everything up to here was deterministic and could be checked against HF
token-for-token. Sampling breaks that, so it is verified in two halves:

  - the **filters** (temperature, top-k, top-p) are pure functions on logits and
    are compared to HF's warpers exactly;
  - the **draw** is checked statistically — with a fixed seed for reproducibility,
    and by confirming empirical frequencies converge to the intended probabilities.

Note that filtering happens in logit space, not probability space. Removed
tokens are set to -inf so softmax maps them to exactly zero and the survivors
renormalize on their own — no explicit division needed.
"""

from __future__ import annotations

import torch

FILTER_VALUE = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide by T. Below 1 sharpens the distribution, above 1 flattens it.

    T is a divisor, so the effect is not symmetric around 1: T=0.5 doubles every
    logit gap, T=2 halves it. As T -> 0 the softmax approaches a one-hot on the
    argmax, which is why temperature 0 is treated as greedy rather than actually
    dividing by zero.
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits, drop the rest.

    A fixed-size shortlist. It ignores the shape of the distribution: k=50 keeps
    50 tokens whether the model is certain (where 50 is far too many) or
    genuinely unsure (where it may be too few). That is what top-p addresses.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # The k-th largest value is the threshold; anything strictly below it goes.
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, FILTER_VALUE)


def apply_top_p(logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches p.

    Adaptive where top-k is fixed — a confident distribution keeps a handful of
    tokens, a flat one keeps many.

    Implemented the way HF does it, sorting **ascending** and removing from the
    bottom while the cumulative mass is still under `1 - p`. Sorting descending
    and cutting once the running sum exceeds `p` is the more obvious phrasing
    but drops the token that straddles the boundary, giving a slightly smaller
    nucleus. Both are defensible; only one matches the reference.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove_sorted = cumulative <= (1.0 - p)
    remove_sorted[..., -min_tokens_to_keep:] = False  # never empty the nucleus
    remove = remove_sorted.scatter(-1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, FILTER_VALUE)


class Sampler:
    """A strategy for turning one row of logits into one token id.

    `logits` is (B, vocab); the return is (B, 1). Greedy and random decoding are
    separate implementations rather than a flag, so no `if greedy` branch runs
    once per generated token. Choosing between them happens once, at
    construction — see `Sampler.build`.
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def is_greedy(self) -> bool:
        return False

    @staticmethod
    def build(
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> Sampler:
        """Pick the implementation from decoding parameters.

        `temperature=0` yields a `GreedySampler`: it is the T -> 0 limit of the
        softmax, and the only sane reading of a division that is otherwise
        undefined. This is the one place the greedy/random decision is made.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if temperature == 0.0:
            return GreedySampler()
        return RandomSampler(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)


class GreedySampler(Sampler):
    """Always take the most likely token.

    Deterministic, so it is the only sampler whose output can be compared to
    `transformers` token-for-token — which is exactly what steps 7 and 8 rely on.
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(-1, keepdim=True)

    @property
    def is_greedy(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "GreedySampler()"


class RandomSampler(Sampler):
    """Filter the logits, then draw from what survives.

    Filters are applied in the order temperature -> top-k -> top-p, which is
    what `transformers` does. The order is not cosmetic: temperature rescales
    the gaps, so running it after top-p would change which tokens the nucleus
    contains rather than just how often they win.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be > 0; use GreedySampler for 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # A private generator, so seeded runs do not depend on — or disturb —
        # global RNG state.
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def filter(self, logits: torch.Tensor) -> torch.Tensor:
        """The deterministic half: logits in, logits out. Compared exactly to HF."""
        logits = apply_temperature(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        return apply_top_p(logits, self.top_p)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.filter(logits).softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=self.generator)

    def __repr__(self) -> str:
        return (
            f"RandomSampler(temperature={self.temperature}, "
            f"top_k={self.top_k or 'off'}, top_p={self.top_p})"
        )


GREEDY = GreedySampler()
