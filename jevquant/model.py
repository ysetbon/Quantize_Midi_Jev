"""Core data types.

All score positions are measured in quarter notes ("beats" below), stored as
``Fraction`` so that triplets stay exact (1/3, 2/3, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class PerfNote:
    """One note of a human performance, times in seconds."""

    pitch: int
    onset: float
    offset: float
    velocity: int = 64
    hand: str | None = None  # "R"/"L" when known (synthetic data); else None


@dataclass
class ScoreNote:
    """One note of a notated score, positions in quarter notes."""

    pitch: int
    onset: Fraction
    duration: Fraction
    velocity: int = 64
    hand: str | None = None


@dataclass
class Chunk:
    """A short piece of performance, optionally with its ground-truth score.

    ``perf[i]`` and ``truth[i]`` describe the same note when truth is present.
    """

    perf: list[PerfNote]
    truth: list[ScoreNote] | None = None
    meta: dict = field(default_factory=dict)
