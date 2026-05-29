"""
detector.py
-----------
Measure-Based Anomaly Detection Engine

Implements the singular measure decomposition framework:
    x(t) = f(t)dt + Σ aᵢ δ(t - tᵢ)

Where:
    f(t)  → absolutely continuous background (robust rolling median)
    δ(.)  → singular atomic event support
    aᵢ    → event mass (resolution-normalized deviation)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class DetectionParams:
    """Resolution-control parameters for the detection pipeline."""
    background_window: int = 500       # coarse-graining scale (samples)
    event_window: int = 200            # min separation between impulses
    k_threshold: float = 3.0          # sigma multiplier for singular support
    eps: float = 1e-6                  # regularization floor
    signal_column: str = "Flow Bytes/s"


@dataclass
class AnomalyEvent:
    """A single detected singular event (Dirac analog)."""
    index: int                         # position in time series
    timestamp: Optional[float]        # wall-clock time if available
    mass: float                        # normalized event mass
    z_score: float                     # peak deviation in sigma units
    raw_value: float                   # original signal value at peak
    background: float                  # estimated background at peak
    severity: str = ""                 # LOW / MEDIUM / HIGH / CRITICAL

    def __post_init__(self):
        if not self.severity:
            if self.z_score >= 10:
                self.severity = "CRITICAL"
            elif self.z_score >= 7:
                self.severity = "HIGH"
            elif self.z_score >= 5:
                self.severity = "MEDIUM"
            else:
                self.severity = "LOW"


@dataclass
class DetectionResult:
    """Full output of one detection run."""
    signal: np.ndarray
    background: np.ndarray             # μ(t) — robust median background
    sigma: np.ndarray                  # σ(t) — MAD-based scale estimate
    z_field: np.ndarray                # deviation field z(t)
    support_mask: np.ndarray           # boolean: |z| > K
    delta_measure: np.ndarray          # sparse singular measure
    events: list[AnomalyEvent] = field(default_factory=list)
    params: DetectionParams = field(default_factory=DetectionParams)
    n_samples: int = 0
    duration_ms: float = 0.0
    signal_column: str = ""

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def critical_count(self) -> int:
        return sum(1 for e in self.events if e.severity == "CRITICAL")

    @property
    def high_count(self) -> int:
        return sum(1 for e in self.events if e.severity == "HIGH")

    @property
    def anomaly_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return self.n_events / self.n_samples * 100


class MeasureAnomalyDetector:
    """
    Three-traversal singular measure decomposition detector.

    Traversal 1: Robust background estimation (absolutely continuous component)
    Traversal 2: Deviation field + singular support identification
    Traversal 3: Singular measure extraction → sparse event impulses
    """

    def __init__(self, params: Optional[DetectionParams] = None):
        self.params = params or DetectionParams()

    def _robust_background(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Traversal 1: Estimate f(t) and σ(t) via rolling median + MAD.
        The 1.4826 factor makes MAD consistent with Gaussian σ.
        """
        p = self.params
        s = pd.Series(x)

        mu = s.rolling(p.background_window, min_periods=1).median().values

        mad = (
            s.rolling(p.background_window, min_periods=1)
            .apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)
            .values
        )
        sigma = 1.4826 * mad

        # Regularize
        mu = np.where(np.isnan(mu), np.nanmedian(mu) if np.any(~np.isnan(mu)) else 0, mu)
        sigma = np.where(np.isnan(sigma), np.nanmedian(sigma) if np.any(~np.isnan(sigma)) else p.eps, sigma)
        sigma = np.maximum(sigma, p.eps)

        return mu, sigma

    def _deviation_field(
        self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Traversal 2: Compute z(t) = (x - μ) / σ and singular support mask.
        """
        z = (x - mu) / sigma
        z = np.where(np.isnan(z), 0, z)
        support = np.abs(z) > self.params.k_threshold
        return z, support

    def _extract_singular_measure(
        self, z: np.ndarray, support: np.ndarray
    ) -> np.ndarray:
        """
        Traversal 3: Collapse contiguous support regions to impulses.
        Each region → single peak with mass = mean |z| over region.
        """
        N = len(z)
        event_signal = np.zeros(N)
        i = 0

        while i < N:
            if support[i]:
                start = i
                while i < N and support[i]:
                    i += 1
                end = i
                region = np.abs(z[start:end])
                mass = np.sum(region) / (end - start)
                peak_local = np.argmax(region)
                event_signal[start + peak_local] = mass
            else:
                i += 1

        # Enforce minimum event separation
        peaks, _ = find_peaks(event_signal, distance=self.params.event_window)
        delta = np.zeros(N)
        delta[peaks] = event_signal[peaks]

        # Clear boundary (undefined background region)
        delta[: self.params.background_window] = 0

        return delta

    def _build_events(
        self,
        delta: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        mu: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> list[AnomalyEvent]:
        """Convert non-zero impulse positions to AnomalyEvent objects."""
        impulse_indices = np.where(delta > 0)[0]
        events = []
        for idx in impulse_indices:
            ts = float(timestamps[idx]) if timestamps is not None else None
            events.append(
                AnomalyEvent(
                    index=int(idx),
                    timestamp=ts,
                    mass=float(delta[idx]),
                    z_score=float(np.abs(z[idx])),
                    raw_value=float(x[idx]),
                    background=float(mu[idx]),
                )
            )
        events.sort(key=lambda e: e.mass, reverse=True)
        return events

    def detect(
        self,
        x: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        signal_column: str = "",
    ) -> DetectionResult:
        """
        Run the full three-traversal detection pipeline.

        Parameters
        ----------
        x           : 1-D float array — the traffic signal
        timestamps  : optional array of wall-clock times aligned with x
        signal_column: label for the signal (e.g. "Flow Bytes/s")

        Returns
        -------
        DetectionResult with all intermediate fields and detected events
        """
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=float)

        mu, sigma = self._robust_background(x)
        z, support = self._deviation_field(x, mu, sigma)
        delta = self._extract_singular_measure(z, support)
        events = self._build_events(delta, z, x, mu, timestamps)

        duration_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            signal=x,
            background=mu,
            sigma=sigma,
            z_field=z,
            support_mask=support,
            delta_measure=delta,
            events=events,
            params=self.params,
            n_samples=len(x),
            duration_ms=duration_ms,
            signal_column=signal_column or self.params.signal_column,
        )

    def detect_from_dataframe(
        self, df: pd.DataFrame, column: Optional[str] = None
    ) -> DetectionResult:
        """Convenience wrapper for a pandas DataFrame."""
        col = column or self.params.signal_column

        if col not in df.columns:
            # Try fuzzy match (strip whitespace)
            matches = [c for c in df.columns if c.strip().lower() == col.strip().lower()]
            if matches:
                col = matches[0]
            else:
                available = ", ".join(df.columns[:10])
                raise ValueError(
                    f"Column '{col}' not found. Available: {available}"
                )

        series = df[col].dropna()
        x = series.values.astype(float)

        timestamps = None
        if "Timestamp" in df.columns:
            timestamps = df.loc[series.index, "Timestamp"].values

        return self.detect(x, timestamps=timestamps, signal_column=col)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CICIDS-style CSV, stripping whitespace from column names."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable as detection signals (numeric, >100 non-null values)."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].dropna().shape[0] > 100
    ]