"""
capture.py
----------
Live packet capture using Scapy.

Captures packets from a network interface, extracts per-flow byte/packet
counts over rolling time windows, and feeds them to the detection engine
in real time.

Usage (standalone):
    sudo python capture.py --iface eth0 --window 5 --threshold 3.0
"""

import time
import threading
import queue
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, get_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

from detector import MeasureAnomalyDetector, DetectionParams, DetectionResult


@dataclass
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))

    def __eq__(self, other):
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol) == \
               (other.src_ip, other.dst_ip, other.src_port, other.dst_port, other.protocol)

    def __str__(self):
        return f"{self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port} [{self.protocol}]"


@dataclass
class CaptureStats:
    """Running statistics for the live capture session."""
    total_packets: int = 0
    total_bytes: int = 0
    total_flows: int = 0
    start_time: float = field(default_factory=time.time)
    anomalies_detected: int = 0
    last_result: Optional[DetectionResult] = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def pps(self) -> float:
        return self.total_packets / max(self.elapsed, 1)

    @property
    def bps(self) -> float:
        return self.total_bytes / max(self.elapsed, 1)


class PacketCapture:
    """
    Live capture engine backed by Scapy.

    Architecture
    ------------
    • sniff() runs in a background thread → packets → packet_queue
    • A processing thread drains the queue, accumulates per-second byte
      totals into a rolling deque, and periodically fires the detector.
    • Results are pushed to result_queue for the Streamlit dashboard.
    """

    def __init__(
        self,
        iface: Optional[str] = None,
        window_seconds: int = 60,
        detect_interval: int = 5,
        params: Optional[DetectionParams] = None,
        on_result: Optional[Callable[[DetectionResult], None]] = None,
    ):
        if not SCAPY_AVAILABLE:
            raise RuntimeError(
                "Scapy is not installed. Run: pip install scapy"
            )

        self.iface = iface
        self.window_seconds = window_seconds
        self.detect_interval = detect_interval
        self.detector = MeasureAnomalyDetector(params or DetectionParams(
            background_window=min(30, window_seconds),
            event_window=5,
            k_threshold=3.0,
        ))
        self.on_result = on_result

        # Rolling per-second byte totals
        self._byte_window: deque = deque(maxlen=window_seconds)
        self._packet_window: deque = deque(maxlen=window_seconds)
        self._ts_window: deque = deque(maxlen=window_seconds)

        # Flow tracking
        self._flow_bytes: defaultdict = defaultdict(int)
        self._flow_packets: defaultdict = defaultdict(int)

        # Threading primitives
        self._packet_queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        self.stats = CaptureStats()
        self._current_second_bytes = 0
        self._current_second_packets = 0
        self._current_second_start = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start capture and processing threads."""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not available")
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="PacketCapture"
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="PacketProcess"
        )
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self):
        """Signal threads to stop."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def get_series(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current (timestamps, byte_series) arrays."""
        ts = np.array(list(self._ts_window), dtype=float)
        bps = np.array(list(self._byte_window), dtype=float)
        return ts, bps

    def get_flow_summary(self, top_n: int = 10) -> list[dict]:
        """Top flows by byte count."""
        flows = sorted(
            self._flow_bytes.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        return [
            {
                "flow": str(k),
                "bytes": v,
                "packets": self._flow_packets[k],
            }
            for k, v in flows
        ]

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Run Scapy sniff() in a thread."""
        try:
            sniff(
                iface=self.iface,
                prn=lambda pkt: self._packet_queue.put_nowait(pkt),
                store=False,
                stop_filter=lambda _: self._stop_event.is_set(),
            )
        except PermissionError:
            print("[capture] Permission denied — run with sudo / admin rights")
        except Exception as e:
            print(f"[capture] Capture error: {e}")

    def _process_loop(self):
        """Drain packet queue, accumulate per-second totals, run detector."""
        last_detect = time.time()

        while not self._stop_event.is_set():
            try:
                pkt = self._packet_queue.get(timeout=0.1)
            except queue.Empty:
                self._flush_second()
                continue

            self._process_packet(pkt)

            now = time.time()
            if now - last_detect >= self.detect_interval:
                self._run_detection()
                last_detect = now

    def _process_packet(self, pkt):
        """Extract flow features from a single packet."""
        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        length = len(pkt)

        self.stats.total_packets += 1
        self.stats.total_bytes += length
        self._current_second_bytes += length
        self._current_second_packets += 1

        # Flow key
        proto = "OTHER"
        src_port = dst_port = 0
        if pkt.haslayer(TCP):
            proto = "TCP"
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            proto = "UDP"
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        elif pkt.haslayer(ICMP):
            proto = "ICMP"

        key = FlowKey(ip.src, ip.dst, src_port, dst_port, proto)
        self._flow_bytes[key] += length
        self._flow_packets[key] += 1

        now = time.time()
        if now - self._current_second_start >= 1.0:
            self._flush_second()

    def _flush_second(self):
        """Push accumulated per-second counters into rolling windows."""
        now = time.time()
        if now - self._current_second_start >= 1.0:
            self._byte_window.append(self._current_second_bytes)
            self._packet_window.append(self._current_second_packets)
            self._ts_window.append(self._current_second_start)
            self._current_second_bytes = 0
            self._current_second_packets = 0
            self._current_second_start = now

    def _run_detection(self):
        """Run the measure-based detector on current byte series."""
        _, bps = self.get_series()
        if len(bps) < self.detector.params.background_window // 10:
            return  # not enough data yet

        try:
            result = self.detector.detect(bps, signal_column="Bytes/s (live)")
            self.stats.anomalies_detected += result.n_events
            self.stats.last_result = result
            if self.on_result:
                self.on_result(result)
        except Exception as e:
            print(f"[capture] Detection error: {e}")


# ------------------------------------------------------------------
# Synthetic traffic generator (no-root demo / testing)
# ------------------------------------------------------------------

def generate_synthetic_traffic(
    n: int = 2000,
    base_rate: float = 5e4,
    noise_scale: float = 8e3,
    n_anomalies: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic network traffic with planted anomalies.

    Returns (timestamps, byte_series) as numpy arrays.
    Useful for demo mode when Scapy / root access is unavailable.
    """
    rng = np.random.default_rng(seed)

    # Diurnal base pattern
    t = np.arange(n)
    diurnal = base_rate * (1 + 0.4 * np.sin(2 * np.pi * t / (n / 3)))
    noise = rng.normal(0, noise_scale, n)
    x = diurnal + noise

    # Plant anomalous bursts
    for _ in range(n_anomalies):
        pos = rng.integers(500, n - 50)
        width = rng.integers(3, 20)
        amplitude = rng.uniform(6, 15) * noise_scale
        x[pos : pos + width] += amplitude

    x = np.maximum(x, 0)
    timestamps = time.time() - n + t.astype(float)
    return timestamps, x


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Measure-Based Live Anomaly Detector")
    parser.add_argument("--iface", default=None, help="Network interface (default: auto)")
    parser.add_argument("--window", type=int, default=120, help="Rolling window in seconds")
    parser.add_argument("--threshold", type=float, default=3.0, help="K sigma threshold")
    parser.add_argument("--interval", type=int, default=5, help="Detection interval (s)")
    parser.add_argument("--demo", action="store_true", help="Use synthetic traffic (no root needed)")
    args = parser.parse_args()

    if args.demo or not SCAPY_AVAILABLE:
        print("[capture] Demo mode — using synthetic traffic")
        ts, x = generate_synthetic_traffic(n=2000)
        detector = MeasureAnomalyDetector(DetectionParams(
            background_window=200,
            event_window=50,
            k_threshold=args.threshold,
        ))
        result = detector.detect(x, timestamps=ts, signal_column="Bytes/s (synthetic)")
        print(f"Detected {result.n_events} anomalous events in {result.n_samples} samples")
        for e in result.events[:5]:
            print(f"  [{e.severity:8s}] idx={e.index:5d}  z={e.z_score:.2f}  mass={e.mass:.3f}")
        return

    params = DetectionParams(
        background_window=min(30, args.window),
        event_window=5,
        k_threshold=args.threshold,
    )

    def on_result(r: DetectionResult):
        if r.n_events:
            print(f"[{time.strftime('%H:%M:%S')}] {r.n_events} event(s) detected | "
                  f"critical={r.critical_count} high={r.high_count}")

    cap = PacketCapture(
        iface=args.iface,
        window_seconds=args.window,
        detect_interval=args.interval,
        params=params,
        on_result=on_result,
    )

    print(f"[capture] Starting capture on {args.iface or 'default interface'}")
    print(f"[capture] Window={args.window}s  K={args.threshold}  Interval={args.interval}s")
    print("[capture] Press Ctrl+C to stop\n")

    cap.start()
    try:
        while True:
            time.sleep(1)
            s = cap.stats
            print(f"\r  pkts={s.total_packets:>8,}  bytes={s.total_bytes:>12,}  "
                  f"anomalies={s.anomalies_detected:>4}  elapsed={s.elapsed:.0f}s", end="")
    except KeyboardInterrupt:
        cap.stop()
        print("\n[capture] Stopped.")


if __name__ == "__main__":
    main()