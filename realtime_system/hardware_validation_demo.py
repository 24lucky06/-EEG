"""
COM-only EEG hardware validation monitor.

This standalone UI reads the custom EEG board from COM in real time. It does
not generate simulated waveforms. Use it to capture report screenshots for:

1. shorted-input noise,
2. 10 Hz sine-input chain check,
3. eyes-open / eyes-closed resting EEG,
4. algorithm result summary from offline_training/results.

The UI intentionally uses ASCII text to avoid font issues in Windows debug
sessions.
"""

from __future__ import annotations

import csv
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg", force=True)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, detrend, sosfiltfilt, welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REALTIME_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REALTIME_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_DIR))

try:
    from hardware_reader import CustomSerialEEGReader, HardwareConfig
except Exception as exc:  # pragma: no cover
    CustomSerialEEGReader = None
    HardwareConfig = None
    HARDWARE_IMPORT_ERROR = exc
else:
    HARDWARE_IMPORT_ERROR = None

try:
    from config.settings import ADS1299_VOLTS_PER_COUNT
except Exception:  # pragma: no cover
    ADS1299_VOLTS_PER_COUNT = 4.5 / 24 / (2**23 - 1) * 0.25


RESULT_DIR = PROJECT_ROOT / "offline_training" / "results"
EXPORT_DIR = PROJECT_ROOT / "report_figures_paperstyle_v2" / "hardware_validation_demo"

SFREQ = 250
MAX_SECONDS = 40
FONT = "Segoe UI"
BAND_RATIO_HIGHPASS_HZ = 1.0

METRIC_LABELS = {
    "total_sleep_time_min": "TST",
    "sleep_efficiency_pct": "SE",
    "sleep_onset_latency_min": "SOL",
    "waso_min": "WASO",
    "awakening_count": "Awakenings",
    "n3_pct_tst": "N3 ratio",
    "rem_pct_tst": "REM ratio",
    "score_0_100": "Score",
}


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def band_power(x_uv: np.ndarray, low: float, high: float) -> float:
    if x_uv.size < SFREQ:
        return 0.0
    freqs, psd = welch(x_uv, fs=SFREQ, nperseg=min(x_uv.size, SFREQ * 2))
    mask = (freqs >= low) & (freqs <= high)
    return integrate(psd[mask], freqs[mask]) if np.any(mask) else 0.0


def clean_for_band_ratios(x_uv: np.ndarray) -> np.ndarray:
    """Suppress slow drift before showing awake-state band ratios."""
    x = np.asarray(x_uv, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    x = x - np.median(x)
    if x.size >= SFREQ:
        x = detrend(x, type="linear")
    if x.size >= SFREQ * 3:
        sos = butter(2, BAND_RATIO_HIGHPASS_HZ, btype="highpass", fs=SFREQ, output="sos")
        x = sosfiltfilt(sos, x)
    return x


def band_ratios(x_uv: np.ndarray) -> dict[str, float]:
    x_uv = clean_for_band_ratios(x_uv)
    values = {
        "Delta": band_power(x_uv, 0.5, 4),
        "Theta": band_power(x_uv, 4, 8),
        "Alpha": band_power(x_uv, 8, 13),
        "Beta": band_power(x_uv, 13, 30),
    }
    total = max(sum(values.values()), 1e-9)
    return {k: v / total * 100 for k, v in values.items()}


def centered_channels(data_uv: np.ndarray) -> np.ndarray:
    if data_uv.size == 0:
        return data_uv.copy()
    return data_uv - np.median(data_uv, axis=1, keepdims=True)


def robust_uv_limit(x_uv: np.ndarray, minimum: float = 6.0) -> float:
    if x_uv.size == 0:
        return minimum
    limit = float(np.percentile(np.abs(x_uv), 99)) * 1.4
    if not np.isfinite(limit):
        return minimum
    return max(minimum, limit)


def read_algorithm_summary() -> list[dict[str, str]]:
    path = RESULT_DIR / "report_ready_algorithm_summary.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["experiment"] == "leave_one_subject_out"]


def read_sleep_quality_summary() -> list[dict[str, str]]:
    path = RESULT_DIR / "report_ready_sleep_quality_error_summary.csv"
    if not path.exists():
        return []
    keep = set(METRIC_LABELS)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["mode"] == "dual2" and r["field"] in keep]


class HardwareValidationMonitor:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("EEG Hardware Validation Monitor")
        self.root.geometry("1320x840")
        self.root.minsize(1180, 760)

        self.bg = "#0f172a"
        self.panel = "#172033"
        self.line = "#2c3d5c"
        self.text = "#e5edf7"
        self.muted = "#98a6ba"
        self.cyan = "#38bdf8"
        self.green = "#22c55e"
        self.yellow = "#fbbf24"
        self.red = "#fb7185"

        self.port_var = tk.StringVar(value="COM5")
        self.baud_var = tk.StringVar(value="230400")
        self.channel_indices_var = tk.StringVar(value="1,0")
        self.live_buffer_uv = np.empty((2, 0), dtype=np.float64)
        self.live_lock = threading.Lock()
        self.msg_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.live_running = False
        self.total_samples = 0
        self.last_render_at = 0.0
        self.figures: dict[str, Figure] = {}
        self.live_views: dict[str, dict[str, object]] = {}
        self.buttons: list[tk.Button] = []

        self._style()
        self._build()
        self._render_all()
        self.root.after(300, self.start_live)
        self.root.after(100, self._poll_messages)

    def _style(self) -> None:
        self.root.configure(bg=self.bg)
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=self.bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=(FONT, 10))
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.panel)],
            foreground=[("selected", self.text)],
        )
        style.configure(
            "Treeview",
            background="#101827",
            foreground=self.text,
            fieldbackground="#101827",
            rowheight=29,
            font=(FONT, 10),
        )
        style.configure(
            "Treeview.Heading",
            background=self.panel,
            foreground=self.text,
            font=(FONT, 10, "bold"),
        )

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=self.bg)
        header.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(
            header,
            text="EEG Hardware Validation Monitor",
            bg=self.bg,
            fg=self.text,
            font=(FONT, 21, "bold"),
        ).pack(side="left")

        controls = tk.Frame(header, bg=self.bg)
        controls.pack(side="left", padx=18)
        self._header_field(controls, "Port", self.port_var, 7)
        self._header_field(controls, "Baud", self.baud_var, 8)
        self._header_field(controls, "HW ch", self.channel_indices_var, 6)

        self._button(header, "Start COM", self.start_live, self.green, "#05210f")
        self._button(header, "Stop COM", self.stop_live, self.red, "#2b0610")
        self._button(header, "Save CSV", self.save_live_csv, self.yellow, "#251a03")
        self._button(header, "Export PNG", self.export_figures, self.cyan, "#031525")

        self.status = tk.Label(header, text="Starting COM...", bg=self.bg, fg=self.muted, font=(FONT, 10))
        self.status.pack(side="right", padx=18)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.noise_tab = self._tab("Live shorted-input noise")
        self.sine_tab = self._tab("Live 10 Hz / spectrum")
        self.rest_tab = self._tab("Live eyes open / closed")
        self.result_tab = self._tab("Algorithm results")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._render_current_tab())

    def _header_field(
        self, parent: tk.Frame, label: str, variable: tk.StringVar, width: int
    ) -> None:
        row = tk.Frame(parent, bg=self.bg)
        row.pack(side="left", padx=(0, 8))
        tk.Label(row, text=label, bg=self.bg, fg=self.muted, font=(FONT, 9)).pack(anchor="w")
        tk.Entry(
            row,
            textvariable=variable,
            width=width,
            bg="#101827",
            fg=self.text,
            insertbackground=self.text,
            relief="flat",
            font=(FONT, 10),
        ).pack()

    def _button(
        self, parent: tk.Frame, text: str, command, bg: str, fg: str
    ) -> tk.Button:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            relief="flat",
            padx=12,
            pady=7,
            font=(FONT, 10, "bold"),
        )
        button.pack(side="right", padx=(8, 0))
        self.buttons.append(button)
        return button

    def _tab(self, title: str) -> tk.Frame:
        frame = tk.Frame(self.notebook, bg=self.bg)
        self.notebook.add(frame, text=title)
        return frame

    def _panel(self, parent: tk.Frame) -> tk.Frame:
        frame = tk.Frame(parent, bg=self.panel, highlightbackground=self.line, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        return frame

    def _clear(self, frame: tk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _figure(
        self, parent: tk.Frame, name: str, figsize: tuple[float, float]
    ) -> tuple[Figure, FigureCanvasTkAgg]:
        fig = Figure(figsize=figsize, dpi=100, facecolor=self.panel)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.figures[name] = fig
        return fig, canvas

    def start_live(self) -> None:
        if self.live_running:
            self.status.configure(text="COM stream already running.")
            return
        if CustomSerialEEGReader is None or HardwareConfig is None:
            self.status.configure(text=f"Hardware reader unavailable: {HARDWARE_IMPORT_ERROR}")
            return
        try:
            baud = int(self.baud_var.get().strip())
            channel_indices = [
                int(item.strip())
                for item in self.channel_indices_var.get().split(",")
                if item.strip()
            ]
        except ValueError:
            self.status.configure(text="Invalid baud or channel index. Example HW ch: 1,0")
            return
        if len(channel_indices) != 2:
            self.status.configure(text="HW ch must contain exactly two indices, for example 1,0")
            return

        with self.live_lock:
            self.live_buffer_uv = np.empty((2, 0), dtype=np.float64)
            self.total_samples = 0
        self._reset_live_tabs()
        self.stop_event.clear()
        self.live_running = True
        self.status.configure(text=f"Connecting {self.port_var.get().strip() or 'COM5'}...")
        self.worker = threading.Thread(
            target=self._serial_worker,
            args=(self.port_var.get().strip() or "COM5", baud, channel_indices),
            daemon=True,
        )
        self.worker.start()

    def stop_live(self) -> None:
        self.stop_event.set()
        self.live_running = False
        self.status.configure(text="Stopping COM stream...")

    def _serial_worker(self, port: str, baud: int, channel_indices: list[int]) -> None:
        try:
            config = HardwareConfig(channel_names=["Fp1", "Fp2"], sfreq=SFREQ, epoch_seconds=30)
            reader = CustomSerialEEGReader(
                config=config,
                serial_port=port,
                baud_rate=baud,
                source_sfreq=SFREQ,
                channel_indices=channel_indices,
                volts_per_count=ADS1299_VOLTS_PER_COUNT,
            )
            gen = reader.iter_chunks(chunk_seconds=0.12, timeout_seconds=3.0)
            try:
                for chunk_v in gen:
                    if self.stop_event.is_set():
                        break
                    if chunk_v is None:
                        self.msg_queue.put(("status", "No serial frames for 3 seconds. Check COM/board."))
                        continue
                    self.msg_queue.put(("data", np.asarray(chunk_v, dtype=np.float64) * 1e6))
            finally:
                close = getattr(gen, "close", None)
                if close:
                    close()
        except Exception as exc:
            self.msg_queue.put(("error", str(exc)))
        finally:
            self.msg_queue.put(("stopped", None))

    def _poll_messages(self) -> None:
        changed = False
        while True:
            try:
                kind, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "data":
                self._append_live_data(payload)
                changed = True
            elif kind == "status":
                self.status.configure(text=str(payload))
            elif kind == "error":
                self.live_running = False
                self.status.configure(text=f"COM error: {payload}")
            elif kind == "stopped":
                self.live_running = False
                if self.stop_event.is_set():
                    self.status.configure(text="COM stream stopped.")

        now = time.monotonic()
        if changed and now - self.last_render_at >= 0.25:
            self.last_render_at = now
            seconds = self.total_samples / SFREQ
            self.status.configure(text=f"Live COM running: {seconds:.1f}s, {self.total_samples} samples")
            self._render_current_tab()
        self.root.after(80, self._poll_messages)

    def _append_live_data(self, data_uv: np.ndarray) -> None:
        with self.live_lock:
            self.live_buffer_uv = np.concatenate([self.live_buffer_uv, data_uv], axis=1)
            max_samples = int(MAX_SECONDS * SFREQ)
            if self.live_buffer_uv.shape[1] > max_samples:
                self.live_buffer_uv = self.live_buffer_uv[:, -max_samples:]
            self.total_samples += data_uv.shape[1]

    def _live_window(self, seconds: float) -> np.ndarray:
        with self.live_lock:
            if self.live_buffer_uv.size == 0:
                return np.empty((2, 0), dtype=np.float64)
            return self.live_buffer_uv[:, -int(seconds * SFREQ) :].copy()

    def _render_all(self) -> None:
        self.live_views.clear()
        self._clear(self.noise_tab)
        self._clear(self.sine_tab)
        self._clear(self.rest_tab)
        self._clear(self.result_tab)
        self._render_noise()
        self._render_spectrum()
        self._render_resting()
        self._render_results()

    def _reset_live_tabs(self) -> None:
        self.live_views.clear()
        for name in (
            "live_shorted_input_noise",
            "live_waveform_2s",
            "live_band_power",
            "live_eyes_open_closed_waveform",
            "live_eyes_open_closed_band",
        ):
            self.figures.pop(name, None)
        self._clear(self.noise_tab)
        self._clear(self.sine_tab)
        self._clear(self.rest_tab)
        self._render_noise()
        self._render_spectrum()
        self._render_resting()

    def _render_current_tab(self) -> None:
        try:
            current = self.notebook.index("current")
        except tk.TclError:
            return
        if current == 0:
            self._render_noise()
        elif current == 1:
            self._render_spectrum()
        elif current == 2:
            self._render_resting()

    def _waiting(self, parent: tk.Frame, message: str) -> None:
        panel = self._panel(parent)
        tk.Label(
            panel,
            text=message,
            bg=self.panel,
            fg=self.text,
            font=(FONT, 17, "bold"),
        ).pack(expand=True)

    def _show_waiting(self, key: str, parent: tk.Frame, message: str) -> None:
        if self.live_views.get(key, {}).get("state") == "waiting":
            return
        self._clear(parent)
        self.live_views[key] = {"state": "waiting"}
        self._waiting(parent, message)

    def _build_noise_view(self) -> dict[str, object]:
        self._clear(self.noise_tab)
        panel = self._panel(self.noise_tab)
        body = tk.Frame(panel, bg=self.panel)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=self.panel)
        right = tk.Frame(body, bg=self.panel)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y", padx=16, pady=18)

        fig, canvas = self._figure(left, "live_shorted_input_noise", (9.5, 5.8))
        ax = fig.add_subplot(111)
        (line,) = ax.plot([], [], color=self.cyan, lw=0.9)
        ax.axhline(0, color="#64748b", lw=0.8)
        ax.set_title("Live COM shorted-input / EEG waveform (AC centered)", color=self.text, fontsize=14)
        ax.set_xlabel("Time / s", color=self.muted)
        ax.set_ylabel("AC amplitude / uV", color=self.muted)
        ax.set_xlim(0, 10)
        ax.set_ylim(-6, 6)
        self._dark_axis(ax)

        metrics = self._metric_card(
            right,
            "Live COM metrics",
            [
                ("Source", "COM live, channel 1"),
                ("Sampling", f"{SFREQ} Hz"),
                ("Window", "0.0 s"),
                ("DC offset", "-- uV"),
                ("AC RMS", "-- uV"),
                ("AC peak-to-peak", "-- uV"),
                ("Samples", "0"),
            ],
        )
        view = {"state": "plot", "line": line, "ax": ax, "canvas": canvas, "metrics": metrics}
        self.live_views["noise"] = view
        return view

    def _build_spectrum_view(self) -> dict[str, object]:
        self._clear(self.sine_tab)
        panel = self._panel(self.sine_tab)
        body = tk.Frame(panel, bg=self.panel)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=self.panel)
        right = tk.Frame(body, bg=self.panel)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="both", expand=True)

        fig1, canvas1 = self._figure(left, "live_waveform_2s", (6.5, 5.8))
        ax1 = fig1.add_subplot(111)
        (line,) = ax1.plot([], [], color=self.green, lw=1.0)
        ax1.set_title("Live COM waveform (last 2 s, AC centered)", color=self.text, fontsize=14)
        ax1.set_xlabel("Time / s", color=self.muted)
        ax1.set_ylabel("AC amplitude / uV", color=self.muted)
        ax1.set_xlim(0, 2)
        ax1.set_ylim(-6, 6)
        self._dark_axis(ax1)

        fig2, canvas2 = self._figure(right, "live_band_power", (6.5, 5.8))
        ax2 = fig2.add_subplot(111)
        labels = ["Delta", "Theta", "Alpha", "Beta"]
        bars = ax2.bar(labels, [0.0] * len(labels), color=["#60a5fa", "#a78bfa", self.green, self.yellow])
        texts = [ax2.text(i, 2, "0.0%", ha="center", color=self.text, fontsize=10) for i in range(len(labels))]
        ax2.set_ylim(0, 100)
        ax2.set_title("Live relative band power (drift-reduced)", color=self.text, fontsize=14)
        ax2.set_ylabel("Relative power / %", color=self.muted)
        self._dark_axis(ax2)

        view = {
            "state": "plot",
            "line": line,
            "ax1": ax1,
            "canvas1": canvas1,
            "bars": bars,
            "texts": texts,
            "canvas2": canvas2,
        }
        self.live_views["spectrum"] = view
        return view

    def _build_resting_view(self) -> dict[str, object]:
        self._clear(self.rest_tab)
        panel = self._panel(self.rest_tab)
        body = tk.Frame(panel, bg=self.panel)
        body.pack(fill="both", expand=True)
        top = tk.Frame(body, bg=self.panel)
        bottom = tk.Frame(body, bg=self.panel)
        top.pack(fill="both", expand=True)
        bottom.pack(fill="both", expand=True)

        fig1, canvas1 = self._figure(top, "live_eyes_open_closed_waveform", (12.5, 3.6))
        ax1 = fig1.add_subplot(111)
        (open_line,) = ax1.plot([], [], color=self.cyan, lw=0.9, label="First half")
        (closed_line,) = ax1.plot([], [], color=self.green, lw=0.9, label="Second half")
        ax1.set_title("Live COM resting EEG: first half vs second half", color=self.text, fontsize=14)
        ax1.set_xlabel("Time / s", color=self.muted)
        ax1.set_ylabel("AC amplitude / uV (offset view)", color=self.muted)
        ax1.set_xlim(0, 8)
        ax1.set_ylim(-140, 140)
        ax1.legend(facecolor=self.panel, edgecolor=self.line, labelcolor=self.text)
        self._dark_axis(ax1)

        fig2, canvas2 = self._figure(bottom, "live_eyes_open_closed_band", (12.5, 3.6))
        ax2 = fig2.add_subplot(111)
        labels = ["Delta", "Theta", "Alpha", "Beta"]
        pos = np.arange(len(labels))
        width = 0.35
        open_bars = ax2.bar(pos - width / 2, [0.0] * len(labels), width, color=self.cyan, label="First half")
        closed_bars = ax2.bar(pos + width / 2, [0.0] * len(labels), width, color=self.green, label="Second half")
        ax2.set_xticks(pos, labels)
        ax2.set_ylim(0, 100)
        ax2.set_title("Live band power comparison (drift-reduced)", color=self.text, fontsize=14)
        ax2.set_ylabel("Relative power / %", color=self.muted)
        ax2.legend(facecolor=self.panel, edgecolor=self.line, labelcolor=self.text)
        self._dark_axis(ax2)

        view = {
            "state": "plot",
            "open_line": open_line,
            "closed_line": closed_line,
            "ax1": ax1,
            "canvas1": canvas1,
            "open_bars": open_bars,
            "closed_bars": closed_bars,
            "canvas2": canvas2,
        }
        self.live_views["resting"] = view
        return view

    def _render_noise(self) -> None:
        data = self._live_window(10)
        if data.shape[1] < int(0.5 * SFREQ):
            self._show_waiting("noise", self.noise_tab, "Waiting for live COM data...")
            return

        view = self.live_views.get("noise")
        if view is None or view.get("state") != "plot":
            view = self._build_noise_view()

        raw = data[0]
        x = centered_channels(data)[0]
        t = np.arange(x.size) / SFREQ
        rms = float(np.sqrt(np.mean(x**2)))
        ptp = float(np.ptp(x))
        dc = float(np.median(raw))
        limit = robust_uv_limit(x)

        line = view["line"]
        ax = view["ax"]
        canvas = view["canvas"]
        line.set_data(t, x)
        ax.set_xlim(0, max(10.0, float(t[-1]) if t.size else 10.0))
        ax.set_ylim(-limit, limit)
        self._set_metric_rows(
            view["metrics"],
            [
                ("Source", "COM live, channel 1"),
                ("Sampling", f"{SFREQ} Hz"),
                ("Window", f"{x.size / SFREQ:.1f} s"),
                ("DC offset", f"{dc:.2f} uV"),
                ("AC RMS", f"{rms:.2f} uV"),
                ("AC peak-to-peak", f"{ptp:.2f} uV"),
                ("Samples", str(self.total_samples)),
            ],
        )
        canvas.draw_idle()

    def _render_spectrum(self) -> None:
        data = self._live_window(10)
        if data.shape[1] < int(1.0 * SFREQ):
            self._show_waiting("spectrum", self.sine_tab, "Waiting for live COM data...")
            return

        view = self.live_views.get("spectrum")
        if view is None or view.get("state") != "plot":
            view = self._build_spectrum_view()

        x = centered_channels(data)[0]
        show = x[-int(2 * SFREQ) :]
        t = np.arange(show.size) / SFREQ
        ratios = band_ratios(x)

        line = view["line"]
        ax1 = view["ax1"]
        canvas1 = view["canvas1"]
        line.set_data(t, show)
        ax1.set_xlim(0, max(2.0, float(t[-1]) if t.size else 2.0))
        limit = robust_uv_limit(show)
        ax1.set_ylim(-limit, limit)
        canvas1.draw_idle()

        labels = ["Delta", "Theta", "Alpha", "Beta"]
        vals = [ratios[k] for k in labels]
        for bar, label, value in zip(view["bars"], view["texts"], vals):
            bar.set_height(value)
            label.set_y(min(value + 2, 96))
            label.set_text(f"{value:.1f}%")
        view["canvas2"].draw_idle()

    def _render_resting(self) -> None:
        data = self._live_window(20)
        if data.shape[1] < int(2 * SFREQ):
            self._show_waiting(
                "resting",
                self.rest_tab,
                "Waiting for live COM data. For test: eyes open first 10s, eyes closed next 10s.",
            )
            return

        view = self.live_views.get("resting")
        if view is None or view.get("state") != "plot":
            view = self._build_resting_view()

        x = centered_channels(data)[0]
        half = x.size // 2
        open_eye = x[:half]
        closed_eye = x[half:]
        open_ratios = band_ratios(open_eye)
        closed_ratios = band_ratios(closed_eye)

        open_show = open_eye[-SFREQ * 8 :]
        closed_show = closed_eye[-SFREQ * 8 :]
        open_t = np.arange(open_show.size) / SFREQ
        closed_t = np.arange(closed_show.size) / SFREQ
        offset = max(70.0, float(np.percentile(np.abs(x), 95)) * 2.2)
        view["open_line"].set_data(open_t, open_show + offset)
        view["closed_line"].set_data(closed_t, closed_show - offset)
        ax1 = view["ax1"]
        ax1.set_xlim(0, 8)
        limit = max(offset + robust_uv_limit(x, minimum=40.0), 140.0)
        ax1.set_ylim(-limit, limit)
        view["canvas1"].draw_idle()

        labels = ["Delta", "Theta", "Alpha", "Beta"]
        for bar, value in zip(view["open_bars"], [open_ratios[k] for k in labels]):
            bar.set_height(value)
        for bar, value in zip(view["closed_bars"], [closed_ratios[k] for k in labels]):
            bar.set_height(value)
        view["canvas2"].draw_idle()

    def _render_results(self) -> None:
        panel = self._panel(self.result_tab)
        left = tk.Frame(panel, bg=self.panel)
        right = tk.Frame(panel, bg=self.panel)
        left.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        right.pack(side="right", fill="both", expand=True, padx=14, pady=14)

        tk.Label(
            left,
            text="Sleep staging performance",
            bg=self.panel,
            fg=self.text,
            font=(FONT, 14, "bold"),
        ).pack(anchor="w")
        algo = ttk.Treeview(
            left,
            columns=("model", "subjects", "epochs", "acc", "f1", "kappa"),
            show="headings",
            height=4,
        )
        for col, title, width in [
            ("model", "Model", 150),
            ("subjects", "Subjects", 80),
            ("epochs", "Epochs", 90),
            ("acc", "Acc", 80),
            ("f1", "Macro-F1", 90),
            ("kappa", "Kappa", 80),
        ]:
            algo.heading(col, text=title)
            algo.column(col, width=width, anchor="center")
        algo.pack(fill="x", pady=(8, 18))
        for row in sorted(read_algorithm_summary(), key=lambda r: r["mode"]):
            model = "32-channel model" if row["mode"] == "all32" else "Dual-channel model"
            algo.insert(
                "",
                "end",
                values=(
                    model,
                    row["n_subjects"],
                    row["total_epochs"],
                    f"{float(row['accuracy_mean']) * 100:.1f}%",
                    f"{float(row['macro_f1_mean']) * 100:.1f}%",
                    f"{float(row['kappa_mean']):.3f}",
                ),
            )

        tk.Label(
            right,
            text="Sleep-quality metric errors (dual-channel)",
            bg=self.panel,
            fg=self.text,
            font=(FONT, 14, "bold"),
        ).pack(anchor="w")
        sleep = ttk.Treeview(right, columns=("metric", "mae", "bias", "unit"), show="headings", height=10)
        for col, title, width in [
            ("metric", "Metric", 150),
            ("mae", "MAE", 110),
            ("bias", "Bias", 90),
            ("unit", "Unit", 70),
        ]:
            sleep.heading(col, text=title)
            sleep.column(col, width=width, anchor="center")
        sleep.pack(fill="both", expand=True, pady=(8, 18))
        for row in read_sleep_quality_summary():
            sleep.insert(
                "",
                "end",
                values=(
                    METRIC_LABELS.get(row["field"], row["field"]),
                    f"{float(row['mean_abs_error']):.2f}",
                    f"{float(row['mean_bias_pred_minus_label']):.2f}",
                    row["unit"].replace("\u6b21", "count").replace("\u5206", "score"),
                ),
            )

        text = (
            "Conclusion: 32-channel Kappa=0.800; dual-channel Kappa=0.708. "
            "Dual-channel TST, SE, WASO and score are close to label-derived references."
        )
        tk.Label(
            left,
            text=text,
            wraplength=560,
            justify="left",
            bg=self.panel,
            fg=self.text,
            font=(FONT, 12),
        ).pack(anchor="w", pady=(8, 0))

    def _metric_card(self, parent: tk.Frame, title: str, rows: list[tuple[str, str]]) -> dict[str, tk.StringVar]:
        tk.Label(parent, text=title, bg=self.panel, fg=self.text, font=(FONT, 16, "bold")).pack(anchor="w", pady=(0, 12))
        values: dict[str, tk.StringVar] = {}
        for label, value in rows:
            row = tk.Frame(parent, bg=self.panel)
            row.pack(fill="x", pady=6)
            tk.Label(row, text=label, width=16, anchor="w", bg=self.panel, fg=self.muted, font=(FONT, 11)).pack(side="left")
            value_var = tk.StringVar(value=value)
            values[label] = value_var
            tk.Label(row, textvariable=value_var, anchor="w", bg=self.panel, fg=self.text, font=(FONT, 12, "bold")).pack(side="left")
        return values

    def _set_metric_rows(self, values: dict[str, tk.StringVar], rows: list[tuple[str, str]]) -> None:
        for label, value in rows:
            value_var = values.get(label)
            if value_var is not None:
                value_var.set(value)

    def _dark_axis(self, ax) -> None:
        ax.set_facecolor("#101827")
        ax.tick_params(colors=self.muted)
        ax.grid(True, color="#26374f", alpha=0.7, lw=0.7)
        for spine in ax.spines.values():
            spine.set_color(self.line)
        ax.title.set_color(self.text)
        ax.xaxis.label.set_color(self.muted)
        ax.yaxis.label.set_color(self.muted)

    def save_live_csv(self) -> None:
        data = self._live_window(MAX_SECONDS)
        if data.shape[1] == 0:
            self.status.configure(text="No live data to save.")
            return
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EXPORT_DIR / f"live_com_{stamp}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "ch1_uv", "ch2_uv"])
            for idx in range(data.shape[1]):
                writer.writerow([f"{idx / SFREQ:.6f}", f"{data[0, idx]:.8f}", f"{data[1, idx]:.8f}"])
        self.status.configure(text=f"Live CSV saved: {path}")

    def export_figures(self) -> None:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(
                EXPORT_DIR / f"{name}.png",
                dpi=180,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
            )
        self.status.configure(text=f"PNG exported to {EXPORT_DIR}")


def main() -> None:
    root = tk.Tk()
    HardwareValidationMonitor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
