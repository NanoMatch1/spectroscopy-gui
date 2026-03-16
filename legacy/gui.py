"""Interactive GUI for comparing THz TDS analysis pipelines.

Launch via:
    python run_comparisons.py --gui
or directly:
    python gui.py
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from models import AnalysisDataset

# ── Colour palette — one per person, consistent everywhere ───────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _person_colour(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


# ── Trace definitions ────────────────────────────────────────────────────────
# Each "trace" is a (label, domain, extractor) tuple.
# extractor(dataset) -> (x, y) or None

# -- Time domain --

def _td_raw_ref(ds: AnalysisDataset):
    if ds.time_domain and ds.time_domain.raw_ref_time_ps is not None:
        return ds.time_domain.raw_ref_time_ps, ds.time_domain.raw_ref_amplitude
    return None

def _td_raw_sam(ds: AnalysisDataset):
    if ds.time_domain and ds.time_domain.raw_sample_time_ps is not None:
        return ds.time_domain.raw_sample_time_ps, ds.time_domain.raw_sample_amplitude
    return None

def _td_win_ref(ds: AnalysisDataset):
    if ds.time_domain:
        return ds.time_domain.ref_time_ps, ds.time_domain.ref_amplitude
    return None

def _td_win_sam(ds: AnalysisDataset):
    if ds.time_domain:
        return ds.time_domain.sample_time_ps, ds.time_domain.sample_amplitude
    return None

def _td_orig_ref(ds: AnalysisDataset):
    return ds.raw_reference.time_ps, ds.raw_reference.amplitude

def _td_orig_sam(ds: AnalysisDataset):
    return ds.raw_sample.time_ps, ds.raw_sample.amplitude

# -- FFT --

def _fft_ref_amp(ds: AnalysisDataset):
    if ds.fft:
        return ds.fft.ref_frequency_thz, ds.fft.ref_amplitude
    return None

def _fft_sam_amp(ds: AnalysisDataset):
    if ds.fft:
        return ds.fft.sample_frequency_thz, ds.fft.sample_amplitude
    return None

def _fft_ref_phase(ds: AnalysisDataset):
    if ds.fft:
        return ds.fft.ref_frequency_thz, ds.fft.ref_phase
    return None

def _fft_sam_phase(ds: AnalysisDataset):
    if ds.fft:
        return ds.fft.sample_frequency_thz, ds.fft.sample_phase
    return None

# -- Transfer function (sample / reference) --

def _tf_amplitude(ds: AnalysisDataset):
    if ds.fft is None:
        return None
    # Interpolate sample onto reference frequency grid for the ratio
    ref_f = ds.fft.ref_frequency_thz
    ref_a = ds.fft.ref_amplitude
    sam_f = ds.fft.sample_frequency_thz
    sam_a = ds.fft.sample_amplitude
    if np.array_equal(ref_f, sam_f):
        ratio = np.where(ref_a != 0, sam_a / ref_a, np.nan)
        return ref_f, ratio
    sam_interp = np.interp(ref_f, sam_f, sam_a)
    ratio = np.where(ref_a != 0, sam_interp / ref_a, np.nan)
    return ref_f, ratio

def _tf_phase(ds: AnalysisDataset):
    if ds.fft is None:
        return None
    ref_f = ds.fft.ref_frequency_thz
    ref_p = ds.fft.ref_phase
    sam_f = ds.fft.sample_frequency_thz
    sam_p = ds.fft.sample_phase
    if np.array_equal(ref_f, sam_f):
        return ref_f, sam_p - ref_p
    sam_interp = np.interp(ref_f, sam_f, sam_p)
    return ref_f, sam_interp - ref_p

# -- Optical constants --

def _oc_n(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.n
    return None

def _oc_k(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.k
    return None

def _oc_eps1(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.eps1
    return None

def _oc_eps2(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.eps2
    return None

def _oc_sigma_re(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.sigma_re
    return None

def _oc_sigma_im(ds: AnalysisDataset):
    if ds.optical_constants:
        return ds.optical_constants.frequency_thz, ds.optical_constants.sigma_im
    return None


# domain -> list of (trace_key, display_label, extractor, y_axis_label)
TRACE_REGISTRY: dict[str, list[tuple[str, str, callable, str]]] = {
    "Time Domain": [
        ("td_orig_ref",  "Original reference",       _td_orig_ref,  "Amplitude (V)"),
        ("td_orig_sam",  "Original sample",           _td_orig_sam,  "Amplitude (V)"),
        ("td_win_ref",   "Windowed reference",        _td_win_ref,   "Amplitude (V)"),
        ("td_win_sam",   "Windowed sample",           _td_win_sam,   "Amplitude (V)"),
        ("td_raw_ref",   "Pre-windowed reference",    _td_raw_ref,   "Amplitude (V)"),
        ("td_raw_sam",   "Pre-windowed sample",       _td_raw_sam,   "Amplitude (V)"),
    ],
    "FFT": [
        ("fft_ref_amp",   "Reference amplitude",  _fft_ref_amp,   "Amplitude"),
        ("fft_sam_amp",   "Sample amplitude",      _fft_sam_amp,   "Amplitude"),
        ("fft_ref_phase", "Reference phase",       _fft_ref_phase, "Phase (°)"),
        ("fft_sam_phase", "Sample phase",          _fft_sam_phase, "Phase (°)"),
    ],
    "Transfer Function": [
        ("tf_amplitude", "|H(f)| amplitude",  _tf_amplitude, "|H(f)|"),
        ("tf_phase",     "Phase difference",  _tf_phase,     "Δφ (°)"),
    ],
    "Optical Constants": [
        ("oc_n",        "Refractive index n",       _oc_n,        "n"),
        ("oc_k",        "Extinction coeff k",       _oc_k,        "k"),
        ("oc_eps1",     "ε₁ (real dielectric)",     _oc_eps1,     "ε₁"),
        ("oc_eps2",     "ε₂ (imag dielectric)",     _oc_eps2,     "ε₂"),
        ("oc_sigma_re", "σ_Re (S/m)",               _oc_sigma_re, "σ_Re (S/m)"),
        ("oc_sigma_im", "σ_Im (S/m)",               _oc_sigma_im, "σ_Im (S/m)"),
    ],
}

DOMAIN_X_LABELS = {
    "Time Domain": "Time (ps)",
    "FFT": "Frequency (THz)",
    "Transfer Function": "Frequency (THz)",
    "Optical Constants": "Frequency (THz)",
}

DOMAIN_ORDER = ["Time Domain", "FFT", "Transfer Function", "Optical Constants"]


# ── Line-style cycling for traces within the same person ─────────────────────

_LINE_STYLES = ["-", "--", "-.", ":"]


# ── Main application ─────────────────────────────────────────────────────────

class ComparisonGUI:
    def __init__(self, datasets: list[AnalysisDataset]):
        self.datasets = datasets
        self.names = [ds.name for ds in datasets]

        # ── Root window ──────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("THz TDS Analysis Comparison")
        self.root.geometry("1400x850")
        self.root.minsize(900, 500)

        # ── State variables ──────────────────────────────────
        # Person toggles
        self.person_vars: dict[str, tk.BooleanVar] = {}
        for name in self.names:
            v = tk.BooleanVar(value=True)
            v.trace_add("write", lambda *_: self._schedule_redraw())
            self.person_vars[name] = v

        # Trace toggles (per trace key)
        self.trace_vars: dict[str, tk.BooleanVar] = {}
        for domain, traces in TRACE_REGISTRY.items():
            for key, label, _, _ in traces:
                # Default-on for the most common traces
                default = key in {
                    "td_win_ref", "td_win_sam",
                    "fft_ref_amp", "fft_sam_amp",
                    "tf_amplitude", "tf_phase",
                    "oc_n", "oc_k",
                }
                v = tk.BooleanVar(value=default)
                v.trace_add("write", lambda *_: self._schedule_redraw())
                self.trace_vars[key] = v

        self._redraw_pending = False

        # Per-domain axis control state
        self.autoscale_vars: dict[str, tk.BooleanVar] = {}
        self.manual_x_enabled: dict[str, tk.BooleanVar] = {}
        self.manual_y_enabled: dict[str, tk.BooleanVar] = {}
        self.manual_x_min: dict[str, tk.StringVar] = {}
        self.manual_x_max: dict[str, tk.StringVar] = {}
        self.manual_y_min: dict[str, tk.StringVar] = {}
        self.manual_y_max: dict[str, tk.StringVar] = {}
        # Saved limits from last redraw (used when autoscale is off)
        self._saved_xlim: dict[str, tuple[float, float]] = {}
        self._saved_ylim: dict[str, tuple[float, float]] = {}

        for domain in DOMAIN_ORDER:
            v = tk.BooleanVar(value=True)
            v.trace_add("write", lambda *_: self._schedule_redraw())
            self.autoscale_vars[domain] = v

            mx = tk.BooleanVar(value=False)
            mx.trace_add("write", lambda *_: self._schedule_redraw())
            self.manual_x_enabled[domain] = mx
            my = tk.BooleanVar(value=False)
            my.trace_add("write", lambda *_: self._schedule_redraw())
            self.manual_y_enabled[domain] = my

            self.manual_x_min[domain] = tk.StringVar(value="")
            self.manual_x_max[domain] = tk.StringVar(value="")
            self.manual_y_min[domain] = tk.StringVar(value="")
            self.manual_y_max[domain] = tk.StringVar(value="")

        # ── Layout ───────────────────────────────────────────
        self._build_sidebar()
        self._build_plot_area()
        self._redraw()

    # ── Sidebar ──────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ttk.Frame(self.root, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        sidebar.pack_propagate(False)

        canvas = tk.Canvas(sidebar, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling on the sidebar canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        self._section_frames: dict[str, ttk.Frame] = {}
        self._section_visible: dict[str, tk.BooleanVar] = {}

        # -- People section --
        people_body = self._build_collapsible(inner, "People")

        for idx, name in enumerate(self.names):
            colour = _person_colour(idx)
            frame = ttk.Frame(people_body)
            frame.pack(anchor="w", padx=4)
            cb = tk.Checkbutton(
                frame, text=f"  {name}", variable=self.person_vars[name],
                fg=colour, selectcolor="#ffffff",
                font=("TkDefaultFont", 10, "bold"),
                anchor="w",
            )
            cb.pack(anchor="w")

        ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6, padx=4)

        # -- Trace sections --
        for domain in DOMAIN_ORDER:
            body = self._build_collapsible(inner, domain)

            btn_frame = ttk.Frame(body)
            btn_frame.pack(anchor="w", padx=4)
            ttk.Button(btn_frame, text="All", width=4,
                       command=lambda d=domain: self._set_domain_traces(d, True)).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="None", width=4,
                       command=lambda d=domain: self._set_domain_traces(d, False)).pack(side=tk.LEFT, padx=2)

            for key, label, _, _ in TRACE_REGISTRY[domain]:
                cb = ttk.Checkbutton(body, text=label, variable=self.trace_vars[key])
                cb.pack(anchor="w", padx=12)

            # -- Axis controls --
            axis_lbl = ttk.Label(body, text="Axis controls", font=("TkDefaultFont", 9, "italic"))
            axis_lbl.pack(anchor="w", padx=8, pady=(4, 0))

            auto_cb = ttk.Checkbutton(body, text="Autoscale",
                                       variable=self.autoscale_vars[domain])
            auto_cb.pack(anchor="w", padx=12)

            # X limits
            xf = ttk.Frame(body)
            xf.pack(anchor="w", padx=12, pady=1)
            ttk.Checkbutton(xf, text="X:", variable=self.manual_x_enabled[domain],
                            width=3).pack(side=tk.LEFT)
            ex_min = ttk.Entry(xf, textvariable=self.manual_x_min[domain], width=8)
            ex_min.pack(side=tk.LEFT, padx=1)
            ex_min.bind("<Return>", lambda e: self._schedule_redraw())
            ttk.Label(xf, text="\u2013").pack(side=tk.LEFT)
            ex_max = ttk.Entry(xf, textvariable=self.manual_x_max[domain], width=8)
            ex_max.pack(side=tk.LEFT, padx=1)
            ex_max.bind("<Return>", lambda e: self._schedule_redraw())

            # Y limits
            yf = ttk.Frame(body)
            yf.pack(anchor="w", padx=12, pady=1)
            ttk.Checkbutton(yf, text="Y:", variable=self.manual_y_enabled[domain],
                            width=3).pack(side=tk.LEFT)
            ey_min = ttk.Entry(yf, textvariable=self.manual_y_min[domain], width=8)
            ey_min.pack(side=tk.LEFT, padx=1)
            ey_min.bind("<Return>", lambda e: self._schedule_redraw())
            ttk.Label(yf, text="\u2013").pack(side=tk.LEFT)
            ey_max = ttk.Entry(yf, textvariable=self.manual_y_max[domain], width=8)
            ey_max.pack(side=tk.LEFT, padx=1)
            ey_max.bind("<Return>", lambda e: self._schedule_redraw())

            ttk.Separator(inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4, padx=4)

    def _build_collapsible(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        """Create a collapsible section.  Returns the body frame."""
        var = tk.BooleanVar(value=True)
        self._section_visible[title] = var

        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=4, pady=(4, 0))

        body = ttk.Frame(parent)
        body.pack(fill=tk.X, padx=4)
        self._section_frames[title] = body

        def _toggle():
            if var.get():
                body.pack_forget()
                var.set(False)
                btn.configure(text="\u25B6")
            else:
                # Re-pack body right after header
                body.pack(fill=tk.X, padx=4, after=header)
                var.set(True)
                btn.configure(text="\u25BC")

        btn = ttk.Button(header, text="\u25BC", width=2, command=_toggle)
        btn.pack(side=tk.LEFT)
        lbl = ttk.Label(header, text=title, font=("TkDefaultFont", 11, "bold"),
                         cursor="hand2")
        lbl.pack(side=tk.LEFT, padx=4)
        lbl.bind("<Button-1>", lambda e: _toggle())

        return body

    def _set_domain_traces(self, domain: str, state: bool):
        for key, *_ in TRACE_REGISTRY[domain]:
            self.trace_vars[key].set(state)

    # ── Plot area ────────────────────────────────────────────────────────

    def _build_plot_area(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig = Figure(figsize=(10, 7), dpi=100, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Redraw logic ─────────────────────────────────────────────────────

    def _schedule_redraw(self):
        """Coalesce rapid toggle changes into a single redraw."""
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after(50, self._redraw)

    def _redraw(self):
        self._redraw_pending = False
        self.fig.clear()

        active_people = [name for name in self.names if self.person_vars[name].get()]

        # Determine which domains have at least one active trace
        active_domains: list[str] = []
        for domain in DOMAIN_ORDER:
            for key, *_ in TRACE_REGISTRY[domain]:
                if self.trace_vars[key].get():
                    active_domains.append(domain)
                    break

        if not active_domains or not active_people:
            ax = self.fig.add_subplot(111)
            ax.set_visible(False)
            ax.text(0.5, 0.5, "Select at least one person and one trace",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="grey")
            ax.set_visible(True)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self.canvas.draw_idle()
            return

        n_subplots = len(active_domains)
        axes = self.fig.subplots(n_subplots, 1, squeeze=False)

        for row_idx, domain in enumerate(active_domains):
            ax = axes[row_idx, 0]
            x_label = DOMAIN_X_LABELS[domain]
            active_traces = [
                (key, label, extractor, ylabel)
                for key, label, extractor, ylabel in TRACE_REGISTRY[domain]
                if self.trace_vars[key].get()
            ]

            y_labels_used: set[str] = set()
            has_any_data = False

            for person_idx, person_name in enumerate(active_people):
                ds_idx = self.names.index(person_name)
                ds = self.datasets[ds_idx]
                colour = _person_colour(ds_idx)

                for trace_i, (key, label, extractor, ylabel) in enumerate(active_traces):
                    result = extractor(ds)
                    if result is None:
                        continue

                    x, y = result
                    # Mask NaN for clean plotting
                    mask = np.isfinite(x) & np.isfinite(y)
                    if not np.any(mask):
                        continue

                    has_any_data = True
                    y_labels_used.add(ylabel)
                    ls = _LINE_STYLES[trace_i % len(_LINE_STYLES)]
                    ax.plot(
                        x[mask], y[mask],
                        color=colour,
                        linestyle=ls,
                        linewidth=1.2,
                        label=f"{person_name}: {label}",
                        alpha=0.85,
                    )

            # Annotate people with missing data for this domain
            missing = []
            for person_name in active_people:
                ds_idx = self.names.index(person_name)
                ds = self.datasets[ds_idx]
                all_none = all(
                    extractor(ds) is None
                    for _, _, extractor, _ in active_traces
                )
                if all_none and active_traces:
                    missing.append(person_name)

            ax.set_xlabel(x_label)
            if len(y_labels_used) == 1:
                ax.set_ylabel(y_labels_used.pop())
            else:
                ax.set_ylabel(domain)
            ax.set_title(domain, fontsize=11, fontweight="bold")

            if has_any_data:
                ax.legend(fontsize=7, loc="best", ncol=max(1, len(active_people)),
                          framealpha=0.8)

            if missing:
                note = "No data: " + ", ".join(missing)
                ax.annotate(
                    note, xy=(0.99, 0.01), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8,
                    color="red", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                )

            ax.grid(True, alpha=0.3)

            # ── Apply axis limits ─────────────────────────────
            # Priority: manual override > saved (non-autoscale) > matplotlib auto
            x_set = False
            y_set = False

            if self.manual_x_enabled[domain].get():
                try:
                    xlo = float(self.manual_x_min[domain].get())
                    xhi = float(self.manual_x_max[domain].get())
                    if xlo < xhi:
                        ax.set_xlim(xlo, xhi)
                        x_set = True
                except (ValueError, TypeError):
                    pass

            if self.manual_y_enabled[domain].get():
                try:
                    ylo = float(self.manual_y_min[domain].get())
                    yhi = float(self.manual_y_max[domain].get())
                    if ylo < yhi:
                        ax.set_ylim(ylo, yhi)
                        y_set = True
                except (ValueError, TypeError):
                    pass

            if not self.autoscale_vars[domain].get():
                if not x_set and domain in self._saved_xlim:
                    ax.set_xlim(self._saved_xlim[domain])
                if not y_set and domain in self._saved_ylim:
                    ax.set_ylim(self._saved_ylim[domain])

            # Save current limits for next non-autoscale redraw
            self._saved_xlim[domain] = ax.get_xlim()
            self._saved_ylim[domain] = ax.get_ylim()

        self.fig.set_tight_layout(True)
        self.canvas.draw_idle()

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── Standalone entry point ───────────────────────────────────────────────────

def launch(datasets: list[AnalysisDataset]):
    """Create and run the comparison GUI."""
    app = ComparisonGUI(datasets)
    app.run()


if __name__ == "__main__":
    import logging
    from loaders import load_all

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    base = os.path.dirname(os.path.abspath(__file__))
    sample_json = os.path.join(base, "sample_details.json")
    datasets = load_all(base, sample_json)
    launch(datasets)
