"""Interactive matplotlib editor for THz acquisition data.

Opens a GUI that lets you include/exclude individual acquisitions and patch
single points.  Works on a plain ``np.ndarray`` with shape
``(N_points, 1 + N_acquisitions)`` — no framework dependencies.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def _edit_single_array(
    raw_array: np.ndarray,
    *,
    filename: str = "array",
    page_size: int = 25,
) -> tuple[np.ndarray, bool]:
    """Open an interactive editor for one raw acquisition matrix.

    Expected shape: ``(N_points, 1 + N_acquisitions)``
      - col 0: x/time axis
      - cols 1…: acquisitions

    Returns ``(edited_array, saved_flag)``.
    """
    raw = np.asarray(raw_array)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return np.array(raw, copy=True), False

    time = raw[:, 0]
    acq_indices = list(range(1, raw.shape[1]))
    work_raw = np.array(raw, copy=True)
    active = {idx: True for idx in acq_indices}
    selected_idx = acq_indices[0]

    fig = plt.figure(figsize=(14, 8))

    L, PW = 0.08, 0.56
    ax      = fig.add_axes([L, 0.44, PW, 0.50])
    ax_std  = fig.add_axes([L, 0.26, PW, 0.14])
    ax_sel  = fig.add_axes([L + 0.11, 0.15, PW - 0.11, 0.03])

    R, RW = 0.70, 0.25
    ax_btn       = fig.add_axes([R, 0.91, RW, 0.05])
    ax_patch_btn = fig.add_axes([R, 0.85, RW, 0.05])
    ax_incl      = fig.add_axes([R, 0.57, RW, 0.26])
    ax_incl_page = fig.add_axes([R, 0.53, RW, 0.03])
    ax_excl      = fig.add_axes([R, 0.21, RW, 0.26])
    ax_excl_page = fig.add_axes([R, 0.17, RW, 0.03])
    ax_save_btn  = fig.add_axes([R, 0.06, RW, 0.05])

    fig.suptitle(f"Acquisition Comparison: {filename}", y=0.97, fontsize=13)

    lines = {}
    for idx in acq_indices:
        (ln,) = ax.plot(time, work_raw[:, idx], alpha=0.5)
        lines[idx] = ln

    (selected_point_marker,) = ax.plot(
        [np.nan], [np.nan],
        marker="o", linestyle="None", markersize=7,
        markeredgecolor="black", markerfacecolor="gold", zorder=6,
    )

    (mean_line,) = ax.plot(
        time, np.full_like(time, np.nan, dtype=float),
        color="tab:red", linewidth=2.0, zorder=4, label="Mean (included)",
    )
    (std_line,) = ax_std.plot(
        time, np.full_like(time, np.nan, dtype=float),
        color="tab:purple", linewidth=1.6, label="Std (included)",
    )

    ax.set_ylabel("Signal Amplitude")
    ax.grid(alpha=0.25)

    ax_std.set_xlabel("Time")
    ax_std.set_ylabel("Std dev")
    ax_std.grid(alpha=0.25)
    ax_std.set_yscale("log")
    ax_std.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_std.set_xlim(ax.get_xlim())

    sel_slider = Slider(
        ax=ax_sel, label="Selected",
        valmin=0, valmax=max(0, len(acq_indices) - 1),
        valinit=0, valstep=1 if len(acq_indices) > 1 else None,
    )

    incl_page_slider = Slider(ax=ax_incl_page, label="", valmin=0, valmax=0, valinit=0, valstep=1)
    excl_page_slider = Slider(ax=ax_excl_page, label="", valmin=0, valmax=0, valinit=0, valstep=1)

    btn_toggle = Button(ax_btn, "Toggle selected (include/exclude)")
    btn_patch_point = Button(ax_patch_btn, "Toggle point patch")
    btn_save = Button(ax_save_btn, "Save changes")

    ax.text(
        0.01, 0.99,
        "Keys: x = include/exclude selected, p = patch/restore point, s = save",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
    )

    selected_point_idx = None
    point_patch_state: dict[tuple[int, int], dict] = {}
    incl_text_artists: list = []
    excl_text_artists: list = []

    state = {"saved": False, "result": np.array(raw, copy=True)}

    def get_included():
        return [i for i in acq_indices if active.get(i, False)]

    def get_excluded():
        return [i for i in acq_indices if not active.get(i, False)]

    def _clear_list(ax_list, artists):
        for t in artists:
            try:
                t.remove()
            except Exception:
                pass
        artists.clear()
        ax_list.cla()
        ax_list.set_xticks([])
        ax_list.set_yticks([])
        for spine in ax_list.spines.values():
            spine.set_visible(True)

    def _render_list(ax_list, title, items, page, artists, pick_prefix):
        _clear_list(ax_list, artists)
        t_title = ax_list.text(
            0.50, 0.97, title, transform=ax_list.transAxes,
            ha="center", va="top", fontsize=10, fontweight="bold",
        )
        artists.append(t_title)

        start = page * page_size
        end = min(len(items), start + page_size)
        view = items[start:end]

        if not view:
            t = ax_list.text(
                0.02, 0.85, "(none on this page)",
                transform=ax_list.transAxes, va="top", fontsize=9, alpha=0.7,
            )
            artists.append(t)
            return

        n = len(view)
        top, bottom = 0.88, 0.03
        step = (top - bottom) / max(1, n)

        for k, idx in enumerate(view):
            y = top - k * step
            t = ax_list.text(
                0.02, y, f"Acq {idx}",
                transform=ax_list.transAxes, va="top", fontsize=9, picker=True,
            )
            t.set_gid(f"{pick_prefix}:{idx}")
            artists.append(t)

    def _update_page_sliders():
        incl = get_included()
        excl = get_excluded()
        incl_pages = max(1, int(np.ceil(len(incl) / page_size)))
        excl_pages = max(1, int(np.ceil(len(excl) / page_size)))
        incl_page_slider.valmax = incl_pages - 1
        excl_page_slider.valmax = excl_pages - 1
        if incl_page_slider.val > incl_page_slider.valmax:
            incl_page_slider.set_val(incl_page_slider.valmax)
        if excl_page_slider.val > excl_page_slider.valmax:
            excl_page_slider.set_val(excl_page_slider.valmax)
        incl_page_slider.ax.set_xlim(incl_page_slider.valmin, incl_page_slider.valmax)
        excl_page_slider.ax.set_xlim(excl_page_slider.valmin, excl_page_slider.valmax)

    def _selected_point_key():
        if selected_point_idx is None:
            return None
        return (selected_idx, selected_point_idx)

    def _update_patch_button_label():
        key = _selected_point_key()
        if key is None:
            btn_patch_point.label.set_text("Toggle point patch (no point)")
            return
        patch_info = point_patch_state.get(key)
        if patch_info and patch_info.get("patched", False):
            btn_patch_point.label.set_text(
                f"Restore point: Acq {selected_idx}, idx {selected_point_idx}"
            )
        else:
            btn_patch_point.label.set_text(
                f"Patch point: Acq {selected_idx}, idx {selected_point_idx}"
            )

    def _update_selected_point_marker():
        if selected_point_idx is None:
            selected_point_marker.set_data([np.nan], [np.nan])
            return
        y_value = work_raw[selected_point_idx, selected_idx]
        selected_point_marker.set_data([time[selected_point_idx]], [y_value])

    def _build_output_array() -> np.ndarray:
        kept = [idx for idx in acq_indices if active.get(idx, False)]
        cols = [time] + [work_raw[:, idx] for idx in kept]
        return np.column_stack(cols) if cols else work_raw[:, [0]]

    def apply_styling():
        nonlocal selected_idx

        for idx, ln in lines.items():
            is_on = active.get(idx, False)
            ln.set_visible(is_on)
            if not is_on:
                continue
            if idx == selected_idx:
                ln.set_color("tab:green")
                ln.set_alpha(1.0)
                ln.set_linewidth(2.2)
                ln.set_zorder(3)
            else:
                ln.set_color("0.5")
                ln.set_alpha(0.5)
                ln.set_linewidth(1.0)
                ln.set_zorder(2)
            ln.set_ydata(work_raw[:, idx])

        included = get_included()
        if included:
            stack = np.column_stack([work_raw[:, idx] for idx in included])
            mean_trace = np.mean(stack, axis=1)
            std_trace = np.std(stack, axis=1, ddof=1 if len(included) > 1 else 0)
        else:
            mean_trace = np.full_like(time, np.nan, dtype=float)
            std_trace = np.full_like(time, np.nan, dtype=float)

        if ax_std.get_yscale() == "log":
            std_trace = np.where(std_trace > 0, std_trace, np.nan)

        mean_line.set_data(time, mean_trace)
        std_line.set_data(time, std_trace)
        _update_selected_point_marker()
        _update_patch_button_label()

        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        ax_std.relim()
        ax_std.autoscale_view(scalex=False, scaley=True)

        valid_std = std_trace[np.isfinite(std_trace) & (std_trace > 0)]
        if valid_std.size > 0:
            ax_std.set_ylim(valid_std.min() * 0.3, valid_std.max() * 3.0)

        sel_slider.label.set_text(f"Selected (Acq {selected_idx})")

        _update_page_sliders()

        incl_page = int(incl_page_slider.val)
        excl_page = int(excl_page_slider.val)

        _render_list(ax_incl, "Included", get_included(), incl_page, incl_text_artists, "incl")
        _render_list(ax_excl, "Excluded", get_excluded(), excl_page, excl_text_artists, "excl")

        fig.canvas.draw_idle()

    def toggle_idx(idx):
        active[idx] = not active.get(idx, True)
        apply_styling()

    def on_sel_slider(_val):
        nonlocal selected_idx, selected_point_idx
        pos = int(sel_slider.val)
        pos = max(0, min(pos, len(acq_indices) - 1))
        selected_idx = acq_indices[pos]
        selected_point_idx = None
        apply_styling()

    sel_slider.on_changed(on_sel_slider)

    def on_incl_page(_val):
        apply_styling()

    def on_excl_page(_val):
        apply_styling()

    incl_page_slider.on_changed(on_incl_page)
    excl_page_slider.on_changed(on_excl_page)

    def on_toggle_selected(_event):
        toggle_idx(selected_idx)

    btn_toggle.on_clicked(on_toggle_selected)

    def _patched_neighbor_value(column_index, point_index):
        col = work_raw[:, column_index]
        if point_index <= 0:
            return float(col[1]) if col.size > 1 else float(col[0])
        if point_index >= col.size - 1:
            return float(col[-2]) if col.size > 1 else float(col[-1])
        return float(0.5 * (col[point_index - 1] + col[point_index + 1]))

    def on_toggle_patch_point(_event):
        key = _selected_point_key()
        if key is None:
            return
        column_index, point_index = key
        if not active.get(column_index, False):
            return

        patch_info = point_patch_state.get(key)
        if patch_info is None:
            patch_info = {
                "original": float(work_raw[point_index, column_index]),
                "patched": False,
            }
            point_patch_state[key] = patch_info

        if patch_info["patched"]:
            work_raw[point_index, column_index] = patch_info["original"]
            patch_info["patched"] = False
        else:
            work_raw[point_index, column_index] = _patched_neighbor_value(
                column_index, point_index,
            )
            patch_info["patched"] = True

        apply_styling()

    btn_patch_point.on_clicked(on_toggle_patch_point)

    def on_save(_event):
        state["result"] = _build_output_array()
        state["saved"] = True
        plt.close(fig)

    btn_save.on_clicked(on_save)

    def on_main_click(event):
        nonlocal selected_point_idx
        if event.inaxes is not ax or event.xdata is None:
            return
        if not active.get(selected_idx, False):
            return
        selected_point_idx = int(np.argmin(np.abs(time - event.xdata)))
        apply_styling()

    def on_key_press(event):
        if event.key is None:
            return
        key = str(event.key).lower()
        if key == "x":
            on_toggle_selected(None)
        elif key == "p":
            on_toggle_patch_point(None)
        elif key == "s":
            on_save(None)

    fig.canvas.mpl_connect("button_press_event", on_main_click)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    def on_pick(event):
        gid = getattr(event.artist, "get_gid", lambda: None)()
        if not gid or ":" not in gid:
            return
        _, idx_str = gid.split(":", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            return

        nonlocal selected_idx
        selected_idx = idx
        try:
            sel_slider.set_val(acq_indices.index(idx))
        except ValueError:
            pass
        toggle_idx(idx)

    fig.canvas.mpl_connect("pick_event", on_pick)

    apply_styling()
    plt.show()

    return np.asarray(state["result"]), bool(state["saved"])


def edit_acquisitions(
    data: np.ndarray | Mapping[str, np.ndarray],
    *,
    filename: str | None = None,
    page_size: int = 25,
) -> np.ndarray | dict[str, np.ndarray]:
    """Interactive editor for scan acquisitions.

    Parameters
    ----------
    data
        Single ndarray ``(N, >=2)`` — ``[time | acq1 | acq2 | …]``, **or**
        a mapping ``{name: ndarray}`` to process several arrays sequentially.
    filename
        Display name when *data* is a single array.
    page_size
        Number of items per include/exclude list page.

    Returns
    -------
    Edited array (single mode) or mapping of edited arrays (dict mode).
    Closing the editor without saving returns the original data unchanged.
    """
    if isinstance(data, np.ndarray):
        edited, saved = _edit_single_array(
            data, filename=filename or "array", page_size=page_size,
        )
        return edited if saved else np.array(data, copy=True)

    if isinstance(data, Mapping):
        output: dict[str, np.ndarray] = {}
        for key, value in data.items():
            edited, saved = _edit_single_array(
                value, filename=str(key), page_size=page_size,
            )
            output[key] = edited if saved else np.array(value, copy=True)
        return output

    raise TypeError(
        "data must be either a numpy.ndarray or a mapping of name -> ndarray"
    )
