"""Plot total/prefill/decode time vs OPT model size.

Style adapted from
    https://github.com/xlab-uiuc/Fast-and-Safe-IO-Memory-Protection/blob/master/plots/siyuan_Evaluation_main.py

Reads the *.log files produced by ``vary_model_size_opt.sh`` and renders a
grouped-bar chart with model size on the x-axis and run time (seconds) on the
y-axis. Each memory-management scheme is a separate series.

Following the convention in ``plot_updated_vary_bsz.py``, all log paths,
legend labels, colors, and plotting parameters are defined inline in
``plot_GH200_opt_vary_model_size()`` rather than passed in via CLI flags.
Just run::

    python plot_vary_model_size.py
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Style helpers (lifted with light edits from siyuan_Evaluation_main.py)
# --------------------------------------------------------------------------- #

# Curated palette of distinct, colorblind-aware colors. Keep in sync with
# plot_updated_vary_bsz.py::PALETTE so figures across scripts stay consistent.
PALETTE = [
    "#D55E00",  # 0  vermillion / burnt orange
    "#CC79A7",  # 1  reddish purple / pink
    "#56B4E9",  # 2  sky blue
    "#009E73",  # 3  bluish green
    "#F0E442",  # 4  yellow
    "#0072B2",  # 5  deep blue
    "#E69F00",  # 6  orange
    "#9B59B6",  # 7  amethyst purple
    "#34495E",  # 8  dark slate
    "#16A085",  # 9  teal
    "#27AE60",  # 10 emerald
    "#E74C3C",  # 11 red
    "#2980B9",  # 12 belize blue
    "#F39C12",  # 13 sunflower
    "#8E44AD",  # 14 wisteria
    "#7F8C8D",  # 15 concrete gray
    "#1ABC9C",  # 16 turquoise
    "#C0392B",  # 17 pomegranate
    "#2C3E50",  # 18 midnight blue
    "#BDC3C7",  # 19 silver
]


def color(i):
    """Return the i-th palette color, cycling if i >= len(PALETTE)."""
    return PALETTE[i % len(PALETTE)]


# Backwards-compat alias for any callers still using ``default_colors``.
default_colors = PALETTE


def calculate_plot_params(num_x_labels, num_series, max_width=None):
    """Dynamically pick figure size, fonts, bar width, etc."""
    base_width = 7.0
    base_height = 3.2
    base_font = 14
    base_bar_width = 0.38
    base_gap = 1.5

    width_scale = max(1.0, min(2.2, 1.0 + (num_x_labels - 3) * 0.10))
    height_scale = max(1.0, min(2.0, 1.0 + (num_series - 2) * 0.1))

    final_width = base_width * width_scale
    if max_width:
        final_width = min(final_width, max_width)
    # Fix the width to the LaTeX text column (6.5in) so the 12pt plot font
    # renders at body (12pt) size when included at \linewidth.
    figsize = (6.5, base_height * height_scale)

    font_scale = 1.0
    if num_x_labels > 10:
        font_scale *= 0.95
    if num_x_labels > 20:
        font_scale *= 0.95
    if num_series > 5:
        font_scale *= 0.95
    font_size = max(7, int(base_font * font_scale)) - 2

    effective_width_per_group = 1.30
    bar_width = effective_width_per_group / max(num_series, 1)
    if num_x_labels > 15:
        bar_width *= 0.92

    if num_x_labels > 20:
        gap_factor = 1.0
    elif num_x_labels > 10:
        gap_factor = 1.2
    elif num_x_labels < 5:
        gap_factor = 2.0
    else:
        gap_factor = base_gap

    return {
        'figsize': figsize,
        'font_size': font_size,
        'bar_width': bar_width,
        'gap_factor': gap_factor,
        'legend_fontsize': font_size,
        'label_fontsize': font_size,
        'legend_ncol': min(3, max(2, num_series // 2)),
        'show_value_labels': True,
    }


def _autocompute_y_break(series_list, gap_ratio=4.0):
    """Find a natural break in a set of bar values.

    Returns ``(low_max, high_min)`` if there's a gap of at least ``gap_ratio``x
    between two consecutive values across all series, else ``None``.
    """
    vals = []
    for s in series_list:
        for v in s.get('values', []):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v = float(v)
            if v > 0:
                vals.append(v)
    vals = sorted(set(vals))
    if len(vals) < 3:
        return None
    best_gap = 0.0
    best_pair = None
    for a, b in zip(vals, vals[1:]):
        if a <= 0:
            continue
        if b / a < gap_ratio:
            continue
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_pair = (a, b)
    if best_pair is None:
        return None
    a, b = best_pair
    # Pad the cut so the bars fully fit: bottom range goes up to ~1.15x the
    # largest "small" value, top range starts at ~0.85x the smallest "large".
    return (a * 1.15, b * 0.85)


def plot_bars_dynamic(series_list, x_labels, title, xlabel, ylabel,
                      output_dir=None, precision=1, show_value_labels=None,
                      log_scale=False, scientific_labels=False, max_width=None,
                      missing_label='OOM', y_break=None):
    """Plot N-series grouped bar chart with dynamically adjusted layout.

    series_list entries: dict with keys
        'label' (str), 'values' (list[float]), optional 'color', 'errors',
        optional 'short_label' (string shown under each bar; falls back to the
        token before the first ". " in 'label').
    A value of ``None`` or ``np.nan`` is treated as missing data (annotated
    with ``missing_label`` and rendered as a zero-height bar).

    ``y_break``: optional ``(low_max, high_min)`` tuple, or the string
    ``'auto'``. When set, the y-axis is "cut" between ``low_max`` and
    ``high_min`` so a small set of dominant outliers doesn't crush the rest of
    the data. The two segments use independent linear scales. Incompatible
    with ``log_scale=True`` (log_scale is silently ignored when broken).
    """
    if not series_list or x_labels is None:
        return

    num_series = len(series_list)
    num_x_labels = len(x_labels)
    params = calculate_plot_params(num_x_labels, num_series, max_width=max_width)

    if show_value_labels is None:
        show_value_labels = params['show_value_labels']

    if y_break == 'auto':
        y_break = _autocompute_y_break(series_list)
    if y_break is not None and log_scale:
        log_scale = False  # broken-axis layout always uses linear segments

    plt.rcParams.update({
        'font.size':         params['font_size'],
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'grid.alpha':        0.35,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.5,
        'axes.linewidth':    0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })

    x = np.arange(num_x_labels) * params['gap_factor']

    if y_break is not None:
        # Top axis is taller than bottom by default since the upper range is
        # usually wider in absolute terms than the bottom range.
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            figsize=params['figsize'],
            gridspec_kw={'height_ratios': [1.6, 1.0], 'hspace': 0.08},
        )
        bar_axes = [ax_top, ax_bot]
        ax_main = ax_bot  # axis that owns x ticks, OOM marks, short tags
    else:
        fig, ax = plt.subplots(figsize=params['figsize'])
        ax_top = ax_bot = ax_main = ax
        bar_axes = [ax]

    # Draw bars. When broken, the same bars are drawn on both axes; the ylim
    # on each axis hides the parts that don't belong to that segment.
    bars_handles = []  # list[BarContainer] over series, taken from ax_main
    series_max_values = []
    missing_marks = []

    for ax_idx, ax in enumerate(bar_axes):
        per_axis_handles = []
        for idx, series in enumerate(series_list):
            offset = (idx - (num_series - 1) / 2) * params['bar_width']
            color = series.get('color', default_colors[idx % len(default_colors)])
            label = series.get('label', f'Series {idx + 1}')
            raw_values = list(series.get('values', []))
            if len(raw_values) != num_x_labels:
                raw_values = (raw_values + [None] * num_x_labels)[:num_x_labels]

            plot_values = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                           for v in raw_values]

            # Only attach the legend label on one axis (top) so the auto legend
            # doesn't duplicate every series.
            label_for_axis = label if ax is ax_top else None
            bars = ax.bar(x + offset, plot_values, params['bar_width'],
                          color=color, alpha=0.88, label=label_for_axis,
                          edgecolor='black', linewidth=0.5)
            per_axis_handles.append(bars)

            if ax is ax_main:
                for j, v in enumerate(raw_values):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        missing_marks.append((x[j] + offset, params['bar_width']))
                if plot_values:
                    series_max_values.append(max(plot_values))
        if ax is ax_main:
            bars_handles = per_axis_handles

    # X-axis labels & xlabel: only on the bottom (or only) axis.
    rotation, ha = 0, 'center'
    if num_x_labels > 15:
        rotation, ha = 45, 'right'
    elif num_x_labels > 8:
        rotation, ha = 30, 'right'
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_labels, rotation=rotation, ha=ha,
                            fontsize=params['font_size'])
    ax_main.set_xlabel(xlabel, fontsize=params['label_fontsize'])

    # Y-axis: shared label between segments when broken; per-axis tick fontsize
    # and grid styling either way.
    for ax in bar_axes:
        ax.tick_params(axis='y', labelsize=params['font_size'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    if y_break is not None:
        # Use a figure-level supylabel so both segments share one label.
        fig.supylabel(ylabel, fontsize=params['label_fontsize'], x=0.02)
    else:
        ax_main.set_ylabel(ylabel, fontsize=params['label_fontsize'])

    if log_scale:
        ax_main.set_yscale('log')

    # Y-limits.
    overall_max = max(series_max_values) if series_max_values else 1.0
    overall_max = max(overall_max, 1e-6)
    if y_break is not None:
        low_max, high_min = y_break
        ax_top.set_ylim(high_min, overall_max * 1.05)
        ax_bot.set_ylim(0, low_max)
        # Hide the spines & tick line where the cut happens.
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        # Diagonal break marks (in axes coordinates so they stay attached to
        # the axis edge regardless of figure resize).
        d_x, d_y = 0.012, 0.020
        kwargs = dict(color='k', clip_on=False, linewidth=0.8)
        ax_top.plot((-d_x, +d_x), (-d_y, +d_y),
                    transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d_x, 1 + d_x), (-d_y, +d_y),
                    transform=ax_top.transAxes, **kwargs)
        ax_bot.plot((-d_x, +d_x), (1 - d_y, 1 + d_y),
                    transform=ax_bot.transAxes, **kwargs)
        ax_bot.plot((1 - d_x, 1 + d_x), (1 - d_y, 1 + d_y),
                    transform=ax_bot.transAxes, **kwargs)
    elif not log_scale:
        ax_main.set_ylim(0, overall_max * 1.10)

    # Legend: force two columns, placed just above ax_top with a small gap.
    legend_ncol = min(2, num_series)
    n_rows = int(np.ceil(num_series / legend_ncol))
    legend_y = 1.0 + 0.015 * n_rows

    ax_top.legend(loc='lower center',
                  bbox_to_anchor=(0.5, legend_y),
                  ncol=legend_ncol,
                  fontsize=params['legend_fontsize'],
                  frameon=False,
                  borderpad=0.3,
                  labelspacing=0.35,
                  handlelength=1.6,
                  handletextpad=0.5,
                  columnspacing=1.4)

    if show_value_labels:
        if num_x_labels > 25:
            y_offset = 0.0075 * overall_max
        elif num_x_labels > 15:
            y_offset = 0.010 * overall_max
        else:
            y_offset = 0.015 * overall_max

        # When broken, render each bar's value on whichever axis it fits in.
        for ax in bar_axes:
            ax_lo, ax_hi = ax.get_ylim()
            for series_handles in (
                    bars_handles if ax is ax_main
                    else [b for b in ax.containers if hasattr(b, 'patches')]):
                container = series_handles
                # ax.containers gives BarContainers which iterate over Rects
                for bar in container:
                    height = bar.get_height()
                    if height <= 0 or height < ax_lo or height > ax_hi:
                        continue
                    label_text = (f'{height:.1e}' if scientific_labels
                                  else f'{height:.{precision}f}')
                    y_pos = (height * 1.15 if log_scale
                             else height + y_offset)
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            y_pos, label_text,
                            ha='center', va='bottom',
                            fontsize=params['font_size'])

    # Annotate missing-data bars (red, rotated, just above the baseline).
    if missing_marks:
        if log_scale:
            ymin, _ = ax_main.get_ylim()
            ymin = max(ymin, 1e-12)
            text_y = ymin * 1.5
        else:
            ax_lo, ax_hi = ax_main.get_ylim()
            text_y = ax_lo + (ax_hi - ax_lo) * 0.02
        for xpos, _bw in missing_marks:
            ax_main.text(xpos, text_y, missing_label,
                         ha='center', va='bottom',
                         fontsize=max(params['font_size'] - 2, 6),
                         color='#b00020', rotation=90)

    # Per-bar short tags (e.g. "C1", "A2", "U0") placed just below the x-axis
    # baseline (between the bar and the OPT-... tick label).
    xaxis_trans = ax_main.get_xaxis_transform()
    short_fontsize = max(params['font_size'] - 3, 6)
    have_short_tags = False
    for idx, series in enumerate(series_list):
        short = series.get('short_label')
        if short is None:
            full = str(series.get('label', ''))
            short = full.split('.', 1)[0].strip()
        if not short:
            continue
        have_short_tags = True
        offset = (idx - (num_series - 1) / 2) * params['bar_width']
        color = series.get('color', default_colors[idx % len(default_colors)])
        for j in range(num_x_labels):
            ax_main.text(x[j] + offset, -0.02, short,
                         transform=xaxis_trans,
                         ha='center', va='top',
                         fontsize=short_fontsize,
                         color=color, fontweight='bold',
                         rotation=0, clip_on=False)

    if have_short_tags:
        ax_main.tick_params(axis='x', pad=short_fontsize + 6)
        ax_main.xaxis.labelpad = short_fontsize + 8

    if y_break is None:
        plt.tight_layout(pad=0.4)
    else:
        # tight_layout fights gridspec_kw['hspace']; set margins manually.
        fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.22,
                            hspace=0.08)

    # Sanitize the title into a filename: lowercase, spaces -> '_', and
    # strip shell/path-unfriendly punctuation like () , = so the resulting
    # files don't need quoting in the terminal.
    safe_title = title.lower().replace(' ', '_')
    for ch in '()[]{}<>,=:;\'"`!?':
        safe_title = safe_title.replace(ch, '')
    # Collapse any runs of underscores produced by stripping.
    while '__' in safe_title:
        safe_title = safe_title.replace('__', '_')
    safe_title = safe_title.strip('_')
    file_name = f"{safe_title}.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = file_name

    plt.savefig(file_path, bbox_inches='tight', format='pdf')
    # Also save a PNG for quick viewing in the IDE.
    plt.savefig(file_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print(f'Saved plot to {file_path}')
    plt.close()


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

# Last "Total: <num> Prefill: <num> Decode: <num>" line in each log
TOTAL_LINE_RE = re.compile(
    r'^Total:\s*([\d.eE+-]+)\s+Prefill:\s*([\d.eE+-]+)\s+Decode:\s*([\d.eE+-]+)'
)


def extract_times(log_path):
    """Return (total, prefill, decode) seconds from the *last* timing line."""
    last = None
    try:
        with open(log_path, 'r', errors='replace') as f:
            for line in f:
                m = TOTAL_LINE_RE.match(line)
                if m:
                    last = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    except FileNotFoundError:
        return None
    return last


def _load_or_none(path):
    """Read (total, prefill, decode) from a log, or return None if missing."""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return None
    times = extract_times(path)
    if times is None:
        print(f"Warning: no timing line found in: {path}")
    return times


def _load_series_data(log_paths, model_sizes, folder):
    """Return {model_size: (total, prefill, decode) | None} for one scheme.

    ``log_paths`` is a list of relative log paths aligned by index with
    ``model_sizes`` (use ``None`` for an entry that should be marked as
    missing). Each path is joined under ``folder`` and parsed for its
    last "Total/Prefill/Decode" line.
    """
    if len(log_paths) != len(model_sizes):
        raise ValueError(
            f"log_paths has {len(log_paths)} entries but model_sizes has "
            f"{len(model_sizes)}; they must be aligned by index."
        )
    out = {}
    for size, rel_path in zip(model_sizes, log_paths):
        if not rel_path:
            out[size] = None
            continue
        out[size] = _load_or_none(os.path.join(folder, rel_path))
    return out


def _values_for_metric(per_size_data, model_sizes, metric_idx):
    """Pull one metric (0=total, 1=prefill, 2=decode) from per-size data."""
    values = []
    for size in model_sizes:
        times = per_size_data.get(size)
        values.append(None if times is None else times[metric_idx])
    return values


# Match a row from the "PyTorch CUDA memory summary" table, e.g.:
#   | GPU reserved memory   |  82730 MiB |  82730 MiB |  82730 MiB |   0 B    |
# We capture the four "<value> <unit>" cells (cur / peak / total alloc / total
# freed) so callers can pick whichever column they care about.
_GPU_RESERVED_ROW_RE = re.compile(
    r'\|\s*GPU reserved memory\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
)

_UNIT_TO_MIB = {
    'B':   1.0 / (1024.0 * 1024.0),
    'KiB': 1.0 / 1024.0,
    'MiB': 1.0,
    'GiB': 1024.0,
    'TiB': 1024.0 * 1024.0,
    # Treat the bare metric prefixes the same as IEC. The PyTorch summary uses
    # IEC (MiB), but accept K/M/G as a defensive fallback.
    'KB':  1.0 / 1024.0,
    'MB':  1.0,
    'GB':  1024.0,
    'TB':  1024.0 * 1024.0,
}


def _parse_peak_gpu_reserved_mib(log_path):
    """Return the *peak* "GPU reserved memory" value (MiB) from a log.

    Reads the PyTorch CUDA memory summary table and returns the second
    column ("Peak Usage") of the "GPU reserved memory" row. Returns
    ``None`` if the table or the row is missing.

    If the log contains multiple summary tables (e.g. one per warmup +
    one per real run), the *last* one wins -- matching the behaviour of
    ``extract_times`` for timing lines.
    """
    if not log_path or not os.path.exists(log_path):
        return None
    last_peak_mib = None
    try:
        with open(log_path, 'r', errors='replace') as f:
            for line in f:
                if 'GPU reserved memory' not in line:
                    continue
                m = _GPU_RESERVED_ROW_RE.search(line)
                if not m:
                    continue
                # Groups: 1=cur_val, 2=cur_unit, 3=peak_val, 4=peak_unit, ...
                peak_val = float(m.group(3))
                peak_unit = m.group(4)
                factor = _UNIT_TO_MIB.get(peak_unit)
                if factor is None:
                    continue
                last_peak_mib = peak_val * factor
    except FileNotFoundError:
        return None
    return last_peak_mib


def compute_oversubscription_ratios(log_paths, model_sizes, folder,
                                    gpu_memory_gib):
    """Read peak "GPU reserved memory" from each log and divide by GPU size.

    ``log_paths`` is aligned by index with ``model_sizes``. Returns a list
    of ratios (peak / gpu_memory) -- ``None`` for entries whose log is
    missing or doesn't contain a memory summary table. Also returns the
    raw peak MiB list for diagnostics.

    Ratio > 1.0 means the run requested more GPU memory than the physical
    GPU has, i.e. it was oversubscribed (only meaningful when running on
    UVM-backed allocators that can spill to host memory).
    """
    if len(log_paths) != len(model_sizes):
        raise ValueError(
            f"log_paths has {len(log_paths)} entries but model_sizes has "
            f"{len(model_sizes)}; they must be aligned by index."
        )
    gpu_mib = float(gpu_memory_gib) * 1024.0
    peaks_mib = []
    ratios = []
    for size, rel_path in zip(model_sizes, log_paths):
        if not rel_path:
            peaks_mib.append(None)
            ratios.append(None)
            continue
        path = os.path.join(folder, rel_path)
        peak = _parse_peak_gpu_reserved_mib(path)
        if peak is None:
            print(f"Warning: no GPU memory summary in: {path}")
        peaks_mib.append(peak)
        ratios.append(None if peak is None else peak / gpu_mib)
    return ratios, peaks_mib


# --------------------------------------------------------------------------- #
# Inline configuration (mirrors plot_updated_vary_bsz.py::plot_GH200_cg1)
# --------------------------------------------------------------------------- #

def plot_GH200_opt_vary_model_size():
    """Plot total/prefill/decode time vs OPT model size on GH200.

    All log paths, legend labels, colors, and plotting parameters are
    defined inline below, mirroring the style of
    ``plot_updated_vary_bsz.py::plot_GH200_cg1``.

    The series order in every produced figure follows the insertion order
    of the ``series`` dict.
    """
    folder = "../vary_batch_size/huggingface/GH200/vary_model_size/opt"
    output_dir = f"{folder}/plots"
    machine_type = "GH200"
    prompt_len = 1920
    decode_len = 128
    bsz = 1
    log_scale = False

    # OPT model sizes (in plotting order, and in billions of parameters).
    # ``log_template`` lists for each scheme are aligned with this by index:
    # entry i is the log path (relative to ``folder``) for ``model_sizes[i]``.
    # Use ``None`` to mark a size as missing for that scheme.
    model_sizes = ['6.7b', '13b', '30b', '66b', '175b']

    series = {
        "no_uvm_dynamic": {
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "no_uvm_offloaded": {
            "label": "A. Application-level KV Cache Offload",
            "color": PALETTE[1],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-no_uvm-offloaded-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-offloaded-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-offloaded-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-no_uvm-offloaded-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-no_uvm-offloaded-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "uvm_managed": {
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "uvm_advise_prefetch": {
            "label": "U1. UVM w/ caching and prepopulate GPU memory (GPU_FIRST)",
            "color": PALETTE[7],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_advise_prefetch-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_advise_prefetch-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_advise_prefetch-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_advise_prefetch-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_advise_prefetch-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "uvm_advise_prefetch_discard": {
            "label": "U2. U1 + discard",
            "color": PALETTE[8],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_advise_prefetch_discard-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_advise_prefetch_discard-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_advise_prefetch_discard-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_advise_prefetch_discard-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_advise_prefetch_discard-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "uvm_gpu_ac": {
            "label": "U3. U1 + Access counter (GPU_AC)",
            "color": PALETTE[9],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_gpu_ac-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_gpu_ac-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_gpu_ac-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_gpu_ac-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-customcaching--no-preferred-loc-uvm_managed_gpu_ac-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
        "uvm_gpu_ac_discarduvm_gpu_ac_discard": {
            "label": "U4. Pytorch-UVM + Access counter",
            "color": PALETTE[10],
            "log_template": [
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_gpu_ac_discard-dynamic-opt-6.7b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_gpu_ac_discard-dynamic-opt-13b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_gpu_ac_discard-dynamic-opt-30b-prompt1920-20260420_001738.log",
                "py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_gpu_ac_discard-dynamic-opt-66b-prompt1920-20260420_001738.log",
                "debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-no-preferred-loc-uvm_managed_gpu_ac_discard-dynamic-opt-175b-prompt1920-20260420_121526.log",
            ],
        },
    }

    # Materialize per-size timings once per series (avoids re-parsing for each metric).
    for key, spec in series.items():
        spec["data"] = _load_series_data(spec["log_template"], model_sizes, folder)
        loaded = sum(1 for v in spec["data"].values() if v is not None)
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"loaded={loaded}/{len(model_sizes)} model sizes")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_map = {
        'total':   (0, 'Total Time (sec)',   'Total Run Time'),
        'prefill': (1, 'Prefill Time (sec)', 'Prefill Time'),
        'decode':  (2, 'Decode Time (sec)',  'Decode Time'),
    }

    x_labels = [f'OPT-{s}' for s in model_sizes]

    for metric, (metric_idx, ylabel, title_metric) in metric_map.items():
        series_list = []
        for spec in series.values():
            series_list.append({
                'label':  spec['label'],
                'color':  spec['color'],
                'values': _values_for_metric(spec["data"], model_sizes, metric_idx),
            })

        # Pretty diagnostic dump: scheme rows, model-size columns
        print(f'\n=== Metric: {metric} ===')
        col_w = 10
        label_w = 60
        header = ' ' * (label_w + 2) + ''.join(f'{f"OPT-{s}":>{col_w}}'
                                               for s in model_sizes)
        print(header)
        print(' ' * (label_w + 2) + '-' * (col_w * len(model_sizes)))
        for s in series_list:
            cells = []
            for v in s['values']:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cells.append(f'{"n/a":>{col_w}}')
                else:
                    cells.append(f'{v:>{col_w}.2f}')
            print(f'  {s["label"]:<{label_w}}' + ''.join(cells))

        title = (f'{title_metric} vs OPT Model Size on {machine_type} '
                 f'(prompt={prompt_len}, decode={decode_len}, bsz={bsz})')

        plot_bars_dynamic(series_list,
                          x_labels=x_labels,
                          title=title,
                          xlabel='OPT model size',
                          ylabel=ylabel,
                          output_dir=output_dir,
                          precision=1,
                          log_scale=log_scale)


def plot_GH200_64K_opt_vary_model_size(gpu_memory_gib=96):
    """Plot total/prefill/decode time vs OPT model size on GH200 (64K run).

    Sibling of ``plot_GH200_opt_vary_model_size``. Uses the
    ``20260423_005725`` sweep, which:
      * adds a new "C2. HF device_map='auto' (model-weights offload)" series
        compared to the original run,
      * drops the "A. Application-level KV Cache Offload" series, and
      * uses the newer customcaching log naming (no ``-no-preferred-loc-``
        infix).

    The "64K" tag in the function name is the run label, not a context
    length: the underlying configuration is still ``prompt=1920``,
    ``decode=128``, ``bsz=1`` (verified from log contents).

    ``gpu_memory_gib`` is the physical GPU HBM size used to annotate each
    x-tick with the U1 oversubscription ratio (peak GPU reserved memory /
    GPU memory). Pass 0 (or None) to skip the annotation.
    """
    folder = "../vary_batch_size/huggingface/GH200/vary_model_size/opt"
    output_dir = f"{folder}/plots_64K"
    machine_type = "GH200"
    prompt_len = 1920
    decode_len = 128
    bsz = 1
    log_scale = False

    model_sizes = ['6.7b', '13b', '30b', '66b', '175b']

    ts = "20260423_005725"

    series = {
        "no_uvm_dynamic": {
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-opt-{s}-prompt1920-{ts}.log"
                for s in model_sizes
            ],
        },
        "no_uvm_devmap_auto": {
            "label": "A2. Application-level model weights offload",
            "color": PALETTE[1],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-devmap_auto-opt-{s}-prompt1920-{ts}.log"
                for s in model_sizes
            ],
        },
        "uvm_managed": {
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-opt-{s}-prompt1920-{ts}.log"
                for s in model_sizes
            ],
        },
        # "uvm_advise_prefetch": {
        #     "label": "U1. UVM w/ caching and prepopulate GPU memory",
        #     "color": PALETTE[7],
        #     "log_template": [
        #         f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_advise_prefetch-dynamic-opt-{s}-prompt1920-{ts}.log"
        #         for s in model_sizes
        #     ],
        # },
        "uvm_advise_prefetch_discard": {
            "label": "U1. Pytorch-UVM",
            "color": PALETTE[8],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-uvm_managed_advise_prefetch_discard-dynamic-opt-{s}-prompt1920-{ts}.log"
                for s in model_sizes
            ],
        },
        # "uvm_gpu_ac": {
        #     "label": "U3. U1 + Access counter (GPU_AC)",
        #     "color": PALETTE[9],
        #     "log_template": [
        #         f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_gpu_ac-dynamic-opt-{s}-prompt1920-{ts}.log"
        #         for s in model_sizes
        #     ],
        # },
        "uvm_gpu_ac_discard": {
            "label": "U2. Pytorch-UVM + Access counter",
            "color": PALETTE[10],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-uvm_managed_gpu_ac_discard-dynamic-opt-{s}-prompt1920-{ts}.log"
                for s in model_sizes
            ],
        },
    }

    for key, spec in series.items():
        spec["data"] = _load_series_data(spec["log_template"], model_sizes, folder)
        loaded = sum(1 for v in spec["data"].values() if v is not None)
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"loaded={loaded}/{len(model_sizes)} model sizes")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_map = {
        'total':   (0, 'Total Time (sec)',   'Total Run Time'),
        'prefill': (1, 'Prefill Time (sec)', 'Prefill Time'),
        'decode':  (2, 'Decode Time (sec)',  'Decode Time'),
    }

    # Augment x-tick labels with the U1 oversubscription ratio
    # ``peak_GPU_reserved / gpu_memory`` so the runtime cliff in the bars
    # is paired with the memory pressure that caused it. We reuse the U1
    # series' log paths (already defined above) instead of duplicating the
    # template here.
    x_labels = [f'OPT-{s}' for s in model_sizes]
    if gpu_memory_gib:
        u1_logs = series['uvm_advise_prefetch_discard']['log_template']
        ratios, _peaks = compute_oversubscription_ratios(
            u1_logs, model_sizes, folder, gpu_memory_gib)
        x_labels = [
            f'{base}\n{r:.2f}\u00d7' if r is not None else base
            for base, r in zip(x_labels, ratios)
        ]

    for metric, (metric_idx, ylabel, title_metric) in metric_map.items():
        series_list = []
        for spec in series.values():
            series_list.append({
                'label':  spec['label'],
                'color':  spec['color'],
                'values': _values_for_metric(spec["data"], model_sizes, metric_idx),
            })

        print(f'\n=== Metric: {metric} (64K run) ===')
        col_w = 10
        label_w = 60
        header = ' ' * (label_w + 2) + ''.join(f'{f"OPT-{s}":>{col_w}}'
                                               for s in model_sizes)
        print(header)
        print(' ' * (label_w + 2) + '-' * (col_w * len(model_sizes)))
        for s in series_list:
            cells = []
            for v in s['values']:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cells.append(f'{"n/a":>{col_w}}')
                else:
                    cells.append(f'{v:>{col_w}.2f}')
            print(f'  {s["label"]:<{label_w}}' + ''.join(cells))

        title = (f'{title_metric} vs OPT Model Size on {machine_type} 64K '
                 f'(prompt={prompt_len}, decode={decode_len}, bsz={bsz})')

        plot_bars_dynamic(series_list,
                          x_labels=x_labels,
                          title=title,
                          xlabel='OPT model size '
                                 f'(2nd line: Oversubscription ratio)',
                          ylabel=ylabel,
                          output_dir=output_dir,
                          precision=1,
                          show_value_labels=False,
                          log_scale=log_scale,
                          y_break='auto')


def plot_GH200_64K_qwen_vary_model_size(gpu_memory_gib=96):
    """Plot total/prefill/decode time vs Qwen model size on GH200 (64K run).

    Sibling of ``plot_GH200_64K_opt_vary_model_size`` for the Qwen sweep
    produced by ``vary_model_size_qwen.sh``. Same five schemes (C1, A2, U0,
    U1, U2). Model sizes are Qwen2.5-{7B,14B,32B,72B} plus the lone dense
    Qwen1.5-110B at the high end. Underlying configuration: ``prompt=1920``,
    ``decode=128``, ``bsz=1`` (verified from the log contents).
    """
    folder = "../vary_batch_size/huggingface/GH200/vary_model_size/qwen"
    output_dir = f"{folder}/plots_64K"
    machine_type = "GH200"
    prompt_len = 1920
    decode_len = 128
    bsz = 1
    log_scale = False

    # ``model_sizes`` doubles as both the in-filename token and the x-axis
    # label here, since each Qwen entry already carries its family + size.
    model_sizes = [
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-32B",
        "Qwen2.5-72B",
        "Qwen1.5-110B",
    ]

    ts = "20260423_053434"

    series = {
        "no_uvm_dynamic": {
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-{m}-prompt1920-{ts}.log"
                for m in model_sizes
            ],
        },
        # py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-devmap_auto-Qwen2.5-72B-prompt1920-20260424_125403.log
        "no_uvm_devmap_auto": {
            "label": "A2. Application-level model weights offload",
            "color": PALETTE[1],
            "log_template": [
                # All entries except 72B use the normal template
                f"debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-devmap_auto-{m}-prompt1920-{ts}.log"
                if m != "Qwen2.5-72B"
                # 72B has a special timestamped log
                else "debug/py313_pytorch_cuda13-cuda-default-no_uvm-dynamic-devmap_auto-Qwen2.5-72B-prompt1920-20260424_125403.log"
                for m in model_sizes
            ],
        },
        "uvm_managed": {
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-uvm_managed-dynamic-{m}-prompt1920-{ts}.log"
                for m in model_sizes
            ],
        },
        # "uvm_advise_prefetch": {
        #     "label": "U1. UVM w/ caching and prepopulate GPU memory",
        #     "color": PALETTE[7],
        #     "log_template": [
        #         f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_advise_prefetch-dynamic-{m}-prompt1920-{ts}.log"
        #         for m in model_sizes
        #     ],
        # },
        "uvm_advise_prefetch_discard": {
            "label": "U1. Pytorch-UVM",
            "color": PALETTE[8],
            "log_template": [
                f"debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-uvm_managed_advise_prefetch_discard-dynamic-{m}-prompt1920-{ts}.log"
                for m in model_sizes
            ],
        },
        # "uvm_gpu_ac": {
        #     "label": "U3. U1 + Access counter (GPU_AC)",
        #     "color": PALETTE[9],
        #     "log_template": [
        #         f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_gpu_ac-dynamic-{m}-prompt1920-{ts}.log"
        #         for m in model_sizes
        #     ],
        # },
        # "uvm_gpu_ac_discard": {
        #     "label": "U2. Pytorch-UVM + Access counter",
        #     "color": PALETTE[10],
        #     "log_template": [
        #         f"debug/py313_pytorch_cuda13-cuda-default-customcaching-discard=Standard-uvm_managed_gpu_ac_discard-dynamic-{m}-prompt1920-{ts}.log"
        #         for m in model_sizes
        #     ],
        # },
    }

    for key, spec in series.items():
        spec["data"] = _load_series_data(spec["log_template"], model_sizes, folder)
        loaded = sum(1 for v in spec["data"].values() if v is not None)
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"loaded={loaded}/{len(model_sizes)} model sizes")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_map = {
        'total':   (0, 'Total Time (sec)',   'Total Run Time'),
        'prefill': (1, 'Prefill Time (sec)', 'Prefill Time'),
        'decode':  (2, 'Decode Time (sec)',  'Decode Time'),
    }

    # Augment x-tick labels with the U1 oversubscription ratio so the
    # runtime cliff in the bars is paired with the memory pressure that
    # caused it. See ``plot_GH200_64K_opt_vary_model_size`` for the rationale.
    x_labels = list(model_sizes)
    if gpu_memory_gib:
        u1_logs = series['uvm_advise_prefetch_discard']['log_template']
        ratios, _peaks = compute_oversubscription_ratios(
            u1_logs, model_sizes, folder, gpu_memory_gib)
        x_labels = [
            f'{base}\n{r:.2f}\u00d7' if r is not None else base
            for base, r in zip(x_labels, ratios)
        ]

    for metric, (metric_idx, ylabel, title_metric) in metric_map.items():
        series_list = []
        for spec in series.values():
            series_list.append({
                'label':  spec['label'],
                'color':  spec['color'],
                'values': _values_for_metric(spec["data"], model_sizes, metric_idx),
            })

        print(f'\n=== Metric: {metric} (Qwen 64K run) ===')
        col_w = 14
        label_w = 60
        header = ' ' * (label_w + 2) + ''.join(f'{m:>{col_w}}'
                                               for m in model_sizes)
        print(header)
        print(' ' * (label_w + 2) + '-' * (col_w * len(model_sizes)))
        for s in series_list:
            cells = []
            for v in s['values']:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    cells.append(f'{"n/a":>{col_w}}')
                else:
                    cells.append(f'{v:>{col_w}.2f}')
            print(f'  {s["label"]:<{label_w}}' + ''.join(cells))

        title = (f'{title_metric} vs Qwen Model Size on {machine_type} 64K '
                 f'(prompt={prompt_len}, decode={decode_len}, bsz={bsz})')

        plot_bars_dynamic(series_list,
                          x_labels=x_labels,
                          title=title,
                          xlabel='Qwen model '
                                 f'(2nd line: Oversubscription ratio)',
                          ylabel=ylabel,
                          output_dir=output_dir,
                          precision=1,
                          show_value_labels=False,
                          log_scale=log_scale,
                          y_break='auto')


def plot_GH200_64K_opt_oversubscription(gpu_memory_gib=96):
    """Bar chart of (peak GPU reserved memory) / (GPU memory) for the OPT
    sweep, parsed from the U1 (uvm_advise_prefetch) logs.

    The U1 scheme is the natural source for this metric because it runs at
    every model size (so we get a complete sweep) *and* it lets the
    allocator request memory beyond physical GPU capacity (UVM-backed),
    so the "peak GPU reserved" value reflects the model+KV-cache working
    set, not a hard cudaMalloc cap.

    ``gpu_memory_gib`` is the physical GPU HBM size used as the
    denominator (default: 96, matching GH200 SXM).
    """
    folder = "../vary_batch_size/huggingface/GH200/vary_model_size/opt"
    output_dir = f"{folder}/plots_64K"
    machine_type = "GH200"
    prompt_len = 1920
    decode_len = 128
    bsz = 1
    ts = "20260423_005725"
    model_sizes = ['6.7b', '13b', '30b', '66b', '175b']

    log_paths = [
        f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_advise_prefetch-dynamic-opt-{s}-prompt1920-{ts}.log"
        for s in model_sizes
    ]

    ratios, peaks_mib = compute_oversubscription_ratios(
        log_paths, model_sizes, folder, gpu_memory_gib)

    print(f'\n=== Oversubscription ratio (OPT, U1, GPU={gpu_memory_gib} GiB) ===')
    col_w = 12
    for s, peak, ratio in zip(model_sizes, peaks_mib, ratios):
        peak_str = (f'{peak/1024:.1f} GiB' if peak is not None else 'n/a')
        ratio_str = (f'{ratio:.2f}x' if ratio is not None else 'n/a')
        print(f'  OPT-{s:<10}  peak={peak_str:>12}   '
              f'oversub={ratio_str:>8}')

    series_list = [{
        'label': f'U1. Peak GPU reserved / {gpu_memory_gib} GiB',
        'short_label': '',  # don't draw a per-bar tag (single series)
        'color': PALETTE[7],
        'values': ratios,
    }]
    x_labels = [f'OPT-{s}' for s in model_sizes]
    title = (f'Oversubscription Ratio (U1) vs OPT Model Size on {machine_type} 64K '
             f'(GPU={gpu_memory_gib} GiB, prompt={prompt_len}, '
             f'decode={decode_len}, bsz={bsz})')

    plot_bars_dynamic(
        series_list,
        x_labels=x_labels,
        title=title,
        xlabel='OPT model size',
        ylabel='Peak GPU reserved / GPU memory',
        output_dir=output_dir,
        precision=2,
        show_value_labels=True,
        log_scale=False,
    )


def plot_GH200_64K_qwen_oversubscription(gpu_memory_gib=96):
    """Qwen counterpart of ``plot_GH200_64K_opt_oversubscription``."""
    folder = "../vary_batch_size/huggingface/GH200/vary_model_size/qwen"
    output_dir = f"{folder}/plots_64K"
    machine_type = "GH200"
    prompt_len = 1920
    decode_len = 128
    bsz = 1
    ts = "20260423_053434"
    model_sizes = [
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-32B",
        "Qwen2.5-72B",
        "Qwen1.5-110B",
    ]

    log_paths = [
        f"debug/py313_pytorch_cuda13-cuda-default-customcaching--uvm_managed_advise_prefetch-dynamic-{m}-prompt1920-{ts}.log"
        for m in model_sizes
    ]

    ratios, peaks_mib = compute_oversubscription_ratios(
        log_paths, model_sizes, folder, gpu_memory_gib)

    print(f'\n=== Oversubscription ratio (Qwen, U1, GPU={gpu_memory_gib} GiB) ===')
    for m, peak, ratio in zip(model_sizes, peaks_mib, ratios):
        peak_str = (f'{peak/1024:.1f} GiB' if peak is not None else 'n/a')
        ratio_str = (f'{ratio:.2f}x' if ratio is not None else 'n/a')
        print(f'  {m:<14}  peak={peak_str:>12}   '
              f'oversub={ratio_str:>8}')

    series_list = [{
        'label': f'U1. Peak GPU reserved / {gpu_memory_gib} GiB',
        'short_label': '',
        'color': PALETTE[7],
        'values': ratios,
    }]
    x_labels = list(model_sizes)
    title = (f'Oversubscription Ratio (U1) vs Qwen Model Size on {machine_type} 64K '
             f'(GPU={gpu_memory_gib} GiB, prompt={prompt_len}, '
             f'decode={decode_len}, bsz={bsz})')

    plot_bars_dynamic(
        series_list,
        x_labels=x_labels,
        title=title,
        xlabel='Qwen model',
        ylabel='Peak GPU reserved / GPU memory',
        output_dir=output_dir,
        precision=2,
        show_value_labels=True,
        log_scale=False,
    )


if __name__ == '__main__':
    # plot_GH200_opt_vary_model_size()
    # plot_GH200_64K_opt_vary_model_size(gpu_memory_gib=96)
    plot_GH200_64K_qwen_vary_model_size(gpu_memory_gib=96)
    # plot_GH200_64K_opt_oversubscription(gpu_memory_gib=96)
    # plot_GH200_64K_qwen_oversubscription(gpu_memory_gib=96)
