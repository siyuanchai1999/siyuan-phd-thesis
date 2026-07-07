import base64
import re

import matplotlib.pyplot as plt
import numpy as np
import os



ERR_GPU_OOM=-1
ERR_OTHER=-2

# Curated palette of distinct, colorblind-aware colors.
# Order is chosen so adjacent indices stay visually separable in bar plots.
# The first 7 entries are the Okabe-Ito palette (colorblind-safe); the rest
# are Tableau / Flat-UI accents that pair well with them.
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


# --------------------------------------------------------------------------- #
# GPU memory parsing (ported from plot_vary_model_size.py)
# --------------------------------------------------------------------------- #

_GPU_RESERVED_ROW_RE = re.compile(
    r'\|\s*GPU reserved memory\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
    r'\s*([\d.]+)\s+([KMGT]?i?B)\s*\|'
)

_ACTIVE_MEMORY_ROW_RE = re.compile(
    r'\|\s*Active memory\s*\|'
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
    'KB':  1.0 / 1024.0,
    'MB':  1.0,
    'GB':  1024.0,
    'TB':  1024.0 * 1024.0,
}


def _parse_per_batch_peak_memory(file_path, row_re):
    """Parse peak memory (MiB) per batch size for the given row regex.

    The regex must capture the four ``Cur Usage / Peak Usage / Tot Alloc /
    Tot Freed`` cells (value + unit) from the PyTorch CUDA summary table.
    Returns a list of floats (or ``None``) aligned with the batch sizes
    found by ``read_data`` on the same file.
    """
    if not file_path or not os.path.exists(file_path):
        return []

    batch_peaks = []
    current_peak = None

    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            if "input:" in line and "bsz:" in line:
                if current_peak is not None or batch_peaks:
                    batch_peaks.append(current_peak)
                current_peak = None
            m = row_re.search(line)
            if m:
                peak_val = float(m.group(3))
                peak_unit = m.group(4)
                factor = _UNIT_TO_MIB.get(peak_unit)
                if factor is not None:
                    current_peak = peak_val * factor
    if current_peak is not None or batch_peaks:
        batch_peaks.append(current_peak)

    return batch_peaks


def parse_per_batch_peak_gpu_memory(file_path):
    """Parse peak GPU reserved memory (MiB) per batch size from a log file."""
    return _parse_per_batch_peak_memory(file_path, _GPU_RESERVED_ROW_RE)


def parse_per_batch_peak_active_memory(file_path):
    """Parse peak active memory (MiB) per batch size from a log file."""
    return _parse_per_batch_peak_memory(file_path, _ACTIVE_MEMORY_ROW_RE)


def read_data(file_path):
    batch_sizes = []
    total_times = []
    prefill_times = []
    decode_times = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        processing_one_batch = False
        for line in lines:
            if "input:" in line and "bsz:" in line:

                if processing_one_batch:
                    total_times.append(ERR_OTHER)
                    prefill_times.append(ERR_OTHER)
                    decode_times.append(ERR_OTHER)
                    processing_one_batch = False
                    # continue

                parts = line.split()
                bsz_index = parts.index("bsz:")
                batch_size = int(parts[bsz_index + 1])
                batch_sizes.append(batch_size)
                processing_one_batch = True
            if processing_one_batch:
                if "Total:" in line:
                    parts = line.split()
                    total_index = parts.index("Total:")
                    if total_index != -1:
                        total_time = float(parts[total_index + 1])
                        total_times.append(total_time)
                        processing_one_batch = False
                    
                if "Prefill:" in line:
                    parts = line.split()
                    prefill_index = parts.index("Prefill:")
                    if prefill_index != -1:
                        prefill_time = float(parts[prefill_index + 1])
                        prefill_times.append(prefill_time)
                if "Decode:" in line:
                    parts = line.split()
                    decode_index = parts.index("Decode:")
                    if decode_index != -1:
                        decode_time = float(parts[decode_index + 1])
                        decode_times.append(decode_time)
                        
                if "CUDA out of memory" in line:
                    total_times.append(ERR_GPU_OOM)
                    prefill_times.append(ERR_GPU_OOM)
                    decode_times.append(ERR_GPU_OOM)
                    processing_one_batch = False

    return batch_sizes, total_times, prefill_times, decode_times



def pad_data(data, target_length):
    batch_sizes, total_times, prefill_times, decode_times = data
    pad_length = target_length - len(batch_sizes)
    if pad_length > 0:
        batch_sizes = np.pad(batch_sizes, (0, pad_length), constant_values=0)

    pad_length = target_length - len(total_times)
    if pad_length > 0:
        total_times = np.pad(total_times, (0, pad_length), constant_values=0)
    
    pad_length = target_length - len(prefill_times)
    if pad_length > 0:
        prefill_times = np.pad(prefill_times, (0, pad_length), constant_values=0)
    
    pad_length = target_length - len(decode_times)
    if pad_length > 0:
        decode_times = np.pad(decode_times, (0, pad_length), constant_values=0)
    
    return batch_sizes, total_times, prefill_times, decode_times


def plot_decode_times_combined(uvm_data, cudaMalloc_data, cudaMallocPluggable_data, ManagedPrefetchPluggable_data):
    batch_sizes_uvm, _, _, decode_times_uvm = uvm_data
    batch_sizes_ManagedPrefetchPluggable, _, _, decode_times_ManagedPrefetchPluggable = ManagedPrefetchPluggable_data
    
    # Determine the maximum length for padding
    max_length = len(batch_sizes_uvm)
    
    # Pad cudaMallocPluggable and ManagedPrefetchPluggable data if necessary
    batch_sizes_cudaMallocPluggable, _, _, decode_times_cudaMallocPluggable = pad_data(cudaMallocPluggable_data, max_length)
    batch_sizes_cudaMalloc, _, _, decode_times_cudaMalloc = pad_data(cudaMalloc_data, max_length)
    
    # Convert batch sizes to string for better x-axis labeling
    batch_sizes_str = [str(bsz) for bsz in batch_sizes_uvm]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color for decode
    color = '#56B4E9'  # light blue

    # Plot bars for decode times
    bar_width = 0.2
    r1 = np.arange(len(batch_sizes_uvm))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    p1 = ax.bar(r1, decode_times_cudaMalloc, bar_width, label='Decode Time (sec)', color=color)
    p2 = ax.bar(r2, decode_times_cudaMallocPluggable, bar_width, color=color)
    p3 = ax.bar(r3, decode_times_ManagedPrefetchPluggable, bar_width, color=color)
    p4 = ax.bar(r4, decode_times_uvm, bar_width, color=color)

    # Adding labels and title
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (sec)')
    ax.set_title('Decode Time Analysis')
    ax.set_xticks([r + 1.5 * bar_width for r in range(len(batch_sizes_str))])
    ax.set_xticklabels(batch_sizes_str)
    
    # Add legend
    ax.legend()

    # Mark the data sources in the x-axis
    for i in range(len(batch_sizes_str)):
        ax.text(r1[i], -0.2 * max(decode_times_uvm), 'cudaMalloc', ha='center', rotation=45)
        ax.text(r2[i], -0.2 * max(decode_times_uvm), 'cudaMalloc Plug', ha='center', rotation=45)
        ax.text(r3[i], -0.2 * max(decode_times_uvm), 'UVM Prefetch', ha='center', rotation=45)
        ax.text(r4[i], -0.2 * max(decode_times_uvm), 'UVM', ha='center', rotation=45)

    # Push the x-ticks further down
    ax.tick_params(axis='x', pad=100)
    
    # Adding the decode time on top of each bar
    for i, total in enumerate(decode_times_cudaMalloc):
        ax.text(r1[i], total + 0.05, f'{total:.1f}', ha='center')

    for i, total in enumerate(decode_times_cudaMallocPluggable):
        ax.text(r2[i], total + 0.05, f'{total:.1f}', ha='center')

    for i, total in enumerate(decode_times_ManagedPrefetchPluggable):
        ax.text(r3[i], total + 0.05, f'{total:.1f}', ha='center')

    for i, total in enumerate(decode_times_uvm):
        ax.text(r4[i], total + 0.05, f'{total:.1f}', ha='center')

    plt.savefig('decode_time_absolute.png', bbox_inches='tight', dpi=300)
    print("Plot saved as 'decode_time_absolute.png'")

def checkGPUOOM(label, error_type):

    if error_type == ERR_GPU_OOM:
        return True
    
    if "all on GPU" in label or "cudaMalloc" in label or "cudamalloc" in label:
        return True
    GPU_OOM_list = [
        'F0. FlexGen all on GPU',
        'F1. FlexGen KV Cache on CPU',
        'C0. CUDACachingAlloc (native)',
        'P0. Pluggable cudaMalloc',
        'PM0. Pluggable Mempool cudaMalloc'
    ]

    if label in GPU_OOM_list:
        return True

def _detect_break_range(values, min_ratio=3.0):
    """Detect a meaningful gap in positive values for a broken y-axis.

    Returns ``(lower_max, upper_min, upper_max)`` if a break is warranted,
    else ``None``. The largest gap (by ratio) where the higher value is at
    least ``min_ratio`` times the lower value is selected.
    """
    pos = sorted([v for v in values if v > 0], reverse=True)
    if len(pos) < 3:
        return None
    best_idx = -1
    best_ratio = min_ratio
    for i in range(len(pos) - 1):
        if pos[i + 1] <= 0:
            continue
        r = pos[i] / pos[i + 1]
        if r > best_ratio:
            best_ratio = r
            best_idx = i
    if best_idx < 0:
        return None
    upper_max = pos[0]
    upper_min = pos[best_idx]
    lower_max = pos[best_idx + 1]
    return lower_max, upper_min, upper_max


def plot_backbone(batch_sizes_str, data_list, x_label, y_label, title, filename,
    colors, labels, sub_labels,
    precision=1, show_legend=True, broken_y=False, broken_y_min_ratio=3.0,
    timeout_value=None, y_max_cap=None, bar_width_scale=1.0):

    num_datasets = len(data_list)
    num_batches = len(batch_sizes_str)

    # ---- dynamic sizing (adapted from plot_vary_model_size.py) ----
    # Target the LaTeX text column so the 12pt plot font renders at body
    # (12pt) size when the figure is included at its natural width.
    # ``TARGET_WIDTH_IN`` is the physical width of the saved figure; keep it
    # at/below \linewidth (6.5in) so no LaTeX down-scaling shrinks the text.
    # The broken-y (dual-panel) layout crops to a narrower tight bbox, so it
    # needs a larger figure width to end up at the same natural width (and
    # hence the same rendered font at \linewidth) as the single-axis figures.
    TARGET_WIDTH_IN = 6.9 if broken_y else 6.5
    height_scale = max(1.0, min(2.0, 1.0 + (num_datasets - 2) * 0.1))
    # Shorter panels so the two stacked subfigures (small/large) don't fill a
    # whole page; width (and hence rendered font size) is unchanged. Keep the
    # taller panel for broken-y figures, which need room for two y-segments.
    # The single-axis panels need enough height for the (vertically centered)
    # y-axis label -- "Total Time (sec)" is ~1.6in tall when rotated -- to fit
    # within the axes; otherwise its top pokes up and collides with the legend.
    figsize = (TARGET_WIDTH_IN, (3.2 if broken_y else 2.9) * height_scale)

    font_size = 12
    if num_batches > 10:
        font_size -= 1
    if num_datasets > 5:
        font_size -= 1
    font_size = max(7, font_size)

    bar_width = 1.30 / max(num_datasets, 1) * bar_width_scale
    gap_factor = 2.0 if num_batches < 5 else 1.5

    # ---- publication-quality rcParams ----
    plt.rcParams.update({
        'font.size':         font_size,
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

    # ---- compute global max & decide whether to break ----
    # Treat any value at or above ``timeout_value`` as the cap so it doesn't
    # dominate axis scaling -- they will be drawn clipped at the cap and
    # annotated with "Timeout".
    def _effective(v):
        if timeout_value is not None and v >= timeout_value and y_max_cap is not None:
            return y_max_cap
        return v

    max_value = 1.0
    for data in data_list:
        if len(data) > 0:
            valid = [_effective(v) for v in data if v > 0]
            if valid:
                max_value = max(max_value, max(valid))
    if y_max_cap is not None:
        max_value = min(max_value, y_max_cap)

    break_info = None
    if broken_y:
        all_vals = [_effective(v) for data in data_list for v in data if v > 0]
        break_info = _detect_break_range(all_vals, min_ratio=broken_y_min_ratio)

    x = np.arange(num_batches) * gap_factor
    short_fontsize = max(font_size - 3, 6)

    if break_info is None:
        # ============ single-axis (default) path ============
        fig, ax = plt.subplots(figsize=figsize)

        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            ax.bar(x + offset, data_list[i], bar_width,
                   label=labels[i], color=colors[i],
                   alpha=0.88, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes_str, fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if show_legend:
            max_label_len = max((len(l) for l in labels), default=10)
            char_w_in = 0.058 * (font_size / 10.0)
            per_col_in = max_label_len * char_w_in + 0.45
            # Round (not floor) so a legend that needs ~1.9 columns still lays
            # out in 2 columns rather than collapsing into a tall single column.
            legend_ncol = max(1, int(round(figsize[0] / per_col_in)))
            legend_ncol = min(legend_ncol, num_datasets)
            n_rows = int(np.ceil(num_datasets / legend_ncol))
            legend_y = 1.02 + 0.06 * n_rows
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, legend_y),
                      ncol=legend_ncol, fontsize=font_size,
                      frameon=False, borderpad=0.3, labelspacing=0.35,
                      handlelength=1.6, handletextpad=0.5, columnspacing=1.4)

        y_offset = 0.015 * max_value
        timeout_y = max_value * 0.55
        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            for j, val in enumerate(data_list[i]):
                xpos = x[j] + offset
                if val <= 0:
                    ax.text(xpos, y_offset, 'OOM',
                            ha='center', va='bottom', fontsize=max(font_size - 2, 6),
                            color='#b00020', rotation=90)
                elif timeout_value is not None and val >= timeout_value:
                    ax.text(xpos, timeout_y, 'Timeout 3h',
                            ha='center', va='center',
                            fontsize=max(font_size - 2, 6),
                            color='#b00020', rotation=90, fontweight='bold')

        xaxis_trans = ax.get_xaxis_transform()
        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            for j in range(num_batches):
                ax.text(x[j] + offset, -0.02, sub_labels[i],
                        transform=xaxis_trans,
                        ha='center', va='top', fontsize=short_fontsize,
                        color=colors[i], fontweight='bold', clip_on=False)
        ax.tick_params(axis='x', pad=short_fontsize + 6)
        ax.xaxis.labelpad = short_fontsize + 8

        if y_max_cap is not None:
            ax.set_ylim(0, y_max_cap)
        else:
            ax.set_ylim(0, max_value * 1.10)
        plt.tight_layout(pad=0.4)

    else:
        # ============ broken-y-axis path ============
        lower_max, upper_min, upper_max = break_info

        # Pad the visible regions a little.
        lower_top = lower_max * 1.15
        upper_bot = upper_min * 0.92
        upper_top = upper_max * 1.05
        if y_max_cap is not None:
            upper_top = min(upper_top, y_max_cap)
            if upper_bot >= upper_top:
                upper_bot = upper_top * 0.85

        # Allocate vertical space between the two axes proportional to data
        # ranges (with sensible bounds so the upper panel is always visible).
        upper_span = max(upper_top - upper_bot, 1.0)
        lower_span = max(lower_top, 1.0)
        ratio_top = upper_span / (upper_span + lower_span)
        ratio_top = float(np.clip(ratio_top, 0.30, 0.55))
        ratio_bot = 1.0 - ratio_top

        fig_w, fig_h = figsize
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True,
            figsize=(fig_w, fig_h * 1.20),
            gridspec_kw={'height_ratios': [ratio_top, ratio_bot], 'hspace': 0.07},
        )

        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            ax_top.bar(x + offset, data_list[i], bar_width,
                       label=labels[i], color=colors[i],
                       alpha=0.88, edgecolor='black', linewidth=0.5)
            ax_bot.bar(x + offset, data_list[i], bar_width,
                       label=labels[i], color=colors[i],
                       alpha=0.88, edgecolor='black', linewidth=0.5)

        ax_top.set_ylim(upper_bot, upper_top)
        ax_bot.set_ylim(0, lower_top)

        # Hide the spines between the two axes and the top axis's x ticks.
        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Diagonal break marks at the gap between panels.
        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color='k',
                      clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs2 = dict(transform=ax_bot.transAxes, color='k',
                       clip_on=False, linewidth=1.0)
        d2 = d * (ratio_top / ratio_bot)
        ax_bot.plot((-d, +d), (1 - d2, 1 + d2), **kwargs2)
        ax_bot.plot((1 - d, 1 + d), (1 - d2, 1 + d2), **kwargs2)

        # X-axis labels / ticks (only on the bottom axis).
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(batch_sizes_str, fontsize=font_size)
        ax_bot.set_xlabel(x_label, fontsize=font_size)

        # Y-axis label centered across both panels.
        fig.supylabel(y_label, fontsize=font_size)

        ax_top.tick_params(axis='y', labelsize=font_size)
        ax_bot.tick_params(axis='y', labelsize=font_size)
        ax_top.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bot.grid(axis='y', linestyle='--', alpha=0.7)

        # Legend above the upper panel.
        if show_legend:
            max_label_len = max((len(l) for l in labels), default=10)
            char_w_in = 0.058 * (font_size / 10.0)
            per_col_in = max_label_len * char_w_in + 0.45
            legend_ncol = max(1, int(round(fig_w / per_col_in)))
            legend_ncol = min(legend_ncol, num_datasets)
            n_rows = int(np.ceil(num_datasets / legend_ncol))
            legend_y = 1.02 + 0.08 * n_rows
            ax_top.legend(loc='lower center', bbox_to_anchor=(0.5, legend_y),
                          ncol=legend_ncol, fontsize=font_size,
                          frameon=False, borderpad=0.3, labelspacing=0.35,
                          handlelength=1.6, handletextpad=0.5, columnspacing=1.4)

        # OOM annotations on the bottom axis (lower-scale offset).
        y_offset = 0.015 * lower_top
        timeout_y_top = (upper_bot + upper_top) / 2.0
        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            for j, val in enumerate(data_list[i]):
                xpos = x[j] + offset
                if val <= 0:
                    ax_bot.text(xpos, y_offset, 'OOM',
                                ha='center', va='bottom',
                                fontsize=max(font_size - 2, 6),
                                color='#b00020', rotation=90)
                elif timeout_value is not None and val >= timeout_value:
                    ax_top.text(xpos, timeout_y_top, 'Timeout 3h',
                                ha='center', va='center',
                                fontsize=max(font_size - 2, 6),
                                color='#b00020', rotation=90,
                                fontweight='bold')

        # Short per-bar tags under the bottom axis.
        xaxis_trans = ax_bot.get_xaxis_transform()
        for i in range(num_datasets):
            offset = (i - (num_datasets - 1) / 2) * bar_width
            for j in range(num_batches):
                ax_bot.text(x[j] + offset, -0.02, sub_labels[i],
                            transform=xaxis_trans,
                            ha='center', va='top', fontsize=short_fontsize,
                            color=colors[i], fontweight='bold', clip_on=False)
        ax_bot.tick_params(axis='x', pad=short_fontsize + 6)
        ax_bot.xaxis.labelpad = short_fontsize + 8

        plt.tight_layout(pad=0.4)

    # ---- save PDF (primary) + PNG (quick preview) ----
    pdf_name = filename.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_name, bbox_inches='tight', format='pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    print(f"Plot saved as '{pdf_name}' and '{filename}'")
    plt.close()


def plot_absolute_times(series,
                        output_dir='output', batch_size_mask=None,
                        machine_type='A100',
                        prompt_len=1920, decode_len=128,
                        show_legend=True,
                        gpu_memory_gib=0,
                        oversub_series_key=None,
                        oversub_log_path=None,
                        broken_y=False,
                        broken_y_min_ratio=3.0,
                        timeout_value=None,
                        y_max_cap=None,
                        bar_width_scale=1.0):
    """Render the full set of TTFT/decode/ITL/total/throughput plots.

    Parameters
    ----------
    series : dict[str, dict]
        Ordered mapping of series-key -> {"data": <read_data() tuple or None>,
                                          "label": <legend label>,
                                          "color": <matplotlib color>,
                                          "sub_label": <optional short tag>}.
        Bar order follows the dict insertion order. Entries whose "data" is
        None are silently skipped. If "sub_label" is omitted, it defaults to
        the prefix of "label" before the first '.'.
    gpu_memory_gib : float
        Physical GPU HBM size in GiB. When > 0, oversubscription ratios are
        computed and appended to the x-tick labels.
    oversub_series_key : str or None
        Key into ``series`` whose log file is used to compute the per-batch
        oversubscription ratio. Ignored when ``gpu_memory_gib`` is 0.
    oversub_log_path : str or None
        Explicit log path for oversub parsing (overrides ``oversub_series_key``).
    """

    data_list = []
    colors = []
    labels = []
    sub_labels = []
    log_paths = {}
    for key, spec in series.items():
        data = spec.get("data")
        label = spec.get("label", key)
        clr = spec.get("color")
        sub_label = spec.get("sub_label", label.split('.')[0])

        if data is None:
            print(f"Warning: Data for '{label}' ({key}) is None, skipping this dataset.")
            continue

        data_list.append(data)
        colors.append(clr)
        labels.append(label)
        sub_labels.append(sub_label)
        if spec.get("log_path"):
            log_paths[key] = spec["log_path"]

    if not data_list:
        print("Warning: no series with data to plot, skipping.")
        return

    max_length = max([len(data[0]) for data in data_list])
    padded_data_list = [pad_data(data, max_length) for data in data_list]

    batch_sizes_str = [str(bsz) for bsz in data_list[0][0]]
    for i in range(len(data_list)):
        if len(data_list[i][0]) > len(batch_sizes_str):
            batch_sizes_str = [str(bsz) for bsz in data_list[i][0]]

    print(batch_sizes_str)
    total_times_list = np.array([data[1] for data in padded_data_list])
    prefill_times_list = np.array([data[2] for data in padded_data_list])
    decode_times_list = np.array([data[3] for data in padded_data_list])

    print(total_times_list.shape)

    mask = np.array([True] * len(batch_sizes_str))
    if batch_size_mask is not None:
        mask = batch_size_mask.copy()

    total_times_list = total_times_list[:, mask]
    prefill_times_list = prefill_times_list[:, mask]
    decode_times_list = decode_times_list[:, mask]
    itl_times_list = np.array(decode_times_list) / decode_len
    batch_sizes_str = [batch_sizes_str[i] for i in range(len(batch_sizes_str)) if mask[i]]

    batch_sizes_int = np.array([int(b) for b in batch_sizes_str])

    # ---- oversubscription ratio annotation on x-tick labels ----
    oversub_xlabel_suffix = ''
    if gpu_memory_gib and gpu_memory_gib > 0:
        gpu_mib = gpu_memory_gib * 1024.0
        ref_log = oversub_log_path
        if ref_log is None and oversub_series_key and oversub_series_key in log_paths:
            ref_log = log_paths[oversub_series_key]
        if ref_log is None:
            for spec in series.values():
                if spec.get("log_path") and spec.get("data") is not None:
                    ref_log = spec["log_path"]
                    break

        if ref_log:
            n_bsz_pre = len(batch_sizes_str)
            bsz_labels_pre = list(batch_sizes_str)

            peaks = parse_per_batch_peak_gpu_memory(ref_log)
            if peaks:
                full_peaks = peaks
                masked_peaks = [full_peaks[i] for i in range(len(full_peaks)) if i < len(mask) and mask[i]] if batch_size_mask is not None else full_peaks
                if len(masked_peaks) == n_bsz_pre:
                    batch_sizes_str = [
                        f'{bsz}\n{p / gpu_mib:.2f}\u00d7' if p is not None else bsz
                        for bsz, p in zip(batch_sizes_str, masked_peaks)
                    ]
                    oversub_xlabel_suffix = '\n(2nd line: Oversubscription ratio)'
                    print(f"Reserved oversub ratios: " + ", ".join(
                        f"bsz={b}: {p/gpu_mib:.2f}x" if p is not None else f"bsz={b}: n/a"
                        for b, p in zip(bsz_labels_pre, masked_peaks)
                    ))

            # Active-memory oversub ratios are computed and logged for
            # diagnostic purposes only -- they are not annotated on the plot.
            active_peaks = parse_per_batch_peak_active_memory(ref_log)
            if active_peaks:
                masked_active = [active_peaks[i] for i in range(len(active_peaks)) if i < len(mask) and mask[i]] if batch_size_mask is not None else active_peaks
                if len(masked_active) == n_bsz_pre:
                    print(f"Active oversub ratios:   " + ", ".join(
                        f"bsz={b}: {p/gpu_mib:.2f}x" if p is not None else f"bsz={b}: n/a"
                        for b, p in zip(bsz_labels_pre, masked_active)
                    ))

    def compute_tokens_per_sec(times_list, tokens_per_request):
        tps = np.zeros_like(times_list, dtype=float)
        for i in range(len(times_list)):
            for j in range(len(times_list[i])):
                t = times_list[i][j]
                if t > 0:
                    tps[i][j] = batch_sizes_int[j] * tokens_per_request / t
                else:
                    tps[i][j] = t
        return tps

    prefill_tps_list = compute_tokens_per_sec(prefill_times_list, prompt_len)
    decode_tps_list = compute_tokens_per_sec(decode_times_list, decode_len)
    e2e_tps_list = compute_tokens_per_sec(total_times_list, prompt_len + decode_len)

    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}')

    # ---- print performance tables ----
    # Strip the oversub annotation (newline + ratio) from batch labels for the
    # table header so columns stay compact.
    bsz_headers = [b.split('\n')[0] for b in batch_sizes_str]
    col_w = max(10, max((len(h) for h in bsz_headers), default=6) + 2)
    label_w = max(len(l) for l in labels) + 2

    metric_tables = {
        'TTFT (sec)':           (prefill_times_list, 2),
        'Decode (sec)':         (decode_times_list, 2),
        'ITL (sec)':            (itl_times_list, 3),
        'Total (sec)':          (total_times_list, 2),
        'Prefill tok/s':        (prefill_tps_list, 0),
        'Decode tok/s':         (decode_tps_list, 0),
        'E2E tok/s':            (e2e_tps_list, 0),
    }

    for metric_name, (data_arr, prec) in metric_tables.items():
        print(f'\n=== {metric_name} ===')
        header = ' ' * (label_w + 2) + ''.join(f'{"bsz=" + h:>{col_w}}' for h in bsz_headers)
        print(header)
        print(' ' * (label_w + 2) + '-' * (col_w * len(bsz_headers)))
        for i, lbl in enumerate(labels):
            cells = []
            for v in data_arr[i]:
                if v <= 0:
                    cells.append(f'{"OOM":>{col_w}}')
                else:
                    cells.append(f'{v:>{col_w}.{prec}f}')
            print(f'  {lbl:<{label_w}}' + ''.join(cells))

    x_label_base = '# of concurrent prompts' + oversub_xlabel_suffix

    backbone_kw = dict(broken_y=broken_y, broken_y_min_ratio=broken_y_min_ratio,
                       bar_width_scale=bar_width_scale)
    # Time-domain plots also get the timeout marker and y-axis cap; the
    # throughput plots below intentionally don't (units are tokens/sec).
    time_kw = dict(backbone_kw, timeout_value=timeout_value, y_max_cap=y_max_cap)

    plot_backbone(batch_sizes_str,
                  prefill_times_list,
                  x_label=x_label_base,
                  y_label='TTFT (sec)',
                  title=f'Time to First Token (TTFT) on {machine_type}',
                  filename=f'{output_dir}/ttft_absolute_combined.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  show_legend=show_legend,
                  **time_kw,
)

    plot_backbone(batch_sizes_str,
                  decode_times_list,
                  x_label=x_label_base,
                  y_label='Decode Time (sec)',
                  title=f'Decode latency on {machine_type}',
                  filename=f'{output_dir}/decode_time_absolute_combined.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  show_legend=show_legend,
                  **time_kw,
)

    plot_backbone(batch_sizes_str,
                  itl_times_list,
                  x_label=x_label_base,
                  y_label='ITL (sec)',
                  title=f'Inter Token Latency (ITL) on {machine_type}',
                  filename=f'{output_dir}/itl_absolute_combined.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  precision=2,
                  show_legend=show_legend,
                  **backbone_kw,
)

    plot_backbone(batch_sizes_str,
                  total_times_list,
                  x_label=x_label_base,
                  y_label='Total Time (sec)',
                  title=f'E2E latency on {machine_type}',
                  filename=f'{output_dir}/total_time_absolute_combined.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  show_legend=show_legend,
                  **time_kw,
)

    plot_backbone(batch_sizes_str,
                  prefill_tps_list,
                  x_label=x_label_base,
                  y_label='Tokens/sec',
                  title=f'Prefill Throughput (prompt_len={prompt_len}) on {machine_type}',
                  filename=f'{output_dir}/prefill_tokens_per_sec.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  precision=0,
                  show_legend=show_legend,
                  **backbone_kw,
)

    plot_backbone(batch_sizes_str,
                  decode_tps_list,
                  x_label=x_label_base,
                  y_label='Tokens/sec',
                  title=f'Decode Throughput (decode_len={decode_len}) on {machine_type}',
                  filename=f'{output_dir}/decode_tokens_per_sec.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  precision=0,
                  show_legend=show_legend,
                  **backbone_kw,
)

    plot_backbone(batch_sizes_str,
                  e2e_tps_list,
                  x_label=x_label_base,
                  y_label='Tokens/sec',
                  title=f'E2E Throughput (prompt={prompt_len}+decode={decode_len}) on {machine_type}',
                  filename=f'{output_dir}/e2e_tokens_per_sec.png',
                  colors=colors,
                  labels=labels,
                  sub_labels=sub_labels,
                  precision=0,
                  show_legend=show_legend,
                  **backbone_kw,
)

def split_plots(series,
        output_dir='output',
        machine_type='A100',
        prompt_len=1920, decode_len=128,
        small_mask_indices=None, large_mask_indices=None,
        all_mask_indices=None,
        show_legend=True,
        gpu_memory_gib=0,
        oversub_series_key=None,
        oversub_log_path=None,
        broken_y=False,
        broken_y_min_ratio=3.0,
        timeout_value=None,
        y_max_cap=None,
        bar_width_scale=1.0):
    """Generate small-batch, large-batch, and all-batch sub-plots."""

    n_bsz = None
    for spec in series.values():
        data = spec.get("data")
        if data is not None:
            n_bsz = len(data[0])
            break
    if n_bsz is None:
        print("Warning: no series with data, nothing to plot.")
        return

    common_kw = dict(
        machine_type=machine_type,
        prompt_len=prompt_len, decode_len=decode_len,
        show_legend=show_legend,
        gpu_memory_gib=gpu_memory_gib,
        oversub_series_key=oversub_series_key,
        oversub_log_path=oversub_log_path,
        broken_y=broken_y,
        broken_y_min_ratio=broken_y_min_ratio,
        timeout_value=timeout_value,
        y_max_cap=y_max_cap,
        bar_width_scale=bar_width_scale,
    )

    small_batch_size_mask = np.array([i in small_mask_indices for i in range(n_bsz)]) if small_mask_indices else None
    large_batch_size_mask = np.array([i in large_mask_indices for i in range(n_bsz)]) if large_mask_indices else None
    all_batch_size_mask = np.array([i in all_mask_indices for i in range(n_bsz)]) if all_mask_indices else None

    if small_batch_size_mask is not None:
        plot_absolute_times(series,
            output_dir=f'{output_dir}/small_batch_size',
            batch_size_mask=small_batch_size_mask, **common_kw)

    if large_batch_size_mask is not None:
        plot_absolute_times(series,
            output_dir=f'{output_dir}/large_batch_size',
            batch_size_mask=large_batch_size_mask, **common_kw)

    plot_absolute_times(series,
        output_dir=f'{output_dir}/all_batch_size',
        batch_size_mask=all_batch_size_mask, **common_kw)
        

def _read_or_none(path):
    """Read data from a log file, or return None if the path is empty/missing."""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return None
    return read_data(path)


def plot_H100(gpu_memory_gib=80):
    """Plot vary-batch-size results for OPT-13B on H100."""
    folder = "../vary_batch_size/huggingface/H100/debug"
    output_dir = folder
    machine_type = "H100"
    prompt_len = 1920
    decode_len = 128
    small_mask_indices = [0, 1, 2]
    large_mask_indices = [3, 4, 5, 6, 7, 8]
    show_legend = True

    oversub_log = f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching-discard=NoUnmap-uvm_managed_advise_prefetch_discard-dynamic-opt-13b-prompt1920-20251118_231650.log"

    series = {
        "CUDACachingAlloc": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-dynamic-opt-13b-prompt1920-20251111_194804.log"),
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
        },
        "hf_dynamic_offload": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-offloaded-opt-13b-prompt1920-20251111_194804.log"),
            "label": "A. Application-level KV Cache Offload",
            "color": PALETTE[1],
        },
        "CUDAPluggableAlloc_uvm_managed": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-uvm_managed-dynamic-opt-13b-prompt1920-20251118_231650.log"),
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
        },
        # "custom_CUDACachingAlloc_uvm_managed_gpu_first": {
        #     "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching--uvm_managed_advise_prefetch-dynamic-opt-13b-prompt1920-20251118_231650.log"),
        #     "label": "U1. UVM w/ caching and prepopulate GPU memory",
        #     "color": PALETTE[7],
        # },
        "custom_CUDACachingAlloc_uvm_managed_advise_prefetch_discard": {
            "data": _read_or_none(oversub_log),
            "label": "U1. Pytorch-UVM",
            "color": PALETTE[8],
            "log_path": oversub_log,
        },
    }

    for key, spec in series.items():
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"data={'<loaded>' if spec.get('data') is not None else None}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_plots(
        series,
        output_dir=output_dir,
        machine_type=machine_type,
        prompt_len=prompt_len,
        decode_len=decode_len,
        small_mask_indices=small_mask_indices,
        large_mask_indices=large_mask_indices,
        show_legend=show_legend,
        gpu_memory_gib=gpu_memory_gib,
        oversub_log_path=oversub_log,
    )


def plot_H100_motivation(gpu_memory_gib=80):
    """Variant of plot_H100 used for the motivation figure.

    Identical to plot_H100 but omits the Pytorch-UVM
    (``custom_CUDACachingAlloc_uvm_managed_advise_prefetch_discard``)
    series. Output goes to a ``motivation/`` subfolder so it does not
    overwrite the regular plot_H100 PDFs/PNGs.
    """
    folder = "../vary_batch_size/huggingface/H100/debug"
    output_dir = f"{folder}/motivation"
    machine_type = "H100"
    prompt_len = 1920
    decode_len = 128
    small_mask_indices = [0, 1, 2]
    large_mask_indices = [3, 4, 5, 6, 7, 8]
    show_legend = True

    oversub_log = f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching-discard=NoUnmap-uvm_managed_advise_prefetch_discard-dynamic-opt-13b-prompt1920-20251118_231650.log"

    series = {
        "CUDACachingAlloc": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-dynamic-opt-13b-prompt1920-20251111_194804.log"),
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
        },
        "hf_dynamic_offload": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-offloaded-opt-13b-prompt1920-20251111_194804.log"),
            "label": "A. Application-level KV Cache Offload",
            "color": PALETTE[1],
        },
        "CUDAPluggableAlloc_uvm_managed": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-uvm_managed-dynamic-opt-13b-prompt1920-20251118_231650.log"),
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
        },
    }

    for key, spec in series.items():
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"data={'<loaded>' if spec.get('data') is not None else None}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_plots(
        series,
        output_dir=output_dir,
        machine_type=machine_type,
        prompt_len=prompt_len,
        decode_len=decode_len,
        small_mask_indices=small_mask_indices,
        large_mask_indices=large_mask_indices,
        show_legend=show_legend,
        gpu_memory_gib=gpu_memory_gib,
        oversub_log_path=oversub_log,
        bar_width_scale=0.5,
    )


def plot_H100_discard(gpu_memory_gib=80):
    """Plot vary-batch-size results for OPT-13B on H100."""
    folder = "../vary_batch_size/huggingface/H100/debug"
    output_dir = f"{folder}/discard"
    machine_type = "H100"
    prompt_len = 1920
    decode_len = 128
    small_mask_indices = [2, 3, 4, 5]
    large_mask_indices = [3, 4, 5, 6, 7, 8]
    # Batch sizes shown in the eval figure (all_batch_size output): 24,32,40,48.
    all_mask_indices = [2, 3, 4, 5]
    show_legend = True

    oversub_log = f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching-discard=NoUnmap-uvm_managed_advise_prefetch_discard-dynamic-opt-13b-prompt1920-20251118_231650.log"

    series = {
        "CUDACachingAlloc": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-dynamic-opt-13b-prompt1920-20251111_194804.log"),
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
        },
        "hf_dynamic_offload": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-offloaded-opt-13b-prompt1920-20251111_194804.log"),
            "label": "A. Application-level KV Cache Offload",
            "color": PALETTE[1],
        },
        "CUDAPluggableAlloc_uvm_managed": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-uvm_managed-dynamic-opt-13b-prompt1920-20251118_231650.log"),
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
        },
         "custom_CUDACachingAlloc_uvm_managed_gpu_first": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching--uvm_managed_advise_prefetch-dynamic-opt-13b-prompt1920-20251118_231650.log"),
            "label": "U1'. Pytorch-UVM (w/o discard)",
            "color": PALETTE[7],
        },
        "custom_CUDACachingAlloc_uvm_managed_advise_prefetch_discard": {
            "data": _read_or_none(oversub_log),
            "label": "U1. Pytorch-UVM (w/ discard)",
            "color": PALETTE[8],
            "log_path": oversub_log,
        },
    }

    for key, spec in series.items():
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"data={'<loaded>' if spec.get('data') is not None else None}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_plots(
        series,
        output_dir=output_dir,
        machine_type=machine_type,
        prompt_len=prompt_len,
        decode_len=decode_len,
        small_mask_indices=small_mask_indices,
        large_mask_indices=large_mask_indices,
        all_mask_indices=all_mask_indices,
        show_legend=show_legend,
        gpu_memory_gib=gpu_memory_gib,
        oversub_log_path=oversub_log,
        broken_y=True,
        broken_y_min_ratio=3.0,
    )


def plot_H100_no_migration(gpu_memory_gib=80):
    """Plot vary-batch-size results for OPT-13B on H100 (no-migration run)."""
    folder = "../vary_batch_size/huggingface/H100/debug"
    output_dir = f"{folder}/no_migration"
    machine_type = "H100"
    prompt_len = 1920
    decode_len = 128
    small_mask_indices = [2, 3, 4, 5]
    large_mask_indices = [3, 4, 5, 6, 7, 8]
    # Batch sizes shown in the eval figure (all_batch_size output): 32,40,48,56,64.
    all_mask_indices = [3, 4, 5, 6, 7]
    show_legend = True
    timeout_seconds = 3600 * 3

    # Keep the original no-discard U1 log as the oversubscription source.
    oversub_log = f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-customcaching-discard=NoUnmap-uvm_managed_advise_prefetch_discard-dynamic-opt-13b-prompt1920-20251118_231650.log"


    series = {
        "CUDACachingAlloc": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-dynamic-opt-13b-prompt1920-20251111_194804.log"),
            "label": "C1. cudaMalloc w/ caching (vanilla Pytorch)",
            "color": PALETTE[0],
        },
        "hf_dynamic_offload": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-no_uvm-offloaded-opt-13b-prompt1920-20251111_194804.log"),
            "label": "A. Application-level KV Cache Offload",
            "color": PALETTE[1],
        },
        "CUDAPluggableAlloc_uvm_managed": {
            "data": _read_or_none(f"{folder}/py313_pytorch_cuda13-cuda-1024Awaken-uvm_managed-dynamic-opt-13b-prompt1920-20251118_231650.log"),
            "label": "U0. UVM w/o caching (Pytorch default)",
            "color": PALETTE[2],
        },
        # "custom_CUDACachingAlloc_uvm_managed_gpu_first": {
        #     "data": _read_or_none(oversub_log),
        #     "label": "U1'. Pytorch-UVM (w/o discard)",
        #     "color": PALETTE[7],
        #     "log_path": oversub_log,
        # },
        "custom_CUDACachingAlloc_uvm_managed_advise_prefetch_discard": {
            "data": _read_or_none(oversub_log),
            "label": "U1. Pytorch-UVM",
            "color": PALETTE[8],
            "log_path": oversub_log,
        },
        # "custom_CUDACachingAlloc_uvm_managed_gpu_first_sam_no_migration": {
        #     "data": _read_or_none(
        #         f"{folder}/py313_pytorch_cuda13-cuda-default-customcaching--"
        #         "uvm_managed_gpu_first_sam-dynamic-opt-13b-prompt1920-20260423_001847.log"
        #     ),
        #     "label": "U2. Pytorch-UVM + SAM (no migration)",
        #     "color": PALETTE[8],
        # },
        # "custom_CUDACachingAlloc_uvm_managed_gpu_ac_no_migration": {
        #     "data": _read_or_none(
        #         f"{folder}/py313_pytorch_cuda13-cuda-default-customcaching--"
        #         "uvm_managed_gpu_ac-dynamic-opt-13b-prompt1920-20260423_001847.log"
        #     ),
        #     "label": "U3. Pytorch-UVM + No migration",
        #     "color": PALETTE[9],
        # },
        "custom_CUDACachingAlloc_uvm_managed_gpu_ac_discard_no_migration": {
            "data": _read_or_none(
                f"{folder}/py313_pytorch_cuda13-cuda-default-customcaching-"
                "discard=NoUnmap-uvm_managed_gpu_ac_discard-dynamic-opt-13b-"
                "prompt1920-20260423_001847.log"
            ),
            "label": "U3. Pytorch-UVM + no migration",
            "color": PALETTE[10],
        },
    }

    # For these no-migration variants, treat missing logs as explicit
    # timeouts so they still appear in the bar charts.
    timeout_series_keys = [
        "custom_CUDACachingAlloc_uvm_managed_gpu_ac_no_migration",
        "custom_CUDACachingAlloc_uvm_managed_gpu_ac_discard_no_migration",
    ]
    reference_data = None
    for spec in series.values():
        if spec.get("data") is not None:
            reference_data = spec["data"]
            break
    if reference_data is not None:
        ref_batch_sizes = np.array(reference_data[0], copy=True)
        ref_n = len(ref_batch_sizes)
        timeout_val = float(timeout_seconds)

        def _coerce_to_timeouts(times, n):
            arr = np.asarray(times, dtype=float)
            if arr.size < n:
                arr = np.pad(arr, (0, n - arr.size), constant_values=timeout_val)
            elif arr.size > n:
                arr = arr[:n]
            arr = np.where(arr <= 0, timeout_val, arr)
            return arr

        for key in timeout_series_keys:
            spec = series.get(key)
            if spec is None:
                continue
            data = spec.get("data")
            if data is None:
                spec["data"] = (
                    ref_batch_sizes.copy(),
                    np.full(ref_n, timeout_val),
                    np.full(ref_n, timeout_val),
                    np.full(ref_n, timeout_val),
                )
                print(
                    f"Info: {key} missing -> using {timeout_seconds}s timeout "
                    f"for all {ref_n} batch sizes."
                )
                continue

            bsz, total_t, prefill_t, decode_t = data
            new_total = _coerce_to_timeouts(total_t, ref_n)
            new_prefill = _coerce_to_timeouts(prefill_t, ref_n)
            new_decode = _coerce_to_timeouts(decode_t, ref_n)

            replaced_idx = [
                int(ref_batch_sizes[i])
                for i in range(ref_n)
                if i < len(total_t) and float(total_t[i]) <= 0
            ]
            extended_idx = [
                int(ref_batch_sizes[i])
                for i in range(len(total_t), ref_n)
            ]
            if replaced_idx or extended_idx:
                print(
                    f"Info: {key} marking timeouts ({timeout_seconds}s) for "
                    f"bsz={replaced_idx + extended_idx}"
                )

            if len(np.asarray(bsz)) < ref_n:
                bsz = ref_batch_sizes.copy()
            spec["data"] = (bsz, new_total, new_prefill, new_decode)

    for key, spec in series.items():
        print(f"{key}: label={spec['label']!r} color={spec['color']} "
              f"data={'<loaded>' if spec.get('data') is not None else None}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_plots(
        series,
        output_dir=output_dir,
        machine_type=machine_type,
        prompt_len=prompt_len,
        decode_len=decode_len,
        small_mask_indices=small_mask_indices,
        large_mask_indices=large_mask_indices,
        all_mask_indices=all_mask_indices,
        show_legend=show_legend,
        gpu_memory_gib=gpu_memory_gib,
        oversub_log_path=oversub_log,
        broken_y=True,
        broken_y_min_ratio=3.0,
        timeout_value=timeout_seconds,
        y_max_cap=7000,
    )


if __name__ == "__main__":
    # plot_H100()
    plot_H100_motivation()
    # plot_H100_discard()
    # plot_H100_no_migration()
    # plot_H100_no_migration()
    # plot_GH200_cg1_managed_64k()
    # plot_GH200_cg1_managed_64k_SAM()

