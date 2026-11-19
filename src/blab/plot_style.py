"""Shared desktop and headless Matplotlib axis styles."""

AUDIO_FREQ_MIN_HZ = 20
AUDIO_FREQ_MAX_HZ = 20000
AUDIO_AXIS_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
AUDIO_AXIS_LABELS = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

PLOT_TITLE_SIZE = 11
PLOT_LABEL_SIZE = 9
PLOT_TICK_SIZE = 9
PLOT_LEGEND_SIZE = 9
PLOT_TITLE_PAD = 7
GRID_LINE_ALPHA = 0.6
MINOR_GRID_LINE_ALPHA = 0.3


def apply_audio_frequency_axis(axes) -> None:
    axes.set_xscale("log")
    axes.set_xlim(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ)
    axes.set_xticks(AUDIO_AXIS_TICKS)
    axes.set_xticklabels(AUDIO_AXIS_LABELS)


def apply_compact_plot_text(axes) -> None:
    axes.title.set_fontsize(PLOT_TITLE_SIZE)
    axes.xaxis.label.set_size(PLOT_LABEL_SIZE)
    axes.yaxis.label.set_size(PLOT_LABEL_SIZE)
    axes.tick_params(axis="both", which="major", labelsize=PLOT_TICK_SIZE)
    legend = axes.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PLOT_LEGEND_SIZE)
