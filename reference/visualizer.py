"""
Generates an interactive HTML report from preformatted BEM visualization data.

The input bundle is expected to be created by format_visualizer_data.py and to
contain balloon-surface arrays plus smoothed horizontal and vertical isobar
heatmaps.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==========================================
# Configuration
# ==========================================
INPUT_NPZ = Path("pressure_data_formatted.npz")
OUTPUT_HTML = Path("directivity_report.html")
ISOBAR_ZSMOOTH = "best"
CUSTOM_COLORS = [
    "#00008F", "#0000FF", "#006FFF", "#00DFFF", "#4FFFBF",
    "#BFFF4F", "#FFDF00", "#FF6F00", "#FF0000", "#8F0000",
]

def load_formatted_data(npz_path: Path) -> dict[str, np.ndarray]:
    required = {
        "freq_hz",
        "balloon_surface_spl",
        "balloon_x",
        "balloon_y",
        "balloon_z",
        "isobar_angle_deg",
        "isobar_freq_hz",
        "horizontal_isobar_spl",
        "vertical_isobar_spl",
        "impedance_real",
        "impedance_imag",
        "min_db",
        "max_db",
    }
    with np.load(npz_path) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"NPZ missing keys: {sorted(missing)}")
        return {key: data[key] for key in required}

def _make_colorscale(colors: list[str]) -> list[list[float | str]]:
    scale_steps = np.linspace(0, 1, len(colors))
    return [[v, c] for v, c in zip(scale_steps, colors)]

def _make_discrete_colorscale(
    colors: list[str],
    min_db: float,
    max_db: float,
) -> list[list[float | str]]:
    boundaries = np.linspace(min_db, max_db, len(colors) + 1)
    norm = (boundaries - min_db) / (max_db - min_db)
    scale: list[list[float | str]] = []
    for i, color in enumerate(colors):
        scale.append([float(norm[i]), color])
        scale.append([float(norm[i + 1]), color])
    return scale

def _add_heatmap(
    fig: go.Figure,
    freqs_interp: np.ndarray,
    angle_deg: np.ndarray,
    spl_matrix: np.ndarray,
    colorscale: list[list[float | str]],
    color_range_db: tuple[float, float],
    row: int,
    col: int,
    name: str,
):
    fig.add_trace(
        go.Heatmap(
            x=freqs_interp,
            y=angle_deg,
            z=spl_matrix,
            colorscale=colorscale,
            zmin=color_range_db[0],
            zmax=color_range_db[1],
            zsmooth=ISOBAR_ZSMOOTH,
            showscale=False,
            name=name,
        ),
        row=row,
        col=col,
    )

def _format_frequency_label(freq: float) -> str:
    return f"{freq:.0f}"


def _build_frequency_report_html(fig: go.Figure, freqs: np.ndarray) -> str:
    plot_div_id = "directivity-report-plot"
    freq_labels = [_format_frequency_label(freq) for freq in freqs]
    figure_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
        div_id=plot_div_id,
    )
    frequency_buttons = "\n".join(
        (
            f'<button class="frequency-item{(" is-active" if idx == 0 else "")}" '
            f'type="button" data-index="{idx}">{label}</button>'
        )
        for idx, label in enumerate(freq_labels)
    )
    freqs_json = json.dumps(freq_labels)
    surface_count = len(freq_labels)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Directivity Report</title>
    <style>
        :root {{
            color-scheme: light;
            font-family: Consolas, "Lucida Console", "Courier New", monospace;
            background: #ececec;
            color: #111;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: #ececec;
        }}

        .report-shell {{
            display: flex;
            align-items: flex-start;
            min-height: 100vh;
        }}

        .plot-shell {{
            flex: 1 1 auto;
            min-width: 0;
            background: #fff;
        }}

        .frequency-panel {{
            position: sticky;
            top: 0;
            display: flex;
            flex-direction: column;
            width: 180px;
            height: 100vh;
            border-left: 1px solid #a8a8a8;
            background: #efefef;
        }}

        .frequency-heading {{
            padding: 12px 10px 10px;
            border-bottom: 1px solid #bdbdbd;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            text-align: center;
        }}

        .frequency-subtitle {{
            padding: 0 10px 10px;
            border-bottom: 1px solid #bdbdbd;
            font-size: 11px;
            color: #555;
            text-align: center;
        }}

        .frequency-list {{
            flex: 1 1 auto;
            overflow-y: auto;
            padding: 8px;
        }}

        .frequency-list::-webkit-scrollbar {{
            width: 12px;
        }}

        .frequency-list::-webkit-scrollbar-thumb {{
            border: 2px solid #efefef;
            border-radius: 999px;
            background: #adadad;
        }}

        .frequency-item {{
            display: block;
            width: 100%;
            margin: 0;
            padding: 6px 8px;
            border: 1px solid transparent;
            background: transparent;
            color: #111;
            font: inherit;
            font-size: 14px;
            text-align: center;
            cursor: pointer;
        }}

        .frequency-item:hover {{
            background: #d9e6f5;
            border-color: #c0d4eb;
        }}

        .frequency-item.is-active {{
            background: #0a74da;
            border-color: #075eb1;
            color: #fff;
        }}

        @media (max-width: 900px) {{
            .report-shell {{
                flex-direction: column;
            }}

            .frequency-panel {{
                position: static;
                width: 100%;
                height: 280px;
                border-left: 0;
                border-top: 1px solid #a8a8a8;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-shell">
        <main class="plot-shell">
            {figure_html}
        </main>
        <aside class="frequency-panel" aria-label="Frequency selector">
            <div class="frequency-heading">Frequencies</div>
            <div class="frequency-list" id="frequency-list">
                {frequency_buttons}
            </div>
        </aside>
    </div>
    <script>
        (function () {{
            const plot = document.getElementById("{plot_div_id}");
            const frequencyLabels = {freqs_json};
            const surfaceCount = {surface_count};
            const buttons = Array.from(document.querySelectorAll(".frequency-item"));

            function setActiveButton(index) {{
                buttons.forEach((button, buttonIndex) => {{
                    const isActive = buttonIndex === index;
                    button.classList.toggle("is-active", isActive);
                    if (isActive) {{
                        button.setAttribute("aria-current", "true");
                    }} else {{
                        button.removeAttribute("aria-current");
                    }}
                }});
            }}

            function updateFrequency(index) {{
                const totalTraces = plot.data.length;
                const visible = Array.from({{ length: totalTraces }}, (_, traceIndex) => {{
                    return traceIndex === index || traceIndex >= surfaceCount;
                }});

                Plotly.restyle(plot, "visible", visible);
                Plotly.relayout(plot, {{ title: `Frequency: ${{frequencyLabels[index]}} Hz` }});
                setActiveButton(index);
            }}

            function initializePanel() {{
                if (!plot || !plot.data || plot.data.length === 0) {{
                    window.setTimeout(initializePanel, 50);
                    return;
                }}

                buttons.forEach((button) => {{
                    button.addEventListener("click", () => {{
                        updateFrequency(Number(button.dataset.index));
                    }});
                }});

                updateFrequency(0);
                Plotly.Plots.resize(plot);
            }}

            window.addEventListener("resize", () => {{
                if (plot) {{
                    Plotly.Plots.resize(plot);
                }}
            }});

            initializePanel();
        }})();
    </script>
</body>
</html>
"""

def _apply_axis_layout(fig: go.Figure, min_db: float, max_db: float):
    max_range = max_db - min_db
    axis_def = dict(range=[-max_range, max_range], title="")

    freq_min_hz = 200
    freq_max_hz = 20000
    log_range = [np.log10(freq_min_hz), np.log10(freq_max_hz)]

    tickvals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ticktext = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

    fig.update_layout(
        scene=dict(
            xaxis=axis_def,
            yaxis=axis_def,
            zaxis=axis_def,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            dragmode="turntable",
        )
    )

    fig.update_xaxes(
        title_text="Frequency (Hz)",
        type="log",
        range=log_range,
        tickvals=tickvals,
        ticktext=ticktext,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="Frequency (Hz)",
        type="log",
        range=log_range,
        tickvals=tickvals,
        ticktext=ticktext,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Frequency (Hz)",
        type="log",
        range=log_range,
        tickvals=tickvals,
        ticktext=ticktext,
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Angle (deg)", range=[-180, 180], row=1, col=2)
    fig.update_yaxes(title_text="Impedance", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg)", range=[-180, 180], row=2, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=9 / 16, row=1, col=2)
    fig.update_yaxes(scaleanchor="x3", scaleratio=9 / 16, row=2, col=2)


def _add_impedance_plot(
    fig: go.Figure,
    freqs: np.ndarray,
    impedance_real: np.ndarray,
    impedance_imag: np.ndarray,
):
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=impedance_real,
            mode="lines",
            name="Impedance Real",
            line=dict(color="#1f77b4", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=impedance_imag,
            mode="lines",
            name="Impedance Imag",
            line=dict(color="#d62728", width=2, dash="dash"),
        ),
        row=2,
        col=1,
    )


def create_balloon_plot(formatted_data: dict[str, np.ndarray]):
    freqs = formatted_data["freq_hz"].astype(float, copy=False)
    balloon_surface_spl = formatted_data["balloon_surface_spl"]
    balloon_x = formatted_data["balloon_x"]
    balloon_y = formatted_data["balloon_y"]
    balloon_z = formatted_data["balloon_z"]
    angle_deg = formatted_data["isobar_angle_deg"].astype(float, copy=False)
    freqs_interp = formatted_data["isobar_freq_hz"].astype(float, copy=False)
    horizontal_spl = formatted_data["horizontal_isobar_spl"]
    vertical_spl = formatted_data["vertical_isobar_spl"]
    impedance_real = formatted_data["impedance_real"].astype(float, copy=False)
    impedance_imag = formatted_data["impedance_imag"].astype(float, copy=False)
    min_db = float(formatted_data["min_db"])
    max_db = float(formatted_data["max_db"])
    color_range_db = (min_db, max_db)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "heatmap"}],
        ],
        subplot_titles=(
            "3D Balloon Directivity",
            "Horizontal Polar Isobar",
            "Radiation Impedance",
            "Vertical Polar Isobar",
        ),
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    colorscale = _make_colorscale(CUSTOM_COLORS)
    discrete_colorscale = _make_discrete_colorscale(
        CUSTOM_COLORS,
        color_range_db[0],
        color_range_db[1],
    )

    print(f"Generating surfaces for {len(freqs)} frequencies...")
    for idx, freq in enumerate(freqs):
        fig.add_trace(
            go.Surface(
                x=balloon_x[idx],
                y=balloon_y[idx],
                z=balloon_z[idx],
                surfacecolor=balloon_surface_spl[idx],
                colorscale=colorscale,
                cmin=color_range_db[0],
                cmax=color_range_db[1],
                colorbar=dict(title="Normalized SPL (dB)"),
                name=f"{freq:.0f} Hz",
                visible=(idx == 0),
                showscale=True,
                lighting=dict(roughness=0.5, ambient=0.8, diffuse=0.8),
            ),
            row=1,
            col=1,
        )

    print("Generating horizontal/vertical isobar plots...")
    _add_heatmap(
        fig,
        freqs_interp,
        angle_deg,
        horizontal_spl,
        discrete_colorscale,
        color_range_db,
        row=1,
        col=2,
        name="Horizontal isobar",
    )
    _add_impedance_plot(
        fig,
        freqs,
        impedance_real,
        impedance_imag,
    )
    _add_heatmap(
        fig,
        freqs_interp,
        angle_deg,
        vertical_spl,
        discrete_colorscale,
        color_range_db,
        row=2,
        col=2,
        name="Vertical isobar",
    )

    fig.update_layout(
        title=f"Frequency: {_format_frequency_label(freqs[0])} Hz",
        height=1200,
        margin=dict(r=0, l=0, b=0, t=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    _apply_axis_layout(fig, min_db, max_db)

    print(f"Saving to {OUTPUT_HTML}...")
    OUTPUT_HTML.write_text(_build_frequency_report_html(fig, freqs), encoding="utf-8")
    print("Done!")

if __name__ == "__main__":
    formatted_data = load_formatted_data(INPUT_NPZ)
    create_balloon_plot(formatted_data)