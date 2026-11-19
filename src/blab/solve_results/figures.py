"""Headless figures for saved-run reports, using desktop axes and scalar projections."""

from io import StringIO

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from blab.observation_planes import ObservationPlaneDisplay
from blab.plot_style import GRID_LINE_ALPHA, PLOT_TITLE_PAD, apply_audio_frequency_axis, apply_compact_plot_text
from blab.ui.observation_plane_results import project_field_scalars


def svg(figure):
    buffer = StringIO()
    figure.savefig(buffer, format="svg")
    return buffer.getvalue().encode("ascii", "xmlcharrefreplace").decode("ascii")


def metric_figure(metric, samples):
    figure = Figure(figsize=(6.5, 3), layout="constrained")
    axes = figure.subplots()
    samples = sorted(samples, key=lambda sample: sample["frequency_hz"])
    frequencies = [sample["frequency_hz"] for sample in samples]
    impedance = metric == "normalized-impedance"
    label_key = "channel_names" if metric == "maximum-spl" else "component_ids"
    names = samples[0][label_key]
    if any(sample[label_key] != names for sample in samples):
        raise ValueError("Report trace identities changed across frequencies.")
    marker = "." if len(samples) == 1 else None
    if impedance:
        for i, name in enumerate(names):
            for component, style in (("real", "-"), ("imag", "--")):
                axes.plot(
                    frequencies,
                    [s["values"][component][i] for s in samples],
                    style,
                    marker=marker,
                    linewidth=1.5,
                    label=f"{name} Z {component}",
                )
        axes.set_ylabel("Normalized Acoustic Impedance (Z / ρcSd)")
        title = "Acoustic Impedance"
    else:
        for i, name in enumerate(names):
            axes.plot(frequencies, [s["values"][i] for s in samples], marker=marker, linewidth=1.5, label=name)
        if metric == "peak-excursion":
            title = "Transducer Excursion"
            axes.set_ylabel("One-way peak excursion (mm)")
            axes.set_ylim(bottom=0)
        else:
            title = "Maximum SPL"
            axes.set_ylabel("SPL (dB)")
            axes.yaxis.set_major_locator(MultipleLocator(10))
            levels = np.asarray([s["values"] for s in samples], dtype=float)
            if np.isfinite(levels).any():
                top = np.ceil(np.nanmax(levels) / 5) * 5
                axes.set_ylim(top - 50, top)
    axes.set_title(title, pad=PLOT_TITLE_PAD)
    axes.set_xlabel("Frequency (Hz)")
    apply_audio_frequency_axis(axes)
    axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
    axes.legend(loc="best")
    apply_compact_plot_text(axes)
    return figure


def plane_figure(plane, arrays, frequency_hz):
    figure = Figure(figsize=(6, 5), layout="constrained")
    axes = figure.subplots()
    nx, ny = plane.sample_shape
    u = np.linspace(-plane.width_m / 2, plane.width_m / 2, nx)
    v = np.linspace(-plane.height_m / 2, plane.height_m / 2, ny)
    if plane.display == ObservationPlaneDisplay.FLOW_ARROWS:
        active = plane.complex_component == "real"
        key = "active_in_plane_intensity_w_per_m2" if active else "reactive_in_plane_intensity_w_per_m2"
        vectors = arrays[key]
        uu, vv = np.meshgrid(u, v)
        magnitude = np.linalg.norm(vectors, axis=1)
        axis_u, axis_v, _ = plane.local_axes
        finite = magnitude[np.isfinite(magnitude)]
        reference = plane.intensity_scale_w_per_m2 or max(np.max(finite, initial=0.0), 1e-30)
        artist = axes.quiver(
            uu.ravel(),
            vv.ravel(),
            vectors @ axis_u,
            vectors @ axis_v,
            magnitude,
            angles="xy",
            scale_units="xy",
            scale=reference / (0.8 * min(u[1] - u[0], v[1] - v[0])),
            cmap="turbo",
            clim=(0, reference),
        )
        label = ("Active" if active else "Reactive") + " in-plane intensity (W/m²)"
    else:
        field = (
            arrays["particle_velocity_m_per_s"]
            if plane.display == ObservationPlaneDisplay.PARTICLE_VELOCITY
            else arrays["pressure_pa"]
        )
        projection = project_field_scalars(
            field,
            plane.display,
            pressure_color_limit_pa=plane.pressure_color_limit_pa,
            spl_min_db=plane.spl_min_db,
            spl_max_db=plane.spl_max_db,
            particle_velocity_limit_m_per_s=plane.particle_velocity_limit_m_per_s,
            normalized_spl_range_db=plane.normalized_spl_range_db,
        )
        artist = axes.pcolormesh(
            u,
            v,
            projection.values.reshape(ny, nx),
            shading="nearest",
            cmap=projection.cmap,
            vmin=projection.clim[0],
            vmax=projection.clim[1],
            rasterized=True,
        )
        label = projection.title
    axes.set(xlabel="Plane X (m)", ylabel="Plane Y (m)", aspect="equal", title=f"{plane.name} at {frequency_hz:g} Hz")
    figure.colorbar(artist, ax=axes, label=label)
    apply_compact_plot_text(axes)
    return figure


def catalog_figure(metric, data):
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize

    figure = Figure(figsize=(7, 4), layout="constrained")
    frequencies = data["frequencies_hz"]
    quantities = data["quantities"]
    title = metric.replace("-", " ").title()
    if metric == "balloon":
        from matplotlib import colormaps
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        from blab.balloon import balloon_surface_points

        if len(frequencies) != 1:
            raise ValueError("Select one --frequency to render a balloon SVG.")
        axes = figure.add_subplot(projection="3d")
        low, high = data["clip_db"]
        values = np.clip(quantities["normalized_spl"]["values"][0], low, high)
        points = balloon_surface_points(data["directions_xyz"], values, low)
        triangles = np.asarray(data["triangle_indices"])
        mesh = Poly3DCollection(points[triangles], cmap=colormaps["turbo"], norm=Normalize(low, high))
        mesh.set_array(values[triangles].mean(axis=1))
        axes.add_collection3d(mesh)
        radius = high - low
        axes.set(
            xlim=(-radius, radius),
            ylim=(-radius, radius),
            zlim=(-radius, radius),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            title=f"Directivity Balloon at {frequencies[0]:g} Hz",
        )
        axes.set_box_aspect((1, 1, 1))
        figure.colorbar(mesh, ax=axes, label="Relative SPL (dB)")
        return figure
    axes = figure.subplots()
    axes.set(xlabel="Frequency (Hz)", title=title)
    apply_audio_frequency_axis(axes)
    if "angles_deg" in data:
        from blab.plotting import VisualizerConfig

        cmap = LinearSegmentedColormap.from_list("boundary_lab_isobar", VisualizerConfig.custom_colors)
        low, high = data["clip_db"]
        step = data["contour_step_db"]
        norm = Normalize(low, high)
        if step > 0:
            inner = np.arange(np.ceil(low / step) * step, high, step)
            bounds = np.r_[low, inner[(inner > low) & (inner < high)], high]
            norm = BoundaryNorm(bounds, max(len(bounds), cmap.N))
        grid = quantities["normalized_spl"]["values"]
        if len(frequencies) == 1:
            image = axes.scatter(
                np.full(len(data["angles_deg"]), frequencies[0]),
                data["angles_deg"],
                c=grid[:, 0],
                cmap=cmap,
                norm=norm,
                marker="s",
            )
        else:
            image = axes.pcolormesh(
                frequencies, data["angles_deg"], grid, shading="nearest", rasterized=True, cmap=cmap, norm=norm
            )
        axes.set_ylabel("Angle (degrees)")
        figure.colorbar(image, ax=axes, label="Normalized SPL (dB)")
    else:
        fields = list(quantities) if metric != "interface-radiation" else ["spl_db"]
        dual = len(fields) == 2 or metric == "spinorama"
        secondary = axes.twinx() if dual else None
        lines = []
        axis_labels = {
            "phase_deg": "Phase (degrees)",
            "magnitude": "Impedance (ohm)",
            "spl_db": "SPL (dB)",
            "propagation_aligned_phase": "Phase (degrees)",
            "level": "Relative SPL (dB)",
            "rms_velocity": "Average normal velocity, RMS (m/s)",
            "group_delay": "Group delay (ms)",
            "real_power": "Real input power (W)",
        }
        for j, field in enumerate(fields):
            item = quantities[field]
            for i, (label, y) in enumerate(zip(data["trace_names"], item["values"], strict=True)):
                right = j == 1 or (metric == "spinorama" and i >= 5)
                target = secondary if right else axes
                target.set_ylabel("DI (dB)" if metric == "spinorama" and right else axis_labels[field])
                lines.extend(
                    target.plot(
                        frequencies,
                        y,
                        "--" if right else "-",
                        label=f"{label} {field}" if len(fields) > 1 else label,
                        linewidth=1.5,
                        marker="." if len(frequencies) == 1 else None,
                    )
                )
        if metric == "spinorama":
            axes.set_ylim(-40, 10)
            secondary.set_ylim(-5, 45)
        axes.legend(handles=lines, loc="best", fontsize=8)
        axes.grid(which="major", color="#808080", linewidth=0.8, alpha=GRID_LINE_ALPHA)
        if secondary is not None:
            apply_compact_plot_text(secondary)
    apply_compact_plot_text(axes)
    return figure
