"""Read, synthesize, and extract quantities from an existing headless run."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from blab.observation_planes import ObservationPlane
from blab.phasor import SOLVER_PHASOR_CONVENTION, convert_phasor
from blab.solve_results.artifact import ResultRun
from blab.solve_results.catalog import CATALOG_METRICS, catalog_report
from blab.solve_results.derived import peak_excursion_mm, phase_deg, pressure_spl_db
from blab.solve_results.model import DIAPHRAGM_VELOCITY_ID
from blab.solve_results.plane_query import sample_plane
from blab.solve_results.reports import maximum_spl, normalized_impedance


def _values(array):
    values = np.asarray(array)
    if np.iscomplexobj(values):
        return {"real": _values(values.real), "imag": _values(values.imag)}
    if np.issubdtype(values.dtype, np.floating):
        return np.where(np.isfinite(values), values, None).tolist()
    return values.tolist()


def _query(run, args, hz):
    q, axes, values, meta = run.sample(
        hz,
        args.quantity,
        excitation=args.excitation,
        response=args.response,
        uncorrected=args.uncorrected,
    )
    axes = list(axes)
    for axis_name, raw_index in args.index:
        if axis_name not in axes:
            raise ValueError(f"Unknown or already selected axis {axis_name!r}.")
        axis = axes.index(axis_name)
        index = int(raw_index)
        if not 0 <= index < values.shape[axis]:
            raise ValueError(f"Index {index} outside axis {axis_name!r}.")
        values = np.take(values, index, axis=axis)
        axes.pop(axis)
    unit = q["unit"]
    if args.view == "spl":
        if unit != "Pa":
            raise ValueError("SPL requires pressure in Pa.")
        values, unit = pressure_spl_db(values), "dB re 20 uPa"
    elif args.view == "peak-excursion":
        if q["quantity"] != "diaphragm_velocity" or unit != "m/s":
            raise ValueError("Peak excursion requires RMS diaphragm velocity.")
        values, unit = peak_excursion_mm(values, meta["freq_hz"]), "mm"
    elif args.view == "phase":
        if args.response is None:
            values = convert_phasor(values, run.manifest["phasor_convention"])
        values, unit = phase_deg(values), "deg"
    elif args.view == "magnitude":
        values = np.abs(values)
    return {
        "frequency_hz": meta["freq_hz"],
        "excitation_port_ids": meta["excitation_port_ids"],
        "axes": axes,
        "target_id": q["target_id"],
        "metadata": q["metadata"],
        "unit": unit,
        "values": _values(values),
    }


def _write_csv(payload):
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["frequency_hz", "quantity", "response", "unit", "indices", "excitation_port_id", "real", "imag"])
    for sample in payload["samples"]:
        data = sample["values"]
        real = np.asarray(data["real"] if isinstance(data, dict) else data)
        imag = np.asarray(data["imag"]) if isinstance(data, dict) else None
        for index in np.ndindex(real.shape):
            indices = dict(zip(sample["axes"], index, strict=True))
            excitation_index = indices.get("excitation", dict(payload["indices"]).get("excitation"))
            excitation_id = payload["excitation"]
            if excitation_index is not None:
                excitation_id = sample["excitation_port_ids"][int(excitation_index)]
            writer.writerow(
                [
                    sample["frequency_hz"],
                    payload["quantity"],
                    payload["response"],
                    sample["unit"],
                    json.dumps(dict(payload["indices"]) | indices),
                    excitation_id,
                    real[index],
                    imag[index] if imag is not None else None,
                ]
            )


def _plane_result(run, args):
    if args.plane_file:
        raw = json.loads(args.plane_file.read_text(encoding="utf-8-sig"))
    else:
        raw = next((p for p in run.project.get("observation_planes", []) if p["id"] == args.plane_id), None)
    definition = ObservationPlane.from_payload(raw)
    if definition is None:
        raise ValueError("Missing or invalid observation plane definition.")
    response = None if args.excitation else args.response or definition.response_id
    if args.uncorrected and not response:
        raise ValueError("--uncorrected requires a synthesized response.")
    if (
        args.pressure_only
        and definition.plane_type.value != "interior"
        and definition.display.value in {"flow_arrows", "particle_velocity"}
        and args.format == "svg"
    ):
        raise ValueError("Velocity and flow figures require exterior gradients. Omit --pressure-only.")
    arrays = sample_plane(
        run,
        definition,
        args.frequency,
        args.excitation,
        response,
        args.uncorrected,
        backend=args.backend,
        julia=args.julia_executable,
        pressure_only=args.pressure_only,
    )
    if args.format == "svg":
        from blab.solve_results.figures import plane_figure, svg

        return svg(plane_figure(definition, arrays, run.completed[args.frequency]["freq_hz"]))
    return {
        "run": str(run.root),
        "status": run.manifest["status"],
        "plane": definition.to_payload(),
        "shape_yx": definition.sample_shape[::-1],
        "frequency_hz": run.completed[args.frequency]["freq_hz"],
        "response": response,
        "excitation": args.excitation,
        "uncorrected": args.uncorrected,
        "phasor_convention": SOLVER_PHASOR_CONVENTION,
        "exterior_backend": args.backend if definition.plane_type.value != "interior" else None,
        "exterior_pressure_only": args.pressure_only,
        "fields": {name: _values(values) for name, values in arrays.items()},
    }


def _report_result(run, args, frequencies):
    payload = {"run": str(run.root), "status": run.manifest["status"], "samples": [], "metric": args.metric}
    if args.spherical and args.metric != "spinorama":
        raise ValueError("--spherical applies to the spinorama report.")
    if args.metric == "electrical-impedance" and args.uncorrected:
        raise ValueError("Electrical impedance is a raw voltage-basis quantity. Omit --uncorrected.")
    if args.metric in CATALOG_METRICS:
        if args.response:
            raise ValueError("These desktop reports use saved channels and expose their named traces.")
        data = catalog_report(run, args.metric, frequencies, uncorrected=args.uncorrected, spherical=args.spherical)
        if args.format == "svg":
            from blab.solve_results.figures import catalog_figure, svg

            return svg(catalog_figure(args.metric, data))
        for quantity in data["quantities"].values():
            quantity["values"] = _values(quantity["values"])
        return {
            "run": str(run.root),
            "status": run.manifest["status"],
            "metric": args.metric,
            "uncorrected": args.uncorrected,
            "requested_frequencies_hz": frequencies,
            **data,
        }
    if args.metric != "peak-excursion" and (args.response or args.uncorrected):
        raise ValueError("Drive selection and correction apply to peak-excursion reports.")
    for hz in frequencies:
        if args.metric == "peak-excursion":
            q, _, velocity, meta = run.sample(
                hz, DIAPHRAGM_VELOCITY_ID, response=args.response or "system", uncorrected=args.uncorrected
            )
            ids = q["metadata"].get("component_ids") or run.domain_array(q["target_id"], "component_id").tolist()
            values = peak_excursion_mm(velocity, meta["freq_hz"])
            info = {
                "component_ids": ids,
                "unit": "mm",
                "response": args.response or "system",
                "uncorrected": args.uncorrected,
            }
        elif args.metric == "maximum-spl":
            info, values = maximum_spl(run, hz)
        else:
            info, values = normalized_impedance(run, hz)
        payload["samples"].append(info | {"frequency_hz": run.completed[hz]["freq_hz"], "values": _values(values)})
    if args.format == "svg":
        from blab.solve_results.figures import metric_figure, svg

        return svg(metric_figure(args.metric, payload["samples"]))
    return payload


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="List committed samples, quantities and domains")
    domain = commands.add_parser("domain", help="Read a coordinate or topology array by name")
    query = commands.add_parser("query", help="Extract a quantity by its artifact ID")
    report = commands.add_parser("report", help="Calculate a desktop metric from the saved result")
    plane = commands.add_parser("plane", help="Extract an interior, exterior, or combined observation plane")
    definition = plane.add_mutually_exclusive_group(required=True)
    definition.add_argument("--plane-file", type=Path, help="Plane JSON in saved-project format")
    definition.add_argument("--plane-id", help="Use a plane from the project snapshot")
    plane.add_argument(
        "--backend",
        choices=("beat_cpu", "beat_cuda", "beat_rocm", "beat_metal"),
        default="beat_cpu",
        help="Local BEAT backend for exterior field evaluation",
    )
    plane.add_argument("--julia-executable", default="julia")
    plane.add_argument("--pressure-only", action="store_true", help="Skip exterior gradients and velocity")
    plane.add_argument("--frequency", required=True, type=float)
    for command in (inspect, query, domain, report, plane):
        command.add_argument("run", type=Path)
        command.add_argument("--json", action="store_true", help="JSON output (default)")
    domain.add_argument("domain_id")
    domain.add_argument("field")
    domain.add_argument("--group", choices=("coordinates", "topology"))
    for command in (plane, report):
        command.add_argument("--format", choices=("json", "svg"), default="json")
    for command in (query, report):
        command.add_argument("--frequency", type=float, action="append", help="Requested frequency in Hz (repeatable)")
    query.add_argument("--format", choices=("json", "csv"), default="json")
    query.add_argument("quantity")
    for command in (query, plane):
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--excitation", help="Select one raw excitation port ID")
        selection.add_argument("--response", help="Synthesize system or channel:NAME using saved drive settings")
        command.add_argument(
            "--uncorrected", action="store_true", help="Explicitly disable saved flat-target correction"
        )
    query.add_argument("--index", nargs=2, metavar=("AXIS", "INDEX"), action="append", default=[])
    query.add_argument("--view", choices=("raw", "magnitude", "phase", "spl", "peak-excursion"), default="raw")
    report.add_argument("metric", choices=("normalized-impedance", "maximum-spl", "peak-excursion", *CATALOG_METRICS))
    inspect.add_argument(
        "--diagnostics", type=float, metavar="HZ", help="Include solver diagnostics for one committed sample"
    )
    report.add_argument(
        "--spherical", action="store_true", help="Use retained sphere samples for spinorama sound power"
    )
    report.add_argument("--response", help="Peak excursion drive (default: system)")
    report.add_argument(
        "--uncorrected", action="store_true", help="Omit saved flat-target correction for drive-dependent reports"
    )
    args = parser.parse_args(argv)
    try:
        run = ResultRun(args.run)
        if args.command == "inspect":
            payload = run.inspect() | {
                "channel_by_port": run.channel_by_port,
                "saved_planes": run.project["observation_planes"],
            }
            if args.diagnostics is not None:
                payload["diagnostics"] = {
                    "frequency_hz": args.diagnostics,
                    "values": run.metadata(args.diagnostics)["diagnostics"],
                }
        elif args.command == "plane":
            payload = _plane_result(run, args)
        elif args.command == "domain":
            payload = {
                "domain": args.domain_id,
                "field": args.field,
                "values": _values(run.domain_array(args.domain_id, args.field, args.group)),
            }
        else:
            frequencies = args.frequency or list(run.completed)
            if not frequencies:
                raise ValueError("Run has no committed frequencies.")
            payload = {"run": str(run.root), "status": run.manifest["status"], "samples": []}
            if args.command == "query":
                if args.uncorrected and not args.response:
                    raise ValueError("--uncorrected requires --response.")
                payload.update(
                    quantity=args.quantity,
                    view=args.view,
                    response=args.response,
                    excitation=args.excitation,
                    indices=args.index,
                    uncorrected=args.uncorrected,
                    phasor_convention=SOLVER_PHASOR_CONVENTION if args.response else run.manifest["phasor_convention"],
                )
                payload["samples"] = [_query(run, args, hz) for hz in frequencies]
                if args.format == "csv":
                    _write_csv(payload)
                    return
            else:
                payload = _report_result(run, args, frequencies)
        print(payload if isinstance(payload, str) else json.dumps(payload, allow_nan=False))
    except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError) as exc:
        print(json.dumps({"error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
