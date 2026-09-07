"""Engine-owned worker negotiation, independent of application models."""

from __future__ import annotations

from . import COMPILED_SYSTEM_VERSION, SYSTEM_RESULT_VERSION, SYSTEM_SOLVE_REQUEST_VERSION, validate_solve_request

WORKER_PROTOCOL_VERSION = 1
FIELD_ARRAY_VERSION = 1


class WorkerCompatibilityError(RuntimeError):
    """The running engine cannot satisfy the client's wire requirements."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkerCompatibilityError(f"Incompatible BEAT worker: {message}")


def validate_worker_ready(info: dict) -> None:
    _require(info.get("type") == "ready", "expected a ready announcement.")
    protocol = info.get("protocol")
    _require(isinstance(protocol, dict), "missing versioned handshake; update the engine worker.")
    _require(protocol.get("name") == "beat-worker", "unknown worker protocol.")
    _require(
        type(protocol.get("version")) is int and protocol["version"] == WORKER_PROTOCOL_VERSION,
        f"worker protocol version must be {WORKER_PROTOCOL_VERSION}.",
    )
    engine = info.get("engine")
    _require(
        isinstance(engine, dict)
        and engine.get("name") == "BEAT Engine"
        and isinstance(engine.get("version"), str)
        and bool(engine["version"]),
        "missing engine identity/version.",
    )
    contracts = info.get("contracts")
    _require(isinstance(contracts, dict), "missing contract versions.")
    for name in ("system_request", "compiled_system", "system_result", "field_array"):
        versions = contracts.get(name)
        _require(
            isinstance(versions, list)
            and bool(versions)
            and all(type(value) is int and value > 0 for value in versions),
            f"invalid {name} versions.",
        )
    for name in ("operations", "precisions", "solve_kinds"):
        values = info.get(name)
        _require(
            isinstance(values, list) and all(isinstance(value, str) and value for value in values),
            f"invalid {name} capabilities.",
        )
    backends = info.get("backends")
    _require(isinstance(backends, dict) and bool(backends), "missing backend availability.")
    for name, backend in backends.items():
        _require(isinstance(backend, dict) and type(backend.get("available")) is bool, f"invalid {name} availability.")
        _require(isinstance(backend.get("reason", ""), str), f"invalid {name} availability reason.")


def negotiate_submission(info: dict, request: dict, operation: str) -> dict:
    """Select the client's supported formats or fail before writing a command."""
    validate_worker_ready(info)
    _require(operation in info["operations"], f"operation {operation!r} is unavailable.")
    contracts = info["contracts"]
    command = {"protocol_version": WORKER_PROTOCOL_VERSION}
    if operation == "solve":
        validate_solve_request(request)
        for name, version in (
            ("system_request", SYSTEM_SOLVE_REQUEST_VERSION),
            ("compiled_system", COMPILED_SYSTEM_VERSION),
            ("system_result", SYSTEM_RESULT_VERSION),
        ):
            _require(version in contracts[name], f"{name} version {version} is unavailable.")
        command["result_schema_version"] = SYSTEM_RESULT_VERSION
        options = request["solver_options"]
        kinds = {region["kind"] for region in request["compiled_system"]["regions"]}
        kind = (
            "interior_fem"
            if "unbounded_air" not in kinds
            else "exterior_bem"
            if "bounded_air" not in kinds
            else "coupled_fem_bem_lem"
        )
        _require(kind in info["solve_kinds"], f"solve kind {kind!r} is unavailable.")
        if "cancel_path" in request:
            _require(info.get("cancellation") == "marker_file", "marker-file cancellation is unavailable.")
        default_precision = "float32" if kind == "exterior_bem" else "float64"
        # Interior FEM uses CPU independently of the exterior BEM option.
        backend_name = "cpu" if kind == "interior_fem" else str(options.get("bem_backend", "cpu")).lower()
    elif operation == "bem_field":
        _require(FIELD_ARRAY_VERSION in contracts["field_array"], "field array version 1 is unavailable.")
        _require(
            type(request.get("binary_array_schema_version")) is int
            and request["binary_array_schema_version"] == FIELD_ARRAY_VERSION,
            "field request requires binary array version 1.",
        )
        command["field_array_schema_version"] = FIELD_ARRAY_VERSION
        options = request
        default_precision = "float32"
        backend_name = str(options.get("bem_backend", "cpu")).lower()
    else:
        raise WorkerCompatibilityError(f"Unsupported client operation: {operation}")
    precision = str(options.get("precision", default_precision)).lower()
    if operation == "solve" and kind != "exterior_bem":
        precision = {"complex64": "float32", "complex128": "float64"}.get(precision, precision)
    _require(precision in info["precisions"], f"precision {precision!r} is unavailable.")
    backend = info["backends"].get(backend_name, {})
    _require(
        backend.get("available") is True,
        f"backend {backend_name!r} is unavailable: {backend.get('reason', 'not advertised by this worker')}",
    )
    return command


def validate_worker_event(event: dict) -> None:
    """Reject a negotiated worker that sends a different result representation."""
    if event.get("type") == "result":
        result = event.get("result")
        _require(
            isinstance(result, dict)
            and type(result.get("schema_version")) is int
            and result["schema_version"] == SYSTEM_RESULT_VERSION,
            f"response must use selected system_result version {SYSTEM_RESULT_VERSION}.",
        )
    elif event.get("type") == "field_result":
        _require(isinstance(event.get("values_binary"), dict), "response must use selected binary field array format.")
