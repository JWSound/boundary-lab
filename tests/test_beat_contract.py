"""BEAT's model-independent contract and the Boundary Lab projection."""

import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from blab.solvers.beat_contract import validate_compiled_system, validate_solve_request

CONTRACT = Path(__file__).resolve().parents[1] / "src/blab/solvers/beat_contract"
CORPUS = json.loads((CONTRACT / "conformance.json").read_text())


def test_schema_uses_only_keywords_supported_by_both_validators():
    supported = {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "minimum",
        "exclusiveMinimum",
    }

    def check(schema):
        assert set(schema) <= supported
        if "$ref" in schema:
            assert schema["$ref"].startswith("#/$defs/")
            assert set(schema) <= {"$ref", "title", "description", "$schema", "$id", "$defs"}
        if "additionalProperties" in schema:
            assert isinstance(schema["additionalProperties"], bool)
        for key in ("properties", "$defs"):
            for child in schema.get(key, {}).values():
                check(child)
        if "items" in schema:
            check(schema["items"])

    check(json.loads((CONTRACT / "system-v1.schema.json").read_text()))


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
def test_engine_conformance_cases(case):
    request = copy.deepcopy(CORPUS["base_request"])
    for change in case["changes"]:
        parent = request
        for key in change["path"][:-1]:
            parent = parent[key]
        key = change["path"][-1]
        if change.get("remove"):
            del parent[key]
        else:
            parent[key] = change["value"]
    if case["valid"]:
        validate_solve_request(request)
    else:
        with pytest.raises(ValueError, match="BEAT contract"):
            validate_solve_request(request)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_or_nonfinite_extension_values_are_rejected(value):
    request = copy.deepcopy(CORPUS["base_request"])
    request["solver_options"]["custom"] = value
    with pytest.raises(ValueError, match="finite JSON"):
        validate_solve_request(request)


def test_adapter_does_not_leak_new_application_dataclass_fields():
    from blab.physical_model import CompiledPhysicalSystem
    from blab.system_contract import compiled_system_from_dict, compiled_system_to_dict

    @dataclass(frozen=True)
    class FutureApplicationSystem(CompiledPhysicalSystem):
        editor_selection: str = "must stay in the application"

    system = compiled_system_from_dict(CORPUS["base_request"]["compiled_system"])
    future = FutureApplicationSystem(**vars(system))
    assert compiled_system_to_dict(future) == compiled_system_to_dict(system)
    assert "editor_selection" not in compiled_system_to_dict(future)


def test_adapter_rejects_bad_wire_before_coercing_it():
    from blab.system_contract import system_solve_request_from_dict

    request = copy.deepcopy(CORPUS["base_request"])
    request["frequencies_hz"] = [True]
    with pytest.raises(ValueError, match="expected number"):
        system_solve_request_from_dict(request)


def test_contract_loads_standalone_and_validates_without_application_or_numpy():
    code = """
import importlib.abc
import importlib.util
import json
import pathlib
import sys

class StandardLibraryOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            raise AssertionError(fullname)

sys.meta_path.insert(0, StandardLibraryOnly())
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('independent_contract', path / '__init__.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = json.loads((path / 'example-exterior-request.json').read_text())
module.validate_solve_request(request)
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(CONTRACT)], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr


def test_compiled_contract_requires_version_even_without_request():
    system = copy.deepcopy(CORPUS["base_request"]["compiled_system"])
    del system["contract_version"]
    with pytest.raises(ValueError, match="contract_version"):
        validate_compiled_system(system)
