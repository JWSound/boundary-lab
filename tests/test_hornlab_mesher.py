"""The HornLab waveguide mesher fast path and its fallback to Ath."""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.ath import (
    AthRunResult,
    HornlabWaveguideGenerationError,
    generate_waveguide_mesh,
    run_hornlab_waveguide,
)

SUPPORTED_CONFIG = "Throat.Diameter = 25.4\nMesh.Quadrants = 1234\n"


def _quadrant_mesh(path: Path) -> None:
    """A two-triangle patch tagged as the Ath driven diaphragm surface."""
    meshio.write(
        path,
        meshio.Mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
                dtype=np.float64,
            ),
            cells=[("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))],
            cell_data={"gmsh:physical": [np.array([2, 2], dtype=np.int32)]},
            field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
        ),
        file_format="gmsh22",
        binary=False,
    )


class FakeMesherRunner:
    """Stands in for the hornlab_mesher subprocess."""

    def __init__(self, **result_overrides) -> None:
        self.result = {
            "mesher_version": "0.1.0",
            "formula": "osse",
            "mode": "bare",
            "units": "mm",
            "quadrants": "1234",
            "n_vertices": 4,
            "n_triangles": 2,
            **result_overrides,
        }
        self.calls: list[tuple[Path, Path]] = []

    def run(self, config_path: Path, msh_path: Path, *, timeout_s: float | None = None) -> dict:
        self.calls.append((config_path, msh_path))
        _quadrant_mesh(msh_path)
        return dict(self.result)

    def stop(self) -> None:  # pragma: no cover - the fast path never cancels here
        pass


def test_mesher_output_becomes_a_normal_ath_run_result(tmp_path: Path) -> None:
    result = run_hornlab_waveguide(
        config_text=SUPPORTED_CONFIG,
        run_root=tmp_path,
        case_name="waveguide",
        runner=FakeMesherRunner(),
    )

    assert isinstance(result, AthRunResult)
    assert result.driven_tag == 2
    assert result.cleaned_msh_path.exists()
    assert result.config_path.read_text(encoding="utf-8") == SUPPORTED_CONFIG
    # The raw mesher mesh is a working file and must not survive the run.
    assert not list(result.output_dir.glob(".*_hornlab_raw.msh"))


def test_each_mesher_run_records_its_package_provenance(tmp_path: Path) -> None:
    result = run_hornlab_waveguide(
        config_text=SUPPORTED_CONFIG,
        run_root=tmp_path,
        case_name="waveguide",
        runner=FakeMesherRunner(),
    )

    provenance = json.loads((result.output_dir / "hornlab_mesher.json").read_text(encoding="utf-8"))
    assert provenance["mesher_version"] == "0.1.0"
    assert provenance["formula"] == "osse"
    assert provenance["units"] == "mm"


def test_a_quadrant_disagreement_never_gets_mirrored(tmp_path: Path) -> None:
    """Mirroring a mesh against the wrong axes would silently change the model."""
    with pytest.raises(HornlabWaveguideGenerationError, match="quadrants"):
        run_hornlab_waveguide(
            config_text="Mesh.Quadrants = 1\n",
            run_root=tmp_path,
            case_name="waveguide",
            runner=FakeMesherRunner(quadrants="1234"),
        )


@pytest.mark.parametrize(
    ("config_text", "reason"),
    (
        ("Throat.Ext.Length = 20\n", "Throat.Ext.Length"),
        ("Source.Contours = {\n", "multi-source"),
        ("LFSource.Diameter = 200\n", "multi-source"),
        ("ABEC.SimType = 2\nMesh.SubdomainSlices = 1\n", "two-subdomain"),
    ),
)
def test_geometry_the_mesher_cannot_build_is_refused_before_meshing(
    tmp_path: Path, config_text: str, reason: str
) -> None:
    runner = FakeMesherRunner()
    with pytest.raises(HornlabWaveguideGenerationError, match=reason):
        run_hornlab_waveguide(config_text=config_text, run_root=tmp_path, runner=runner)
    assert runner.calls == []


@pytest.mark.parametrize(
    "config_text",
    (
        "Throat.Ext.Length = 0\n",
        "Throat.Ext.Angle = 20\n",
        "Source.VelocityProfile = profile.txt\n",
        "ABEC.SimType = 1\nMesh.SubdomainSlices = 1\n",
    ),
)
def test_supported_geometry_is_not_diverted_to_ath(tmp_path: Path, config_text: str) -> None:
    runner = FakeMesherRunner()
    run_hornlab_waveguide(config_text=config_text, run_root=tmp_path, runner=runner)
    assert len(runner.calls) == 1


def test_unsupported_geometry_falls_back_to_the_ath_toolchain(tmp_path: Path) -> None:
    ath_result = AthRunResult(
        output_dir=tmp_path,
        msh_path=tmp_path / "waveguide.msh",
        config_path=tmp_path / "waveguide.cfg",
        driven_tag=2,
        radiators=(),
    )
    statuses: list[str] = []

    class FakeAthRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, **_kwargs) -> AthRunResult:
            self.calls += 1
            return ath_result

    ath_runner = FakeAthRunner()
    mesher_runner = FakeMesherRunner()

    result = generate_waveguide_mesh(
        ath_exe=tmp_path / "ath202608.exe",
        config_text="Throat.Ext.Length = 20\n",
        run_root=tmp_path,
        runner=ath_runner,
        mesher_runner=mesher_runner,
        status_callback=statuses.append,
    )

    assert result is ath_result
    assert ath_runner.calls == 1
    assert mesher_runner.calls == []
    assert any("running Ath" in status for status in statuses)


def test_a_failing_ath_fallback_reports_both_failures(tmp_path: Path) -> None:
    class FailingAthRunner:
        def run(self, **_kwargs) -> AthRunResult:
            raise RuntimeError("Ath failed with exit code 1")

    with pytest.raises(RuntimeError) as excinfo:
        generate_waveguide_mesh(
            ath_exe=tmp_path / "ath202608.exe",
            config_text="Throat.Ext.Length = 20\n",
            run_root=tmp_path,
            runner=FailingAthRunner(),
            mesher_runner=FakeMesherRunner(),
        )

    message = str(excinfo.value)
    assert "Throat.Ext.Length" in message
    assert "exit code 1" in message
