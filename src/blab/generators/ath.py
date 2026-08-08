"""Ath implementation of the geometry-generator contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from blab.ath import (
    AthCancelledError,
    AthProcessRunner,
    AthRunResult,
    ath_mirror_axes_for_result,
    clean_ath_mesh_output,
    detect_ath_radiators,
    find_physical_tag_by_name,
)
from blab.generators.base import (
    GeneratedGeometry,
    GenerationCancelledError,
    GenerationRequest,
    GeneratorBackend,
    GeneratorCapabilities,
    GeneratorDocument,
    GeneratorSession,
)

ATH_PROVIDER_ID = "ath"
ATH_SOURCE_FORMAT = "ath_cfg"
ATH_PROVIDER_SCHEMA_VERSION = 1


def ath_source(config_text: str) -> dict[str, Any]:
    return {"format": ATH_SOURCE_FORMAT, "text": str(config_text)}


def ath_source_text(document: GeneratorDocument) -> str:
    if document.provider_id != ATH_PROVIDER_ID:
        raise ValueError(f"Document {document.id!r} is not an Ath design.")
    source_format = str(document.source.get("format", ATH_SOURCE_FORMAT))
    if source_format != ATH_SOURCE_FORMAT:
        raise ValueError(f"Unsupported Ath source format: {source_format}")
    return str(document.source.get("text", ""))


def with_ath_source_text(document: GeneratorDocument, config_text: str) -> GeneratorDocument:
    return replace(document, source=ath_source(config_text))


def ath_result_to_generated_geometry(result: AthRunResult) -> GeneratedGeometry:
    return GeneratedGeometry(
        provider_id=ATH_PROVIDER_ID,
        output_dir=result.output_dir,
        mesh_path=result.msh_path,
        source_path=result.config_path,
        radiators=result.radiators,
        cleaned_mesh_path=result.cleaned_msh_path,
        reduced_cleaned_mesh_path=result.reduced_cleaned_msh_path,
        mirror_axes=ath_mirror_axes_for_result(result),
        quality_warning=result.quality_warning,
        provider_metadata={"driven_tag": result.driven_tag},
    )


def generated_geometry_to_ath_result(result: GeneratedGeometry) -> AthRunResult:
    if result.provider_id != ATH_PROVIDER_ID:
        raise ValueError(f"Cannot convert provider {result.provider_id!r} to an Ath result.")
    driven_tag = result.provider_metadata.get("driven_tag")
    if driven_tag is None:
        driven_tag = find_physical_tag_by_name(result.solver_mesh_path, "SD1D1001")
    return AthRunResult(
        output_dir=result.output_dir,
        msh_path=result.mesh_path,
        config_path=result.source_path or result.output_dir / "config.txt",
        driven_tag=int(driven_tag),
        radiators=result.radiators,
        cleaned_msh_path=result.cleaned_mesh_path,
        reduced_cleaned_msh_path=result.reduced_cleaned_mesh_path,
        quality_warning=result.quality_warning,
    )


class AthGeneratorSession:
    def __init__(self, request: GenerationRequest, *, ath_exe: Path):
        self.request = request
        self.ath_exe = ath_exe
        self._runner = AthProcessRunner()

    def generate(self, *, status_callback=None, stop_requested=None) -> GeneratedGeometry:
        if status_callback is not None:
            status_callback("Running Ath...")
        try:
            raw_result = self._runner.run(
                ath_exe=self.ath_exe,
                config_text=str(self.request.source.get("text", "")),
                run_root=self.request.run_root,
                case_name=self.request.case_name,
            )
            if stop_requested is not None and stop_requested():
                raise GenerationCancelledError("Geometry generation cancelled")
            if status_callback is not None:
                status_callback("Cleaning generated mesh...")
            result = ath_result_to_generated_geometry(clean_ath_mesh_output(raw_result))
            if stop_requested is not None and stop_requested():
                raise GenerationCancelledError("Geometry generation cancelled")
            return result
        except AthCancelledError as exc:
            raise GenerationCancelledError(str(exc)) from exc

    def stop(self) -> None:
        self._runner.stop()


class AthGeneratorBackend(GeneratorBackend):
    provider_id = ATH_PROVIDER_ID
    label = "Ath"
    capabilities = GeneratorCapabilities(source_formats=(ATH_SOURCE_FORMAT,))

    def __init__(self, *, ath_exe: str | Path | None = None, **_options: Any):
        self.ath_exe = None if ath_exe is None else Path(ath_exe)

    def create_session(self, request: GenerationRequest) -> GeneratorSession:
        if request.provider_id != self.provider_id:
            raise ValueError(f"Ath backend cannot generate provider {request.provider_id!r}.")
        source_format = str(request.source.get("format", ATH_SOURCE_FORMAT))
        if source_format != ATH_SOURCE_FORMAT:
            raise ValueError(f"Unsupported Ath source format: {source_format}")
        if self.ath_exe is None:
            raise ValueError("Ath executable path was not provided.")
        return AthGeneratorSession(request, ath_exe=self.ath_exe)

    def restore(self, document: GeneratorDocument) -> GeneratedGeometry | None:
        artifact = document.artifact
        if artifact is None:
            return None
        mesh_path = Path(artifact.mesh_path)
        if not mesh_path.exists():
            return None
        cleaned_path = Path(artifact.cleaned_mesh_path) if artifact.cleaned_mesh_path else None
        solver_path = cleaned_path if cleaned_path is not None and cleaned_path.exists() else mesh_path
        try:
            driven_tag = find_physical_tag_by_name(solver_path, "SD1D1001")
            ath_result = AthRunResult(
                output_dir=Path(artifact.output_dir),
                msh_path=mesh_path,
                config_path=(
                    Path(artifact.source_path) if artifact.source_path else Path(artifact.output_dir) / "config.txt"
                ),
                driven_tag=driven_tag,
                radiators=detect_ath_radiators(solver_path),
                cleaned_msh_path=cleaned_path if cleaned_path is not None and cleaned_path.exists() else None,
                reduced_cleaned_msh_path=(
                    Path(artifact.reduced_cleaned_mesh_path)
                    if artifact.reduced_cleaned_mesh_path and Path(artifact.reduced_cleaned_mesh_path).exists()
                    else None
                ),
            )
            return ath_result_to_generated_geometry(ath_result)
        except Exception:
            return None
