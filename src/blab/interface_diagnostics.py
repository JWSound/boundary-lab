"""Construction diagnostics independent of Qt and of saved solver interfaces."""

from dataclasses import dataclass

from blab.interface_conform import InterfaceConformError


@dataclass(frozen=True)
class InterfaceDiagnostic:
    bounded_boundary_id: str
    unbounded_boundary_id: str | None
    status: str
    detail: str


class InterfaceBuildError(InterfaceConformError):
    """An unsuccessful build with identified interface sides."""

    def __init__(self, message: str, diagnostics: tuple[InterfaceDiagnostic, ...]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
