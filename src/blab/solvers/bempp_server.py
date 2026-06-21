"""Compatibility imports for the neutral HTTP solve-server backend."""

from __future__ import annotations

from blab.solvers.http_server import HttpServerBackend, HttpServerSession

BemppServerBackend = HttpServerBackend
BemppServerSession = HttpServerSession

__all__ = [
    "HttpServerBackend",
    "HttpServerSession",
    "BemppServerBackend",
    "BemppServerSession",
]
