# SPDX-License-Identifier: Apache-2.0

from .quark_scheme import QuarkLinearScheme, QuarkMoEScheme


def _missing_scheme(name: str, exc: Exception):
    class _MissingScheme:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"{name} is unavailable because an optional dependency is missing: {exc}"
            ) from exc

        @classmethod
        def get_min_capability(cls) -> int:
            return 0

    _MissingScheme.__name__ = name
    return _MissingScheme


try:
    from .quark_w4a4_mxfp4 import QuarkW4A4MXFP4
except ModuleNotFoundError as exc:
    QuarkW4A4MXFP4 = _missing_scheme("QuarkW4A4MXFP4", exc)

try:
    from .quark_w4a4_mxfp4_moe import QuarkW4A4MXFp4MoE
except ModuleNotFoundError as exc:
    QuarkW4A4MXFp4MoE = _missing_scheme("QuarkW4A4MXFp4MoE", exc)

try:
    from .quark_w8a8_fp8 import QuarkW8A8Fp8
except ModuleNotFoundError as exc:
    QuarkW8A8Fp8 = _missing_scheme("QuarkW8A8Fp8", exc)

try:
    from .quark_w8a8_fp8_moe import QuarkW8A8FP8MoE
except ModuleNotFoundError as exc:
    QuarkW8A8FP8MoE = _missing_scheme("QuarkW8A8FP8MoE", exc)

__all__ = [
    "QuarkLinearScheme",
    "QuarkMoEScheme",
    "QuarkW4A4MXFP4",
    "QuarkW8A8Fp8",
    "QuarkW4A4MXFp4MoE",
    "QuarkW8A8FP8MoE",
]
