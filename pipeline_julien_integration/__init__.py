"""Module d'integration : v2 simple fallback pour la pipeline de Julien.

Usage minimal (sans dampening, bit-exact sur cas normaux) :
    from pipeline_julien_integration import apply_v2_fallback

Usage avec dampening hair+hat (+2% de gain en CV) :
    from pipeline_julien_integration import apply_v2_fallback_with_dampening

Voir fallback_v2.py et README.md pour la documentation complete d'integration.
"""
from .fallback_v2 import (
    apply_v2_fallback,
    apply_v2_fallback_with_dampening,
    detect_plante_regime,
)

__all__ = [
    "apply_v2_fallback",
    "apply_v2_fallback_with_dampening",
    "detect_plante_regime",
]
