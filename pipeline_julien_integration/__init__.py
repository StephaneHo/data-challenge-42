"""Module d'integration : v2 simple fallback pour la pipeline de Julien.

Usage :
    from pipeline_julien_integration import apply_v2_fallback

Voir fallback_v2.py pour la documentation complete d'integration.
"""
from .fallback_v2 import apply_v2_fallback, detect_plante_regime

__all__ = ["apply_v2_fallback", "detect_plante_regime"]
