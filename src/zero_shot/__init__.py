"""Zero-shot face occlusion estimation via face parsing.

Pipeline:
  image -> SegFormer face parsing (19 classes) -> feature extraction (per-class areas)
        -> linear calibration to IDEMIA labels -> predicted occlusion ratio
"""
