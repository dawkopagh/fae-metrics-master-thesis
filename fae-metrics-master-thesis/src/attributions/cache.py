"""
Disk caching layer for pre-computed attribution maps.

Responsibilities (see docs/thesis_plan.md §6, Attribution Generation):
    - Save per-(model, FAE method, image_id) attribution maps to
      attributions_cache/ in a versioned, content-addressed format
    - Load cached attributions and return them in the canonical (H, W) shape
    - Invalidate cache entries when the model checkpoint or FAE configuration changes
    - Avoid recomputing attributions that already exist on disk

Not responsible for: generating attributions from scratch (see generate.py),
ensembling attribution maps (see ensembling.py),
or metric computation (see metrics/).
"""
