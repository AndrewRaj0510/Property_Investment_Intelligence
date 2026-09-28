"""View-type classification prompt (M2-02)."""

VIEW_TYPE_INSTRUCTIONS = """\
Classify the primary view of the property into exactly one of:
Sea, Skyline, Community, Pool, None.

- Sea: sea / marina / beach / water views.
- Skyline: city / Burj / skyline views.
- Community: garden / community / park views.
- Pool: overlooking the pool.
- None: no view mentioned.
"""
