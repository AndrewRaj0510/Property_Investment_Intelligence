"""Finish-tier classification prompt (M2-01)."""

FINISH_TIER_INSTRUCTIONS = """\
You classify the interior finish quality of a Dubai rental property from its
listing description into exactly one of: Basic, Mid, High, Luxury.

Guidance:
- Basic: builder-standard, functional, no upgrades mentioned.
- Mid: tasteful/modern finishing, some upgrades.
- High: high-end upgraded finishes, premium fittings.
- Luxury: bespoke, ultra-premium materials, designer fit-out.

Examples:
- "functional builder-standard unit" -> Basic
- "modern finishing with built-in wardrobes" -> Mid
- "high-end upgraded kitchen and marble bathrooms" -> High
- "bespoke luxury fit-out with imported Italian marble" -> Luxury
"""
