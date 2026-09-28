"""Amenity-richness scoring prompt (M2-03)."""

AMENITY_SCORE_INSTRUCTIONS = """\
Score the amenity richness from 0 to 5 (integer):
0 = none mentioned
1-2 = basic (e.g. parking, shared pool)
3-4 = good (gym + pool + parking + security)
5 = exceptional (gym + pool + parking + concierge + spa/beach/extras)
"""
