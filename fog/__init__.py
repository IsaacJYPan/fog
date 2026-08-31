"""FOG — a Gen 9 OU battle agent that plays the fog of war in both directions.

Encodes the acting player's private team, the current public board, and ordered public turn history
separately. A battle-conditioned set-autoregressive belief model reads the opponent's hidden team;
policy and value share the same three legal information sources.

See ``ARCHITECTURE.md`` for the tracked design and implementation boundary.
"""

__version__ = "0.0.1"
