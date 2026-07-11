"""FOG — a Gen 9 OU battle agent that plays the fog of war in both directions.

Reads the opponent's hidden team via a battle-conditioned belief model, and controls what it reveals of
its own via an information-leakage penalty (measured against oracle upper bounds).

See the build spec (docstring in each subpackage) and README.md for what exists and how to run it.
"""

__version__ = "0.0.1"
