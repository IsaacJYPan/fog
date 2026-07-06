"""Environment: poke-env wrappers, POC formats, and the vectorized self-play runner (spec §0).

The Showdown simulator (Node) is the throughput bottleneck, not the GPU. The runner spins up N parallel
Node Showdown processes via poke-env's async interface, all pushing trajectories to a single GPU
learner over a queue. Measure battles/sec on day one (M0). Every schedule in the spec is stated in
battles, not wall-clock.
"""

from fog.env.player import FogPlayer

__all__ = ["FogPlayer"]
