# Upstream components

FOG's native integration uses these components without claiming their original playing strength as a project contribution:

| Component | Source | License |
| --- | --- | --- |
| Metamon 1.6.0 | [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) | MIT |
| AMAGO 3.4.0 | [UT-Austin-RPL/amago](https://github.com/UT-Austin-RPL/amago) | MIT |
| Native poke-env fork | [UT-Austin-RPL/poke-env](https://github.com/UT-Austin-RPL/poke-env) | MIT |
| Kakuna checkpoint 34 | [jakegrigsby/metamon](https://huggingface.co/jakegrigsby/metamon) | Apache-2.0 |
| Foul Play | [pmariglia/foul-play](https://github.com/pmariglia/foul-play) | GPL-3.0 |
| Pokémon Showdown simulator | [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) | MIT |

The Foul Play bridge, sampling adapter, and process-pool adapter in this directory are distributed under GPL-3.0, with their license in `LICENSE.foul-play`. They run in a separate process and import a pinned upstream checkout. The sampling adapter replaces an unbounded move-proposal rejection loop with its equivalent first-success distribution, terminating when remaining proposal probabilities are zero. The process adapter reuses a spawn-based search pool, avoiding Linux forks from the asynchronous decision thread. Both adapters check the pinned source hash; upstream source files, search budgets, and MCTS move selection remain unchanged. The remaining FOG modules use the repository's MIT license. Downloaded weights retain their Apache-2.0 license, supplied in `LICENSE.kakuna`; they are excluded from Git.

Private experiment packages retain upstream license files, pinned source revisions, and checkpoint/data provenance. Model and data downloads are separately attributed in each manifest; their licenses are separate from the FOG source license.
