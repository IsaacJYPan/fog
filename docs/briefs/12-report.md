# Brief 12: replay/live parity (stage 3 running)

**Answer.** On 200 games (smoke), own PP differed on every baseline row and now matches exactly; every one of 2,566 replacement requests aligns. The live first-action overwrite is fixed, and replay history retains the live 60-turn window. All requested fields exceed 99.5% exact matches except own HP and history: their remaining differences are explained row by row by public HP rounding and undisclosed or overridden submitted choices. The three rollout digests change because history preserves the first choice. Trained play is not yet measured.

| Smoke result | Before | After |
| --- | ---: | ---: |
| Aligned decisions | 24,149 | 26,143 |
| Missing replay / extra replay | 3,195 / 0 | 1,201 / 0 |
| Exact own PP | 0% | 100% |
| Rows containing overwritten live actions | 21,107 | 0 |

CPU only; pinned Showdown `e440c4a`, random R4 weights, seed 12000. The tool captures actual forward tensors, uses exact own sheets only in this diagnostic, and compares serialized loader inputs. It also fixes own-state blanking, nickname aliases, item refresh, Tera-form switch labels and stored request phase. All forced switches are retained, including one-option rows with zero policy cross-entropy. Current-turn history remains excluded, matching live play.

Evidence: `12-artifacts/parity-before.json`, `parity-final.json` (named blocks, largest differences and every residual row); the ledger has the full before/after table. `unrecoverable-choices.txt` demonstrates sleep, faint pre-emption and faster Encore. The eleven model-consumed candidate columns exceed 99.9% exact matches; raw unused-column and stat-fallback differences are documented. 52 covering tests pass; ruff is clean.

Not verified: stage 3 is running—22,000-game v12 reconversion, paired old/new 1,000-game R4 training (same seed), common-row holdout metrics, new replacement metrics, and 400 greedy games against each scripted opponent for each model. No play-strength claim or new gate result is made.
