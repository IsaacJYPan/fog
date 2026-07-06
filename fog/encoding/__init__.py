"""State encoding (spec §2). Decision-point observation = (entities, field, history, my_team_sheet).

  * `entities`  -> [12, D_ent]  (§2.1) slots 0-5 mine, 6-11 opponent
  * `field`     -> [D_field]    (§2.2) ~40 dims
  * `history`   -> [T, D_turn]  (§2.3) T<=60, padded/truncated from the front

`my_team_sheet` is not a separate input: it lives inside entity slots 0-5 with all epistemic flags set
"known". Unknown vs. known-absent is distinguished by the epistemic flags. The only lossy step anywhere
is spread clustering (§1.6), verified in tests.
"""

from fog.encoding.dims import Dims, DIMS
from fog.encoding.observation import Observation

__all__ = ["Dims", "DIMS", "Observation"]
