"""State encoding. Decision-point observation = (entities, field, history, my_team_sheet).

  * `entities`  -> [12, D_ent]  (§2.1) slots 0-5 mine, 6-11 opponent
  * `field`     -> [D_field]    (§2.2) ~40 dims
  * `history`   -> [T, D_turn]  (§2.3) T<=60, padded/truncated from the front

`my_team_sheet` is not a separate input: it lives inside entity slots 0-5 with all epistemic flags
set "known". Unknown vs. known-absent is distinguished by the epistemic flags. Submitted nature
and EVs are represented natively; each EV is stored as its exact stat-contribution unit EV//4.
"""

from fog.encoding.dims import DIMS, Dims
from fog.encoding.observation import Observation

__all__ = ["DIMS", "Dims", "Observation"]
