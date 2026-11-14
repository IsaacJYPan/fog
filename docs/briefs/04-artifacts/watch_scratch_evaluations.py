"""Score finished 100k snapshots during the fixed-team run, using one evaluation CPU thread."""
from pathlib import Path
import time

from run_scratch import evaluate


def main():
    work = Path('runs/rl_check/scratch/checkpoints')
    while not (work/'ppo_final.pt').exists():
        checkpoints = sorted(
            [p for p in work.glob('ppo_[0-9]*.pt') if '.state.' not in p.name],
            key=lambda p: int(p.stem.split('_')[1]),
        )
        for weights in checkpoints:
            # The sidecar is opened only after the weights have finished writing.
            if weights.with_name(weights.stem+'.state.pt').exists():
                evaluate('scratch', weights.stem.split('_')[1], weights)
        time.sleep(20)


if __name__ == '__main__':
    main()
