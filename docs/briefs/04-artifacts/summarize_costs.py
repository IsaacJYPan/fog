"""Summarize brief-04 costs from saved measurements; launches no experiments."""
import csv
import json
from pathlib import Path


ARTIFACTS = Path(__file__).parent


def read(name):
    return json.loads((ARTIFACTS / name).read_text())


def training_cost(decisions, wall, collection, update, optimization, optimizer_rows):
    parts = {
        'collection': collection,
        'update': update,
        'outside_training_loop': wall - collection - update,
    }
    assert min(parts.values()) >= 0
    return {
        'decisions': decisions,
        'wall_seconds': wall,
        'end_to_end_decisions_per_second': decisions / wall,
        'optimization_seconds': optimization,
        'optimizer_rows': optimizer_rows,
        'effective_epochs': optimizer_rows / decisions,
        'optimization_rows_per_second': optimizer_rows / optimization,
        'parts': {
            name: {
                'seconds': seconds,
                'minutes_per_training_hour': 60 * seconds / wall,
                'hypothetical_decisions_per_second_if_free': decisions / (wall - seconds),
            } for name, seconds in parts.items()
        },
    }


def main():
    stage0 = read('stage0-summary.json')
    stage1 = read('scratch-summary.json')
    profile = read('stage0-profile.json')
    result = {'smoke': True}
    for name, data in [('stage0', stage0), ('stage1', stage1)]:
        result[name] = training_cost(
            data['decisions'], data['wall_seconds'], data['collection_seconds'],
            data['update_seconds'], data['optimization_seconds'], data['optimizer_rows'],
        )
    shares = {
        'node_roundtrip_and_json': profile['node_roundtrip_seconds'] / profile['profiled_seconds'],
        'observation_building': profile['observation_seconds'] / profile['profiled_seconds'],
        'acting_forward': profile['forward_seconds'] / profile['profiled_seconds'],
    }
    shares['other_collection'] = 1 - sum(shares.values())
    projected_parts = {name: stage0['collection_seconds'] * share for name, share in shares.items()}
    projected_parts['update'] = stage0['update_seconds']
    projected_parts['outside_training_loop'] = (
        stage0['wall_seconds'] - stage0['collection_seconds'] - stage0['update_seconds']
    )
    result['stage0_collection_projection'] = {
        'method': 'Apply single-worker cProfile wall shares to four-worker collection wall time; '
                  'approximate projection, not a four-worker trace. Node CPU time overlaps and '
                  'is not added. Free-part rates are Amdahl arithmetic, not measured speedups.',
        'profile_shares': shares,
        'parts': {
            name: {
                'projected_seconds': seconds,
                'minutes_per_training_hour': 60 * seconds / stage0['wall_seconds'],
                'hypothetical_decisions_per_second_if_free': (
                    stage0['decisions'] / (stage0['wall_seconds'] - seconds)
                ),
            } for name, seconds in projected_parts.items()
        },
    }
    result['stage1']['first_observed_decisions_per_ten_points'] = (
        stage1['first_decisions_per_ten_points']
    )
    result['stage1']['full_budget_decisions_per_ten_points'] = (
        stage1['final_decisions_per_ten_points']
    )
    if (ARTIFACTS / 'blueprint-summary.json').exists():
        stage2 = read('blueprint-summary.json')
        commands = [json.loads(line) for line in (ARTIFACTS / 'commands.jsonl').read_text().splitlines()]
        attempts = [row for row in commands if 'fog.train --config runs/rl_check/blueprint/config.yaml'
                    in row['command']]
        assert attempts
        rows = list(csv.DictReader((ARTIFACTS / 'blueprint-metrics.csv').open()))
        logged = int(float(rows[-1]['decisions']))
        assert logged == stage2['last_logged_decisions']
        sums = {key: sum(float(row[key]) for row in rows) for key in (
            'collection_seconds', 'update_seconds', 'optimization_seconds', 'optimizer_rows',
        )}
        result['stage2'] = training_cost(
            logged, sum(row['seconds'] for row in attempts), sums['collection_seconds'],
            sums['update_seconds'], sums['optimization_seconds'], sums['optimizer_rows'],
        )
        result['stage2']['checkpoint_decisions'] = stage2['checkpoint_decisions']
        result['stage2']['attempts'] = len(attempts)
        result['stage2']['wall_note'] = (
            'Trainer-process wall time observed by controller, including polling overhead and '
            'CPU contention from intermediate evaluations; baseline/final evaluation excluded.'
        )
        baseline = next(row for row in read('blueprint-0-pool_a.json')['rows']
                        if row['b'] == 'simpleheuristics')
        points = []
        for path in sorted(ARTIFACTS.glob('blueprint-*-pool_a.json'),
                           key=lambda p: int(p.stem.split('-')[1])):
            data = json.loads(path.read_text())
            assert data['status'] == 'complete'
            row = next(row for row in data['rows'] if row['b'] == 'simpleheuristics')
            gain = row['a_win_rate_decisive'] - baseline['a_win_rate_decisive']
            counter = int(path.stem.split('-')[1])
            if gain >= .1 and row['wilson_95_low'] > baseline['wilson_95_high']:
                points.append({'decisions': counter, 'gain_percentage_points': 100 * gain,
                               'decisions_per_ten_points': counter / (10 * gain)})
        result['stage2']['passing_training_pool_points'] = points
        result['stage2']['cost_note'] = (
            'PPO decisions only; excludes blueprint reconstruction. First observed checkpoint '
            'does not identify the first actual crossing. No held-out gain is implied.'
        )
    (ARTIFACTS / 'cost-breakdown.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
