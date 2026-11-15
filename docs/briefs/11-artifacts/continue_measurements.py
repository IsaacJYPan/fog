"""Resume brief 11's fixed source measurements; run from the repository root.

Uses the existing shard runner. Stops for diagnosis on a fidelity or gate failure. A restart
resumes complete four-game blocks. Only stage reports, the ledger and roadmap are published.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess as sp
import sys
import time
from zoneinfo import ZoneInfo

ART = Path('docs/briefs/11-artifacts')
SOURCE = 'aa38e468f03f4aa9fa78d1569984043c058240db'
STATE = ART / 'measurement-progress.json'
ORDER = [(800, 8), (200, 8), (3200, 8), (800, 1), (800, 32),
         (200, 1), (200, 32), (3200, 1), (3200, 32)]
state = json.loads(STATE.read_text()) if STATE.exists() else {
    'source_commit': SOURCE, 'budget_core_seconds': 0.0, 'jobs': {}, 'published': []}


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def source_fixed():
    if sp.check_output(['git', 'diff', SOURCE, '--', 'fog', 'tests']):
        raise RuntimeError('application or test source changed; inspect before resuming')


def command(name, games, shards, seed, budget, k, opponents):
    return [sys.executable, '-m', 'fog.eval.run_shards', '--shards', str(shards),
            '--games', str(games), '--seed', str(seed), '--out-dir', str(ART / name), '--',
            '--pool', 'data/teams/replay_v1/validation', '--device', 'cpu',
            '--controller', 'mcts=fog.search.engine_mcts:showdown_controller',
            '--controller-kwargs', 'mcts=' + json.dumps({'k': k, 'budget_ms': budget,
                                                        'threads': 1}, separators=(',', ':')),
            *[arg for opp in opponents for arg in ('--opponent', opp)],
            '--max-turns', '800', '--progress-every', '4']


def running_smoke():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'fog.eval.run_shards' in args and str(ART / 'fidelity-smoke').encode() in args:
            return True
    return False


def run(name, games, shards, seed, budget, k, opponents, capped=False):
    source_fixed()
    merged = ART / name / 'merged.json'
    if merged.exists():
        return json.loads(merged.read_text())
    argv = command(name, games, shards, seed, budget, k, opponents)
    prior = state['jobs'].get(name, {})
    job = {**prior, 'command': shlex.join(argv), 'fog_commit': sp.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'status': 'running'}
    state['jobs'][name] = job
    save()
    cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = float(state['budget_core_seconds'])
    started = time.monotonic()
    with (ART / (name + '.txt')).open('a') as log:
        process = sp.Popen(argv, stdout=log, stderr=sp.STDOUT, start_new_session=True)
        job['pid'] = process.pid
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if capped:
                    # Charge every allocated shard core while the cell is live, including
                    # startup and a finished shard waiting for the slowest. This is a
                    # conservative wall-core budget; actual child CPU is retained too.
                    state['budget_core_seconds'] = before + shards * elapsed
                job['wall_seconds_this_attempt'] = elapsed
                save()
                if capped and state['budget_core_seconds'] >= 40 * 3600:
                    job['status'] = 'stopped_at_40_core_hours'
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except sp.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    save()
                    return None
                try:
                    remaining = ((40 * 3600 - state['budget_core_seconds']) / shards
                                 if capped else 10.0)
                    process.wait(timeout=max(0.01, min(10.0, remaining)))
                except sp.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    job['wall_seconds_this_attempt'] = time.monotonic() - started
    if capped:
        state['budget_core_seconds'] = before + shards * job['wall_seconds_this_attempt']
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    job['child_cpu_seconds_this_attempt'] = (after.ru_utime + after.ru_stime
                                           - cpu_start.ru_utime - cpu_start.ru_stime)
    job['status'] = 'complete' if process.returncode == 0 else 'failed'
    job['exit_code'] = process.returncode
    save()
    if process.returncode:
        raise RuntimeError(f'{name}: exit {process.returncode}; resume the same script')
    return json.loads(merged.read_text())


def qualified(row, minimum=0, threshold=0.0):
    fidelity = row['a_fidelity']
    return (row['finished_games'] >= minimum and row['failures'] == 0
            and row['truncated'] == 0 and row['a_score'] >= threshold
            and fidelity['fallback_rate'] < 0.05 and fidelity['rejected'] == 0
            and fidelity['order_errors'] == 0)


def write_report():
    def describe(name):
        path = ART / name / 'merged.json'
        if not path.exists():
            return 'not measured'
        return '; '.join(
            f"{r['b']}: {r['completed_games']} games, {r['a_win_rate_decisive']:.3%} "
            f"[95% {r['wilson_95_low']:.3%}, {r['wilson_95_high']:.3%}], "
            f"fallbacks {r['a_fidelity']['fallbacks']}/{r['a_fidelity']['decisions']}, "
            f"rejected {r['a_fidelity']['rejected']}, "
            f"{r['a_fidelity']['seconds_per_decision']:.3f} s/decision"
            for r in json.loads(path.read_text())['rows'])
    cells = []
    for budget, k in ORDER:
        path = ART / f'budget-{budget}-k{k}' / 'merged.json'
        if path.exists():
            row = json.loads(path.read_text())['rows'][0]
            cells.append((row['a_score'], budget, k, row))
    best = max(cells, key=lambda cell: cell[0]) if cells else None
    best_text = (f"best observed complete cell: {best[1]} ms, K={best[2]}, "
                 f"{best[0]:.3%}; intervals overlap, so this does not rank close cells."
                 if best else 'No budget cell has completed.')
    status = state.get('status', 'in progress')
    text = f'''# Brief 11: EngineMCTS gate G1

**Answer:** G1 {status}. Gate: {describe('gate')}. Budget curve: {best_text}

**Defects fixed.** Teal-Tera Ogerpon matches its submitted-world slot after re-entry;
active identity, HP and used Tera survive. Pending public charge turns reach the engine,
with Power Herb / immediate animation completion respected. Tests cover both defects.
On the identical 200-game Showdown fuzzer schedule: 14,362 positions, zero exceptions,
1,214 → 1,043 mismatching positions. Opponent HP/Tera/active and missing Phantom Force /
Meteor Beam differences disappear. Counts overlap. Most residual positions are hidden
Regenerator healing (821). Engine round-trip: 3,526 positions, 11 residual core mismatches;
extended last-move, Wish, Future Sight, toxic counts and timers remain approximate.

**Measurements.** Default-budget fidelity smoke: {describe('fidelity-smoke')}.
Budget allocated so far: {state['budget_core_seconds'] / 3600:.3f} core-hours (40-hour cap).
All runs use validation teams, pinned Showdown `e440c4a`, CPU, one search thread, fixed
matched blocks, no sequential stop. The gate requests 1,008 games per opponent so all four
shards have equal whole-block sizes (252 each); the merger requires equal shard protocols.
Commands, launch commits, completed/partial JSONs, counters and printed rows are in
`11-artifacts/`; actual seconds per decision are pooled from the shards as run.

**Validation / not verified.** 45 focused and 84 integration tests passed; ruff clean.
Source `aa38e46`; no test-split files read, no threshold or search variant changed.
Unfinished stages and not-run cells are explicit in `measurement-progress.json`.
Residual Flash Fire, Unburden, locked moves, screen/item bookkeeping and engine chance
approximations remain. A failed gate requires manual loss/mechanics diagnosis before a retry.

Resume from the repository root: `.venv/bin/python docs/briefs/11-artifacts/continue_measurements.py`.
'''
    Path('docs/briefs/11-report.md').write_text(text)


def publish(stage, note, name=None):
    if stage in state['published']:
        return
    from fog.eval.showdown_direct_selection import ledger_row

    date = dt.datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    rows = []
    if name:
        report = json.loads((ART / name / 'merged.json').read_text())
        cmd = state['jobs'].get(name, {}).get('command', json.loads(
            (ART / 'commands.json').read_text())['fidelity_smoke'])
        rows = [ledger_row(row, backend='showdown_direct', commit=SOURCE[:7],
                           command=cmd, date=date) for row in report['rows']]
    with Path('docs/results-ledger.md').open('a') as ledger:
        ledger.write('\n### ' + date + ': ' + stage + '\n\n' + note + '\n\n'
                     + '\n'.join(rows) + '\n')
    state['published'].append(stage)
    save()
    write_report()
    # Preserve printed shard results (the .log files themselves are ignored).
    for path in ART.glob('*/shard_*.log'):
        path.with_suffix('.txt').write_text(path.read_text())
    sp.run(['git', 'add', 'docs/briefs/11-artifacts', 'docs/briefs/11-report.md',
            'docs/results-ledger.md', 'ROADMAP.md'], check=True)
    sp.run(['git', 'commit', '-m', f'Record EngineMCTS {stage} [skip ci]'], check=True)
    sp.run(['git', 'push', 'origin', 'work/g1'], check=True)


def main():
    source_fixed()
    state['status'] = 'not yet measured; fidelity smoke running'
    save()
    while running_smoke():
        time.sleep(10)
    smoke = run('fidelity-smoke', 200, 2, 11000, 1600, 8, ['simpleheuristics'])
    if not qualified(smoke['rows'][0], 200):
        state['status'] = 'open: fidelity smoke failed; reconstruction diagnosis required'
        publish('fidelity smoke failed', 'Stage 1 did not meet fidelity acceptance.', 'fidelity-smoke')
        return
    roadmap = Path('ROADMAP.md')
    roadmap.write_text(roadmap.read_text().replace('- [ ] **P1.3 Fidelity counters.**',
                                                  '- [x] **P1.3 Fidelity counters.**'))
    state['status'] = 'not yet measured; budget curve running'
    publish('fidelity smoke complete', 'Stage 1 passes: default-budget 200-game smoke, '
            'fallbacks below 5%, no rejected actions. P1.3 complete. This is smoke, not G1.',
            'fidelity-smoke')
    for budget, k in ORDER:
        if state['budget_core_seconds'] >= 40 * 3600:
            break
        name = f'budget-{budget}-k{k}'
        report = run(name, 400, 4, 12000, budget, k, ['simpleheuristics'], capped=True)
        if report is None:
            break
        publish(f'budget {budget} ms K={k}', '400-game smoke; same seed blocks across cells.', name)
    completed_cells = [f'{b} ms K={k}' for b, k in ORDER
                       if (ART / f'budget-{b}-k{k}' / 'merged.json').exists()]
    state['not_run_budget_cells'] = [f'{b} ms K={k}' for b, k in ORDER
                                   if not (ART / f'budget-{b}-k{k}' / 'merged.json').exists()]
    if len(completed_cells) == 9:
        roadmap.write_text(roadmap.read_text().replace('- [ ] **P1.4 Budget curve.**',
                                                      '- [x] **P1.4 Budget curve.**'))
    state['status'] = 'not yet measured; gate matches running'
    publish('budget curve stage complete', f"Budget charged {state['budget_core_seconds']/3600:.4f} "
            f"allocated core-hours. Complete cells: {completed_cells}. Unfinished/not run: "
            f"{state['not_run_budget_cells']}. The 40-core-hour cap stops further cells.")
    gate = run('gate', 1008, 4, 13000, 1600, 8, ['simpleheuristics', 'maxdamage'])
    passed = all(qualified(row, 1000, {'simpleheuristics': .85, 'maxdamage': .95}[row['b']])
                 for row in gate['rows']) and len(gate['rows']) == 2
    state['status'] = 'passed' if passed else 'failed; loss and mechanics diagnosis required'
    save()
    roadmap.write_text(roadmap.read_text().replace('**If it fails:** the cause is reconstruction',
        f"_Brief 11: G1 {state['status']}; see docs/briefs/11-report.md and the ledger._\n\n"
        '**If it fails:** the cause is reconstruction', 1))
    publish('gate G1 measured', f"G1 {state['status']}. Default K=8, 1600 ms, at least "
            '1,000 matched games per opponent on validation; no sequential stopping.', 'gate')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state['status'] = f'stopped: {type(exc).__name__}: {exc}'
        save()
        write_report()
        raise
