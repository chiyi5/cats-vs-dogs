"""Run the section2 experiment queue and record its status; see plan.md section 5.2.

Skips runs that already have metrics.json. Stop on the first failure so a broken
configuration does not silently consume the whole queue.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_QUEUE = [
    ('M0', 42), ('M1', 42), ('M2', 42), ('M3', 42), ('M4', 42),
    ('MF', 42), ('MF', 123), ('MF', 2026),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=Path('data/datasets'))
    p.add_argument('--output', type=Path, default=Path('section2/results'))
    p.add_argument('--status', type=Path, default=Path('section2/artifacts/queue_status.json'))
    p.add_argument('--queue', nargs='+', metavar='EXP:SEED',
                   help='override the default queue, e.g. MF:42 MF:123')
    p.add_argument('--epochs', type=int, help='forwarded to train.py for smoke tests')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()

    queue = ([tuple(item.split(':')) for item in a.queue]
             if a.queue else DEFAULT_QUEUE)
    queue = [(experiment, int(seed)) for experiment, seed in queue]
    status = {'state': 'running', 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
              'runs': []}
    a.status.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        a.status.write_text(json.dumps(status, indent=2))

    flush()
    for experiment, seed in queue:
        run_dir = a.output / f'{experiment}_seed{seed}'
        record = {'experiment': experiment, 'seed': seed, 'status': 'pending'}
        status['runs'].append(record)
        if (run_dir / 'metrics.json').exists():
            record['status'] = 'skipped'
            flush()
            continue
        command = [sys.executable, 'section2/train.py', '--data', str(a.data),
                   '--experiment', experiment, '--seed', str(seed),
                   '--output', str(a.output), '--workers', str(a.workers)]
        if a.epochs is not None:
            command += ['--epochs', str(a.epochs)]
        if a.amp:
            command.append('--amp')
        record['status'] = 'running'
        flush()
        print('$ ' + ' '.join(command), flush=True)
        result = subprocess.run(command)
        record['returncode'] = result.returncode
        record['status'] = 'complete' if result.returncode == 0 else 'failed'
        flush()
        if result.returncode != 0:
            status['state'] = 'failed'
            flush()
            raise SystemExit(f'{experiment}_seed{seed} failed; see the log above')

    status['state'] = 'complete'
    status['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    flush()
    print(f'queue complete: {len(queue)} runs, see {a.status}')


if __name__ == '__main__':
    main()
