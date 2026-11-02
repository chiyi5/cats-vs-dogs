"""Sequential formal experiment queue; completed runs are skipped, failures stop the queue."""
import json
import csv
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
out = root / 'section1/results/formal'
out.mkdir(parents=True, exist_ok=True)
status = root / 'section1/artifacts/queue_status.json'
for exp in ['E1', 'E2', 'E3', 'E4', 'E5']:
    run = out / f'{exp}_seed42'
    if (run / 'metrics.json').exists():
        continue
    cmd = [sys.executable, '-u', 'section1/run_experiment.py', '--data', 'data/datasets',
           '--experiment', exp, '--epochs', '20', '--seed', '42', '--batch-size', '16',
           '--workers', '2', '--amp', '--exclude', 'section1/artifacts/excluded_train.json', '--output', str(out)]
    status.write_text(json.dumps({'state': 'running', 'experiment': exp, 'command': cmd}, indent=2))
    with (out / f'{exp}_seed42.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '2'})
    if result.returncode:
        status.write_text(json.dumps({'state': 'failed', 'experiment': exp, 'returncode': result.returncode}, indent=2))
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable, 'section1/summarize_results.py'], check=True, stdout=subprocess.DEVNULL)
checks = []
shared_files = None
for exp in ['E1', 'E2', 'E3', 'E4', 'E5']:
    run = out / f'{exp}_seed42'
    m = json.loads((run / 'metrics.json').read_text())
    c = json.loads((run / 'config.json').read_text())
    with (run / 'val_predictions.csv').open() as f:
        predictions = list(csv.DictReader(f))
    assert len(predictions) == 5000
    accuracy = sum(r['label'] == r['prediction'] for r in predictions) / len(predictions)
    assert abs(accuracy - m['val_accuracy']) < 1e-12
    assert c['train_count'] == 19994 and c['val_count'] == 5000
    files = ((run / 'train_files.csv').read_bytes(), (run / 'val_files.csv').read_bytes())
    # Compare parsed relative file lists since independently launched runs may use different path prefixes.
    if shared_files is None:
        shared_files = files
    assert files == shared_files, 'Different data lists between comparisons'
    if exp != 'E5':
        with (run / 'history.csv').open() as f: history = list(csv.DictReader(f))
        assert len(history) == 20
        assert max(float(r['val_accuracy']) for r in history) == m['val_accuracy']
    checks.append({'experiment': exp, 'validation_predictions': len(predictions), 'accuracy_verified': True})
(root / 'section1/artifacts/validation_checks.json').write_text(json.dumps(checks, indent=2))
subprocess.run([sys.executable, 'section1/summarize_results.py'], check=True, stdout=subprocess.DEVNULL)
status.write_text(json.dumps({'state': 'complete', 'experiments': ['E1', 'E2', 'E3', 'E4', 'E5'], 'checks_passed': True}))
