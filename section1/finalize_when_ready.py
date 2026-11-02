"""Validate the existing long-running queue once all runs finish."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
status = root / 'section1/artifacts/queue_status.json'
while True:
    try:
        state = json.loads(status.read_text()).get('state')
    except (FileNotFoundError, json.JSONDecodeError):
        state = 'running'
    if state == 'failed':
        raise SystemExit('Experiment queue failed; see queue_status.json and run log.')
    if state == 'complete':
        # All metrics already exist, so run_all skips training and verifies outputs.
        result = subprocess.run([sys.executable, str(root / 'section1/run_all.py')], cwd=root)
        if result.returncode:
            status.write_text(json.dumps({'state': 'validation_failed', 'returncode': result.returncode}))
        raise SystemExit(result.returncode)
    time.sleep(30)
