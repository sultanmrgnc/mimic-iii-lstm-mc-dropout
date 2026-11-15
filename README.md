# Simplified Research Implementation

## Project Overview

This repository contains a simplified, public, incomplete implementation of a research workflow for retrospective mechanical-ventilation analysis using MIMIC-III. It includes no patient data, trained model, execution output, metric, or reproduced result.

## Research Objective

The provisional objective is to study an LSTM-based retrospective ICU-stay label derived from recorded ventilation procedures. The included label is not a validated future prediction target, clinical tool, or prospective claim.

## Dataset and Access Requirements

MIMIC-III is available through [PhysioNet](https://physionet.org/content/mimiciii/1.4/) under credentialing, training, and data-use requirements. Data is not included. `DATA_DIR = None` prevents automatic access; authorized users must explicitly call a reader with a local path.

Do not provide restricted data to online services unless their terms and deployment provide verified zero retention, no training, and no human review. Unverifiable cloud services must not be used. Colab or other cloud execution is not automatically endorsed.

## Methodology Overview

The notebook retains schema names, selected vital-sign and laboratory item IDs, a provisional procedure-derived stay-level label, a 24-step feature window constant, an LSTM builder, and MC Dropout. Full cohort definition, temporal cutoff, preprocessing, sequence construction, training, calibration, and evaluation are incomplete.

Leakage controls required for a complete study include splitting by patient `subject_id`, not ICU stay; applying the temporal cutoff before constructing the label; and fitting all preprocessing on training data only. No unimplemented pipeline is implied.

## LSTM Architecture

The preserved architecture is: input, LSTM(64, `return_sequences=True`), dropout, LSTM(32), dropout, dense(16, ReLU), dropout, and dense(1, sigmoid). Compilation retains Adam at `1e-3`, binary cross-entropy, AUROC, precision, recall, and accuracy. The notebook does not instantiate or train the model automatically.

## Monte Carlo Dropout

`mc_dropout_predict` preserves stochastic inference exactly by repeatedly calling `model(X, training=True)`, converting to NumPy, flattening each prediction, and stacking 50 samples by default. It is never called automatically.

## Repository Structure

- `public_notebook.ipynb`: cleaned, definition-only simplified implementation.
- `README.md`: scope, safety, limitations, and local validation.
- `requirements.txt`: minimal direct imports; versions are not locked or validated.
- `.gitignore`: root-level deny-all allowlist for these four files.

## Installation and Usage

Use an isolated local environment on an authorized machine:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter notebook public_notebook.ipynb
```

Dependency installation is never performed by the notebook. Do not execute data-reader functions without authorized local data access.

Run the persistent, dataset-free repository validation from this repository root in bash/WSL:

```bash
python3 - <<'PY'
import ast
import json
import re
import subprocess
from pathlib import Path

root = Path.cwd().resolve()
git_root_result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root, capture_output=True, text=True)
if git_root_result.returncode == 0:
    git_root = Path(git_root_result.stdout.strip()).resolve()
    if git_root != root:
        raise SystemExit(f'Stop: current directory {root} differs from Git root {git_root}')
else:
    git_root = None
    print('Git checks skipped: no Git repository found.')
allowed = {'.gitignore', 'README.md', 'public_notebook.ipynb', 'requirements.txt'}
local_dirs = {'.git', '.venv', '.ipynb_checkpoints'}
names = {path.name for path in root.iterdir() if path.name not in local_dirs}
assert names <= allowed, f'Unexpected root names: {sorted(names - allowed)}'
assert allowed <= names, f'Missing public files: {sorted(allowed - names)}'
notebook = json.loads((root / 'public_notebook.ipynb').read_text(encoding='utf-8'))
assert notebook['nbformat'] == 4
code = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'] if cell.get('cell_type') == 'code')
ast.parse(code)
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        assert cell.get('execution_count') is None and cell.get('outputs') == []
blocked = [r'!pip', r'google\.colab', r'drive\.mount', r'/content/', r'os\.makedirs', r'to_csv\s*\(', r'to_pickle\s*\(', r'\.fit\s*\(', r'(?<!def )build_lstm_model\s*\(WINDOW_SIZE']
assert not any(re.search(pattern, code, re.I) for pattern in blocked)
public = '\n'.join((root / name).read_text(encoding='utf-8') for name in allowed if (root / name).exists())
secrets = [r'-----BEGIN [A-Z ]*PRIVATE KEY-----', r'AKIA[0-9A-Z]{16}', r'gh[pousr]_[A-Za-z0-9_]{20,}', r'(password|api[_-]?key|secret)\s*=\s*["\'][^"\']+["\']']
assert not any(re.search(pattern, public, re.I) for pattern in secrets)
ignore = (root / '.gitignore').read_text(encoding='utf-8').splitlines()
assert ignore == ['/*', '!/.gitignore', '!/README.md', '!/public_notebook.ipynb', '!/requirements.txt']
if git_root is not None:
    staged = set(subprocess.run(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root, check=True, capture_output=True, text=True).stdout.split('\0')) - {''}
    assert staged <= allowed, f'Unexpected staged names: {sorted(staged - allowed)}'
    tracked = set(subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True, capture_output=True, text=True).stdout.split('\0')) - {''}
    assert tracked <= allowed, f'Unexpected tracked names: {sorted(tracked - allowed)}'
    has_commits = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root, capture_output=True).returncode == 0
    if has_commits:
        historic = set(subprocess.run(['git', 'log', '--all', '--name-only', '--pretty=format:'], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines())
        historic.discard('')
        unexpected = sorted(historic - allowed)[:20]
        assert not unexpected, f'Unexpected historical filenames (contents not audited): {unexpected}'
    else:
        print('Git history checks skipped: repository has no commits.')
print('Public-file validation passed; no dataset accessed.')
PY
```

The validator reads only the four allowlisted working-tree files. Git checks inspect bounded filename lists only; historical contents and patient content are not audited. `load_authorized_tables` eagerly loads every listed table into RAM. `read_filtered_events` uses chunks for filtering but accumulates the filtered result before returning; neither function claims global memory efficiency.

### Dataset-free reader checks

Run this Python check from the repository root. It extracts only the reader function, creates temporary synthetic CSV data, and never executes notebook top-level code or accesses restricted data:

```python
import ast
import json
import tempfile
from pathlib import Path
import pandas as pd

notebook = json.loads(Path('public_notebook.ipynb').read_text(encoding='utf-8'))
source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
reader = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'read_csv_required')
exec(compile(ast.Module(body=[reader], type_ignores=[]), '<reader-test>', 'exec'))

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'events.csv'
    pd.DataFrame({'ID': [1], 'Value': [2]}).to_csv(path, index=False)
    try:
        read_csv_required(path, ['id'], usecols=['id'])
    except ValueError as error:
        assert 'usecols' in str(error)
    else:
        raise AssertionError('usecols must be rejected')
    try:
        read_csv_required(path, ['missing'])
    except ValueError as error:
        assert 'Missing required columns' in str(error)
    else:
        raise AssertionError('missing columns must be rejected')
    result = read_csv_required(path, ['id'], sep=',', encoding='utf-8', low_memory=True)
    assert result.to_dict('list') == {'id': [1]}
```

The synthetic check covers the unsupported-argument contract, missing-column reporting, and ordinary header-based reading.

## Current Implementation Status

The implementation provides robust case-insensitive required-column checks, nullable numeric parsing, timestamp parsing, and empty-chunk handling. Data reads are explicit opt-in functions. Training, patient-level splitting, temporal labeling, preprocessing, sequence creation, and evaluation are intentionally absent. No results are claimed.

## Reproducibility Limitations

This simplified notebook cannot reproduce the research. Dependencies are not locked or validated, the full experimental pipeline is absent, and no original outputs were retained. The local validator checks public-file structure and common disclosure risks, not scientific correctness, hidden history contents, legal compliance, or absence of every possible secret.

## Data Privacy and PhysioNet DUA

MIMIC-III remains restricted data. Follow the [official DUA](https://physionet.org/content/mimiciii/view-dua/1.4/) and [PhysioNet guidance on responsible use with LLMs](https://physionet.org/news/post/llm-responsible-use/). Do not upload patient data, derived patient-level content, credentials, or private paths.

DUA clause 9 states that openly disseminated results require contribution of the code used to produce those results. This simplified demonstration may not fulfill that requirement. The repository owner must resolve contribution of the complete experiment code separately with PhysioNet; this repository makes no claim of compliance, approval, or legal guarantee.

## References and Citation

Johnson, A. E. W., Pollard, T. J., & Mark, R. G. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26
