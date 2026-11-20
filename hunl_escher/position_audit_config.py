"""Pinned policies and the original P32 development library for Batch audits."""
import hashlib
import json
from pathlib import Path

from .continuation_sources import source_profile

LIBRARY_PATH = Path(__file__).with_name('data') / 'strategic_positions_v1.json'
LIBRARY_SHA256 = '3575612cbdcb672fc7d877d93fe041a7e8629134c661684a50012f5a5763dfc8'
DEVELOPMENT_POSITIONS = 374
ASSESSMENT_POSITIONS = 152
AUDIT_SECONDS = 600


def policies(hours):
    target = source_profile(hours)
    return [dict(label='P32', iteration=1658,
                 uri='gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857',
                 policy_sha256='73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129'),
            dict(label=f'P{hours}', iteration=target['iteration'], uri=target['uri'],
                 policy_sha256=target['policy_sha256'])]


def frozen_library(path=LIBRARY_PATH):
    data = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if (digest != LIBRARY_SHA256 or data.get('smoke') or
            sum(p['partition'] == 'development' for p in data['positions']) != DEVELOPMENT_POSITIONS or
            sum(p['partition'] == 'assessment' for p in data['positions']) != ASSESSMENT_POSITIONS):
        raise ValueError('Audit requires the unchanged original P32 position library')
    return data
