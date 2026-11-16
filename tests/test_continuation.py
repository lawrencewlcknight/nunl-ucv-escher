"""Bounded real continuation and evaluation checks; no network or paid VMs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from hunl_escher.continuation import LABELS, run_continuation, trainer_command
from hunl_escher.continuation_cloud import extract_source
from hunl_escher.continuation_evaluate import SECONDARY_ALPHA, gate_secondary
from hunl_escher.evaluation_stats import estimate
from hunl_escher.experiment import publish_policy

REPO=Path(__file__).resolve().parents[1]


def launcher_module():
    sys.path.insert(0,str(REPO/'tools'))
    try:
        spec=importlib.util.spec_from_file_location('continuation_launcher',REPO/'tools/launch_continuation.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


@pytest.fixture(scope='module')
def continued(tmp_path_factory):
    directory=tmp_path_factory.mktemp('continuation')
    archived=directory/'archived'
    module=launcher_module()
    module.source_archive(REPO,directory/'source.tar.gz')
    extract_source(directory/'source.tar.gz',archived)
    baseline=directory/'baseline';baseline.mkdir()
    env={**os.environ,'PYTHONPATH':str(archived),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    result=subprocess.run([sys.executable,'-m','hunl_escher.train','--smoke','--iterations','1',
        '--workers','1','--threads','1','--seed','123','--policy-every','0',
        '--policy-seconds','1000000000','--output',str(baseline/'training')],
        cwd=archived,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-4000:]
    publish_policy(baseline/'training',baseline,'P8')
    output=directory/'continued'
    # Real subprocesses and exact original source checks on every stage.
    report=run_continuation(baseline,archived,output,stage_seconds=.001,
                            wall_seconds=180,allow_smoke=True)
    assert report['status']=='TRAINING_COMPLETE'
    return baseline,archived,output


def test_full_continuation_preserves_baseline_and_exports_four_policies(continued):
    baseline,archived,root=continued
    report=json.loads((root/'continuation.json').read_text())
    assert report['final_recovery_verified']
    assert list(report['checkpoints'])==list(LABELS)
    assert [report['checkpoints'][label]['outer_iteration'] for label in LABELS]==[1,2,3,4,5]
    assert (baseline/'training/state_000001').exists()
    assert [p.name for p in (root/'training').glob('state_*')]==['state_000005']
    assert (root/'P8.pkl').read_bytes()==(baseline/'P8.pkl').read_bytes()
    for label in LABELS:
        assert (root/f'{label}.pkl').exists()
    assert json.loads((root/'reload-P32.json').read_text())['status']=='VERIFIED'
    assert json.loads((root/'training/manifest.json').read_text())==json.loads((baseline/'training/manifest.json').read_text())


def test_changed_archived_source_is_rejected(continued,tmp_path):
    baseline,archived,_=continued
    import shutil
    source=tmp_path/'changed'
    shutil.copytree(archived,source)
    with (source/'hunl_escher/config.py').open('a') as handle:handle.write('\n# incompatible source\n')
    with pytest.raises(RuntimeError,match='reload verification failed'):
        run_continuation(baseline,source,tmp_path/'out',stage_seconds=.001,wall_seconds=30,allow_smoke=True)
    assert not (tmp_path/'out/P14.pkl').exists()
    assert json.loads((tmp_path/'out/continuation.json').read_text())['status']=='FAILED'


def test_production_rejects_smoke_and_same_directory(continued,tmp_path):
    baseline,archived,_=continued
    with pytest.raises(ValueError,match='smoke'):
        run_continuation(baseline,archived,tmp_path/'out')
    with pytest.raises(ValueError,match='separate'):
        run_continuation(baseline,archived,baseline,allow_smoke=True)


def test_stage_command_keeps_resume_contract(continued):
    baseline,_,_=continued
    contract=json.loads((baseline/'training/manifest.json').read_text())
    command=trainer_command(baseline/'training',contract,21600)
    assert command[command.index('--policy-seconds')+1]==str(contract['policy_seconds'])
    assert command[command.index('--additional-training-seconds')+1]=='21600'
    assert '--resume' in command


def test_secondary_plan_and_primary_gate():
    result=estimate([1.,2.]*100,alpha=SECONDARY_ALPHA)
    assert result['confidence']==.99
    assert gate_secondary(result,False,False)['confirmatory'] is False


def test_real_offline_evaluation_suite(continued,tmp_path):
    from hunl_escher.continuation_evaluate import run_suite
    _,_,root=continued
    report=run_suite(root,tmp_path/'evaluation',workers=1,max_seconds=240,smoke=True)
    assert report['status']=='COMPLETE'
    assert len(report['head_to_head'])==5
    assert all(not x['result']['confirmatory'] for x in report['head_to_head'].values())
    assert report['head_to_head']['P14-vs-P8']['result']['confidence']==.99
    assert (tmp_path/'evaluation/continuation_report.md').exists()


def test_archive_rejects_traversal(tmp_path):
    import io
    with tarfile.open(tmp_path/'bad.tar.gz','w:gz') as handle:
        info=tarfile.TarInfo('../outside');info.size=1;handle.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='safe regular'):
        extract_source(tmp_path/'bad.tar.gz',tmp_path/'extract')
    assert not (tmp_path/'outside').exists()


def test_launch_bundle_has_one_bounded_vm_and_no_secrets(tmp_path):
    module=launcher_module()
    plan=module.build_bundle(tmp_path/'bundle',project='test-project',zone='europe-west1-b',
        bucket='gs://test-bucket/hunl',service_account='runner@test-project.iam.gserviceaccount.com',
        run_name='hunl-continuation-test',source_uri='gs://test-bucket/hunl/original',expected_p8_sha256='a'*64)
    creates=[x for x in plan['commands'] if x[1:4]==['compute','instances','create']]
    assert len(creates)==1 and '--max-run-duration' in creates[0] and '36h' in creates[0]
    assert 'c4-highmem-16' in creates[0]
    assert plan['additional_active_hours']==24 and plan['stage_hours']==6
    script=(tmp_path/'bundle/startup.sh').read_text()
    subprocess.run(['bash','-n',str(tmp_path/'bundle/startup.sh')],check=True)
    assert 'continuation_cloud' in script and "trap 'shutdown -h now' EXIT" in script
    with tarfile.open(tmp_path/'bundle/source.tar.gz') as handle:
        names=handle.getnames()
    assert 'hunl_escher/verify_recovery.py' in names
    assert all('.env' not in x and not x.startswith(('outputs/','.git/')) for x in names)


def test_six_hour_stage_boundaries_preserve_learning_state(continued,tmp_path):
    """Four restart/export boundaries must match uninterrupted learner updates."""
    import shutil
    import torch
    from hunl_escher.training_state import read_training_state
    baseline,archived,root=continued
    training=tmp_path/'training'
    shutil.copytree(baseline/'training',training)
    contract=json.loads((training/'manifest.json').read_text())
    command=trainer_command(training,contract,1000000)
    command[command.index('--iterations')+1]='5'
    result=subprocess.run(command,cwd=archived,env={**os.environ,'PYTHONPATH':str(archived)},
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-3000:]
    a=read_training_state(training/'state_000005')
    b=read_training_state(root/'training/state_000005')
    def equal(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:equal(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):equal(xx,yy)
        elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        elif isinstance(x,torch.Tensor):assert torch.equal(x,y)
        else:assert x==y
    for key in ('regret_trainers','q_ensemble','calibration','gate_controller','rng','actors','replay_rng','parallel_contract'):
        equal(a[key],b[key])
    equal(a['average_policy_trainer']['buffer'],b['average_policy_trainer']['buffer'])


def test_final_slumbot_only_p32_offline_and_pilot_excluded(continued, tmp_path):
    from hunl_escher.api import SlumbotClient
    from hunl_escher.continuation_benchmark import run_final_benchmark
    from test_remote import MockServer
    _, _, root = continued
    server = MockServer('slumbot')
    def factory(provider, menu):
        assert provider == 'slumbot'
        return SlumbotClient(action_menu=menu, transport=server)
    result = run_final_benchmark(root, tmp_path/'slumbot', smoke=True, client_factory=factory)
    assert result['status'] == 'COMPLETE'
    assert result['completed_hands'] == 8 and result['completed_pilot_hands'] == 4
    assert result['estimate']['observations'] == [8]
    assert not result['estimate']['confirmatory']
    assert server.number == 12
    journal = (tmp_path/'slumbot/hands.jsonl').read_text()
    assert 'SECRET_TOKEN' not in journal
    rows = [json.loads(line) for line in journal.splitlines()]
    assert {r['policy'] for r in rows if 'policy' in r} == {'P32'}
    assert result['historical_comparison'] is None
    with pytest.raises(ValueError, match='offline'):
        run_final_benchmark(root, tmp_path/'live-smoke', smoke=True)


def test_final_slumbot_ambiguous_hand_is_not_retried_or_scored(continued, tmp_path):
    from hunl_escher.api import SlumbotClient, AmbiguousOutcome
    from hunl_escher.continuation_benchmark import run_final_benchmark
    from test_remote import MockServer
    _, _, root = continued
    class Broken(MockServer):
        failed = 0
        def request(self, method, path, body=None):
            if self.number == 7 and path == '/slumbot/api/new_hand':
                self.failed += 1
                raise AmbiguousOutcome('Uncertain outcome')
            return super().request(method, path, body)
    server = Broken('slumbot')
    result = run_final_benchmark(root, tmp_path/'partial', smoke=True,
        client_factory=lambda _, menu: SlumbotClient(action_menu=menu, transport=server))
    assert result['status'] == 'INCOMPLETE_PROVIDER_ERROR'
    assert result['completed_hands'] == 3 and server.failed == 1
    assert result['estimate']['observations'] == [3]
    assert not result['estimate']['confirmatory']


def test_historical_baseline_requires_completed_matching_run(continued, tmp_path):
    from hunl_escher.continuation_benchmark import historical_comparison
    _, _, root = continued
    marker = json.loads((root/'P8.json').read_text())
    (tmp_path/'P8.json').write_text(json.dumps(marker))
    report = dict(status='COMPLETE', provider='slumbot', primary_metric='raw_bb',
        plan={'selected': 4}, scores={p: dict(main=[-1., 2., -3., 4.], pilot=[1000.]*4) for p in ('P4','P8')})
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    result = historical_comparison(path, root, [4., 5., 6., 7.])
    assert result['status'] == 'AVAILABLE' and not result['confirmatory']
    assert result['observations'] == [4, 4] and result['bb_per_100'] == 500
    report['status'] = 'MAIN'
    path.write_text(json.dumps(report))
    assert historical_comparison(path, root, [4., 5.])['status'] == 'BASELINE_INCOMPLETE'
    marker['sha256'] = '0'*64
    (tmp_path/'P8.json').write_text(json.dumps(marker))
    assert historical_comparison(path, root, [4., 5.])['status'] == 'INVALID_OR_UNAVAILABLE_BASELINE'


def test_policy_tamper_and_unfinished_training_rejected_before_evaluation(continued, tmp_path):
    import shutil
    from hunl_escher.continuation_evaluate import verified_checkpoint
    _, _, root = continued
    for name in ('P32.pkl', 'P32.json', 'continuation.json'):
        shutil.copy2(root/name, tmp_path/name)
    with (tmp_path/'P32.pkl').open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        verified_checkpoint(tmp_path, 'P32', smoke=True)
    report = json.loads((tmp_path/'continuation.json').read_text())
    report['final_recovery_verified'] = False
    (tmp_path/'continuation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='reload'):
        verified_checkpoint(tmp_path, 'P32', smoke=True)


def test_slumbot_launcher_waits_for_verified_p32_and_uses_small_vm(tmp_path):
    spec = importlib.util.spec_from_file_location('slumbot_launcher', REPO/'tools/launch_continuation_slumbot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = launcher_module().build_bundle(tmp_path/'train', project='test-project', zone='us-central1-a',
        bucket='gs://test-bucket/hunl', service_account='runner@test-project.iam.gserviceaccount.com',
        run_name='hunl-continuation-test', source_uri='gs://test-bucket/hunl/original', expected_p8_sha256='a'*64)
    ready = dict(status='READY', policy='P32', source_uri=plan['source_uri'], sha256='b'*64)
    training = dict(status='TRAINING_COMPLETE', final_recovery_verified=True, smoke=False,
                    checkpoints={'P8': {'sha256': 'a'*64}, 'P32': {'sha256': 'b'*64}})
    marker = {'sha256': 'b'*64}
    expected = module.validate_ready(plan, ready, training, marker)
    result = module.build_bundle(plan, tmp_path/'external', expected)
    creates = [c for c in result['commands'] if c[1:4] == ['compute','instances','create']]
    assert len(creates) == 1 and 'c4-standard-4' in creates[0] and '26h' in creates[0]
    assert result['commands'][0][3] == '--if-generation-match=0'
    script = tmp_path/'external/startup.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    assert '--role slumbot' in script.read_text()
    assert 'GTO' not in script.read_text()
    training['final_recovery_verified'] = False
    with pytest.raises(ValueError, match='not ready'):
        module.validate_ready(plan, ready, training, marker)
