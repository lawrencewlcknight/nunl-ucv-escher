"""Pinned production baselines for Batch continuation experiments."""


def source_profile(hours):
    common = dict(seed=20261115, workers=8, threads=8)
    if hours == 56:
        return dict(common, hours=56, iteration=2941,
            uri='gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p32-plus24-20261117-121800',
            policy_sha256='a1ec410adc7aab98c07045a78ea37b42e7a4c7c3a6bb648afa7545a18d4ce94a',
            fingerprint='a3b341129c3b2af43b93f9799482169393bd07b0dc79692e10a5d631ad045809')
    if hours == 80:
        return dict(common, hours=80, iteration=4160,
            uri='gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p56-plus24-batch-20261119-050144',
            policy_sha256='a489504dada7fcb98501a6655c705f4ad679a46cd7979aad15bdbe63483acc5c',
            fingerprint='b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b',
            archive_sha256='323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1')
    if hours == 104:
        return dict(common, hours=104, iteration=5359,
            uri='gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p80-plus24-batch-20261120',
            policy_sha256='b717b47b9afa0a8e1c2efc6214fb48d46d8cb069b85e873465fbf695a93c8307',
            fingerprint='b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b',
            archive_sha256='323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1')
    raise ValueError('Batch baseline must be 56, 80 or 104')


def validate_source_metadata(read, hours):
    """Check published provenance; the worker additionally hashes and reloads state."""
    p = source_profile(hours)
    label, iteration = f'P{hours}', p['iteration']
    state = f'state_{iteration:06d}'
    report, policy, pointer, contract, reload = [read(name) for name in (
        'continuation.json', f'{label}.json', 'training/latest.json',
        'training/manifest.json', f'reload-{label}.json')]
    if (report['status'] != 'TRAINING_COMPLETE' or not report.get('final_recovery_verified') or
            report['labels'][-1] != label or
            report['checkpoints'][label]['sha256'] != p['policy_sha256'] or
            policy['sha256'] != p['policy_sha256'] or policy['outer_iteration'] != iteration or
            policy['recovery'] != state or pointer['state'] != state or pointer['iteration'] != iteration or
            contract['source_fingerprint'] != p['fingerprint'] or contract['seed'] != p['seed'] or
            contract['workers'] != p['workers'] or contract['threads'] != p['threads'] or contract['smoke'] or
            reload['status'] != 'VERIFIED' or reload['iteration'] != iteration or
            reload['source_fingerprint'] != p['fingerprint']):
        raise ValueError(f'{label} source identity/recovery is not the approved baseline')
    if hours >= 80:
        resume = read('resume.json')
        diagnostics = policy['metrics']['policy_validation']
        if (resume['source_fingerprint'] != p['fingerprint'] or
                resume['training_source_archive_sha256'] != p['archive_sha256'] or
                resume['training_source_archive'] != p['uri'] + '/training_source.tar.gz' or
                resume['policy_group_sampling'] != 'objective_mass' or
                diagnostics.get('group_sampling') != 'objective_mass' or
                diagnostics.get('policy_sampling_seed') != [p['seed'], iteration]):
            raise ValueError(f'{label} does not attest to the corrected archived sampler')
    return dict(status='VERIFIED_METADATA', baseline_sha256=p['policy_sha256'], iteration=iteration,
                source_fingerprint=p['fingerprint'],
                full_reload='Repeated on Batch before training')
