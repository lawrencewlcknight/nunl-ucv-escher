"""Immutable production identities and fixed external benchmark budgets."""
from .continuation_sources import source_profile


def benchmark_profile(hours=128):
    if hours == 128:
        manifest = 'a08210f79bcddebc564109886673d38cb7c8002326b682b2b66f085eb54c048b'
        counts = {'slumbot': (0, 20000), 'gtow': (0, 8000)}
    elif hours == 200:
        manifest = '55b08d9fe715a671e5bf30576bcb87a22a7ef67e9fcf32861778a90647ec81b2'
        # User-selected sample cap, not a claim about a provider's daily quota.
        counts = {'slumbot': (0, 8000)}
    else:
        raise ValueError('External benchmark policy must be P128 or P200')
    return dict(source_profile(hours), label=f'P{hours}', manifest_sha256=manifest,
                counts=counts)
