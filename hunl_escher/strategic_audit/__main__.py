"""Freeze a position library, validate it, or audit saved HUNL policies offline."""
import argparse
import json
from pathlib import Path

from .library import make_library, validate_library, write_new
from .workflow import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    lib = sub.add_parser('library', help='Create a fixed development/assessment library')
    lib.add_argument('--output', type=Path, required=True)
    lib.add_argument('--seed', type=int, default=20261118)
    lib.add_argument('--smoke', action='store_true')
    validate = sub.add_parser('validate')
    validate.add_argument('--library', type=Path, required=True)
    audit = sub.add_parser('run')
    audit.add_argument('--library', type=Path, required=True)
    audit.add_argument('--policy', action='append', required=True, metavar='LABEL=CHECKPOINT',
                       help='Repeat for comparisons; smoke controls: LABEL=rule:check_call/random/all_in/fold')
    audit.add_argument('--output', type=Path, required=True)
    audit.add_argument('--partition', choices=('development', 'assessment'), default='development')
    audit.add_argument('--unlock-assessment', action='store_true')
    audit.add_argument('--max-seconds', type=float, default=300)
    audit.add_argument('--threads', type=int, default=1)
    audit.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'library':
        data = make_library(args.seed, args.smoke)
        write_new(args.output, data)
        result = dict(status='CREATED', positions=len(data['positions']), path=str(args.output))
    elif args.command == 'validate':
        data = validate_library(json.loads(args.library.read_text()))
        result = dict(status='VALID', positions=len(data['positions']))
    else:
        policies = []
        for value in args.policy:
            if '=' not in value:
                parser.error('--policy must be LABEL=CHECKPOINT or LABEL=rule:NAME')
            policies.append(tuple(value.split('=', 1)))
        result = run(args.library, policies, args.output, partition=args.partition,
            unlock_assessment=args.unlock_assessment, max_seconds=args.max_seconds,
            threads=args.threads, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('CREATED', 'VALID', 'COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
