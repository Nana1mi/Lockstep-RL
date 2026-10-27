"""Write follow-up configs only; no GPU execution or downloads."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configurations(model):
    base = json.loads((ROOT / 'configs/train-demo-simple.json').read_text())
    result = {}
    for seed in (1, 2):
        config = json.loads(json.dumps(base))
        config.update(model=model, seed=seed,
            validation_status='GPU_PENDING; follow-up seed replication, not executed')
        result[f'replication-seed{seed}'] = config
    for seed in (0, 1, 2):
        for anchor in ('rollout', 'recomputed'):
            config = json.loads(json.dumps(base))
            config.update(model=model, seed=seed,
                validation_status='GPU_PENDING; matched anchor/update_steps2 comparison, not executed; clipping activation must be measured')
            config['grpo'].update(anchor=anchor, update_steps=2)
            result[f'anchor-{anchor}-seed{seed}'] = config
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, default=ROOT / '.build/followups')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, config in configurations(args.model).items():
        path = args.output / (name + '.json')
        if path.exists():
            raise ValueError('Use a fresh output directory; do not overwrite reviewed configs')
        path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
        paths.append(str(path))
    print(json.dumps({'status': 'GPU_PENDING; configurations only', 'files': paths}, indent=2))


if __name__ == '__main__':
    main()
