#!/usr/bin/env python3
import argparse, pathlib, sys
import pandas as pd

DOMAINS = ['biology', 'chemistry', 'material', 'physics']


def question(row):
    return str(row['prompt'][1]['content'])


def answer(row):
    return row['reward_model']['ground_truth']


def audit(root: pathlib.Path):
    summary, changed, removed = [], [], []
    for d in DOMAINS:
        for split in ['train', 'test']:
            o = pd.read_parquet(root / 'original' / f'sciknoweval-{d}' / f'{split}.parquet')
            c = pd.read_parquet(root / 'curated' / f'sciknoweval-{d}' / f'{split}.parquet')
            idx = [r['extra_info'].get('orig_idx') for _, r in c.iterrows()]
            if any(i is None for i in idx) or len(set(idx)) != len(idx) or max(idx) >= len(o):
                sys.exit(f'{d}/{split}: orig_idx is missing, duplicated or out of range')
            rel = rew = 0
            for (_, r), i in zip(c.iterrows(), idx):
                ro = o.iloc[i]
                same = question(r) == question(ro)
                if same and answer(ro) == answer(r):
                    continue
                action = 'relabelled' if same else 'rewritten'
                rel += action == 'relabelled'
                rew += action == 'rewritten'
                changed.append(dict(domain=d, split=split, action=action,
                                    verdict=r['extra_info'].get('audit_verdict'),
                                    confidence=r['extra_info'].get('audit_confidence'),
                                    original_answer=answer(ro), curated_answer=answer(r),
                                    replaced_option_text=r['extra_info'].get('audit_orig_option_text') or '',
                                    question=question(r)))
            keep = set(idx)
            for i in range(len(o)):
                if i not in keep:
                    removed.append(dict(domain=d, split=split, original_answer=answer(o.iloc[i]),
                                        question=question(o.iloc[i])))
            summary.append(dict(domain=d, split=split, original=len(o), curated=len(c),
                                unchanged=len(c) - rel - rew, relabelled=rel, rewritten=rew,
                                removed=len(o) - len(c)))
    return pd.DataFrame(summary), pd.DataFrame(changed), pd.DataFrame(removed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    root = pathlib.Path(a.data_dir)
    sm, ch, rm = audit(root)
    print(sm.to_string(index=False))
    print(f'\nchanged: {len(ch)}   removed: {len(rm)}')
    if a.write:
        out = root / 'curation'
        sm.to_csv(out / 'summary.csv', index=False)
        ch.to_csv(out / 'changed_items.csv', index=False)
        rm.to_csv(out / 'removed_items.csv', index=False)
        print(f'written to {out}')
