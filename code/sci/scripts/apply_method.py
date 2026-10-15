#!/usr/bin/env python3
import argparse, pathlib, shutil, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['grpo', 'sdpo', 'rlsd', 'lead'])
    ap.add_argument('--framework', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent
    src = here.parent / 'methods' / a.method / 'files'
    dst = pathlib.Path(a.framework).resolve()
    if not src.is_dir():
        sys.exit(f'missing method tree: {src}')
    if not (dst / 'verl').is_dir():
        sys.exit(f'not a framework checkout: {dst}')
    n = 0
    for f in sorted(src.rglob('*')):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        out = dst / rel
        print(('would copy ' if a.dry_run else 'copy ') + str(rel))
        if not a.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        n += 1
    print(f'{n} files for method {a.method}')

if __name__ == '__main__':
    main()
