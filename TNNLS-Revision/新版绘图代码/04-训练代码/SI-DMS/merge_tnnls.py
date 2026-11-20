"""Combine separately run TNNLS evaluations without mixing duplicate conditions."""
import argparse
import json
from pathlib import Path
from sidms.reporting import _modern_groups

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); rows=[]
    for path in args.inputs:
        part=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(part,list) or not part: raise ValueError(f'Expected nonempty evaluation list: {path}')
        rows.extend(part)
    _modern_groups(rows)
    versions={r.get('protocol_version') for r in rows}
    if len(versions)!=1 or None in versions:
        raise ValueError('Missing or inconsistent protocol_version; verify older runs separately before combining')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(rows,f,indent=2)
    print(f'Combined {len(rows)} records in {args.output}')
if __name__=='__main__': main()
