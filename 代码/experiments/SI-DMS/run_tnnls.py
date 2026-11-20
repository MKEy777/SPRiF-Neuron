"""Run TNNLS SI-DMS controls with separate clean/intervention training outputs."""
import argparse
import json
from pathlib import Path
import torch
from sidms.config import ExperimentConfig
from sidms.engine import train_model, evaluate_grid

DISPLAY = {'sprif_scalar':'SPRiF','memory_reset5':'MEMORY_RESET5','real_modes3':'REAL_MODES3','asrnn':'ASRNN','brf':'BRF','lif':'LIF'}
FORMAL_SEEDS = [0, 1, 2, 42, 1111]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/default.yaml')
    p.add_argument('--output',required=True,help='New output directory; existing runs are never overwritten')
    p.add_argument('--models',nargs='+',choices=list(DISPLAY),default=['real_modes3'])
    p.add_argument('--training',nargs='+',choices=['Clean','Intervention'],default=['Clean','Intervention'])
    p.add_argument('--seeds',nargs='+',type=int,default=FORMAL_SEEDS)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--eval-batches',type=int,default=20)
    p.add_argument('--smoke',action='store_true',help='Mark reduced-budget output as smoke; never mix into formal output')
    args=p.parse_args()
    if not args.smoke and (args.steps != 3000 or args.eval_batches != 20 or args.seeds != FORMAL_SEEDS):
        p.error('Formal mode requires 3000 steps, 20 evaluation batches and the five specified seeds; use --smoke for development')
    if args.steps < 1 or args.eval_batches < 1: p.error('steps and eval-batches must be positive')
    root=Path(args.output)
    if root.exists() and any(root.iterdir()): raise FileExistsError(f'Output directory is not empty: {root}')
    cfg=ExperimentConfig.from_yaml(args.config);cfg.train.steps=args.steps;cfg.train.seeds=args.seeds
    rows_all=[]
    for name in args.models:
        for condition in args.training:
            cfg.task.train_intervention_counts=[0] if condition=='Clean' else [0,1,2,4,8]
            for seed in args.seeds:
                out=root/condition/name/f'seed_{seed}';out.mkdir(parents=True,exist_ok=False)
                print(f'Training {name} {condition} seed {seed}',flush=True)
                def progress(row):
                    if row['step'] == 1 or row['step'] % 100 == 0 or row['step'] == args.steps:
                        print(f"step {row['step']}/{args.steps} loss={row['loss']:.4f}",flush=True)
                model, history=train_model(name,cfg,seed,progress=progress,tnnls_protocol=True)
                torch.save({'model':model.state_dict(),'config':cfg.to_dict(),'name':name,'training':condition,'seed':seed,'smoke':args.smoke},out/'checkpoint.pt')
                (out/'train_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
                rows=evaluate_grid(model,cfg,seed,batches=args.eval_batches,paired=True)
                for r in rows:
                    r['model_id']=r['model'];r['model']=DISPLAY[name];r['training']=condition;r['smoke']=args.smoke
                (out/'eval_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                rows_all.extend(rows)
                (root/'evaluations.partial.json').write_text(json.dumps(rows_all,indent=2),encoding='utf-8')
                print(f'Saved {out}',flush=True)
    (root/'evaluations.json').write_text(json.dumps(rows_all,indent=2),encoding='utf-8')
if __name__=='__main__':main()
