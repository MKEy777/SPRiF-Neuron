"""SI-DMS composite: matched replay, five-delay seed curves, and delay/K maps."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
SKILL_SCRIPTS = Path.home()/'.codex/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

MODELS = ['SPRiF', 'MEMORY_RESET5', 'REAL_MODES3', 'ASRNN', 'BRF', 'LIF']
SEEDS = {0, 1, 2, 42, 1111}
DELAYS = {200, 400, 800, 1600, 2500}
COLORS = ['#176b80', '#b65c38', '#695695', '#bd873b', '#68747b']

def load_metrics(path, fraction):
    df = pd.DataFrame(json.loads(path.read_text(encoding='utf-8-sig')))
    required = ['training','model','seed','delay_ms','intervention_count','intervention_fraction','accuracy']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Missing fields: {sorted(set(required)-set(df.columns))}; legacy data are not TNNLS results')
    for c in required[2:]:
        df[c] = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(df[c]).all(): raise ValueError(f'Non-finite {c}')
    if not df.accuracy.between(0,1).all(): raise ValueError('accuracy must be in [0,1]')
    df = df[(df.intervention_count == 0) | np.isclose(df.intervention_fraction, fraction)].copy()
    key = ['training','model','seed','delay_ms','intervention_count']
    if df.duplicated(key).any(): raise ValueError('Duplicate seed/condition; do not repeat K=0 for each fraction')
    if set(zip(df.training,df.model)) != {(t,m) for t in ('Clean','Intervention') for m in MODELS}:
        raise ValueError('Require six specified models under both Clean and Intervention training')
    common_grid = None
    for (training,model), g in df.groupby(['training','model']):
        if set(g.delay_ms) != DELAYS: raise ValueError(f'Five required delays missing: {training}/{model}')
        grid = set(zip(g.delay_ms,g.intervention_count))
        if common_grid is None: common_grid = grid
        if grid != common_grid: raise ValueError('All model/training groups must share delay/K grid')
        for (delay,k), cell in g.groupby(['delay_ms','intervention_count']):
            if set(cell.seed) != SEEDS: raise ValueError(f'Five seeds required: {training}/{model}/{delay}/{k}')
    ks = sorted(df.intervention_count.unique())
    if not {0,1,2,4,8,16,32,40}.issubset(ks) or any((d,k) not in common_grid for d in DELAYS for k in ks):
        raise ValueError('Need complete delay × K grid, including 0, 1, 2, 4, 8, 16, 32, 40')
    return df

def load_replay(path, training):
    with np.load(path, allow_pickle=False) as z:
        required = ['input_clean','input_stress','spikes_clean','spikes_stress','forced_stress','dt_ms','cue1_start_ms','cue1_end_ms','cue2_start_ms','cue2_end_ms','seed','sample_id','model','training','checkpoint_id','intervention_count','logits_clean','logits_stress','first_side','second_side']
        missing = set(required)-set(z.files)
        if missing: raise ValueError(f'Replay missing fields: {sorted(missing)}')
        x = {k:np.array(z[k]) for k in required}
    if str(x['model'].item()) != 'SPRiF' or str(x['training'].item()) != training: raise ValueError('Replay model/training mismatch')
    if int(x['seed'].item()) not in SEEDS or int(x['intervention_count'].item()) != 8: raise ValueError('Replay requires a formal seed and K=8')
    if not str(x['checkpoint_id'].item()).strip() or not str(x['sample_id'].item()).strip(): raise ValueError('Replay provenance missing')
    if x['input_clean'].shape != x['input_stress'].shape or not np.array_equal(x['input_clean'],x['input_stress']): raise ValueError('Clean/stress inputs must be identical')
    clean, stress, forced = (x[k] for k in ('spikes_clean','spikes_stress','forced_stress'))
    if clean.ndim != 2 or clean.shape != stress.shape or clean.shape != forced.shape or x['input_clean'].shape[0] != clean.shape[0]: raise ValueError('Replay arrays must match [time, neuron]')
    if not all(np.isin(v,[0,1]).all() for v in (clean,stress,forced)): raise ValueError('Spikes and forced hits must be binary')
    if np.any(forced > stress) or np.count_nonzero(np.any(forced,axis=1)) != 8: raise ValueError('Forced hits must be spikes at eight timesteps')
    dt = float(x['dt_ms']); bounds=[float(x[k]) for k in ('cue1_start_ms','cue1_end_ms','cue2_start_ms','cue2_end_ms')]
    if dt <= 0 or not (0 <= bounds[0] < bounds[1] < bounds[2] < bounds[3] <= len(clean)*dt): raise ValueError('Invalid cue timing')
    times = np.flatnonzero(np.any(forced,axis=1))*dt
    if np.any((times < bounds[1]) | (times >= bounds[2])): raise ValueError('Forced events must occur during delay')
    for key in ('logits_clean','logits_stress'):
        if x[key].shape != (len(clean),2) or not np.isfinite(x[key]).all(): raise ValueError('Readout logits must be finite [time,2]')
    if int(x['first_side']) != 0 or int(x['second_side']) != 0: raise ValueError('Original layout requires a declared A-A trial')
    return x

def make_figure(df, trace, training, out, fraction):
    import legacy_layout as layout
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                        'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
                        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False})
    common = dict(dt=float(trace['dt_ms']),pre=float(trace['cue1_start_ms']),
                  cue=float(trace['cue1_end_ms']-trace['cue1_start_ms']),
                  delay=float(trace['cue2_start_ms']-trace['cue1_end_ms']))
    pair=[]
    for condition in ('clean','stress'):
        hits=np.zeros_like(trace['spikes_clean'],dtype=bool) if condition=='clean' else trace['forced_stress'].astype(bool)
        pair.append(dict(common,spikes=trace['spikes_'+condition],hits=hits,intervention=hits,
                         logits=trace['logits_'+condition],input=trace['input_'+condition]))
    # Same dimensions, grid ratios, spans and spacing as the historical main composite.
    fig=plt.figure(figsize=(7.0,4.8))
    outer=fig.add_gridspec(1,2,width_ratios=[.44,.56],left=.065,right=.965,top=.95,bottom=.14,wspace=.25)
    left=outer[0,0].subgridspec(3,1,height_ratios=[1.18,.44,.96],hspace=layout.LEFT_COLUMN_HSPACE)
    trial_grid=left[0].subgridspec(2,1,hspace=.08)
    clean_ax=fig.add_subplot(trial_grid[0]);stress_ax=fig.add_subplot(trial_grid[1],sharex=clean_ax)
    details=left[1].subgridspec(1,2,width_ratios=[1,1.08],wspace=.36)
    activity_ax=fig.add_subplot(details[0]);readout_ax=fig.add_subplot(details[1]);curve_ax=fig.add_subplot(left[2])
    right=outer[0,1].subgridspec(1,2,width_ratios=[1,.055],wspace=.18)
    heat_grid=right[0,0].subgridspec(3,2,hspace=.27,wspace=.21)
    layout.plot_paired_trial(clean_ax,stress_ax,activity_ax,readout_ax,*pair)
    data=df[df.training==training]
    handles=layout.plot_accuracy(curve_ax,data)
    cax=fig.add_subplot(right[0,1]);layout.plot_heatmaps(fig,heat_grid,data,cax)
    ordered=['SPRiF','MEMORY_RESET5','ASRNN','BRF','REAL_MODES3','LIF']
    fig.legend([handles[m] for m in ordered],ordered,ncol=layout.MODEL_LEGEND_NCOL,
               loc='lower center',bbox_to_anchor=(layout.MODEL_LEGEND_X,layout.MODEL_LEGEND_Y),
               fontsize=5.35,handlelength=1.2,columnspacing=.62,handletextpad=.35,borderpad=.15,labelspacing=.28)
    out.mkdir(parents=True,exist_ok=True);stem=out/f'si_dms_composite_{training.lower()}'
    # Audit each comparable group; intentional unequal detail widths retained from original.
    require_matplotlib_panel_alignment(fig,axes=[clean_ax,stress_ax],json_out=str(stem)+'.raster-alignment.json',strict=True)
    heat_axes=[ax for ax in fig.axes if ax.images]
    require_matplotlib_panel_alignment(fig,axes=heat_axes,json_out=str(stem)+'.heatmap-alignment.json',strict=True)
    selected=layout.select_paired_raster_units(*pair).tolist()
    Path(str(stem)+'.selection.json').write_text(json.dumps(dict(training=training,seed=int(trace['seed']),
        sample_id=str(trace['sample_id']),checkpoint_id=str(trace['checkpoint_id']),displayed_neurons=selected,
        total_neurons=int(trace['spikes_clean'].shape[1]),rule='Historical paired non-tonic unit ranking; 10 displayed',
        heatmap_slot='REAL_MODES3 occupies the historical lambda0 position; these are different models'),indent=2),encoding='utf-8')
    fig.savefig(str(stem)+'.pdf',bbox_inches='tight');fig.savefig(str(stem)+'.svg',bbox_inches='tight');fig.savefig(str(stem)+'.png',dpi=600,bbox_inches='tight')
    plt.close(fig)
    return stem

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=BASE/'原始数据'/'evaluations.json')
    p.add_argument('--replay-clean',type=Path,default=BASE/'原始数据'/'replay_clean.npz')
    p.add_argument('--replay-intervention',type=Path,default=BASE/'原始数据'/'replay_intervention.npz')
    p.add_argument('--output',type=Path,default=BASE/'输出图片')
    p.add_argument('--fraction',type=float,default=.15)
    args=p.parse_args(); df=load_metrics(args.input,args.fraction)
    traces={t:load_replay(path,t) for t,path in [('Clean',args.replay_clean),('Intervention',args.replay_intervention)]}
    args.output.mkdir(parents=True,exist_ok=True)
    seed_means=df.groupby(['training','model','seed','intervention_count'],as_index=False).accuracy.mean()
    seed_means.to_csv(args.output/'five_delay_per_seed.csv',index=False)
    for training in ('Clean','Intervention'):
        print('Saved',make_figure(df,traces[training],training,args.output,args.fraction))
if __name__=='__main__': main()
