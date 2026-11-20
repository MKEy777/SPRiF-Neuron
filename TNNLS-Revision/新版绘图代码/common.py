"""Shared TNNLS11 parameter loading, local replay and plotting utilities."""
import numpy as np

def parameters(path):
    with np.load(path, allow_pickle=False) as z:
        p = {k: np.array(z[k]) for k in ('alpha','rho','omega','eta','bar_G','threshold')}
    n = p['alpha'].size
    for k, shape in {'alpha':(n,), 'rho':(n,), 'omega':(n,), 'eta':(n,2), 'bar_G':(n,2,3)}.items():
        if p[k].shape != shape or not np.isfinite(p[k]).all():
            raise ValueError(f'Invalid {k}: expected {shape}')
    for k in ('alpha','rho','eta'):
        if not ((p[k] > 0) & (p[k] < 1)).all(): raise ValueError(f'Invalid decay {k}')
    if not ((p['omega'] >= 0) & (p['omega'] <= np.pi)).all(): raise ValueError('Invalid omega')
    p['threshold'] = np.broadcast_to(p['threshold'], (n,))
    if not (np.isfinite(p['threshold']) & (p['threshold'] > 0)).all(): raise ValueError('Invalid threshold')
    return p

def replay(p, currents, memory_reset=False, forced=None, spikes=True):
    """Fixed-current local replay, NOT a recurrent-network intervention experiment."""
    n = len(p['alpha']); currents = np.asarray(currents)
    if currents.ndim != 2 or currents.shape[1] != n or not np.isfinite(currents).all():
        raise ValueError('currents must be finite [time, neuron]')
    forced = np.zeros_like(currents, dtype=bool) if forced is None else forced
    if forced.shape != currents.shape: raise ValueError('forced shape mismatch')
    x = np.zeros((n,3)); h = np.zeros(n); u = np.zeros(n)
    B = np.stack([1-p['alpha'],1-p['rho'],np.zeros(n)],axis=-1)
    bhat = B / np.linalg.norm(B,axis=-1,keepdims=True)
    rows = []
    for t, current in enumerate(currents):
        a,b,c = x.T; w = p['omega']; r = p['rho']
        xp = np.stack([p['alpha']*a, r*(np.cos(w)*b-np.sin(w)*c), r*(np.sin(w)*b+np.cos(w)*c)],axis=-1)+B*current[:,None]
        proj = np.einsum('nk,npk->np',xp,p['bar_G'])
        drive = h + proj[:,0]
        up = p['eta'][:,0]*u + drive
        hn = p['eta'][:,1]*h + proj[:,1]
        # Force discharge by overwriting candidate, as in SI-DMS; local current held fixed.
        up = np.where(forced[t],p['threshold'],up)
        z = (up >= p['threshold']).astype(float) if spikes else np.zeros(n)
        x = xp - (p['threshold']*z)[:,None]*bhat if memory_reset else xp.copy()
        u = up-p['threshold']*z; h = hn
        rows.append((xp.copy(),x.copy(),h.copy(),up.copy(),u.copy(),z.copy(),drive.copy()))
    return {k:np.stack([row[i] for row in rows]) for i,k in enumerate(('x_pre','x_post','h','u_pre','u_post','spike','drive'))}

def save_plot(out, name, xs, series, xlabel, ylabel, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(6.5,3.2),layout='constrained')
    for label, y, sd in series:
        line, = ax.plot(xs,y,label=label)
        if sd is not None: ax.fill_between(xs,y-sd,y+sd,color=line.get_color(),alpha=.15)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title); ax.legend(fontsize=7)
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'): fig.savefig(out/f'{name}.{ext}',dpi=300)
    plt.close(fig)
