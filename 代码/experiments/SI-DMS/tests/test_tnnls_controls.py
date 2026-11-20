import torch
from sidms.cells import build_cell
from sidms.config import ExperimentConfig
from sidms.data import make_batch


def test_real_modes_are_driven_and_preserve_scalar_reset():
    cell=build_cell('real_modes3',2,3,5,1.0)
    state=cell.initial_state(1,'cpu')
    with torch.no_grad():
        cell.alpha_raw.fill_(torch.logit(torch.tensor(.2)))
        cell.rho_raw.fill_(torch.logit(torch.tensor(.4)))
        cell.omega_raw.fill_(torch.logit(torch.tensor(.7)))
        cell.input.weight.zero_();cell.input.bias.fill_(1)
    forced=torch.ones(1,3,dtype=torch.bool)
    z, new, diag=cell.step(torch.zeros(1,2),state,forced)
    expected=torch.tensor([.8,.6,.3]).view(1,1,3).expand(1,3,3)
    torch.testing.assert_close(new['x'],expected)
    assert torch.all(z==1) and torch.all(diag['forced_hit'])
    assert sum(p.numel() for n,p in cell.named_parameters() if n in {'alpha_raw','rho_raw','omega_raw','eta_raw','bar_G'}) == 11*cell.hidden_size


def test_memory_reset_matches_normalized_drive_direction():
    base=build_cell('sprif_scalar',2,3,5,1.0)
    ctrl=build_cell('memory_reset5',2,3,5,1.0);ctrl.load_state_dict(base.state_dict())
    state=base.initial_state(1,'cpu');x=torch.zeros(1,2);mask=torch.ones(1,3,dtype=torch.bool)
    z,a,_=base.step(x,state,mask);z2,b,_=ctrl.step(x,state,mask)
    alpha,rho=torch.sigmoid(base.alpha_raw),torch.sigmoid(base.rho_raw)
    B=torch.stack((1-alpha,1-rho,torch.zeros_like(alpha)),-1)
    torch.testing.assert_close(a['x']-b['x'],z.unsqueeze(-1)*B/torch.linalg.vector_norm(B,dim=-1,keepdim=True))
    torch.testing.assert_close(a['h'],b['h']);torch.testing.assert_close(a['u'],b['u']);torch.testing.assert_close(z,z2)


def test_clean_and_stress_inputs_share_seed_with_separate_mask_stream():
    cfg=ExperimentConfig();seed=3_000_000_000_123;mask_seed=4_000_000_000_123
    a=make_batch(cfg,4,200,0,64,seed=seed,fraction=0,mask_seed=mask_seed)
    b=make_batch(cfg,4,200,8,64,seed=seed,fraction=.15,mask_seed=mask_seed)
    assert torch.equal(a.x,b.x) and torch.equal(a.y,b.y)
    assert not a.intervention.any() and b.intervention.any()
