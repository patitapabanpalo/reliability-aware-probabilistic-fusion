import torch
from reliability_fusion.models.components import GaussianEvidence, ReliabilityHead
from reliability_fusion.models.fusion import gaussian_poe, reliability_weighted_poe
from reliability_fusion.models.haim import HAIMProbabilisticModel
from reliability_fusion.losses import consistency_loss

def reference_poe(experts, weighted):
    ps=torch.ones_like(experts[0]["mu"]); pms=torch.zeros_like(ps)
    for e in experts:
        p=1.0/torch.exp(e["logvar"]).clamp_min(1e-6)
        if weighted: p=p*e["reliability"].clamp(0,1)
        ps=ps+p; pms=pms+p*e["mu"]
    v=1.0/ps.clamp_min(1e-6)
    return v*pms,v

def make_experts():
    torch.manual_seed(7)
    return [{"mu":torch.randn(5,32),"logvar":torch.randn(5,32).clamp(-6,4),"reliability":torch.sigmoid(torch.randn(5,1))} for _ in range(3)]

def test_gaussian_poe_matches_reference():
    e=make_experts(); mu,v=reference_poe(e,False); out=gaussian_poe(e)
    assert torch.allclose(out["mu"],mu); assert torch.allclose(out["variance"],v)

def test_reliability_poe_matches_reference():
    e=make_experts(); mu,v=reference_poe(e,True); out=reliability_weighted_poe(e)
    assert torch.allclose(out["mu"],mu); assert torch.allclose(out["variance"],v)

def test_gaussian_evidence_clamp_and_names():
    m=GaussianEvidence(128,32); out=m(torch.randn(4,128))
    assert out["mu"].shape==(4,32); assert out["logvar"].min()>=-6; assert out["logvar"].max()<=4
    keys=m.state_dict().keys(); assert "mu.weight" in keys and "logvar.weight" in keys

def test_reliability_shape_and_range():
    m=ReliabilityHead(32); r=m(torch.randn(4,32),torch.randn(4,32)); assert r.shape==(4,1); assert ((r>0)&(r<1)).all()

def test_dataset_specific_consistency_spaces_differ():
    a=torch.tensor([0.0,2.0]); b=torch.tensor([1.0,-1.0])
    assert not torch.allclose(consistency_loss(a,b,"logit"),consistency_loss(a,b,"probability"))

def test_haim_checkpoint_key_compatibility():
    keys=HAIMProbabilisticModel("reliability").state_dict().keys()
    assert "gaussian_heads.cxr.mu.weight" in keys
    assert "gaussian_heads.cxr.logvar.weight" in keys
    assert "reliability_heads.cxr.network.0.weight" in keys
