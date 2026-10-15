"""Canonical HAIM model, refactored from haim_probabilistic_models.py without changing operations."""
from torch import nn
from ..encoders.haim import HAIMModalityEncoder, MODALITY_DIMS
from .components import GaussianEvidence, ReliabilityHead, PredictionHead
from .fusion import mean_fusion, gaussian_poe, reliability_weighted_poe
MODALITIES = tuple(MODALITY_DIMS)

class HAIMProbabilisticModel(nn.Module):
    def __init__(self, fusion_type="gaussian", embedding_dim=128, latent_dim=32):
        super().__init__()
        if fusion_type not in {"mean","gaussian","reliability"}: raise ValueError(f"Invalid fusion_type: {fusion_type}")
        self.fusion_type=fusion_type; self.embedding_dim=embedding_dim; self.latent_dim=latent_dim
        self.encoders=nn.ModuleDict({m:HAIMModalityEncoder(d,embedding_dim) for m,d in MODALITY_DIMS.items()})
        # Original attribute name retained for checkpoint compatibility.
        self.gaussian_heads=nn.ModuleDict({m:GaussianEvidence(embedding_dim,latent_dim) for m in MODALITIES})
        self.reliability_heads=nn.ModuleDict({m:ReliabilityHead(latent_dim) for m in MODALITIES})
        self.prediction_head=PredictionHead(latent_dim)
    def encode_modality(self, modality, x):
        h=self.encoders[modality](x); e=self.gaussian_heads[modality](h); r=self.reliability_heads[modality](e["mu"],e["logvar"])
        return {"embedding":h,"mu":e["mu"],"logvar":e["logvar"],"reliability":r}
    def encode_available(self,batch,missing=None):
        missing=set() if missing is None else set(missing); experts=[]; by_modality={}
        for m in MODALITIES:
            if m in missing: continue
            if m not in batch: raise KeyError(f"Batch is missing '{m}'.")
            e=self.encode_modality(m,batch[m]); experts.append(e); by_modality[m]=e
        if not experts: raise ValueError("All modalities are missing.")
        return experts,by_modality
    def forward(self,batch,missing=None,return_details=False):
        experts,evidence=self.encode_available(batch,missing)
        fused={"mean":mean_fusion,"gaussian":gaussian_poe,"reliability":reliability_weighted_poe}[self.fusion_type](experts)
        logits=self.prediction_head(fused["mu"])
        if not return_details: return logits
        return {"logits":logits,"fused_mu":fused["mu"],"fused_variance":fused["variance"],"evidence":evidence}

def build_model_a(): return HAIMProbabilisticModel("mean")
def build_model_b(): return HAIMProbabilisticModel("gaussian")
def build_model_c(): return HAIMProbabilisticModel("reliability")
