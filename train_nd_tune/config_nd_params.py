from dataclasses import dataclass, field

#config_and_params.py
@dataclass
class MambaConfig:
    block_size = 128
    d_model: int = 128
    n_layer: int = 4
    vocab_size: int = 0
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
        
@dataclass
class TrainerParams:
    batch_size = 16
    lr = 1e-3