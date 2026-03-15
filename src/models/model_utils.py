from typing import List, Optional

import torch
import torch.nn as nn

from .custom_attention_processor import MVAttnProcessor2_0, MVXFormersAttnProcessor, set_supermat_unet_2d_condition_attn_processor

def set_supermat_mv_self_attention(model: nn.Module, num_views: int, attn_positions: Optional[List[str]] = None, use_xformers: bool = True):
    mv_attn_cls = MVXFormersAttnProcessor if use_xformers else MVAttnProcessor2_0
    def set_self_attn_proc_func(name, hs, cad, ap):
        if attn_positions is None or any([pos in name for pos in attn_positions]):
            return mv_attn_cls(num_views=num_views)
        return ap
    
    set_supermat_unet_2d_condition_attn_processor(
        model,
        set_self_attn_proc_func=set_self_attn_proc_func
    )