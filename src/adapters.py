from typing import Optional

from .models.uv_refine_unet_2d_condition import UVRefineUNet2DConditionModel
from .models.supermat_unet_2d_condition import SuperMatUNet2DConditionModel

class AdapterWrapper:
    @staticmethod
    def convert(pipeline: object, *args, **kwargs) -> object:
        raise NotImplementedError
    
class UVRefineAdapterWrapper(AdapterWrapper):
    @staticmethod
    def convert(
        pipeline: object,
        use_camera_embeddings: bool,
        camera_embeddings_dim: Optional[int] = None,
        replicate_num: int = 2
    ):
        unet_kwargs = {
            "in_channels": 8,
        }
        if use_camera_embeddings:
            unet_kwargs.update({
                "class_embed_type": "projection",
                "projection_class_embeddings_input_dim": camera_embeddings_dim
            })
        unet = UVRefineUNet2DConditionModel.from_config(pipeline.unet.config, **unet_kwargs)
        incompatible_keys = unet.load_state_dict({k: v for k, v in pipeline.unet.state_dict().items() if k != "conv_in.weight"}, strict=False)
        print(incompatible_keys)

        unet.conv_in.weight.data[:,:4] = pipeline.unet.conv_in.weight
        
        unet.replicate(replicate_num=replicate_num, shared_blocks_lora=False)
        pipeline.unet = unet
        return pipeline
    
class SuperMatAdapterWrapper(AdapterWrapper):
    @staticmethod
    def convert(
        pipeline: object,
        use_camera_embeddings: bool,
        camera_embeddings_dim: Optional[int] = None,
        replicate_num: int = 2
    ):
        unet_kwargs = {}
        if use_camera_embeddings:
            unet_kwargs.update({
                "class_embed_type": "projection",
                "projection_class_embeddings_input_dim": camera_embeddings_dim
            })
        unet = SuperMatUNet2DConditionModel.from_config(pipeline.unet.config, **unet_kwargs)
        incompatible_keys = unet.load_state_dict(pipeline.unet.state_dict(), strict=False)
        print(incompatible_keys)

        unet.replicate(replicate_num=replicate_num, shared_blocks_lora=False)
        pipeline.unet = unet
        return pipeline
