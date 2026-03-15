# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UVRefineStableDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        input_uv: torch.Tensor,
        input_uv_position: torch.Tensor,
        input_uv_mask: torch.Tensor,
        num_inference_steps: int = 1,
        num_images_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "pil",
        replicate_num: int = 1,
        **kwargs,
    ):
        device = self._execution_device
        
        input_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )['input_ids'].to(self.device)
        encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0].float()
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        image_pt = input_uv * 2.0 - 1.0  # scale to [-1, 1]
        image_pt = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor

        latents = [torch.zeros_like(image_pt, device=device, dtype=image_pt.dtype) for _ in range(replicate_num)]
        latents = torch.concat(latents, dim=0)

        downsampled_height = image_pt.shape[-2]
        downsampled_width = image_pt.shape[-1]
        downsampled_uv_position = F.interpolate(input_uv_position, size=(downsampled_height, downsampled_width), mode='bilinear')
        downsampled_uv_mask = F.interpolate(input_uv_mask, size=(downsampled_height, downsampled_width), mode='bilinear')
        image_pt = torch.cat([image_pt, downsampled_uv_mask, downsampled_uv_position], dim=1)

        for i, t in enumerate(timesteps):
            latent_model_input = [image_pt]
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
            )[0]
            
            noise_pred = torch.concat(noise_pred, dim=0)
            latents = self.scheduler.step(noise_pred, t, latents).pred_original_sample

        latents = latents.chunk(replicate_num)
        images = []
        for i, latent in enumerate(latents):
            image = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            images.append(image)
        
        return images
