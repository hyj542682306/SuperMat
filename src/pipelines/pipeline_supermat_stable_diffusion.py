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

from typing import Dict, List, Optional, Union

import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SuperMatStableDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 1,
        num_images_per_prompt: Optional[int] = 1,
        source_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        image_condition: bool = False, 
        replicate_num: int = 2,
        camera_embeds: Optional[torch.FloatTensor] = None,
        use_fp32_input: bool = True,
        **kwargs,
    ):
        device = self._execution_device

        if image_condition:
            clip_image_mean = torch.as_tensor(self.feature_extractor.image_mean)[:, None, None].to(self.device)
            clip_image_std = torch.as_tensor(self.feature_extractor.image_std)[:, None, None].to(self.device)
            imgs_in_proc = TF.resize(source_image, (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), interpolation=InterpolationMode.BICUBIC)
            imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std)
            encoder_hidden_states = self.image_encoder(imgs_in_proc).image_embeds.unsqueeze(1)
        else:
            input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )['input_ids'].to(device)
            encoder_hidden_states = self.text_encoder(input_ids, return_dict=False)[0]
            if use_fp32_input:
                encoder_hidden_states = encoder_hidden_states.float()
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        image_pt = source_image * 2.0 - 1.0  # scale to [-1, 1]
        image_pt = self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor

        latents = [torch.zeros_like(image_pt, device=device, dtype=image_pt.dtype) for _ in range(replicate_num)]
        latents = torch.concat(latents, dim=0)

        for i, t in enumerate(timesteps):
            latent_model_input = image_pt
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=camera_embeds,
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
