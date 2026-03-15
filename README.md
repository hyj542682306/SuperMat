# SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates

### [Project Page](https://hyj542682306.github.io/SuperMat/) | [Paper](https://arxiv.org/pdf/2411.17515)

Official implementation of SuperMat, SuperMatMV and UV Refinement Network in *SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates*.

Yijia Hong<sup>1,2</sup>, Yuan-Chen Guo<sup>4</sup>, Ran Yi<sup>1</sup>, Yulong Chen<sup>3</sup>, Yan-Pei Cao<sup>4</sup>, Lizhuang Ma<sup>1</sup><br>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Harbin Institute of Technology, <sup>4</sup>VAST

<img src="assets/overview.png" alt="overview">

## Installation

The code has been tested on Python 3.12.0.

```jsx
conda create --name supermat python==3.12
conda activate supermat

git clone https://github.com/hyj542682306/SuperMat
cd SuperMat
pip install -r requirements.txt
```

## Checkpoints

We provide the pretrained parameters for four models:
- `supermat.pth`: parameters for the SuperMat model
- `supermat_mv.pth`: parameters for the multi-view version of SuperMat
- `uv_refine_bc.pth`: parameters for the UV refinement network handling albedo materials
- `uv_refine_rm.pth`: parameters for the UV refinement network handling roughness & metallic materials

Please download the pretrained models from [Hugging Face](https://huggingface.co/oyiya/SuperMat). These models are independent of each other, so you only need to download the ones required for your inference. 

By default, pretrained models should be placed in the `checkpoints` folder.

Besides, all models are built upon the base model `stabilityai/stable-diffusion-2-1`. By default, it will be downloaded automatically when running the code. If you prefer to use a local copy, you can specify the path via arguments to load the model offline.

> Please note that the official `stabilityai/stable-diffusion-2-1` base model has been removed. You may need to obtain the base model parameters through alternative sources, such as `sd2-community/stable-diffusion-2-1` on Hugging Face.

## Usage

SuperMat prefer RGBA images where only the target object appears as foreground, with alpha values set to `0` for all other regions. During inference, the input image is alpha-composited with a gray background `(0.5, 0.5, 0.5)`.

SuperMat models work best at `512x512` resolution and UV refinement networks work best at `1024×1024` resolution.

### 1) SuperMat Single-Image Inference

Inputs can be one image or a folder of images. For each image, outputs are written into one subfolder.

```bash
python inference_supermat.py \
  --input examples/ring_rendered_2views \ # /path/to/image_or_folder
  --output-dir outputs \ # /path/to/output
  --checkpoint checkpoints/supermat.pth \ # /path/to/supermat.pth
  --base-model stabilityai/stable-diffusion-2-1 \ # sd2-community/stable-diffusion-2-1
  --device cuda:0 \
  --image-size 512
```

### 2) SuperMat Multi-View Inference

Use this when each case has multi-view images.

```bash
python inference_supermat_mv.py \
  --input examples/bag_rendered_6views \ # /path/to/case_dir
  --output-dir outputs_mv \ # /path/to/output
  --checkpoint checkpoints/supermat_mv.pth \ # /path/to/supermat_mv.pth
  --base-model stabilityai/stable-diffusion-2-1 \ # sd2-community/stable-diffusion-2-1
  --device cuda:0 \
  --image-size 512 \
  --num_views 6 \
  --use-camera-embeds
```

Please note the following requirements for the multi-view model:
- The multi-view version of SuperMat processes **6 orthogonal** views by default. For each view, the **camera-to-world (c2w) matrix** is provided to the model as camera embeddings.
- All inputs for a single case should be organized in one folder. Refer to the folder structure in `examples/bag_rendered_6views` as an example.
- The input images must follow the same naming convention as in the example.
- Camera information is stored in `meta.json`. Please refer to `examples/bag_rendered_6views/meta.json` for the required format.

### 3) UV Refine Inference

Use this for refining the albedo UV map.

```bash
python inference_uv_refine.py \
  --input-uv examples/axe_uv/uv_bc.png \ # /path/to/uv_albedo
  --input-uv-position examples/axe_uv/uv_position.png \ # /path/to/uv_position
  --input-uv-mask examples/axe_uv/uv_mask.png \ # /path/to/uv_mask \
  --output-dir outputs_uv_bc \ # /path/to/output
  --checkpoint checkpoints/uv_refine_bc.pth \ # /path/to/uv_refine_bc.pth
  --base-model stabilityai/stable-diffusion-2-1 \ # sd2-community/stable-diffusion-2-1
  --device cuda:0 \
  --image-size 1024
```

And use this for refining the RM map.
```bash
python inference_uv_refine.py \
  --input-uv examples/axe_uv/uv_rm.png \ # /path/to/uv_rm
  --input-uv-position examples/axe_uv/uv_position.png \ # /path/to/uv_position
  --input-uv-mask examples/axe_uv/uv_mask.png \ # /path/to/uv_mask \
  --output-dir outputs_uv_rm \ # /path/to/output
  --checkpoint checkpoints/uv_refine_rm.pth \ # /path/to/uv_refine_rm.pth
  --base-model stabilityai/stable-diffusion-2-1 \ # sd2-community/stable-diffusion-2-1
  --device cuda:0 \
  --image-size 1024
```

The RM UV map is organized as follows: the `R` channel is set to `0`, the `G` channel stores the `Roughness` values, and the `B` channel stores the `Metallic` values.


## Acknowledgments

Our project benefits from the amazing [Stable Diffusion](https://github.com/compvis/stable-diffusion). We would like to thank the authors of the project for their contributions to the community.

## Citation

If you find our work useful in your research, please cite:

```
@inproceedings{hong2025supermat,
  title={Supermat: Physically consistent pbr material estimation at interactive rates},
  author={Hong, Yijia and Guo, Yuan-Chen and Yi, Ran and Chen, Yulong and Cao, Yan-Pei and Ma, Lizhuang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={25083--25093},
  year={2025}
}
```
