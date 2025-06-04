import torch

import torch.utils.data as data
from basicsr.archs.kiseki_arch import Kiseki
from basicsr.data.kiseki_inference_dataset import KisekiInMemoryInferenceDataset
from basicsr.data.kiseki_parallel_inference_dataset import (
    KisekiParallizedInMemoryInferenceDataset,
)
from basicsr.models.kiseki_model import ModelInference
from basicsr.data.pbc_inference_dataset import PaintBucketInferenceDataset
from kiseki.logging import Profiler


def load_params(model_path):
    full_model = torch.load(model_path, map_location="cpu", weights_only=False)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model


def main(args):

    ckpt_path = "ckpt/basicpbc.pth"
    model = Kiseki(
        ch_in=6,
        descriptor_dim=128,
        keypoint_encoder=[32, 64, 128],
        GNN_layer_num=9,
        use_clip=True,
        encoder_resolution=(640, 640),
        clip_resolution=(640, 640),
    )
    model.load_state_dict(load_params(ckpt_path))

    opt = {"root": args.path, "multi_clip": args.multi_clip, "mode": args.mode}
    dataset = KisekiInMemoryInferenceDataset(opt)
        
    model_inference = ModelInference(model, dataset.samples)
    if args.mode == "reference":
        model_inference.inference_multi_gt_sequential(args.path)
    else:
        model_inference.inference_multi_gt(args.path)
