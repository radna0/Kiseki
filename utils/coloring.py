import torch
import torch.utils.data as data
from basicsr.archs.basicpbc_arch import BasicPBC
from basicsr.models.pbc_model import ModelInference
from basicsr.data.pbc_inference_dataset import PaintBucketInferenceDataset
from utils.logging import Profiler


def load_params(model_path):
    full_model = torch.load(model_path, map_location="cpu", weights_only=True)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model


def main(args):

    ckpt_path = "ckpt/basicpbc.pth"
    model = BasicPBC(
        ch_in=6,
        descriptor_dim=128,
        keypoint_encoder=[32, 64, 128],
        GNN_layer_num=9,
        use_clip=True,
        encoder_resolution=(640, 640),
        raft_resolution=(args.raft_res, args.raft_res),
        clip_resolution=(640, 640),
    )
    model.load_state_dict(load_params(ckpt_path))
    model.eval()

    opt = {"root": args.path, "multi_clip": args.multi_clip, "mode": args.mode}
    dataset = PaintBucketInferenceDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=1)

    model_inference = ModelInference(model, dataloader)
    with Profiler("Coloring Inference Time", limit=5):
        if args.mode == "reference":
            model_inference.inference_multi_gt_sequential(args.path, args.keep_line)
        else:
            model_inference.inference_multi_gt(args.path, args.keep_line)
