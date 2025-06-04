import numpy as np
import os
import os.path as osp
import random
import shutil
import torch
from collections import OrderedDict
from glob import glob
from skimage import io
from torch import nn as nn
from torch.nn import init as init
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.registry import MODEL_REGISTRY
from kiseki.paint import (
    colorize_label_image,
    dump_json,
    eval_json_folder,
    evaluate,
    load_json,
    read_img_2_np,
    recolorize_gt,
    merge_color_line,
)

import concurrent.futures

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from kiseki.logging import Profiler, logger


CPU_COUNT = os.cpu_count()
MAX_WORKERS = 8
THREADS_PER_WORKER = CPU_COUNT // MAX_WORKERS


@MODEL_REGISTRY.register()
class Kiseki(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l_ce = build_loss(train_opt["l_ce"]).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.data = data
        white_list = ["file_name"]
        for key in data.keys():
            if key not in white_list:
                self.data[key] = data[key].to(self.device)

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.data)

        for k, v in self.data.items():
            self.data[k] = v[0]
        pred = {**self.data, **self.output}

        if pred["skip_train"]:
            return

        l_total = 0
        loss_dict = OrderedDict()

        loss = pred["loss"]  # / self.opt['datasets']['train']['batch_size_per_gpu']
        loss_dict["acc"] = torch.tensor(pred["accuracy"]).to(self.device)
        loss_dict["area_acc"] = torch.tensor(pred["area_accuracy"]).to(self.device)
        loss_dict["valid_acc"] = torch.tensor(pred["valid_accuracy"]).to(self.device)
        loss_dict["loss_total"] = self.l_ce(loss)

        l_total += loss
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.data)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.data)

        if not hasattr(self, "net_g_ema"):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        gt_folder_path = dataloader.dataset.opt["root"]
        with_metrics = self.opt["val"].get("metrics") is not None
        save_img = self.opt["val"].get("save_img", False)
        save_csv = self.opt["val"].get("save_csv", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if hasattr(self, "net_g_ema"):
            model_inference = ModelInference(self.net_g_ema, dataloader)
        else:
            model_inference = ModelInference(self.net_g, dataloader)

        self.net_g.train()
        save_path = osp.join(
            self.opt["path"]["visualization"], str(current_iter), dataset_name
        )
        model_inference.inference_frame_by_frame(save_path, save_img)
        results = eval_json_folder(save_path, gt_folder_path, "")
        if save_csv:
            csv_save_path = os.path.join(save_path, "metrics.csv")
            avg_dict, _, _ = evaluate(
                results, mode=dataset_name, save_path=csv_save_path
            )
        else:
            avg_dict, _, _ = evaluate(results, mode=dataset_name)

        self.metric_results["acc"] = avg_dict["acc"]
        self.metric_results["acc_thres"] = avg_dict["acc_thres"]
        self.metric_results["pix_acc"] = avg_dict["pix_acc"]
        self.metric_results["pix_acc_wobg"] = avg_dict["pix_acc_wobg"]
        self.metric_results["bmiou"] = avg_dict["bmiou"]
        self.metric_results["pix_bmiou"] = avg_dict["pix_bmiou"]

        if with_metrics:
            for metric in self.metric_results.keys():
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # Just output the line for test
        out_dict["line"] = self.data["line_ref"].detach().cpu()
        """
        out_dict['result']= self.blend.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        """
        return out_dict


class ModelInference:
    def __init__(self, model, samples, seed=42):
        self._set_seed(seed)
        self.samples = samples
        self.model = model

    def _set_seed(self, seed):
        self.py_rng_state0 = random.getstate()
        self.np_rng_state0 = np.random.get_state()
        self.torch_rng_state0 = torch.get_rng_state()
        set_random_seed(seed)

    def _recover_seed(self):
        if hasattr(self, "py_rng_state0"):
            random.setstate(self.py_rng_state0)
        if hasattr(self, "np_rng_state0"):
            np.random.set_state(self.np_rng_state0)
        if hasattr(self, "torch_rng_state0"):
            torch.set_rng_state(self.torch_rng_state0)

    def dis_data_to_cuda(self, data):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                import torch_xla as xla

                data[key] = data[key].to(xla.device())
        return data

    def preprocess_character_folder(self, samples, save_path):
        characters = set()

        for test_data in samples:
            line_root, line_name = osp.split(test_data["file_name"])
            logger.info(f"preprocessing {test_data['file_name']}")
            save_folder, _ = osp.split(line_root)
            _, character_name = osp.split(save_folder)
            res_folder = osp.join(save_path, character_name)

            if character_name not in characters:
                characters.add(character_name)
                os.makedirs(res_folder, exist_ok=True)
                gt_root = line_root.replace("line", "ref")
                logger.info(f'ref folder: {glob(osp.join(gt_root, "*.png"))}')
                for gt_path in glob(osp.join(gt_root, "*.png")):
                    # process the ground truth
                    json_path = gt_path.replace("ref", "seg").replace("png", "json")
                    shutil.copy(gt_path, res_folder)
                    shutil.copy(json_path, res_folder)
                    logger.info(f"{gt_path} is given.")

    def inference_single_frame(self, test_data, save_path, threads_per_worker=None):
        with torch.no_grad():
            logger.info(f"threads_per_worker: {threads_per_worker}")
            """ if threads_per_worker is not None:
                # force PyTorch to use 1 thread for intra‐op (convolution, matrix multiply, etc.)
                torch.set_num_threads(1)
                # and if you’re using any multithreaded data‐loading, also limit inter_op
                torch.set_num_interop_threads(1) """

            line_root, line_name = osp.split(test_data["file_name"])
            logger.info(f"processing {test_data['file_name']}")
            save_folder, _ = osp.split(line_root)
            _, character_name = osp.split(save_folder)
            res_folder = osp.join(save_path, character_name)

            _, ref_name = osp.split(test_data["file_name_ref"])
            json_path_ref = osp.join(res_folder, ref_name + ".json")
            color_dict = load_json(json_path_ref)
            json_save_path = osp.join(res_folder, line_name + ".json")
            res = self.model(test_data)
            match_scores = res["match_scores"].cpu().numpy()

            color_next_frame = {}
            unmatch_color = [0] * len(list(color_dict.values())[0])

            for i, scores in enumerate(match_scores):
                color_lookup = np.array(
                    [
                        (
                            color_dict[str(i + 1)]
                            if str(i + 1) in color_dict
                            else unmatch_color
                        )
                        for i in range(len(scores))
                    ]
                )
                unique_colors = np.unique(color_lookup, axis=0)
                accumulated_probs = [
                    np.sum(scores[np.all(color_lookup == color, axis=1)])
                    for color in unique_colors
                ]
                color_next_frame[str(i + 1)] = unique_colors[
                    np.argmax(accumulated_probs)
                ].tolist()

            dump_json(color_next_frame, json_save_path)
            label_path = osp.join(save_folder, "seg", line_name + ".png")
            img_save_path = json_save_path.replace(".json", ".png")

            colorize_label_image(label_path, json_save_path, img_save_path)
            logger.info(f"{img_save_path} created.\n")

    # processpool will be used to parallelize the inference process
    def inference_multi_gt_parallel(self, save_path):
        self.preprocess_character_folder(self.samples, save_path)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(MAX_WORKERS, len(self.samples))
        ) as executor:
            threads_per_worker = int(
                CPU_COUNT // len(self.samples)
                if len(self.samples) < MAX_WORKERS
                else THREADS_PER_WORKER
            )

            futures = [
                executor.submit(
                    self.inference_single_frame,
                    test_data,
                    save_path,
                    threads_per_worker,
                )
                for test_data in self.samples
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    # this function will be used to sequentialize the inference process
    def inference_multi_gt_sequential(self, save_path):
        self.preprocess_character_folder(self.samples, save_path)
        for test_data in self.samples:
            self.inference_single_frame(
                test_data,
                save_path,
            )
