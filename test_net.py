import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from iCLIP.config import cfg
from iCLIP.dataset import make_data_loader
from iCLIP.engine.inference import inference
from iCLIP.modeling.detector import build_detection_model
from iCLIP.utils.checkpoint import ActionCheckpointer
from torch.utils.collect_env import get_pretty_env_info
from iCLIP.utils.comm import synchronize, get_rank
from iCLIP.utils.IA_helper import has_memory
from iCLIP.utils.logger import setup_logger
#pytorch issuse #973
import resource
import time

# webber: prompt
from prompt import text_prompt

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="config_files/iCLIP.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    # Merge config file.
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Print experimental infos.
    save_dir = ""
    logger = setup_logger("iCLIP", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    device = torch.device("cuda")

    # get action list from all label
    actionlist, actiondict, actiontoken = text_prompt(dateset_label=cfg.TESTSET_LABEL, clipbackbone='ViT-B/16', device=device)

    # Build the model.
    model = build_detection_model(cfg, actionlist, actiondict, actiontoken, device)
    model.to(device)
    #print(model.roi_heads.action.text_feature_generator.embedding.weight)

    # load weight.
    output_dir = cfg.OUTPUT_DIR
    checkpointer = ActionCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    mem_active = has_memory(cfg.MODEL.ICLIP_STRUCTURE)
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            output_folders[idx] = output_folder

    # Do inference.
    data_loaders_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_test in zip(output_folders, dataset_names, data_loaders_test):
        inference(
            model,
            data_loader_test,
            dataset_name,
            mem_active=mem_active,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
