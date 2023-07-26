#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import os
import sys
import time


from loguru import logger

# logger.info("--" * 10)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import traceback

from saicinpainting.evaluation.refinement import refine_predict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import (
    make_dataset_single,
)
from saicinpainting.training.trainers import load_checkpoint


# @hydra.main(config_path="../configs/prediction", config_name="default.yaml")
def remove_object_func(input_image, output_image, mask_image=None) -> bool:
    start_time = time.time()

    try:

        logger.info("Start loading...")
        # register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log
        if mask_image is None:
            mask_image = input_image.split(".")[0] + "_mask.png"

        predict_config: OmegaConf = OmegaConf.create(
            {
                "img_suffix": os.path.splitext(input_image)[1],
            }
        )
        with hydra.initialize(config_path="../configs/prediction"):
            cfg = hydra.compose(config_name="uthus_config.yaml")
            predict_config = OmegaConf.create(cfg)

        cfg = OmegaConf.to_object(predict_config)
        # print(f"Predict config: {cfg}")

        train_config_path = os.path.join(predict_config.model.path, "config.yaml")

        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        checkpoint_path = os.path.join(
            predict_config.model.path, "models", predict_config.model.checkpoint
        )
        # logger.info("Start Load model... ")
        device = torch.device(predict_config.device)
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location=device
        )
        model.freeze()
        logger.info("Model loaded!")
        # logger.info(f"Device: {device}")

        # Remove object stage
        dataset = make_dataset_single(
            input_image=input_image,
            mask_image=mask_image,
            **predict_config.dataset,
        )
        cur_out_fname = output_image

        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = default_collate([dataset[0]])
        # if predict_config.get("refine", False):
        assert "unpad_to_size" in batch, "Unpadded size is required for the refinement"
        # image unpadding is taken care of in the refiner, so that output image
        # is same size as the input image
        logger.info(f"Refining ${batch['image'].shape}")
        cur_res = refine_predict(batch, model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)

        return cur_out_fname
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as ex:
        logger.critical(f"Prediction failed due to {ex}:\n{traceback.format_exc()}")
        # sys.exit(1)
        raise ex
    finally:
        end_time = time.time()
        logger.info(f"Prediction took {end_time - start_time:.3f} seconds")
        # logger.info(f"Device: {device}")
        # if device == "cuda":
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    remove_object_func(
        input_image="/home/s2110149/WORKING/ai-customize-image/external_services/image_inpainting/LaMa_test_images2/2023-06-12.jpg",
        output_image="/home/s2110149/WORKING/ai-customize-image/external_services/image_inpainting/output/123_output.png",
        mask_image="/home/s2110149/WORKING/ai-customize-image/external_services/image_inpainting/LaMa_test_images2/2023-06-12_mask.jpg",
    )
