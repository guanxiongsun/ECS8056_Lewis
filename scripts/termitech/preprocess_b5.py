#!/usr/bin/env python3
"""preprocess_b5.py: BtP plan B5. Pre-compute OpenVLA's evaluation pre-processing for the frozen scenes.

Runs in the separate `sgvla-tf` env (TensorFlow CPU), so the torch env stays untouched:
    conda activate sgvla-tf
    python scripts/termitech/preprocess_b5.py --data $SGVLA_DATA --out $SGVLA_ROOT/v2_pre

Each path writes <out>/<path>/<construct_id>_<transform>.png (224x224 RGB) for the 400 frozen scenes:
  official    OpenVLA's Bridge evaluation code applied to our frames
              (experiments/robot/bridge/bridgev2_utils.py, resize_image): JPEG encode/decode, then
              tf.image.resize to 224x224 with lanczos3 and antialias, rounded and clipped
  bridgeorig  an approximation of the training images: first resize to 256x256, the bridge_orig
              release's resolution (area resampling, an assumption), then the official path
Mirroring is applied to the 640x480 frame before pre-processing, as for a flipped camera image.
run_gpu.py reads these files when SGVLA_IMAGE_ROOT=<out>/<path>.
"""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def official(img: np.ndarray) -> np.ndarray:
    x = tf.image.encode_jpeg(img)                                   # as in OpenVLA's resize_image
    x = tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8)
    x = tf.image.resize(x, (224, 224), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()


def bridgeorig(img: np.ndarray) -> np.ndarray:
    x = tf.image.resize(img, (256, 256), method="area")
    x = tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()
    return official(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = pd.read_csv(os.path.join(args.data, "constructed", "evaluation_set.csv"))["construct_id"]
    paths = {"official": official, "bridgeorig": bridgeorig}
    for name in paths:
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    for k, cid in enumerate(ids, 1):
        img = np.asarray(Image.open(os.path.join(args.data, "constructed", "frames", f"{cid}.png")).convert("RGB"))
        for transform, im in (("original", img), ("mirror", np.ascontiguousarray(img[:, ::-1]))):
            for name, fn in paths.items():
                Image.fromarray(fn(im)).save(os.path.join(args.out, name, f"{cid}_{transform}.png"))
        if k % 100 == 0:
            print(f"{k}/{len(ids)}", flush=True)
    print("done", len(ids), "scenes x 2 transforms x", len(paths), "paths")


if __name__ == "__main__":
    main()
