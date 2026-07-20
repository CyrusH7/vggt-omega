import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT-Omega on 7-Scenes office/seq-01.")
    parser.add_argument(
        "--data-dir",
        default="/root/autodl-tmp/vggt-omega/dataset/7-Scenes/office/seq-01",
    )
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/vggt-omega/ckpts/vggt_omega_1b_512.pt",
    )
    parser.add_argument("--output", default="outputs/7scenes_office_seq01_vggt_omega.npz")
    parser.add_argument("--total-frames", type=int, default=1000)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def sample_images(data_dir, total_frames, num_frames):
    image_paths = sorted(glob.glob(str(Path(data_dir) / "frame-*.color.png")))[:total_frames]
    if len(image_paths) < total_frames:
        raise ValueError(f"Found {len(image_paths)} color frames, expected at least {total_frames}.")

    step = total_frames // num_frames
    indices = range(0, total_frames, step)
    return [image_paths[i] for i in indices]


def to_numpy(tensor):
    array = tensor.detach().float().cpu().numpy()
    return array[0] if array.shape[0] == 1 else array


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VGGT-Omega inference.")

    image_names = sample_images(args.data_dir, args.total_frames, args.num_frames)

    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    images = load_and_preprocess_images(image_names, image_resolution=args.image_resolution).to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), tqdm(total=1, desc="Inference", unit="seq") as pbar:
        predictions = model(images)
        torch.cuda.synchronize()
        pbar.update(1)
    elapsed = time.perf_counter() - start

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsics
    predictions["intrinsic"] = intrinsics

    predictions_np = {
        key: to_numpy(value)
        for key, value in predictions.items()
        if isinstance(value, torch.Tensor)
    }
    predictions_np["image_names"] = np.array(image_names)
    predictions_np["camera_tokens"] = predictions_np["camera_and_register_tokens"][:, :1]
    predictions_np["registers"] = predictions_np["camera_and_register_tokens"][:, 1:]
    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **predictions_np)

    print(f"Saved: {output_path}")
    print(f"Inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()