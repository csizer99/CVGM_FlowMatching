"""
Training script for the PianoMotion10M flow-matching model (and, with
--model_type baseline, the comparison regressor).

READ GUIDE.md FIRST -- especially the "assumptions to verify" section.
This script deliberately reuses PianoMotion10M's OWN dataset class
(datasets.PianoPose.PianoPose) rather than re-parsing the annotation JSON
files ourselves. That class already knows the real file layout; writing a
second, independent parser would just be a second place to get it wrong.

REQUIRED SETUP:
1. Clone https://github.com/agnJason/PianoMotion10M somewhere on disk.
2. Set the environment variable PIANOMOTION10M_REPO to that path, e.g.
   export PIANOMOTION10M_REPO=/home/you/PianoMotion10M
   (or edit the default below).
3. Make sure the unzipped dataset (annotation/, audio/, midi/, train.txt,
   test.txt, valid.txt) lives under one folder, and pass that as --data_root.
"""

import argparse
import os
import random
import sys

import torch
import torch.optim as optim
from tqdm import tqdm

PIANOMOTION10M_REPO = os.environ.get("PIANOMOTION10M_REPO", "./PianoMotion10M")
sys.path.insert(0, PIANOMOTION10M_REPO)

try:
    from datasets.PianoPose import PianoPose  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Could not import datasets.PianoPose from '{PIANOMOTION10M_REPO}'. "
        "Set PIANOMOTION10M_REPO to your clone of "
        "https://github.com/agnJason/PianoMotion10M (see GUIDE.md Step 1)."
    ) from e

from model import AudioEncoder, FlowMatchingTransformer, BaselineRegressor  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["flow", "baseline"], default="flow")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_examples_per_epoch", type=int, default=None,
                         help="Cap examples per epoch for a quick smoke test. "
                              "Try e.g. 20 the FIRST time you run this -- see GUIDE.md Step 5.")
    parser.add_argument("--save_path", type=str, default="checkpoint.pt")
    # The following args exist because PianoPose's own constructor expects
    # them -- names and defaults copied directly from the repo's draw.py.
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_sec", type=int, default=-1)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--is_random", action="store_true")
    parser.add_argument("--adjust", action="store_true")
    parser.add_argument("--up_list", nargs="+", default=[])
    parser.add_argument("--return_beta", action="store_true")
    return parser.parse_args()


def get_item(dataset, idx):
    """
    Wraps dataset.__getitem__ the same way the repo's own draw.py calls it:
    dataset.__getitem__(idx, True) -> (batch_dict, para_dict)
    ASSUMPTION (unverified beyond draw.py's usage): this signature and
    return shape is stable across phase='train'/'valid'/'test'. If this
    throws, open datasets/PianoPose.py in the cloned repo and adjust this
    function -- see GUIDE.md troubleshooting, item #1.
    """
    batch, para = dataset.__getitem__(idx, True)
    return batch, para


def build_pose(batch, device):
    """
    Concatenate right + left hand pose into one [1, T, D] tensor.
    ASSUMPTION: batch['right'] / batch['left'] are [T, D_hand] arrays.
    NOTE: the repo's own draw.py slices right_pose[..., 1:52] before
    rendering, which hints column 0 may not be part of the pose itself
    (a timestamp or validity flag, most likely). This function currently
    uses the FULL width. If your loss refuses to go down (see GUIDE.md
    Step 6), try slicing off column 0 here as the first thing to test.
    """
    right = torch.as_tensor(batch["right"]).float().to(device)
    left = torch.as_tensor(batch["left"]).float().to(device)
    pose = torch.cat([right, left], dim=-1)  # [T, D_right + D_left]
    return pose.unsqueeze(0)  # [1, T, D]


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    args.mode = "train"
    train_set = PianoPose(args=args, phase="train")
    print(f"[info] train_set has {len(train_set)} examples")

    # --- peek one example to discover pose_dim / num_frames at RUNTIME ----
    sample_batch, sample_para = get_item(train_set, 0)
    print(f"[info] sample batch keys: {list(sample_batch.keys())}")
    print(f"[info] sample para: {sample_para}")
    right_shape = torch.as_tensor(sample_batch["right"]).shape
    left_shape = torch.as_tensor(sample_batch["left"]).shape
    print(f"[info] right shape: {tuple(right_shape)}, left shape: {tuple(left_shape)}")
    pose_dim = right_shape[-1] + left_shape[-1]
    num_frames = right_shape[0]
    print(f"[info] inferred pose_dim={pose_dim}, num_frames={num_frames}")
    print("[info] STOP AND CHECK: do these numbers look sane? See GUIDE.md Step 3.")

    audio_encoder = AudioEncoder(hidden_dim=args.hidden_dim).to(device)

    if args.model_type == "flow":
        backbone = FlowMatchingTransformer(
            pose_dim=pose_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_heads=args.num_heads,
            max_frames=num_frames,
        ).to(device)
    else:
        backbone = BaselineRegressor(
            pose_dim=pose_dim, hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, num_heads=args.num_heads,
            max_frames=num_frames,
        ).to(device)

    optimizer = optim.Adam(
        list(audio_encoder.parameters()) + list(backbone.parameters()), lr=args.lr
    )

    n_examples = len(train_set)
    indices = list(range(n_examples))

    for epoch in range(args.epochs):
        random.shuffle(indices)
        epoch_indices = indices[: args.max_examples_per_epoch] if args.max_examples_per_epoch else indices

        audio_encoder.train()
        backbone.train()
        pbar = tqdm(epoch_indices, desc=f"epoch {epoch}")
        running_loss = []
        for idx in pbar:
            batch, _ = get_item(train_set, idx)
            x1 = build_pose(batch, device)  # [1, T, D]
            audio = torch.as_tensor(batch["audio"]).float().unsqueeze(0).to(device)  # [1, T_audio]

            cond = audio_encoder(audio, num_frames=x1.shape[1])  # [1, T, hidden]

            if args.model_type == "flow":
                B, T, D = x1.shape
                t = torch.rand(B, 1, device=device)
                x0 = torch.randn_like(x1)
                t_broadcast = t.view(B, 1, 1)
                xt = (1 - t_broadcast) * x0 + t_broadcast * x1
                v_target = x1 - x0
                v_pred = backbone(xt, t, cond)
                loss = ((v_pred - v_target) ** 2).mean()
            else:
                pred = backbone(cond)
                loss = ((pred - x1) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = sum(running_loss) / max(len(running_loss), 1)
        print(f"[info] epoch {epoch} avg loss: {avg_loss:.4f}")

        torch.save(
            {
                "audio_encoder": audio_encoder.state_dict(),
                "backbone": backbone.state_dict(),
                "pose_dim": pose_dim,
                "num_frames": num_frames,
                "model_type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
            },
            args.save_path,
        )
        print(f"[info] saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
