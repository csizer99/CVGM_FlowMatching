"""
Reference-free motion-quality comparison: velocity/acceleration/jerk
statistics for comparing your flow-matching model's output against your
baseline regressor's output on the SAME out-of-training audio (e.g. Fur
Elise). This is reference-free -- no ground truth pose needed -- which is
exactly the situation for a piece with no motion-capture ground truth. See
GUIDE.md for why this is different from, and not a replacement for, real
held-out-test-set metrics.

For actual FGD/WGD-style metrics against held-out PianoMotion10M ground
truth, reuse the PianoMotion10M repo's OWN eval.py rather than reimplementing
that here -- it's already built against the exact data format.

Usage:
    python eval.py --flow_pose generated_pose_flow.pt --baseline_pose generated_pose_baseline.pt
"""

import argparse

import torch


def motion_stats(pose_seq, fps):
    """
    pose_seq: [T, D] tensor, a single generated sequence with the batch
    dimension already removed.
    fps: frames per second of the sequence. CHECK this against the actual
    annotation rate PianoMotion10M was captured at (see GUIDE.md) -- an
    incorrect fps silently rescales every number here and makes the
    comparison meaningless, without producing any error.
    """
    dt = 1.0 / fps
    velocity = (pose_seq[1:] - pose_seq[:-1]) / dt
    acceleration = (velocity[1:] - velocity[:-1]) / dt
    jerk = (acceleration[1:] - acceleration[:-1]) / dt

    return {
        "velocity_mean_abs": velocity.abs().mean().item(),
        "velocity_std": velocity.std().item(),
        "acceleration_mean_abs": acceleration.abs().mean().item(),
        "jerk_mean_abs": jerk.abs().mean().item(),
    }


def compare(flow_pose_seq, baseline_pose_seq, fps):
    flow_stats = motion_stats(flow_pose_seq, fps=fps)
    baseline_stats = motion_stats(baseline_pose_seq, fps=fps)

    print(f"{'metric':22s} {'flow-matching':>16s} {'baseline (MSE)':>18s}")
    for key in flow_stats:
        print(f"{key:22s} {flow_stats[key]:16.4f} {baseline_stats[key]:18.4f}")

    print()
    print("What to look for (see GUIDE.md for the full reasoning):")
    print("- baseline velocity_std much LOWER than flow-matching's -> sign of")
    print("  mode-averaging / blurred-together motion, the exact failure this")
    print("  whole project exists to avoid.")
    print("- Either model's jerk_mean_abs wildly higher than the other's may")
    print("  indicate physically implausible, jittery output -- worth watching")
    print("  the rendered video, not just trusting this number alone.")

    return flow_stats, baseline_stats


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_pose", type=str, required=True)
    parser.add_argument("--baseline_pose", type=str, required=True)
    parser.add_argument("--fps", type=float, required=True,
                         help="See GUIDE.md for how to find PianoMotion10M's real "
                              "annotation frame rate -- do not guess this.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    flow_pose = torch.load(args.flow_pose).squeeze(0)
    baseline_pose = torch.load(args.baseline_pose).squeeze(0)
    compare(flow_pose, baseline_pose, fps=args.fps)
