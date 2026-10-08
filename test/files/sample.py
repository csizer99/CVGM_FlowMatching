"""
Generate a hand-pose sequence for a NEW audio clip (e.g. a Fur Elise
recording) using a trained flow-matching checkpoint.

IMPORTANT PIVOT FROM OUR ORIGINAL PLAN: conditioning here is on AUDIO, not
MIDI. Reason: PianoPose's loader hands you (audio, pose) pairs that are
ALREADY time-aligned for you -- there is no ready-made MIDI-to-frame
alignment step in the PianoMotion10M repo, and building one from raw .mid
files is real extra engineering beyond this v1 (see GUIDE.md). So: to
generate a trajectory "for Fur Elise", you need an AUDIO recording of it
(an mp3/wav of someone actually playing it), not a MIDI file.

Usage:
    python sample.py --checkpoint checkpoint.pt --audio_path fur_elise.wav
"""

import argparse

import torch
import torchaudio

from model import AudioEncoder, FlowMatchingTransformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="generated_pose.pt")
    parser.add_argument("--num_frames", type=int, default=None,
                         help="Defaults to the training window length stored in the "
                              "checkpoint. Generating a longer/shorter sequence than "
                              "training used is untested -- see GUIDE.md.")
    parser.add_argument("--num_steps", type=int, default=30,
                         help="Number of Euler integration steps. Try 30 first; if "
                              "output looks off, try 100 before suspecting the model.")
    parser.add_argument("--expected_sample_rate", type=int, default=None,
                         help="If PianoMotion10M's audio files use a specific sample "
                              "rate (check this -- see GUIDE.md), set it here so your "
                              "input audio gets resampled to match.")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if ckpt.get("model_type", "flow") != "flow":
        raise ValueError(
            "This checkpoint was trained with --model_type baseline, which has no "
            "flow-matching sampling loop (it predicts pose directly, one forward pass, "
            "no t/noise). Use it directly: BaselineRegressor(cond) -> pose. See GUIDE.md."
        )

    pose_dim = ckpt["pose_dim"]
    num_frames = args.num_frames or ckpt["num_frames"]
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 4)
    num_heads = ckpt.get("num_heads", 4)

    audio_encoder = AudioEncoder(hidden_dim=hidden_dim).to(device)
    audio_encoder.load_state_dict(ckpt["audio_encoder"])
    audio_encoder.eval()

    backbone = FlowMatchingTransformer(
        pose_dim=pose_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, num_heads=num_heads, max_frames=num_frames,
    ).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()

    waveform, sr = torchaudio.load(args.audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # collapse to mono

    if args.expected_sample_rate and sr != args.expected_sample_rate:
        print(f"[info] resampling audio from {sr} Hz to {args.expected_sample_rate} Hz "
              "to match training data")
        resampler = torchaudio.transforms.Resample(sr, args.expected_sample_rate)
        waveform = resampler(waveform)

    audio = waveform.to(device)  # [1, T_audio]

    with torch.no_grad():
        cond = audio_encoder(audio, num_frames=num_frames)  # [1, num_frames, hidden]

        x = torch.randn(1, num_frames, pose_dim, device=device)
        dt = 1.0 / args.num_steps
        for i in range(args.num_steps):
            t = torch.full((1, 1), i * dt, device=device)
            v = backbone(x, t, cond)
            x = x + v * dt

    torch.save(x.cpu(), args.out_path)
    print(f"[info] saved generated pose sequence with shape {tuple(x.shape)} to {args.out_path}")
    print("[info] NEXT: visualize this. The PianoMotion10M repo's own datasets/show.py "
          "(used by draw.py) already knows how to render a [T, D] pose sequence -- "
          "reuse that rather than writing a new renderer. See GUIDE.md Step 8.")


if __name__ == "__main__":
    main()
