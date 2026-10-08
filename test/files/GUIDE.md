# Guide: running the PianoMotion10M flow-matching starter

This is written the way I'd want it written if I were about to run this
myself for the first time, with no chance to ask follow-up questions. Every
assumption I'm not fully certain of is called out explicitly, with what to do
if it turns out wrong. Read this whole thing once before running anything.

## 0. What this package is, and isn't

It's a working *skeleton*: a real flow-matching model, a real training loop,
a real sampling loop, wired up against the *real* PianoMotion10M dataset
loader (not a reimplementation of it). It is not a tuned, SOTA system — it's
the smallest thing that should actually run end to end and demonstrate the
mechanism on real data, matching everything we built up to in the toy
notebook, just with a real backbone and real conditioning instead of a
3-layer MLP and a scalar 0/1.

It will almost certainly need at least one or two small fixes on your end,
because I built this against the dataset's documented folder layout and its
own `draw.py` usage pattern, not against the actual `datasets/PianoPose.py`
source code, which I did not fetch. Section 3 tells you exactly where to
look first if something breaks, and why that's the most likely spot.

## 1. The one big pivot from our earlier plan — read this before anything else

Our original proposal conditioned generation on **MIDI**. This code
conditions on **audio** instead. Here's why, plainly: PianoMotion10M's own
dataset loader (`PianoPose`) hands you `audio` and hand-pose already time-
aligned, frame for frame, ready to use. There is no equivalent ready-made
alignment between raw `.mid` files and pose frames anywhere in the repo —
building that yourself (deciding exact onset timing, quantization, how to
turn a MIDI event stream into a per-frame feature vector, all synced to the
video's frame rate) is a real, separate piece of engineering, not something
that falls out of the dataset for free. Since the whole point right now is
getting a working end-to-end pipeline, I used what the loader already gives
you for free.

**Consequence you need to know about now:** generating a trajectory "for Fur
Elise" means feeding in an *audio recording* of Fur Elise (an mp3 or wav of
someone playing it), not the MIDI file. If you specifically want MIDI-driven
generation later, that's a real follow-up project — extending `PianoPose`
or writing your own MIDI-to-frame aligner — not a small tweak to this code.

## 2. Setup, in order

1. **Clone the PianoMotion10M repo** somewhere on disk (separately from this
   folder): `git clone https://github.com/agnJason/PianoMotion10M`.
   This code imports `datasets.PianoPose.PianoPose` directly from that repo
   — it does not duplicate that logic.
2. **Set an environment variable** pointing at that clone:
   `export PIANOMOTION10M_REPO=/path/to/your/PianoMotion10M/clone`
   (or edit the default at the top of `train.py`.)
3. **Confirm your downloaded dataset** is unzipped into one folder containing
   `annotation/`, `audio/`, `midi/`, `train.txt`, `test.txt`, `valid.txt` as
   siblings — this is the `--data_root` you'll pass in.
4. **Install dependencies**: `pip install -r requirements.txt`, plus whatever
   `PianoMotion10M`'s own `requirements.txt` lists (check that repo — its
   `PianoPose` class may need additional packages I don't have visibility
   into, e.g. for JSON/audio parsing).
5. **Read `datasets/PianoPose.py`** in the cloned repo yourself, at least
   skim it, before running anything. I built this against inferred behavior,
   not that source file — you looking at it once now will save you time
   later if step 3's assumption check (below) doesn't match.

## 3. Assumptions to verify — do this before trusting any training run

These are the specific places I'm inferring rather than certain, ranked by
how likely each is to need a fix:

**A. The `__getitem__` call signature.** I call
`dataset.__getitem__(idx, True)` and unpack it as `(batch_dict, para_dict)`,
because that's exactly how the repo's own `draw.py` calls it. I don't know
what the *default* (omitting the second argument) does, or whether this
signature is identical across `phase='train'` vs `'valid'` vs `'test'`. If
`train.py` errors immediately on this call, open `datasets/PianoPose.py` and
check `__getitem__`'s actual signature — this is the single most likely
first failure point.

**B. The keys and shapes inside `batch_dict`.** I assume `batch['audio']`,
`batch['right']`, `batch['left']` exist, with `right`/`left` shaped
`[T, D_hand]`. This is confirmed by `draw.py`'s usage. What I *don't* know:
the exact value of `D_hand`, and whether every dimension in it is meaningful
pose data. Notably, `draw.py` slices `right_pose[..., 1:52]` before
rendering — that strongly suggests index 0 of the raw array is *not* part
of the pose itself (maybe a timestamp, maybe a validity flag). `train.py`
currently uses the *full* width. **The first time you run training**, look
at the printed `[info] right shape: ...` line, and if things aren't
training well later, come back and try slicing off column 0 in
`build_pose()` in `train.py` as your first experiment.

**C. Frame rate (fps).** I never hardcode this anywhere in the model or
training code (correctly — training doesn't need it). But `eval.py` does
need it, and I made it a required argument rather than guessing, on
purpose. Find the real value by checking the repo's docs/paper (116 hours /
10M frames implies an average, not necessarily the exact fps) or by reading
it directly out of one of the annotation JSON files if it's stored there
(open one `_seq_0000.json` file by hand and look).

**D. Audio sample rate.** `sample.py` has an `--expected_sample_rate` flag
that does nothing unless you set it. Find PianoMotion10M's actual audio
sample rate (check the repo/paper, or just load one of the dataset's own
`.mp3` files with `torchaudio.load` and print `sr`) and pass it in when
generating from your own Fur Elise recording — a mismatched sample rate
won't error, it'll just quietly feed the model something different in scale
from what it was trained on.

If any of A–D turn out wrong, that's expected, not a sign this whole
approach is broken — it means you now know the dataset better than I could
infer from documentation, which is exactly the point of you doing this part
yourself.

## 4. Run order

Go through these in order. Do not skip the "check" after each one — that's
where problems get caught cheaply instead of two hours into a real run.

### Step 1 — smoke test the dataset loader alone, before touching my code

Before running anything from this folder, in a plain Python shell, with
`PIANOMOTION10M_REPO` on your path:

```
from datasets.PianoPose import PianoPose
# build args the same way draw.py does, phase='train'
```

Confirm you can construct it and call `__getitem__` at least once without
error, and look at what comes back. This isolates "is my data path right at
all" from "does my model code work" — two very different classes of bug,
and you want to know which one you're looking at.

### Step 2 — run training with a tiny cap, flow model

```
python train.py --data_root /path/to/dataset --model_type flow \
    --max_examples_per_epoch 20 --epochs 2
```

**What to check:** it should print `[info] sample batch keys: ...`,
`[info] right shape: ...`, `[info] inferred pose_dim=..., num_frames=...`
— read these and sanity check them against what you learned in Step 1
before letting it proceed further (Ctrl-C if something looks wrong; better
to stop now than debug after a full run).

### Step 3 — confirm loss is actually decreasing

Watch the tqdm loss numbers. It doesn't need to reach some target value —
just confirm it's trending down, not flat or `nan`. If it's `nan`
immediately: almost always exploding audio input scale (raw waveform can
have values very different in magnitude from what a from-scratch conv net
expects) — try normalizing `audio` to roughly unit variance before encoding,
as your first fix.

### Step 4 — train the baseline regressor the same way

```
python train.py --data_root /path/to/dataset --model_type baseline \
    --max_examples_per_epoch 20 --epochs 2 --save_path checkpoint_baseline.pt
```

Same checks as Step 3.

### Step 5 — once both smoke-test runs look sane, remove the cap

Drop `--max_examples_per_epoch` and increase `--epochs`, and let both models
actually train. How long this takes depends entirely on your hardware and
how big your `train.txt` split actually is — no way for me to estimate that
for you.

### Step 6 — generate from a held-out PianoMotion10M example first

Before jumping to Fur Elise, sanity-check generation on something from your
own dataset's `valid.txt` split — you have a real pose to eyeball your
output against there, which you don't have for Fur Elise. (`sample.py` as
written takes an audio *file path*, not a dataset index — for this step,
you'll need to either export one valid-split audio clip to a file first, or
adapt the loading in `sample.py` to pull directly from the `PianoPose`
valid split. Either is fine; pick whichever's less friction for you.)

### Step 7 — generate from your Fur Elise audio recording

```
python sample.py --checkpoint checkpoint.pt --audio_path fur_elise.wav \
    --expected_sample_rate <whatever you found in 3D>
```

Repeat with `checkpoint_baseline.pt` — but remember `sample.py`'s flow-
matching loop doesn't apply to the baseline checkpoint (it's a direct
forward pass, no stepping); the script will tell you this if you try.

### Step 8 — look at it, don't just trust numbers

Render both outputs. The PianoMotion10M repo's own `datasets/show.py`
(used inside `draw.py`) already renders a pose sequence to video — reuse
that rather than writing a new renderer from scratch; you already saw the
call pattern in `draw.py`: `render_result(path, audio, right_pose, left_pose)`.

### Step 9 — compare quantitatively

```
python eval.py --flow_pose generated_pose_flow.pt \
    --baseline_pose generated_pose_baseline.pt --fps <whatever you found in 3C>
```

Read the printed interpretation notes — specifically, check whether the
baseline's `velocity_std` is noticeably lower than the flow model's. That's
your concrete, numeric version of "did we actually avoid the mode-averaging
problem," the thing this whole project set out to test.

## 5. Things deliberately left out of this v1, and why

- **No pretrained audio encoder.** `AudioEncoder` is a from-scratch conv
  stack so v1 has no extra model downloads to debug. Once the pipeline
  runs, swapping in a pretrained wav2vec2/HuBERT encoder (what
  PianoMotion10M's own baseline uses) is a clean, contained upgrade —
  replace `AudioEncoder`'s internals, keep its input/output contract
  the same.
- **No decoupled left/right hand streams.** We discussed this — the
  published follow-up work found joint single-stream modeling of both
  hands captures their coordination less well than separate streams. This
  version concatenates them into one stream for simplicity. Revisit once
  the simple version is working.
- **No batching.** Every training step here processes one example at a
  time. This is slower than it needs to be, but sidesteps guessing at how
  `PianoPose` behaves inside a `DataLoader`'s default collation, which I
  have no visibility into. Once things work, that's a real, worthwhile
  optimization.
- **No classifier-free guidance.** Not needed for a v1 whose job is proving
  the pipeline runs and the flow-matching mechanism behaves as expected;
  add it once you have a working baseline to improve on.

## 6. If you get stuck

Given everything we talked about earlier in this conversation: try to
predict what should happen *before* running each step (what the loss curve
shape should roughly look like, whether output should look noisy or
structured early on) and check reality against that prediction — that's
still the fastest way to catch a real misunderstanding versus a trivial
bug. Come back with the specific error message or the specific thing that
looked wrong, not just "it didn't work" — the more precise the report, the
faster the actual problem surfaces.
