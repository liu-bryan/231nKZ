# YOLOv8 and RT-DETR Fine-tuning -> IDM / VPT Policy Gameplay Pipeline

Fine-tune a pretrained YOLOv8 and RT-DETR detection model on your custom Katana Zero dataset, with configurable augmentation, validation, and export. Then, use an IDM / VPT Policy to behavior-clone on real and pseudo-labelled gameplay to learn how to roughly play the game.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch is strongly recommended; install the matching build from <https://pytorch.org/get-started/locally/> first if you need it.

## 2. Expected layout

Ultralytics expects this on disk (you said you already have it):

```
dataset/
  images/
    train/  *.jpg
    val/    *.jpg
  labels/
    train/  *.txt   # YOLO format: "class cx cy w h" (normalized)
    val/    *.txt
data.yaml
```

`data.yaml` should look like:

```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: Player
  1: Crosshair
... etc.
```

## 3. Train

Defaults live in `configs/train_config.yaml`. CLI flags override the YAML.

```bash
python -m src.train --config configs/train_config.yaml --data data.yaml
```

Common overrides:

```bash
python -m src.train --data data.yaml --model yolov8m.pt --epochs 200 --batch 32 --imgsz 768 --device 0
```

Resume an interrupted run:

```bash
python -m src.train --name finetune --resume
```

After training, the script automatically runs a final validation pass on `best.pt`. Use `--no-val` to skip it, or `--export onnx` to also export the model.

## 4. Augmentation

Two layers, both already wired up:

1. **Online (default, recommended).** YOLO applies mosaic, mixup, HSV jitter, affine, flips, and random erasing every batch. All knobs are exposed in `configs/train_config.yaml` — tune `mosaic`, `mixup`, `degrees`, `scale`, `hsv_*`, `fliplr`, etc. `close_mosaic: 10` turns mosaic off for the final 10 epochs, which usually nudges mAP up.
2. **Offline (optional).** If your dataset is small, expand it on disk with Albumentations before training:

   ```bash
   python -m src.augment \
     --images dataset/images/train --labels dataset/labels/train \
     --out-images dataset/images/train_aug --out-labels dataset/labels/train_aug \
     --n 3 --imgsz 640
   ```

   Then point your `data.yaml` `train:` at a directory (or list of dirs) that includes both `train` and `train_aug`.

## 5. Predict

```bash
python -m src.predict \
  --weights runs/detect/finetune/weights/best.pt \
  --source path/to/images_or_video \
  --save
```

## 6. RT-DETR (optional transformer baseline)

Train RT-DETR on the **same YOLO-format dataset** for a fair architecture comparison:

```bash
python -m src.train_rtdetr --data data_grouped.yaml --device mps
```

Defaults: `configs/train_rtdetr_config.yaml` → weights in `runs/rtdetr/finetune/weights/best.pt`. Use `batch: 4` if you hit OOM on M3.

## 7. Compare YOLO vs RT-DETR

After training both on the **same** `data_*.yaml`:

```bash
python -m src.compare_models \
  --data data_grouped.yaml \
  --yolo runs/detect/finetune/weights/best.pt \
  --rtdetr runs/rtdetr/finetune/weights/best.pt \
  --device mps \
  --benchmark
```

Prints mAP50, mAP50-95, precision, recall side-by-side, plus optional mean inference ms/image on the val folder. Prefer `data_grouped.yaml` so neither model sees near-duplicate val frames during training.

> **Phase 2:** YOLO and RT-DETR are **two separate pipelines** (separate trajectories, IDM, policy, play). See [Two independent pipelines](#two-independent-pipelines).

## 8. Tips for fine-tuning

- Start from `yolov8s.pt` or `yolov8m.pt`; only go larger if val mAP is still climbing at the end of training.
- `lr0=0.001` with AdamW is a safer default than the from-scratch `0.01` SGD recipe.
- If your classes are very different from COCO, do not freeze the backbone — let it adapt.
- Watch `runs/detect/finetune/results.png` and the val loss/mAP curves; early stopping is on via `patience: 25`.
- If you have few samples per class, lower `mosaic` to ~0.5 and `mixup` to 0.0 — heavy mixing can hurt small datasets.

## 9. Visualize

Interactive viewer for debugging labels, predictions, and trajectories.

```bash
# Ground truth on val images
python -m src.visualize labels --data data_grouped.yaml --split val --show

# Model predictions (YOLO or RT-DETR weights)
python -m src.visualize predict --weights runs/detect/finetune/weights/best.pt \
  --source dataset_grouped/images/val --show

# Side-by-side GT | predictions
python -m src.visualize compare --weights runs/detect/finetune/weights/best.pt \
  --data data_grouped.yaml --split val --show

# Trajectory overlay on video (objects + aim arrow + optional button labels)
python -m src.visualize trajectory --trajectory path/to/session.npz \
  --video recordings/session.mp4 --show --draw-velocity
```

**Interactive keys:** `n` / Space = next, `p` = previous, `s` = save current frame, `q` / Esc = quit.

Without `--show`, all frames are written to `runs/visualize/` automatically. Player and Crosshair are highlighted in green/orange; trajectory mode draws the aim vector between them.

---

# Phase 2: VPT-style policy on object detections

Train a policy on **object lists** (not pixels): detector → tracker → transformer + LSTM → buttons + aim. **YOLO and RT-DETR are two fully separate pipelines** — each has its own trajectories, IDM, policy checkpoint, and `play` invocation. Object lists from the two detectors are **never combined** into one observation.

> **Phase 1 prerequisite for aim:** the **in-game crosshair must be a labeled detection class.** Set `player_class_id` / `cursor_class_id` in the pipeline config to match `data.yaml`.

## Two independent pipelines

```
Pipeline A (YOLO)                         Pipeline B (RT-DETR)
─────────────────                         ─────────────────────
video → YOLO(best.pt)                     video → RT-DETR(best.pt)
     → tracker → object list A                 → tracker → object list B
     → IDM (yolo) → pseudo-label               → IDM (rtdetr) → pseudo-label
     → policy (yolo) → play                    → policy (rtdetr) → play

Config:  configs/vpt_config_yolo.yaml       configs/vpt_config_rtdetr.yaml
Data:    data/trajectories/yolo/            data/trajectories/rtdetr/
Ckpts:   checkpoints/yolo/idm|policy/       checkpoints/rtdetr/idm|policy/
```

Shared policy block (same architecture in both; **separate weights** trained on that pipeline’s trajectories only):

```
object list [(type, x,y,w,h,vx,vy), ...]   # from ONE detector only
   → ObjectEncoder (transformer) → LSTM
   → ButtonHead + AimHead
```

Use `--pipeline yolo` or `--pipeline rtdetr` on training/play CLIs to pick the config. `src/vpt/compare_policy` runs both pipelines on the same video for **evaluation only** (two independent inferences per frame, not a merged list).

Three training stages, mirroring the VPT paper:

1. **IDM** — train an inverse-dynamics model that predicts the action between two consecutive object lists, using a small amount of human-labeled gameplay.
2. **Pseudo-label** — run the IDM over YouTube/Twitch VODs to auto-generate action labels.
3. **BC** — behaviorally clone the policy on (labeled + pseudo-labeled) data. Optional RL fine-tuning if you have a sim hook.

### Policy model architecture (`src/vpt/model.py`)

| Component | Policy (`VPTPolicy`) | IDM (pseudo-label only) |
|-----------|----------------------|-------------------------|
| Input | Up to 32 objects × `(type, x,y,w,h,vx,vy)` per frame | Pairs of frames concatenated |
| Encoder | 2-layer transformer, `d_model=128`, 4 heads | 3-layer transformer + frame-id embed (0/1) |
| Temporal | 1-layer LSTM, `hidden=256` | — (single-step between frames) |
| Heads | 7× BCE buttons + 16-way aim CE | MLP → button logits only |

Configurable in `configs/vpt_config.yaml` (`model.*`, `action_space.*`, `observation.*`). No external pretrained policy weights — train from scratch on your trajectories (YOLO or RT-DETR object lists).

## File layout

```
configs/vpt_config.yaml          # action space, model dims, training hparams
src/katana_logger.py             # records keys + click timing during play
src/vpt/
    model.py                     # ObjectEncoder, VPTPolicy (button + aim heads), IDM, losses
    aim.py                       # compute_aim: player->cursor direction -> bin
    tracking.py                  # CentroidTracker (assigns identities + velocities)
    data.py                      # TrajectoryDataset, FramePairDataset
    pipeline.py                  # pipeline name -> config paths
    detector.py                  # load ONE detector (YOLO or RT-DETR) per call
    extract_objects.py           # video -> trajectory .npz for one detector
    compare_policy.py            # eval: two pipelines side-by-side (no merge)
    log_to_actions.py            # katana_logger JSONL + trajectory -> labeled .npz (CLI)
    clean_log.py                 # strip mouse noise from OLD logs -> current schema (CLI)
    attach_actions.py            # generic CSV + trajectory -> labeled .npz (CLI)
    train_idm.py                 # train the inverse dynamics model (CLI)
    pseudo_label.py              # IDM over unlabeled trajectories (CLI)
    train_bc.py                  # behavioral cloning of the policy (CLI)
    runner.py                    # detector + policy inference (YOLO or RT-DETR)
src/play.py                      # live window capture + runner + pynput
```

## Expected data layout

```
data/trajectories/
  yolo/          unlabeled/  labeled/  pseudo_labeled/   # Pipeline A only
  rtdetr/        unlabeled/  labeled/  pseudo_labeled/   # Pipeline B only
checkpoints/
  yolo/          idm/best.pt  policy/best.pt
  rtdetr/        idm/best.pt  policy/best.pt
configs/
  vpt_config_yolo.yaml
  vpt_config_rtdetr.yaml
  vpt_config.yaml          # legacy flat layout (avoid for new work)
runs/
  detect/finetune/weights/best.pt
  rtdetr/finetune/weights/best.pt
```

### Pipeline A — YOLO (end-to-end)

```bash
# 1) Objects from YOLO only
python -m src.vpt.extract_objects --pipeline yolo \
  --weights runs/detect/finetune/weights/best.pt \
  --source recordings/session.mp4 \
  --out data/trajectories/yolo/unlabeled/session.npz

# 2) Label (same log/video for both pipelines if you run B too)
python -m src.vpt.log_to_actions \
  --trajectory data/trajectories/yolo/unlabeled/session.npz \
  --log logs/run_session.jsonl \
  --out data/trajectories/yolo/labeled/session.npz

# 3) Train
python -m src.vpt.train_idm --pipeline yolo
python -m src.vpt.pseudo_label --pipeline yolo
python -m src.vpt.train_bc --pipeline yolo

# 4) Play
python -m src.play --pipeline yolo --dry-run
python -m src.play --pipeline yolo
```

### Pipeline B — RT-DETR (end-to-end)

```bash
python -m src.vpt.extract_objects --pipeline rtdetr \
  --weights runs/rtdetr/finetune/weights/best.pt \
  --detector rtdetr \
  --source recordings/session.mp4 \
  --out data/trajectories/rtdetr/unlabeled/session.npz

python -m src.vpt.log_to_actions \
  --trajectory data/trajectories/rtdetr/unlabeled/session.npz \
  --log logs/run_session.jsonl \
  --out data/trajectories/rtdetr/labeled/session.npz

python -m src.vpt.train_idm --pipeline rtdetr
python -m src.vpt.pseudo_label --pipeline rtdetr
python -m src.vpt.train_bc --pipeline rtdetr

python -m src.play --pipeline rtdetr --dry-run
python -m src.play --pipeline rtdetr
```

### Katana ZERO action mapping

The pipeline expects this 7-button multi-binary vector (configured in `configs/vpt_config.yaml`):

| Slot | Button name | Physical input |
|------|-------------|----------------|
| 0 | `jump`      | `W` or `Space` |
| 1 | `left`      | `A` |
| 2 | `down`      | `S` (game treats as dodge/roll on ground) |
| 3 | `right`     | `D` |
| 4 | `interact`  | `F` |
| 5 | `slow_mo`   | `Shift` |
| 6 | `attack`    | Left mouse button |

Plus a **parallel aim head**: a softmax over `action_space.aim_bins` (default 16) directions giving the slash direction. Aim is *not* taken from the mouse — `katana_logger.py` records only click **timing**. Instead aim is read from the video: the in-game **crosshair is a labeled detection class**, so aim is computed as the direction from the player box to the cursor box (`src/vpt/aim.py`), quantized into 16 sectors. This is why the logger needs no mouse coordinates and is immune to the game running in a movable window.

Velocity (`vx`, `vy`) is derived per-track inside `extract_objects.py`. No game-state channel (HP, slow_mo gauge) is logged, so `observation.global_dim` stays at 0; bump it later if you ever wire up a memory reader or pixel-tap for those.

> **Required class ids.** `observation.player_class_id` and `observation.cursor_class_id` in `configs/vpt_config.yaml` MUST match the class indices in your `data.yaml` (same for YOLO and RT-DETR — both use identical labels). Aim labels are silently empty if they're wrong — `log_to_actions.py` prints a warning when no frame has both the player and the cursor detected.

## Comparing the two pipelines

| What | Command |
|------|---------|
| Detection mAP only | `python -m src.compare_models` (Phase 1 §7) |
| Full pipeline actions (eval) | `python -m src.vpt.compare_policy --source recording.mp4` |
| Live agent | `python -m src.play --pipeline yolo` **or** `--pipeline rtdetr` |

`compare_policy` runs **two separate** `Runner`s (YOLO+YOLO policy, RT-DETR+RT-DETR policy). It does not merge detections. Optional `--shared-policy` is an ablation only.

`src/vpt/detector.py` loads **one** detector per call; `extract_objects` / `Runner` / `play` each produce a single object list per frame.

## 1. Record gameplay

`katana_logger.py` and OBS are both bound to F9. Hit F9 once to start (creates `logs/run_<ts>.jsonl` and the matching OBS recording), play, hit F9 again to stop. Each session yields a paired `(recording.mp4, run.jsonl)`. The log contains keydown/keyup events and `mousedown`/`mouseup` (click timing only — no coordinates or movement).

> **Old logs (pre-cleanup).** If you have logs from the previous logger that recorded the dense `mousemove` stream and click coordinates, normalize them to the current schema first:
>
> ```bash
> python -m src.vpt.clean_log --in-dir logs --out-dir logs_clean   # or --in-place (backs up to *.jsonl.bak)
> ```
>
> This drops every `mousemove` and strips coordinates from clicks, keeping keys, click timing, and session markers. Aim is unaffected because it comes from the video, not the log.

## 2. Extract object trajectories from each video

Run **once per pipeline** — write under `data/trajectories/yolo/` or `data/trajectories/rtdetr/` only. Same `.npz` schema; `detector` metadata tag records which pipeline created the file.

See [Pipeline A](#pipeline-a--yolo-end-to-end) / [Pipeline B](#pipeline-b--rt-detr-end-to-end) for full commands. Tracker: `src/vpt/tracking.py` (centroid); consider Ultralytics `model.track()` for production.

## 3. Attach inputs from katana_logger

Align the JSONL log to **that pipeline’s** trajectory `.npz` (same video/log can label both trees if you extracted both):

```bash
python -m src.vpt.log_to_actions \
  --trajectory data/trajectories/yolo/unlabeled/session.npz \
  --log logs/run_session.jsonl \
  --out data/trajectories/yolo/labeled/session.npz
```

Repeat with `data/trajectories/rtdetr/...` for Pipeline B.

The script anchors frame 0 to the JSONL's `session_start` event (same F9 press that started OBS), then samples held-button state at each video frame using FPS embedded in the NPZ. If you notice systematic lag between the model's reactions and the video during sanity checks, pass `--offset-ms 50` (or `-50`) to shift the input timeline.

Move the unlabeled `.npz` out of `unlabeled/` once you've labeled it (or run `log_to_actions.py` and delete the original — labeled NPZs are a superset).

`attach_actions.py` remains as a generic CSV → labeled NPZ tool if you ever want to label something other than Katana ZERO gameplay.

## 4. Train the IDM

```bash
python -m src.vpt.train_idm --pipeline yolo
# or
python -m src.vpt.train_idm --pipeline rtdetr
```

The IDM is 3 transformer layers over both frames' objects (with a frame-id embedding so it knows "before" vs "after"), then an MLP head. Watch the validation accuracy — it should reach high exact-match accuracy on the held-out pairs before you trust it for pseudo-labeling. If it caps out at low accuracy, your action labels are probably misaligned with the frames or the action depends on info not present in object lists (e.g. menu state).

The IDM predicts **buttons only**. Aim is directly observable (the cursor is a detected object), so it does not need to be inferred — it's computed deterministically from the object list for both labeled and pseudo-labeled trajectories.

## 5. Pseudo-label unlabeled VODs

```bash
python -m src.vpt.pseudo_label --pipeline yolo
python -m src.vpt.pseudo_label --pipeline rtdetr
```

## 6. Train the policy via BC

```bash
python -m src.vpt.train_bc --pipeline yolo
python -m src.vpt.train_bc --pipeline rtdetr
```

Pseudo-labeled samples are down-weighted by `bc.pseudo_label_weight` (default 0.5) — bump it up if your IDM is very accurate, down if it's noisy. The policy trains two heads jointly: buttons (BCE) and aim (16-way CE, masked to frames where both player and cursor are detected, weighted by `bc.aim_loss_weight`). Each epoch prints `acc` (per-button) and `aim_acc` (direction-bin accuracy on supervised frames).

## 7. Run the policy

### Video replay (`src/vpt/runner`)

One pipeline per invocation — one detector, one object list, one policy:

```bash
python -m src.vpt.runner --pipeline yolo --source recordings/test.mp4
python -m src.vpt.runner --pipeline rtdetr --source recordings/test.mp4
```

| Flag | Purpose |
|------|---------|
| `--pipeline` | `yolo` or `rtdetr` (sets config, default weights, policy path) |
| `--weights` / `--yolo` | Override detector `.pt` |
| `--policy` | Override policy checkpoint |
| `--source` | Video file |

### Compare pipelines (`src/vpt/compare_policy`)

Evaluation only: same frame → **Pipeline A** and **Pipeline B** independently (no merged object list). Default: `checkpoints/yolo/policy/best.pt` vs `checkpoints/rtdetr/policy/best.pt`.

```bash
python -m src.vpt.compare_policy --source recordings/test.mp4 --device mps
```

Reports button agreement, aim agreement, mean object-count delta, and ms/frame per pipeline.

### Live play (`src/play.py`)

```bash
python -m src.play --pipeline yolo --dry-run
python -m src.play --pipeline yolo

python -m src.play --pipeline rtdetr --dry-run
python -m src.play --pipeline rtdetr
```

| Flag | Purpose |
|------|---------|
| `--pipeline` | **`yolo` or `rtdetr`** — picks config + default detector + policy |
| `--weights` / `--yolo` | Override detector weights |
| `--policy` | Override policy path |
| `--detect-only` | Detector only (one object list, no policy) |
| `--dry-run` | Print actions; no `pynput` injection |
| `--window-title` / `--rect` | Window capture: auto-detect by title on **macOS**; on **Windows/Linux** use `--rect L,T,W,H` |
| `--kill-key` | Emergency stop (default `f10`) |

**F10** = stop + release keys. **Ctrl+C** = clean exit.

## Practical notes (from the design transcript)

- **Object detection quality is the bottleneck.** If detections are jittery or drop objects, both IDM and policy will be noisy. Re-train YOLO/RT-DETR on more data before you blame the policy. Run `compare_policy` to see whether policy disagreement is mostly from missed objects vs. box jitter.
- **Two pipelines, never merge lists.** Train and play with `--pipeline yolo` or `--pipeline rtdetr` only. `compare_policy` is for side-by-side eval, not production input.
- **RT-DETR on Apple Silicon (MPS)** is slower (CPU fallback for some ops). On **Windows/Linux with CUDA**, RT-DETR is usually faster than on M3 but still heavier than YOLO at inference. Prefer YOLO for live play unless RT-DETR wins on `compare_policy`.
- **Velocity matters more than position.** A still-frame snapshot can't tell you an enemy is winding up an attack; velocity often can. Both are included in `feats=(x,y,w,h,vx,vy)`.
- **Add global features for hidden state.** Things like "slow-mo gauge", current HP, or menu-open flag aren't visible to YOLO. Set `observation.global_dim > 0` and pass them via `--globals` in `attach_actions.py`.
- **multi_binary vs discrete.** Default is `multi_binary` (independent Bernoulli per button, BCE loss) because in real games you hold combos. Switch to `discrete` in the config for a single-softmax head if your action space is genuinely mutually exclusive.
- **Aim depends on the crosshair detection.** If YOLO misses the cursor on a frame, that frame contributes no aim supervision (training) and falls back to the previous mouse position (inference). If `aim_acc` stays low, check that the crosshair is detected reliably during combat and that 16 bins is fine-grained enough (raise `action_space.aim_bins` for finer control). Because aim comes from the video, it's also pseudo-labeled on VODs for free — no IDM needed.
- **Optional RL fine-tuning.** Katana ZERO is a Unity game, so you can mod it to expose rewards and run PPO against the BC-pretrained checkpoint — exactly the VPT recipe. Reward shaping suggestions: `+kill_enemy, +complete_room, -take_damage, +slow_mo_during_attack`.

