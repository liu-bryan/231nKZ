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

> **Phase 2:** YOLO and RT-DETR are **two detector paths** into one shared controller (one IDM, one policy). Object lists are **never merged**. See [Two detector paths, one policy](#two-detector-paths-one-shared-policy).

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

Train a policy on **object lists** (not pixels): detector → tracker → transformer + LSTM → buttons + aim.

**Two detectors, one controller.** YOLOv8 and RT-DETR are fine-tuned separately (Phase 1) and each produces its **own** per-frame object lists (`data/trajectories/yolo/` vs `data/trajectories/rtdetr/`). Lists are **never fused**. The **same** inverse-dynamics model (IDM) and **same** policy weights $\pi_\theta$ are used for both paths: at play time you only swap which detector feeds the policy.

> **Phase 1 prerequisite for aim:** the **in-game crosshair must be a labeled detection class.** Set `player_class_id` / `cursor_class_id` in either pipeline config to match `data.yaml`.

## Two detector paths, one shared policy

```
                    ┌── YOLO(best.pt)  → tracker → list A ──┐
video (same game) ──┤                                        ├──► shared IDM (train once)
                    └── RT-DETR(best.pt) → tracker → list B ┘         │
                                                                        ▼
                                                              shared policy π_θ (BC once)
                                                                        │
                                        ┌───────────────────────────────┴───────────────────────────────┐
                                        ▼                                                               ▼
                              play / runner --pipeline yolo                              play / runner --pipeline rtdetr
                              (list A + shared π_θ)                                    (list B + shared π_θ)
```

| What differs per path | What is shared |
|----------------------|----------------|
| Detector weights (`runs/detect/...`, `runs/rtdetr/...`) | IDM: `checkpoints/shared/idm/best.pt` |
| Trajectory folders (`data/trajectories/yolo/` vs `rtdetr/`) | Policy: `checkpoints/shared/policy/best.pt` |
| `configs/vpt_config_{yolo,rtdetr}.yaml` (detector + data dirs only) | `play.py` control loop, action heads, class ids |

Policy block (same architecture and **same checkpoint** for both paths):

```
object list [(type, x,y,w,h,vx,vy), ...]   # from ONE detector only — never merged
   → ObjectEncoder (transformer) → LSTM
   → ButtonHead + AimHead
```

Use `--pipeline yolo` or `--pipeline rtdetr` on `extract_objects`, `pseudo_label`, `play`, and `runner` to pick the detector and trajectory directories. Both configs point at the shared IDM/policy paths above.

**Training order** (mirrors VPT; IDM/BC run **once**, not per detector):

1. **IDM** — train on human-labeled trajectories (default: YOLO object lists in `data/trajectories/yolo/labeled/`). Predicts buttons between consecutive frames.
2. **Pseudo-label** — run that IDM on unlabeled VODs **per detector** (YOLO lists and RT-DETR lists in separate folders).
3. **BC** — behavior-clone the shared policy on labeled (+ optional pseudo-labeled) data. We used ground-truth keystroke labels on the YOLO trajectory tree; the same $\pi_\theta$ is then evaluated with either detector at inference.

### Policy model architecture (`src/vpt/model.py`)

| Component | Policy (`VPTPolicy`) | IDM (pseudo-label only) |
|-----------|----------------------|-------------------------|
| Input | Up to 32 objects × `(type, x,y,w,h,vx,vy)` per frame | Pairs of frames concatenated |
| Encoder | 2-layer transformer, `d_model=128`, 4 heads | 3-layer transformer + frame-id embed (0/1) |
| Temporal | 1-layer LSTM, `hidden=256` | — (single-step between frames) |
| Heads | 7× BCE buttons + 16-way aim CE | MLP → button logits only |

Configurable in `configs/vpt_config_yolo.yaml` / `vpt_config_rtdetr.yaml` (`model.*`, `action_space.*`, `observation.*` — identical between files). No external pretrained policy weights; train from scratch on your labeled gameplay.

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
  yolo/          unlabeled/  labeled/  pseudo_labeled/   # YOLO object lists only
  rtdetr/        unlabeled/  labeled/  pseudo_labeled/   # RT-DETR object lists only
checkpoints/
  shared/        idm/best.pt   policy/best.pt            # one IDM + one policy for both paths
configs/
  vpt_config_yolo.yaml       # detector A + yolo data dirs
  vpt_config_rtdetr.yaml     # detector B + rtdetr data dirs (same idm/bc ckpt paths)
runs/
  detect/finetune/weights/best.pt
  rtdetr/finetune/weights/best.pt
```

### Step 0 — Shared IDM + policy (run once)

After you have human-labeled YOLO trajectories (`data/trajectories/yolo/labeled/*.npz`):

```bash
python -m src.vpt.train_idm --pipeline yolo
python -m src.vpt.pseudo_label --pipeline yolo
python -m src.vpt.train_bc --pipeline yolo
```

Writes `checkpoints/shared/idm/best.pt` and `checkpoints/shared/policy/best.pt`. Both pipeline configs reference these paths.

### Path A — YOLO detections

```bash
# 1) Objects from YOLO only
python -m src.vpt.extract_objects --pipeline yolo \
  --weights runs/detect/finetune/weights/best.pt \
  --source recordings/session.mp4 \
  --out data/trajectories/yolo/unlabeled/session.npz

# 2) Attach human keys (same JSONL can be reused for path B after RT-DETR extract)
python -m src.vpt.log_to_actions \
  --trajectory data/trajectories/yolo/unlabeled/session.npz \
  --log logs/run_session.jsonl \
  --out data/trajectories/yolo/labeled/session.npz

# 3) If not done in Step 0: train shared IDM/BC on yolo/labeled (commands above)

# 4) Play with YOLO lists + shared policy
python -m src.play --pipeline yolo --dry-run
python -m src.play --pipeline yolo
```

### Path B — RT-DETR detections

```bash
python -m src.vpt.extract_objects --pipeline rtdetr \
  --weights runs/rtdetr/finetune/weights/best.pt \
  --detector rtdetr \
  --source recordings/session.mp4 \
  --out data/trajectories/rtdetr/unlabeled/session.npz

# Optional: labeled RT-DETR .npz for analysis (same keystroke log as path A)
python -m src.vpt.log_to_actions \
  --trajectory data/trajectories/rtdetr/unlabeled/session.npz \
  --log logs/run_session.jsonl \
  --out data/trajectories/rtdetr/labeled/session.npz

# Pseudo-label RT-DETR unlabeled VODs with the shared IDM (do not re-train IDM)
python -m src.vpt.pseudo_label --pipeline rtdetr

# Play: RT-DETR lists + same policy checkpoint as YOLO
python -m src.play --pipeline rtdetr --dry-run
python -m src.play --pipeline rtdetr
```

### Katana ZERO action mapping

The pipeline expects this 7-button multi-binary vector (configured in `configs/vpt_config_yolo.yaml` / `vpt_config_rtdetr.yaml`):

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

> **Required class ids.** `observation.player_class_id` and `observation.cursor_class_id` in the pipeline configs MUST match the class indices in your `data.yaml` (identical for YOLO and RT-DETR). Aim labels are silently empty if they're wrong — `log_to_actions.py` prints a warning when no frame has both the player and the cursor detected.

## Comparing the two detector paths

| What | Command |
|------|---------|
| Detection mAP only | `python -m src.compare_models` (Phase 1 §7) |
| Full pipeline actions (eval) | `python -m src.vpt.compare_policy --source recording.mp4` |
| Live agent | `python -m src.play --pipeline yolo` **or** `--pipeline rtdetr` (same policy weights) |

`compare_policy` runs two `Runner`s on each frame: YOLO→list A→$\pi_\theta$ and RT-DETR→list B→$\pi_\theta$. By default both use `checkpoints/shared/policy/best.pt` (same weights, different observations). It does **not** merge detections. `--shared-policy` overrides that path only if you want an explicit checkpoint argument.

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

## 4. Train the IDM (once)

```bash
python -m src.vpt.train_idm --pipeline yolo
```

Uses `data/trajectories/yolo/labeled/` by default; saves to `checkpoints/shared/idm/best.pt`. **Do not** train a second IDM for RT-DETR—the same weights pseudo-label both trajectory trees.

The IDM is 3 transformer layers over consecutive object lists (frame-id embeddings for $t$ vs $t{+}1$), then button logits. Watch validation accuracy before trusting pseudo-labels. Low accuracy usually means misaligned logs or missing state in the lists.

The IDM predicts **buttons only**. Aim comes from the crosshair detection (`src/vpt/aim.py`) on labeled and pseudo-labeled trajectories.

## 5. Pseudo-label unlabeled VODs (per detector)

```bash
python -m src.vpt.pseudo_label --pipeline yolo
python -m src.vpt.pseudo_label --pipeline rtdetr
```

Each call reads/writes under that pipeline's `unlabeled` / `pseudo_labeled` dirs but loads the **shared** IDM checkpoint from config.

## 6. Train the policy via BC (once)

```bash
python -m src.vpt.train_bc --pipeline yolo
```

Trains `checkpoints/shared/policy/best.pt` on `yolo/labeled` + `yolo/pseudo_labeled` by default. RT-DETR play still uses this same policy; only the detector-fed object lists change.

Pseudo-labeled samples are down-weighted by `bc.pseudo_label_weight` (default 0.5) — bump it up if your IDM is very accurate, down if it's noisy. The policy trains two heads jointly: buttons (BCE) and aim (16-way CE, masked to frames where both player and cursor are detected, weighted by `bc.aim_loss_weight`). Each epoch prints `acc` (per-button) and `aim_acc` (direction-bin accuracy on supervised frames).

## 7. Run the policy

### Video replay (`src/vpt/runner`)

One detector per invocation — one object list, **shared** policy weights:

```bash
python -m src.vpt.runner --pipeline yolo --source recordings/test.mp4
python -m src.vpt.runner --pipeline rtdetr --source recordings/test.mp4
```

| Flag | Purpose |
|------|---------|
| `--pipeline` | `yolo` or `rtdetr` (detector + trajectory dirs; same `bc.ckpt` in both configs) |
| `--weights` / `--yolo` | Override detector `.pt` |
| `--policy` | Override policy checkpoint |
| `--source` | Video file |

### Compare pipelines (`src/vpt/compare_policy`)

Evaluation only: same frame → YOLO path and RT-DETR path independently (no merged list). Default: both use `checkpoints/shared/policy/best.pt`.

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
| `--pipeline` | **`yolo` or `rtdetr`** — picks detector; policy defaults to `checkpoints/shared/policy/best.pt` |
| `--weights` / `--yolo` | Override detector weights |
| `--policy` | Override policy path |
| `--detect-only` | Detector only (one object list, no policy) |
| `--dry-run` | Print actions; no `pynput` injection |
| `--window-title` / `--rect` | Window capture: auto-detect by title on **macOS**; on **Windows/Linux** use `--rect L,T,W,H` |
| `--kill-key` | Emergency stop (default `f10`) |

**F10** = stop + release keys. **Ctrl+C** = clean exit.

## Practical notes (from the design transcript)

- **Object detection quality is the bottleneck.** If detections are jittery or drop objects, both IDM and policy will be noisy. Re-train YOLO/RT-DETR on more data before you blame the policy. Run `compare_policy` to see whether policy disagreement is mostly from missed objects vs. box jitter.
- **Two detectors, one policy, never merge lists.** Extract and play with `--pipeline yolo` or `--pipeline rtdetr`; IDM and BC train once. `compare_policy` is side-by-side eval (same $\pi_\theta$, different object lists).
- **RT-DETR on Apple Silicon (MPS)** is slower (CPU fallback for some ops). On **Windows/Linux with CUDA**, RT-DETR is usually faster than on M3 but still heavier than YOLO at inference. Prefer YOLO for live play unless RT-DETR wins on `compare_policy`.
- **Velocity matters more than position.** A still-frame snapshot can't tell you an enemy is winding up an attack; velocity often can. Both are included in `feats=(x,y,w,h,vx,vy)`.
- **Add global features for hidden state.** Things like "slow-mo gauge", current HP, or menu-open flag aren't visible to YOLO. Set `observation.global_dim > 0` and pass them via `--globals` in `attach_actions.py`.
- **multi_binary vs discrete.** Default is `multi_binary` (independent Bernoulli per button, BCE loss) because in real games you hold combos. Switch to `discrete` in the config for a single-softmax head if your action space is genuinely mutually exclusive.
- **Aim depends on the crosshair detection.** If YOLO misses the cursor on a frame, that frame contributes no aim supervision (training) and falls back to the previous mouse position (inference). If `aim_acc` stays low, check that the crosshair is detected reliably during combat and that 16 bins is fine-grained enough (raise `action_space.aim_bins` for finer control). Because aim comes from the video, it's also pseudo-labeled on VODs for free — no IDM needed.
- **Optional RL fine-tuning.** Katana ZERO is a Unity game, so you can mod it to expose rewards and run PPO against the BC-pretrained checkpoint — exactly the VPT recipe. Reward shaping suggestions: `+kill_enemy, +complete_room, -take_damage, +slow_mo_during_attack`.

