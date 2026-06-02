# ForceVLA Developer Guide

A practical guide to training, configuring, and deploying ForceVLA for custom robot manipulation tasks.

## Architecture Overview

ForceVLA extends pi0 (Physical Intelligence's flow-matching VLA) with force/torque awareness via LIMoE (Learned Interleaving Mixture of Experts).

```
┌─────────────────────────────────────────────────────────────────┐
│                        ForceVLA (Pi0_Guidance)                  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐    ┌─────────────────────────┐     │
│  │  SigLIP   │  │ Language  │    │   Gemma 2B (VLM)        │     │
│  │  Vision   │──│ Tokens   │───▶│   paligemma_variant     │     │
│  │  Encoder  │  └──────────┘    │   processes prefix      │     │
│  └──────────┘                   └────────────┬────────────┘     │
│                                              │                  │
│  ┌──────────┐                   ┌────────────▼────────────┐     │
│  │ Force/   │──force_in_proj───▶│       LIMoE Block       │     │
│  │ Torque   │    (6→2048)       │   4 experts, 8 heads    │     │
│  │ (6-dim)  │                   │   routes force+vision   │     │
│  └──────────┘                   └────────────┬────────────┘     │
│                                              │ (residual add)   │
│  ┌──────────┐                   ┌────────────▼────────────┐     │
│  │ Proprio  │──state_proj──────▶│   Gemma 300M (Action)   │     │
│  │ (7-dim)  │    (7→1024)       │   action_expert_variant │     │
│  └──────────┘                   │   processes suffix      │     │
│                                 └────────────┬────────────┘     │
│  ┌──────────┐                                │                  │
│  │ Noisy    │──action_in_proj               │                  │
│  │ Actions  │──action_time_mlp──────────────▶│                  │
│  │ + Time   │                   ┌────────────▼────────────┐     │
│  └──────────┘                   │   action_out_proj       │     │
│                                 │   (1024→action_dim)     │     │
│                                 └─────────────────────────┘     │
│                                         ▼                       │
│                                  7D action output               │
│                           [dx,dy,dz,drx,dry,drz,grip]          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

- **Dual-Gemma LLM**: Two Gemma models process tokens jointly. The larger one (2B) handles vision+language (prefix). The smaller one (300M) handles actions (suffix). They share attention via a joint forward pass.
- **SigLIP**: Vision encoder (So400m/14, 224x224). Converts camera images to patch tokens.
- **LIMoE**: Mixture-of-experts block that fuses force/torque information with visual features. 4 experts with top-1 routing, 8 attention heads. Different experts specialize in different force regimes (free motion, contact, insertion).
- **Flow Matching**: Actions are generated via 10-step Euler denoising from noise to clean actions (t=1.0 to t=0.0).

### Model Variants

| Variant | VLM | Action Expert | Trainable | Use Case |
|---------|-----|---------------|-----------|----------|
| `gemma_2b` + `gemma_300m` | Full 2B | Full 300M | All params | Full fine-tune |
| `gemma_2b_lora` + `gemma_300m_lora` | 2B + LoRA(r=16) | 300M + LoRA(r=32) | LoRA + LIMoE + projections | Recommended |
| `dummy` + `dummy` | 64-dim | 64-dim | All | Debugging |

LoRA fine-tuning is recommended: trains ~5% of parameters while keeping backbone frozen.

### Joint State Extension (`use_joint_state=True`)

When `use_joint_state=True`, the model creates an additional `joint_in_proj` linear layer (12→1024) that projects `[joint_pos(6), joint_vel(6)]` into an extra suffix token for the action expert. This gives the model direct access to joint-level kinematics alongside the Cartesian state.

```
                                 ┌────────────────────┐
  ┌──────────┐                   │  Gemma 300M        │
  │ Joint    │──joint_in_proj───▶│  (Action Expert)   │  ← extra suffix token
  │ State    │    (12→1024)      │                    │
  │ (12-dim) │                   └────────────────────┘
  └──────────┘
  joint_pos(6) + joint_vel(6)
```

The `joint_in_proj` layer is randomly initialized (not present in the pi0_base checkpoint) and trained from scratch alongside LoRA adapters and LIMoE.

## Data Format

ForceVLA uses LeRobot v2.1 datasets. Structure:

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   ├── observation.images.center/
│   │   ├── episode_000000.mp4
│   │   └── ...
│   ├── observation.images.left/
│   └── observation.images.right/
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

### Required Parquet Columns

| Column | Shape | Description |
|--------|-------|-------------|
| `observation.state.ee_pos` | (3,) | TCP position xyz in base frame |
| `observation.state.ee_quat` | (4,) | TCP quaternion (xyzw) |
| `observation.state.gripper_pos` | (2,) | Gripper finger positions |
| `observation.state.wrench` | (6,) | Wrist force/torque (Fx,Fy,Fz,Tx,Ty,Tz) |
| `action` | (7,) | Per-timestep action deltas |
| `observation.images.center` | video | Center camera (stored as MP4) |
| `observation.images.left` | video | Left camera |
| `observation.images.right` | video | Right camera |

Optional (with `use_joint_state=True` and `include_joints=True`):

| Column | Shape | Description |
|--------|-------|-------------|
| `observation.state.joint_pos` | (6,) | Joint angles (radians) |
| `observation.state.joint_vel` | (6,) | Joint velocities (rad/s) |

If your dataset has `joint_pos` but not `joint_vel`, compute it offline via finite differences:
```python
joint_vel[t] = (joint_pos[t] - joint_pos[t-1]) * FPS  # e.g., FPS=20
```

### meta/info.json

```json
{
  "codebase_version": "v2.1",
  "robot_type": "ur5e",
  "total_episodes": 475,
  "total_frames": 275500,
  "total_tasks": 10,
  "fps": 20,
  "chunks_size": 1000,
  "total_chunks": 1,
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
  "splits": {"train": "0:475"},
  "features": {
    "observation.state.ee_pos": {"dtype": "float32", "shape": [3]},
    "observation.state.ee_quat": {"dtype": "float32", "shape": [4]},
    "observation.state.gripper_pos": {"dtype": "float32", "shape": [2]},
    "observation.state.wrench": {"dtype": "float32", "shape": [6]},
    "action": {"dtype": "float32", "shape": [7]},
    "observation.images.center": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.images.left": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.images.right": {"dtype": "video", "shape": [480, 640, 3]}
  }
}
```

### meta/tasks.jsonl

```jsonl
{"task_index": 0, "task": "Insert the SFP cable module in SFP_PORT_0 on NIC_CARD at NIC_RAIL_0"}
{"task_index": 1, "task": "Insert the SFP cable module in SFP_PORT_1 on NIC_CARD at NIC_RAIL_0"}
```

### meta/episodes.jsonl

```jsonl
{"episode_index": 0, "task_index": 0, "length": 580}
{"episode_index": 1, "task_index": 1, "length": 580}
```

### SFP Insertion Dataset Versions

Three dataset versions exist for SFP cable insertion, each fixing an issue:

| Version | Dataset | State | Actions | HuggingFace |
|---------|---------|-------|---------|-------------|
| v1 | `aic_gt_sfp_all_trimmed` | 13D (no joints) | `action[3:5]=0` (no rotation) | [tshiamor/aic_gt_sfp_all_trimmed](https://huggingface.co/datasets/tshiamor/aic_gt_sfp_all_trimmed) |
| v2 | `aic_gt_sfp_all_trimmed_v2` | 13D (no joints) | Orientation deltas from ee_quat | [tshiamor/aic_gt_sfp_all_trimmed_v2](https://huggingface.co/datasets/tshiamor/aic_gt_sfp_all_trimmed_v2) |
| v3 | `aic_gt_sfp_all_trimmed_v3` | 25D (with joints) | Same as v2 + joint_vel added | [tshiamor/aic_gt_sfp_all_trimmed_v3](https://huggingface.co/datasets/tshiamor/aic_gt_sfp_all_trimmed_v3) |

All three share the same 475 episodes, 275,500 frames, 10 SFP tasks, and 3 cameras. The differences are in which observation and action fields are populated.

**v1 failure**: `RecordCheatCode.py` hardcoded `np.zeros(3)` for rotation actions. The model could never learn the ~21-degree pitch alignment needed for insertion, and norm_stats had `std=0` for those channels — collapsing them permanently to zero at inference via denormalization.

**v2 fix**: Computed axis-angle rotation deltas offline from consecutive `ee_quat` values: `delta = (R_{t-1}^{-1} * R_t).as_rotvec()`. No re-collection needed.

**v3 addition**: Added `observation.state.joint_vel` via finite differences: `vel[t] = (pos[t] - pos[t-1]) * FPS`. Enables `use_joint_state=True` for joint-aware training.

### Parquet Gotchas

- **Video struct columns**: LeRobot v2.1 stores video references as struct columns in parquet. ForceVLA reads videos separately via MP4 decoding. If your parquets contain struct-type image columns, strip them before training:
  ```python
  import pyarrow.parquet as pq
  t = pq.read_table("episode_000000.parquet")
  video_cols = [c for c in t.column_names if "images" in c and str(t.schema.field(c).type).startswith("struct")]
  if video_cols:
      pq.write_table(t.drop(video_cols), "episode_000000.parquet")
  ```
- **episodes_stats.jsonl**: ForceVLA's data loader expects this file. Create a stub if missing:
  ```python
  import json
  with open("meta/episodes_stats.jsonl", "w") as f:
      for i in range(num_episodes):
          f.write(json.dumps({"episode_index": i, "stats": {}}) + "\n")
  ```

## State and Action Spaces

### State Vector (13D default, 25D with joints)

The model receives a single state vector assembled by `SfpStateTransform`:

```
Index   Field              Source
───── ─────────────────── ──────────────────────────────
0-2     ee_pos              observation.state.ee_pos (xyz)
3-5     axis_angle          quaternion_to_axis_angle(ee_quat)
6       gripper             mean(gripper_pos)
7-12    wrench              observation.state.wrench (Fx,Fy,Fz,Tx,Ty,Tz)
───── ─────────────────── ──────────────────────────────
13-18   joint_pos           (optional) observation.state.joint_pos
19-24   joint_vel           (optional) observation.state.joint_vel
```

Internally, the model splits this vector:
- `state[:7]` (proprio) → `state_proj` → action expert suffix token
- `state[7:13]` (wrench) → `force_in_proj` → VLM prefix, then through LIMoE
- `state[13:25]` (joints) → `joint_in_proj` → extra suffix token (optional)

### Action Vector (7D)

```
Index   Field       Description
───── ─────────── ────────────────────────────────
0-2     dx,dy,dz    Position delta (meters/step)
3-5     drx,dry,drz Orientation delta as rotation vector (radians/step)
6       gripper     Gripper command
```

Actions are per-timestep deltas at the dataset's FPS (e.g., 20Hz). The model predicts `action_horizon` (50) future steps at once.

### Action Horizon

The model outputs a chunk of 50 future actions in one forward pass. At inference, typically only the first action is used (or a few via ActionChunkBroker). This is the flow-matching approach: denoise all 50 timesteps simultaneously.

## Normalization (norm_stats)

### What It Is

`norm_stats` contains per-dimension statistics (mean, std, q01, q99) for the `state` and `actions` vectors. It is used to normalize inputs to zero-mean unit-variance before the model sees them, and to denormalize outputs back to physical units.

### Why It's Critical

Without normalization:
- Position deltas (~0.0005 m) and wrench values (~5 N) differ by 4 orders of magnitude
- The flow matching loss treats all dimensions equally
- The model would learn to minimize loss on high-magnitude dimensions while ignoring small ones
- Result: the model outputs garbage actions

With normalization:
- All dimensions are scaled to comparable ranges
- The model allocates capacity across all action/state dimensions

### How to Compute

```bash
cd ~/ForceVLA
python scripts/compute_norm_stats.py --config-name=your_config_name
```

This iterates the full dataset (after repack + data transforms, but before normalization), computing running mean, std, and percentiles (q01, q99) per dimension. Saved to `./assets/{config_name}/{repo_id}/norm_stats.json`.

### How It's Applied

**Training input normalization** (z-score):
```
normalized = (x - mean) / (std + 1e-6)
```

**Inference output denormalization**:
```
original = x * (std + 1e-6) + mean
```

The `1e-6` epsilon prevents division by zero for constant dimensions (e.g., gripper always 0).

### Failure Mode: Zero-Std Dimensions

If a dimension has `std=0` and `q01=q99=0` (e.g., rotation actions that were always zero in training):
- Normalization maps everything to 0
- The model learns to always output 0 for that dimension
- **Denormalization**: `output * (0 + 1e-6) + 0 = ~1e-6` — effectively zero regardless of model output

This is exactly what happened with v1 rotation actions. The fix was to compute real rotation deltas from the quaternion data.

### Where norm_stats Are Stored

1. **During training**: `./assets/{config_name}/{repo_id}/norm_stats.json`
2. **In checkpoints**: `./checkpoints/{config_name}/{exp}/{step}/assets/{repo_id}/norm_stats.json`
3. **At inference**: loaded from the checkpoint's `assets/` subdirectory

The checkpoint embeds a copy so inference doesn't depend on the training assets directory.

## Training Pipeline

### Step-by-Step

```bash
# 1. Prepare dataset (LeRobot v2.1 format)
# Ensure parquets, videos, info.json, episodes.jsonl, tasks.jsonl exist

# 2. Set up dataset symlink for LeRobot
export HF_LEROBOT_HOME=~/datasets
mkdir -p $HF_LEROBOT_HOME/your_username
ln -s /path/to/your_dataset $HF_LEROBOT_HOME/your_username/your_dataset

# 3. Add config to config.py (see "Adding a Custom Config" below)

# 4. Compute norm stats
export HF_HUB_OFFLINE=1
python scripts/compute_norm_stats.py --config-name=your_config

# 5. Train
python scripts/train.py your_config \
    --exp-name=exp1 \
    --batch_size 4 \
    --num_train_steps 50000 \
    --save_interval 5000 \
    --fsdp_devices 1

# 6. Serve
python scripts/serve_policy.py policy:checkpoint \
    --policy.config your_config \
    --policy.dir checkpoints/your_config/exp1/49999

# 7. Infer
from openpi_client.websocket_client_policy import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(observation_dict)
actions = result["actions"]  # (50, 7)
```

### Flow Matching Loss

ForceVLA uses conditional flow matching for action generation:

```python
# Training: learn the velocity field
noise = random_normal(actions.shape)
time = Beta(1.5, 1) * 0.999 + 0.001     # Biased toward t=1 (noisy)
x_t = time * noise + (1-time) * actions   # Interpolate
u_t = noise - actions                     # Ground truth velocity
v_t = model(x_t, time, observation)       # Predicted velocity
loss = mean((v_t - u_t)^2)               # MSE loss

# Inference: 10-step Euler integration from noise to actions
x = random_normal(...)                    # Start at t=1 (pure noise)
for step in range(10):
    v = model(x, t, observation)
    x = x + (-0.1) * v                   # dt = -1/10
    t = t - 0.1
actions = x                              # t=0, denoised actions
```

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (b1=0.9, b2=0.95, eps=1e-8) |
| Weight decay | 1e-10 |
| Gradient clipping | 1.0 |
| Learning rate | Cosine decay: warmup 1000 steps, peak 2.5e-5, min 2.5e-6 |
| Batch size | 4 (for LoRA on 32GB GPU) |
| EMA | Disabled for LoRA (`ema_decay=None`) |
| Image augmentation | Random crop (95%), resize, rotation (+-5 deg), color jitter |
| LoRA rank | 16 (VLM), 32 (action expert) |

### SFP Insertion Training Configs

Three pre-defined configs for SFP insertion, matching the dataset versions:

| Config | Dataset | State | Key Flags | Model |
|--------|---------|-------|-----------|-------|
| `forcevla_sfp_all_trimmed` | v1 | 13D | (default) | [tshiamor/forcevla-sfp-all-trimmed](https://huggingface.co/tshiamor/forcevla-sfp-all-trimmed) |
| `forcevla_sfp_all_trimmed_v2` | v2 | 13D | (default) | [tshiamor/forcevla-sfp-all-trimmed-v2](https://huggingface.co/tshiamor/forcevla-sfp-all-trimmed-v2) |
| `forcevla_sfp_all_trimmed_v3` | v3 | 25D | `use_joint_state=True`, `include_joints=True` | -- |

The v3 config differs from v2 in two flags:

```python
# v2 (no joints — 13D state)
TrainConfig(
    name="forcevla_sfp_all_trimmed_v2",
    model=pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
    ),
    data=SfpInsertDataConfig(
        repo_id="tshiamor/aic_gt_sfp_all_trimmed_v2",
        base_config=DataConfig(prompt_from_task=True),
    ),
    ...
)

# v3 (with joints — 25D state)
TrainConfig(
    name="forcevla_sfp_all_trimmed_v3",
    model=pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
        use_joint_state=True,           # <-- creates joint_in_proj layer
    ),
    data=SfpInsertDataConfig(
        repo_id="tshiamor/aic_gt_sfp_all_trimmed_v3",
        base_config=DataConfig(prompt_from_task=True),
        include_joints=True,            # <-- adds joint_pos/joint_vel to repack + state
    ),
    ...
)
```

`use_joint_state=True` on the model and `include_joints=True` on the data config must both be set. The model flag creates the `joint_in_proj` projection layer; the data flag adds `joint_pos` and `joint_vel` to the repack transform and state assembly.

### Training Tips

- **VRAM**: LoRA fine-tuning uses ~22 GB on RTX 5090. Reduce batch_size if OOM.
- **Steps**: 50k steps is a good starting point for 400+ episodes.
- **Checkpoints**: `save_interval=5000` + `keep_period=10000` keeps every 10k permanently, prunes 5k intermediates.
- **Wandb**: Enabled by default. Set `WANDB_API_KEY` or disable with `--wandb_enabled=False`.
- **Resume**: Use `--resume` to continue from the latest checkpoint in the same experiment dir.

## Adding a Custom Config

### 1. Create a DataConfigFactory

In `src/openpi/training/config.py`:

```python
@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs, model_config):
        repack_map = {
            "center_image": "observation.images.center",   # your camera key
            "left_image": "observation.images.left",
            "right_image": "observation.images.right",
            "ee_pos": "observation.state.ee_pos",
            "ee_quat": "observation.state.ee_quat",
            "wrench": "observation.state.wrench",
            "gripper_pos": "observation.state.gripper_pos",
            "actions": "action",
            "prompt": "prompt",
        }
        repack_transform = _transforms.Group(inputs=[
            _transforms.RepackTransform(repack_map),
            SfpStateTransform(),                            # assembles 13-dim state
        ])
        data_transforms = _transforms.Group(
            inputs=[forcevla_policy.Forcevla_inputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type)],
            outputs=[forcevla_policy.Forcevla_outputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repo_id=self.repo_id,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```

### 2. Add TrainConfig

```python
TrainConfig(
    name="forcevla_my_robot",
    model=pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,                              # adjust to your action space
    ),
    data=MyRobotDataConfig(
        repo_id="your_username/your_dataset",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.Pi0GuidanceWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=50_000,
    freeze_filter=pi0_force.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    ema_decay=None,
    batch_size=4,
),
```

### 3. Things to Watch

- **action_dim must match your data**: If your actions are 7D, set `action_dim=7`. The state_proj, action_in_proj, and action_out_proj layers are sized accordingly.
- **State assembly**: `SfpStateTransform` concatenates `[ee_pos, axis_angle(ee_quat), gripper, wrench]`. If your robot has different state fields, create a custom state transform.
- **Image count**: ForceVLA expects 3 cameras. If you have fewer, the missing ones are zero-filled and masked.
- **Prompt format**: `prompt_from_task=True` reads from `tasks.jsonl`. Each task description becomes the language conditioning.
- **Action format**: Actions should be per-timestep deltas at your dataset's FPS. No additional delta/absolute conversion is applied (unlike ALOHA/DROID configs).

## Weight Loading

### Pi0GuidanceWeightLoader

Loads the pretrained pi0 base checkpoint and merges with randomly initialized ForceVLA-specific parameters:

```
Loaded from pi0_base:     PaliGemma (SigLIP + Gemma 2B), Gemma 300M action expert
Randomly initialized:     LoRA adapters, LIMoE block, force_in_proj, joint_in_proj
Shape-matched:           state_proj, action_in_proj, action_out_proj (if action_dim matches)
```

If `action_dim` differs from the base checkpoint (32 → 7), the projection layers are randomly initialized and a warning is logged.

### First Run

The first run downloads the pi0_base checkpoint from Google Cloud Storage (~12 GB). It's cached at `~/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/`.

## Inference Fixes (Fork-Specific)

Two bugs in the upstream ForceVLA that only manifest during inference (`sample_actions()`), not training:

### 1. prefix_out_fix (commit 0d5140e)

During KV-cached denoising, the VLM returns `prefix_out=None` for steps 2-10 (prefix is handled via cache). But LIMoE needs the actual prefix embeddings. Fix: cache them from step 1.

### 2. LIMoE Padding (commit 489b463)

LIMoE's MoE router requires `sequence_length % num_experts == 0`. At batch_size=1 inference, the sequence length (e.g., 817) is not divisible by 4. Fix: pad to nearest multiple of 4, then slice output back.

Both fixes are required for any ForceVLA inference deployment.

## Inference with Joint State Models (v3)

When serving a v3 model (trained with `use_joint_state=True`), the server expects a 25D state vector instead of 13D. The client must send joint_pos and joint_vel alongside the standard state.

### Serving

```bash
cd ~/ForceVLA && conda activate forcevla_eval
python scripts/serve_policy.py policy:checkpoint \
    --policy.config forcevla_sfp_all_trimmed_v3 \
    --policy.dir checkpoints/forcevla_sfp_all_trimmed_v3/v3/49999
```

### Client State Vector

The client must build a 25D state (not 13D):

```python
state = np.concatenate([
    ee_pos,         # (3,) TCP position
    axis_angle,     # (3,) from quaternion
    gripper,        # (1,) mean of finger joints
    wrench,         # (6,) force/torque
    joint_pos,      # (6,) joint angles
    joint_vel,      # (6,) joint velocities
])  # total: 25D
```

If the client sends 13D to a 25D server, normalization will fail with:
```
ValueError: operands could not be broadcast together with shapes (13,) (25,)
```

### AIC RunForceVLA Policy

The `RunForceVLA` ROS policy supports both modes via the `FORCEVLA_USE_JOINTS` environment variable:

```bash
# v2 model (13D state, no joints) — default
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunForceVLA

# v3 model (25D state, with joints)
FORCEVLA_USE_JOINTS=1 pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunForceVLA
```

When `FORCEVLA_USE_JOINTS=1`:
- `joint_pos` is read from `obs.joint_states.position[:6]`
- `joint_vel` is computed at runtime via finite differences: `(pos[t] - pos[t-1]) * CONTROL_HZ`
- State vector is 25D: `[ee_pos, axis_angle, gripper, wrench, joint_pos, joint_vel]`

### Mismatched State Dimensions

A common error is serving a v3 model but running a v2 client (or vice versa). The state dimension must match between the norm_stats in the checkpoint and the state vector sent by the client:

| Config | norm_stats state dim | Client state dim | `FORCEVLA_USE_JOINTS` |
|--------|---------------------|-------------------|----------------------|
| v2 | 13 | 13 | `0` (default) |
| v3 | 25 | 25 | `1` |

## Quick Reference

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_LEROBOT_HOME` | Dataset root directory | `~/.cache/lerobot` |
| `HF_HUB_OFFLINE` | Skip HuggingFace Hub downloads | `0` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX GPU memory fraction | `0.75` |
| `OPENPI_DATA_HOME` | Cache for downloaded checkpoints | `~/.cache/openpi` |
| `WANDB_API_KEY` | Weights & Biases API key | (none) |
| `FORCEVLA_USE_JOINTS` | Include joint_pos/joint_vel in state (for v3 models) | `0` |
| `FORCEVLA_HOST` | ForceVLA inference server host | `localhost` |
| `FORCEVLA_PORT` | ForceVLA inference server port | `8000` |

### File Paths

| What | Path |
|------|------|
| Training configs | `src/openpi/training/config.py` |
| Model definition | `src/openpi/models/pi0_force.py` |
| Policy transforms | `src/openpi/policies/forcevla_policy.py` |
| Norm stats | `assets/{config_name}/{repo_id}/norm_stats.json` |
| Checkpoints | `checkpoints/{config_name}/{exp_name}/{step}/` |
| Training script | `scripts/train.py` |
| Norm stats script | `scripts/compute_norm_stats.py` |
| Serving script | `scripts/serve_policy.py` |
| Client package | `packages/openpi-client/src/openpi_client/` |
