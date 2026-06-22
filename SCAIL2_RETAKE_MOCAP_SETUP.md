# SCAIL-2: Video Retakes (Character Replacement) + MoCap / Pose Extraction Setup

**Focus**: Pure SCAIL-2 for **retaking videos** (retarget motion from a driving/source video onto a new reference character — "replacement mode") and **making MoCap stuff** (extract 3D-aware poses, render mocap-like data, 3D retarget from videos using SCAIL-Pose tools).

This is the correct tool for your use case. No Sonic Forage / DEMON bleed.

Fork: https://github.com/TheMindExpansionNetwork/SCAIL-2 (wan-scail2 branch)

## Core Concepts from Reading the Folder

**Retakes / Replacement (main "retake videos" workflow)**:
- Take a **driving video** (source motion/actors doing things).
- Provide a **reference character** (new identity: ref.jpg + ref_mask.jpg).
- Preprocess with `SCAIL-Pose/NLFPoseExtract/process_replacement.py --subdir <your_dir>` (or --matchnearest if multiple people in driving).
  - Outputs: rendered_v2.mp4 (driving copy or pose render), replace_mask.mp4, plus ref_image.png / ref_mask.png if needed.
- Run `generate.py --replace_flag` with:
  - --image / --mask_image : your new character ref
  - --pose : the rendered driving
  - --mask_video : the replace_mask
  - --prompt : **detailed description of the FINAL video** (new character performing the source motion in the source scene, with correct clothing, objects, interactions).
- Result: The source video "retaken" with your new character.

**MoCap / Pose Extraction** (make mocap stuff out of videos):
- SCAIL-Pose is built for this: NLF (3D depth-aware poses, identity-agnostic), DWPose, SAM3 masks.
- Tools:
  - `NLFPoseExtract/process_animation_aio.py --subdir <dir> --e2e_mode` (end-to-end masks + driving copy) or without for pose-driven skeleton renders.
  - Older: `v1_process_pose.py --use_align` for 3D retarget.
  - `render_3d/` : cylinder renders, taichi_cylinder for 3D visualizations.
  - `pose_draw/` : draw body/hand/face poses (2D/3D).
  - Can produce rendered pose videos or raw pose data for mocap export / driving other systems.
- Great for turning any video into usable mocap (3D poses, retargeted, multi-person).

**Fast / Light / Distilled Inference ("light X distilled")**:
- Use the **Lightx2v LoRA** (mentioned in README): `Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors`
- With `--lora_path ... --lora_alpha 1.0 --sample_steps 8 --sample_shift 1 --sample_guide_scale 1.0`
- Much faster (distilled steps).

**Prompts for Retakes**:
- Must describe the *after-replacement* video, not the instruction.
- Our `prompt_enhancer_deepseek.py` (already in fork) is perfect: feed short instruction + video caption + ref desc → DeepSeek (your credits) outputs rich final prompt. No Gemini needed.

**Examples in folder**:
- `examples/replace_001/`: driving + ref + masks for replacement test.
- `examples/001/`, `002/`: animation examples.

**Pre-reqs**:
- Main model: `hf download zai-org/SCAIL-2` (then `python convert.py` for .safetensors on wan branch).
- SCAIL-Pose: its own setup (mmpose env, NLF/DWPose/SAM3 weights, SAM2 optional). Heavy — run preprocessing locally or separate lane.
- For fast: download the Lightx2v LoRA.

## Practical Retake Workflow (Video Retake)

1. Prepare a subdir like the examples:
   ```
   mkdir -p my_retake_test
   cp /path/to/your_driving_video.mp4 my_retake_test/driving.mp4
   cp /path/to/your_new_character.jpg my_retake_test/ref.jpg
   # (create or extract ref_mask.jpg with any mask tool or SAM)
   ```

2. Preprocess for replacement (in SCAIL-Pose dir or with proper env):
   ```
   cd SCAIL-2/SCAIL-Pose
   python NLFPoseExtract/process_replacement.py --subdir ../my_retake_test
   # or --matchnearest if driving has multiple people and you want only one replaced
   ```
   Outputs will appear in the subdir: rendered_v2.mp4, replace_mask.mp4, etc.

3. Enhance prompt with DeepSeek (fast, your credits):
   ```
   python ../prompt_enhancer_deepseek.py \
     --instruction "replace the person in the video with the character in the reference image, keep the exact actions and scene" \
     --video_caption "person walking and interacting on street" \
     --ref_description "the new character: [detailed description of your ref]" \
     --output my_retake_test/enhanced_prompt.txt
   ```

4. Run the retake (with Lightx2v fast distilled if you have the LoRA):
   ```
   python generate.py \
     --model SCAIL-14B \
     --ckpt_dir /path/to/SCAIL-2 \
     --scail_path /path/to/SCAIL-2.safetensors \
     --replace_flag \
     --image my_retake_test/ref.jpg \
     --mask_image my_retake_test/ref_mask.jpg \
     --pose my_retake_test/rendered_v2.mp4 \
     --mask_video my_retake_test/replace_mask.mp4 \
     --prompt "$(cat my_retake_test/enhanced_prompt.txt)" \
     --save_file my_retake_test/retaken_output.mp4 \
     --sample_steps 8 --lora_path /path/to/lightx2v_...safetensors --lora_alpha 1.0   # for light X distilled speed
   ```

## MoCap / Pose Extraction Workflow

To turn any video into mocap data or pose-driven material:

- Use `SCAIL-Pose/NLFPoseExtract/` tools (see README in SCAIL-Pose for full).
- Basic 3D-aware:
  ```
  python NLFPoseExtract/v1_process_pose.py --subdir <your_video_dir> --use_align --resolution 512,896
  ```
- For multi or advanced: v1_process_pose_multi.py, process_animation_aio.py (gets masks + poses).
- Render 3D mocap viz: look in `render_3d/` (cylinder renders from poses, taichi for GPU).
- Pose drawing: `pose_draw/` for 2D/3D skeletons you can export or use as mocap input for other tools.
- The extracted NLF poses are depth-aware and good for retargeting (identity agnostic).

You can then feed the rendered pose video back into SCAIL-2 generate (pose-driven mode) or export the raw pose data.

## Modal Serverless Setup (Pre-cached + Fast Retakes)

We have `modal_app.py` in the folder (cleaned for this use case).

- Pre-caches SCAIL-2 weights in persistent Volume (`scail2-models`) → this is your **snapshot** for fast serverless (run download once).
- Image has everything pre-installed.
- Functions for: download (pre-cache), DeepSeek prompt enhance (retake prompts), and inference with support for `--replace_flag` + Lightx2v LoRA (light X distilled fast mode).

**Setup commands** (run after `pip install modal` + token + secret):

```bash
# 1. Pre-cache the model (the build/snapshot step — do this first, once)
modal run modal_app.py::download_scail_models

# 2. For a retake test (use a prepared subdir or example)
modal run modal_app.py::enhance_prompt_deepseek --instruction "..." --video_caption "..." --ref_description "..."

# 3. Run retake inference (fast with LoRA if you mount the LoRA too or pre-cache it)
# Pass the 4 inputs + enhanced prompt. Add lora params in the call or hardcode in app.
modal run modal_app.py::run_scail_inference \
  --image path/to/ref.jpg --mask_image path/to/ref_mask.jpg \
  --pose path/to/rendered_v2.mp4 --mask_video path/to/replace_mask.mp4 \
  --prompt "the enhanced prompt" \
  --replace_flag   # key for retakes
```

Edit `modal_app.py` to hardcode Lightx2v LoRA path (download it separately to volume or local) and default to replacement mode + low steps for speed.

The volume snapshot makes repeated retakes / mocap tests fast on serverless GPU.

## Next Steps to Get It Running

1. Download the main SCAIL-2 weights locally or let Modal do it via the volume function.
2. Set up SCAIL-Pose env/weights (follow SCAIL-Pose/README — it's the mocap/retarget engine).
3. Pick or upload a driving video (e.g. the one in /opt/data/cache/videos/video_802aedc6519c.mov or prepare like examples/replace_001).
4. Prepare a ref character image + rough mask.
5. Run the preprocess + DeepSeek enhance + generate (local first for small test, then Modal for heavy).
6. For pure mocap output: run the NLF / render tools and inspect the .npy or rendered pose files.

We can iterate: pick a specific video from your drops/workspace, prepare a test subdir, run a real retake here (local smoke or trigger Modal), extract some mocap data, etc.

The folder is fully read — SCAIL-Pose + generate.py + our DeepSeek enhancer + the Lightx2v fast path give you exactly video retakes + mocap extraction.

Let me know a specific driving video or ref to test with, or say "run the preprocess on [video]" and we'll set up the exact commands + any scripts needed. Ready to make those retakes and mocap assets.