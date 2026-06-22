#!/usr/bin/env python3
"""
Simple wrapper for SCAIL-2 video retakes (character replacement) + basic MoCap prep.
Reads a driving video + ref character, runs SCAIL-Pose preprocess for replacement,
uses DeepSeek for prompt, then calls generate (with optional Lightx2v fast distilled LoRA).

Usage example:
  python retake_video.py \
    --driving /path/to/driving.mp4 \
    --ref_image /path/to/my_character.jpg \
    --ref_mask /path/to/my_character_mask.jpg \
    --output retaken.mp4 \
    --instruction "replace the person with my new character, keep all actions and scene exactly" \
    --use_deepseek \
    --light_distilled   # uses Lightx2v LoRA if path provided or default

Requires SCAIL-Pose env for preprocess step (or run preprocess separately).
For full serverless, use the Modal app after pre-caching.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving", required=True, help="Driving/source video to retake motion from")
    parser.add_argument("--ref_image", required=True, help="Reference character image (new identity)")
    parser.add_argument("--ref_mask", required=True, help="Mask for the reference character")
    parser.add_argument("--output", default="retaken_output.mp4", help="Output video path")
    parser.add_argument("--instruction", default="replace the main person/actor in the video with the character from the reference image. Keep the exact motion, actions, timing, camera, and scene.", help="Short instruction for prompt enhancer")
    parser.add_argument("--subdir", default=None, help="Working subdir (default: auto from driving name)")
    parser.add_argument("--use_deepseek", action="store_true", help="Use DeepSeek enhancer (requires DEEPSEEK_API_KEY)")
    parser.add_argument("--light_distilled", action="store_true", help="Use Lightx2v LoRA for fast distilled inference (8 steps)")
    parser.add_argument("--lora_path", default=None, help="Path to Lightx2v LoRA safetensors (if not default)")
    parser.add_argument("--model", default="SCAIL-14B", help="SCAIL model")
    parser.add_argument("--ckpt_dir", default="./SCAIL-2", help="Path to SCAIL-2 checkpoint dir")
    parser.add_argument("--scail_path", default="./SCAIL-2.safetensors", help="Path to converted safetensors")
    args = parser.parse_args()

    driving = Path(args.driving).resolve()
    ref_img = Path(args.ref_image).resolve()
    ref_mask = Path(args.ref_mask).resolve()
    out = Path(args.output).resolve()

    if not driving.exists():
        print(f"Driving video not found: {driving}")
        sys.exit(1)

    subdir = Path(args.subdir) if args.subdir else Path("retake_" + driving.stem)
    subdir.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing retake subdir: {subdir} ===")
    # Copy inputs into subdir for SCAIL-Pose preprocess
    (subdir / "driving.mp4").write_bytes(driving.read_bytes())
    (subdir / "ref.jpg").write_bytes(ref_img.read_bytes())  # SCAIL-Pose expects .jpg or will convert
    (subdir / "ref_mask.jpg").write_bytes(ref_mask.read_bytes())

    # 1. Preprocess for replacement (SCAIL-Pose)
    print("=== Running SCAIL-Pose replacement preprocess (this does the MoCap/pose + mask extraction) ===")
    pose_cmd = [
        "python", "SCAIL-Pose/NLFPoseExtract/process_replacement.py",
        "--subdir", str(subdir),
    ]
    if args.instruction and "match" in args.instruction.lower():
        pose_cmd.append("--matchnearest")
    print(" ".join(pose_cmd))
    try:
        subprocess.check_call(pose_cmd, cwd=Path.cwd())
    except subprocess.CalledProcessError as e:
        print("Preprocess failed (common if SCAIL-Pose env not set up — install per SCAIL-Pose/README).")
        print("You can run the preprocess manually in the right env, then continue.")
        # Continue anyway if outputs exist

    rendered = subdir / "rendered_v2.mp4"
    replace_mask = subdir / "replace_mask.mp4"
    if not rendered.exists() or not replace_mask.exists():
        print("Warning: preprocess outputs not found. Using driving as fallback for pose (may be less accurate).")
        # Fallback: copy driving as rendered for e2e-like
        if not rendered.exists():
            (subdir / "rendered_v2.mp4").write_bytes(driving.read_bytes())
        if not replace_mask.exists():
            # Simple mask fallback not ideal — user should run proper preprocess
            print("Please run proper SCAIL-Pose preprocess for best retake results.")

    # 2. Prompt (DeepSeek or simple)
    prompt = args.instruction
    if args.use_deepseek:
        print("=== Enhancing prompt with DeepSeek ===")
        enh_cmd = [
            "python", "prompt_enhancer_deepseek.py",
            "--instruction", args.instruction,
            "--video_caption", f"driving video: {driving.name}",
            "--ref_description", "the reference character from the image",
            "--output", str(subdir / "enhanced_prompt.txt"),
        ]
        try:
            subprocess.check_call(enh_cmd)
            prompt = (subdir / "enhanced_prompt.txt").read_text().strip()
        except Exception as e:
            print(f"DeepSeek enhance failed: {e}. Using raw instruction as prompt.")
            prompt = args.instruction

    print(f"Final prompt (first 200 chars): {prompt[:200]}...")

    # 3. Generate retake
    print("=== Running SCAIL-2 generate (retake / replacement) ===")
    gen_cmd = [
        "python", "generate.py",
        "--model", args.model,
        "--ckpt_dir", args.ckpt_dir,
        "--scail_path", args.scail_path,
        "--replace_flag",
        "--image", str(subdir / "ref.jpg"),
        "--mask_image", str(subdir / "ref_mask.jpg"),
        "--pose", str(rendered),
        "--mask_video", str(replace_mask),
        "--prompt", prompt,
        "--save_file", str(out),
        "--sample_steps", "40",
    ]

    if args.light_distilled:
        lora = args.lora_path or "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
        gen_cmd += [
            "--lora_path", lora,
            "--lora_alpha", "1.0",
            "--sample_steps", "8",
            "--sample_shift", "1",
            "--sample_guide_scale", "1.0",
        ]
        print("Using Lightx2v distilled LoRA for fast inference (light X distilled mode).")

    print(" ".join(gen_cmd))
    try:
        subprocess.check_call(gen_cmd)
        print(f"\n✅ Retake complete: {out}")
        print("This is your video with the new character performing the driving motion (retake).")
    except subprocess.CalledProcessError:
        print("Generate failed. Check model paths, safetensors, and that you have the SCAIL-2 checkpoint converted.")
        print("For fast serverless: run the Modal pre-cache first (modal run modal_app.py::download_scail_models).")

    # Bonus: basic MoCap note
    print("\nFor pure MoCap extraction from the driving video (separate from retake):")
    print("  cd SCAIL-Pose")
    print("  python NLFPoseExtract/process_animation_aio.py --subdir ../" + str(subdir) + " --e2e_mode")
    print("  # or v1_process_pose.py --use_align for 3D retarget mocap data")
    print("  # Outputs in subdir: rendered poses, masks, NLF 3D data you can use/export as mocap.")

if __name__ == "__main__":
    main()