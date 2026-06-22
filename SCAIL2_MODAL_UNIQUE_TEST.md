# SCAIL-2 Fork + Modal Serverless + Pre-cache Snapshot + DeepSeek + Unique Cool Test

**Fork**: https://github.com/TheMindExpansionNetwork/SCAIL-2 (wan-scail2 branch, full history preserved).

**Test online free footage (driving source)**: 
- Pexels "Video Of People Walking" (royalty-free, CC0, perfect for character animation test).
- Direct page: https://www.pexels.com/video/video-of-people-walking-855564/
- Download the HD video from the page (or use their download button). This is your driving footage.
- Then run SCAIL-Pose preprocessing (from the submodule) on it to produce:
  - rendered_v2.mp4 (pose/driving)
  - rendered_mask_v2.mp4 (mask)
- Pair with a **ref.jpg** + **ref_mask.jpg** of your cool unique character (e.g., a dystopian sonic forager with glowing implants, tattered tech coat — generate or photograph one).

**"Change it into something really cool unique" (to see if it works)**:
- Use the walking footage as driving motion.
- Your ref character (the "identity").
- DeepSeek (extra credits, fast/light) generates a unique dystopian prompt.
- Run SCAIL-2 inference with **light/fast distilled settings** (low steps=20, fast solver path, optimized res) for speed.
- Result: the ref character animated with the walking motion but placed in a **rainy neon cyberpunk megacity at night, atmospheric, mysterious cool vibe** (or any unique scene you instruct).

Example unique instruction for DeepSeek:
"turn the walking footage into a cool unique dystopian sonic forager character in rainy neon cyberpunk megacity at night with glowing blue implants and tattered high-tech coat, atmospheric rain reflections, cinematic"

This tests the full pipeline end-to-end with creative control.

## Pre-cache Models in Volume + Snapshot (for fast serverless)

Everything pre-installed in the Modal image (requirements.txt + openai for DeepSeek + HF tools + ffmpeg).

The heavy SCAIL-2 checkpoint (Wan VAE + T5 + model weights) lives in a persistent Modal Volume. 

**The "snapshot"**: Run the download function **once**. After it succeeds, the Volume contains the pre-cached models (and convert if run). Combined with the pre-built Image, this is your fast serverless snapshot — cold starts are quick because models are already there, no re-download on every call.

## Files Added/Changed in the Fork
- `prompt_enhancer_deepseek.py` — DeepSeek replacement/augmentation for prompt enhancement (light/fast, uses your credits, text-based for unique scenes).
- `modal_app.py` — Full Modal serverless app with:
  - Pre-cache download function (the build/snapshot step).
  - DeepSeek prompt enhancer function.
  - Inference function with **light/fast distilled settings** (sample_steps=20, fast solver, comments for speed).
  - All deps pre-installed in image.
- This README (SCAIL2_MODAL_UNIQUE_TEST.md).

## How to Set Up & Run (Modal)

1. Install Modal CLI locally (one time):
   ```
   pip install modal
   modal token new   # follow login
   ```

2. Create DeepSeek secret (uses your extra credits):
   ```
   modal secret create deepseek DEEPSEEK_API_KEY=sk-...
   ```

3. Pre-cache the models (the build + snapshot step — run once, can take 10-30+ min depending on size):
   ```
   modal run modal_app.py::download_scail_models
   ```
   This populates the `scail2-models` volume. This **is** the snapshot for fast future runs.

4. Test DeepSeek prompt enhancer (unique cool prompt):
   ```
   modal run modal_app.py::enhance_prompt_deepseek \
     --instruction "turn the walking footage into a cool unique dystopian sonic forager character in rainy neon cyberpunk megacity at night with glowing blue implants and tattered high-tech coat, atmospheric rain reflections, cinematic" \
     --video_caption "people walking on a city street" \
     --ref_description "lone character with glowing implants and tech coat"
   ```
   (Set DEEPSEEK_API_KEY env or use the secret.)

5. Run the unique cool animation (after you have the 4 inputs from the Pexels footage + your ref + the enhanced prompt above):
   ```
   modal run modal_app.py::run_scail_inference \
     --image /path/to/your/ref.jpg \
     --mask_image /path/to/your/ref_mask.jpg \
     --pose /path/to/rendered_v2.mp4 \
     --mask_video /path/to/rendered_mask_v2.mp4 \
     --prompt "THE ENHANCED PROMPT FROM STEP 4"
   ```
   Output: a short unique cool dystopian character animation video (the ref character walking in the rainy neon cyber city using the driving motion).

   The function uses light/fast distilled settings for speed.

## Local Smoke / Quick Test (no GPU)

```bash
# Test the DeepSeek enhancer (needs DEEPSEEK_API_KEY)
python prompt_enhancer_deepseek.py \
  --instruction "turn the walking footage into a cool unique dystopian sonic forager..." \
  --video_caption "people walking" \
  --ref_description "glowing implants tech coat character" \
  --output /tmp/test_prompt.txt

cat /tmp/test_prompt.txt
```

The forked repo + new files are ready. The Modal app pre-caches everything, uses DeepSeek for the prompt part, and runs inference with light/fast distilled settings so you can iterate quickly on unique cool animations.

## Verification Gates (per pipeline patterns)
- [x] Forked to TheMindExpansionNetwork/SCAIL-2
- [x] Local clone + submodule init verified
- [x] DeepSeek prompt enhancer written and syntax-checked
- [x] Modal app written with pre-cache volume, image (pre-installed), DeepSeek function, inference with light/fast settings
- [x] Free test footage identified (Pexels walking)
- [x] Unique cool test case defined (dystopian sonic forager in neon rain cyber city)
- [ ] Run download_scail_models (pre-cache/snapshot) — do this next
- [ ] Run enhancer + inference with real inputs from the Pexels footage + your cool ref image

Once the pre-cache build is done, the whole thing is a fast serverless snapshot. Ready to make really cool unique character animations.

Ties to fast inference, DeepSeek credits, pre-cached volume snapshot, and the free footage test. 

Let’s go — run the download first!