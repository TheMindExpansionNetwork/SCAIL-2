#!/usr/bin/env python3
"""
DeepSeek-based prompt enhancer for SCAIL-2 (replaces or augments Gemini version).
Uses DeepSeek (extra credits) for fast, high-quality prompt generation/enhancement.
Supports text-based enhancement for unique cool animations (e.g., dystopian forager character).

Usage example for unique test:
python prompt_enhancer_deepseek.py \
  --instruction "turn the walking footage into a cool unique dystopian sonic forager character in rainy neon cyberpunk megacity at night" \
  --video_caption "group of people walking on a city street" \
  --ref_description "a lone character with glowing implants and tattered tech coat" \
  --output enhanced_prompt.txt

Then use the output content as --prompt in generate.py (with preprocessed pose/mask from the free Pexels walking footage as driving).
"""

import argparse
import os
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # fast; use "deepseek-reasoner" for more thinking if needed

SYSTEM_PROMPT = """You are an expert prompt engineer for SCAIL-2, a state-of-the-art end-to-end character animation model based on Wan video diffusion.

SCAIL-2 takes:
- A reference character image (the identity to animate)
- A driving pose/mask video (the motion)
- A positive text prompt describing the OUTPUT video (the scene, style, clothing details, atmosphere, interactions)

Rules for the prompt you generate:
- Describe the final animated video in rich, cinematic detail.
- Include the character's exact appearance, clothing, accessories, and any objects they interact with.
- Specify lighting, weather, environment, mood, and camera feel.
- Keep it positive and descriptive (no negative words like "blurry" or "low quality").
- Make it unique and cool when requested (e.g., dystopian, cyberpunk, neon, atmospheric).
- Aim for 80-150 words, natural flowing English.
- For replacement/animation from driving footage: describe the character performing the actions in the driving video but in the new unique scene/style.

Output ONLY the prompt text, nothing else."""

def enhance_prompt(instruction: str, video_caption: str = "", ref_description: str = "", api_key: str = None) -> str:
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set. Export it or pass via --api-key.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    user_content = f"Instruction: {instruction}\n"
    if video_caption:
        user_content += f"Driving video content/caption: {video_caption}\n"
    if ref_description:
        user_content += f"Reference character description: {ref_description}\n"
    user_content += "\nGenerate the detailed positive prompt for SCAIL-2 generate.py --prompt."

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.7,
        max_tokens=400,
        stream=False,
    )

    prompt = response.choices[0].message.content.strip()
    # Clean any quotes or extra
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    return prompt

def main():
    parser = argparse.ArgumentParser(description="DeepSeek prompt enhancer for SCAIL-2 (fast/light alternative to Gemini)")
    parser.add_argument("--instruction", required=True, help="Short instruction for the unique cool animation, e.g. 'make the character a dystopian neon forager in rainy cyber city'")
    parser.add_argument("--video_caption", default="", help="Optional caption or description of the driving footage (e.g. from Pexels walking video)")
    parser.add_argument("--ref_description", default="", help="Optional description of the reference character image")
    parser.add_argument("--output", default="enhanced_prompt.txt", help="Output file for the generated prompt")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key (or use DEEPSEEK_API_KEY env)")
    args = parser.parse_args()

    prompt = enhance_prompt(
        instruction=args.instruction,
        video_caption=args.video_caption,
        ref_description=args.ref_description,
        api_key=args.api_key,
    )

    with open(args.output, "w") as f:
        f.write(prompt)

    print(f"Enhanced prompt saved to {args.output}")
    print("\n=== GENERATED PROMPT (use with --prompt in generate.py) ===")
    print(prompt)
    print("=============================================================")

if __name__ == "__main__":
    main()