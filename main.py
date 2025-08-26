# =========================
# main.py — CSV仅三列 + YAML其余参数
# =========================
# 先把 Hugging Face 缓存固定到 D 盘（必须在任何 import 之前）
import os
os.environ.setdefault("HF_HOME", r"D:\HF_models")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\HF_models\hub")
os.environ.setdefault("DIFFUSERS_CACHE", r"D:\HF_models\hub")
# 不再设置 TRANSFORMERS_CACHE，避免 FutureWarning

import argparse
import csv
import time
import yaml
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid


# -------- YAML 配置 --------
def load_cfg(path: str) -> dict:
    """
    读取 config.yaml；不存在也能跑（使用内置默认）。
    你可以在同目录放一个 config.yaml 覆盖默认值。
    """
    cfg = {
        "cache_dir": r"D:\HF_models\hub",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "dtype": "float16",   # float16 / bfloat16 / float32
        "variant": "fp16",    # CPU 时会自动忽略
        "use_safetensors": True,
        "defaults": {
            "out_dir": "./output",
            "width": 512,
            "height": 512,
            "steps": 25,
            "guidance_scale": 7.0,
            "strength": 0.75,
            "negative_prompt": "",
            "seed": None
        }
    }
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if k == "defaults":
                cfg["defaults"].update(v or {})
            else:
                cfg[k] = v
    return cfg


# -------- Pipeline 构建（只加载一次）--------
def build_pipelines(cfg: dict):
    cache_dir = cfg["cache_dir"]
    model_id = cfg["model_id"]
    dtype_str = str(cfg.get("dtype", "float16")).lower()
    variant = cfg.get("variant", "fp16")
    use_safetensors = bool(cfg.get("use_safetensors", True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU 不支持 float16/bfloat16：自动回退 float32，并忽略 fp16 变体
    if device == "cpu" and dtype_str in ("float16", "bfloat16"):
        print("[info] CUDA 不可用，自动切为 float32 并忽略 variant=fp16。")
        dtype_str, variant = "float32", None

    torch_dtype = getattr(torch, dtype_str)

    params = dict(
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
        cache_dir=cache_dir,
        resume_download=True,
    )
    if variant:
        params["variant"] = variant

    # 注意：使用“位置参数”传入模型路径，兼容你的 diffusers 版本
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(model_id, **params).to(device)
    # 共享权重构建 img2img，不会重复下载
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)

    print(f"[env] device={device}  torch={torch.__version__}  cuda={torch.version.cuda}")
    return pipe_t2i, pipe_i2i, device


# -------- 单条任务 --------
def run_one(pipe_t2i, pipe_i2i, device,
            prompt: str, image_path: str | None, name: str,
            out_dir: str, width: int, height: int, steps: int,
            guidance_scale: float, strength: float, seed: int | None,
            negative_prompt: str = "") -> str:

    os.makedirs(out_dir, exist_ok=True)
    init_image = load_image(image_path).resize((width, height)) if image_path else None
    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None

    t0 = time.time()
    if init_image is not None:
        img = pipe_i2i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=width, height=height,
            generator=generator
        ).images[0]
    else:
        img = pipe_t2i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=width, height=height,
            generator=generator
        ).images[0]
    dt = time.time() - t0

    out_path = os.path.join(out_dir, f"{name}_output.png")
    img.save(out_path)

    print(f"[OK] {name} -> {out_path}  用时 {dt:.2f}s")
    return out_path


# -------- 批处理（CSV 仅三列：name,prompt,image）--------
def run_batch(csv_path: str, cfg: dict):
    if not os.path.exists(csv_path):
        raise SystemExit(f"❌ 找不到 CSV：{csv_path}")

    # 读取 YAML 默认
    d = cfg["defaults"]
    out_dir = d["out_dir"]
    width, height = int(d["width"]), int(d["height"])
    steps = int(d["steps"])
    guidance_scale = float(d["guidance_scale"])
    strength = float(d["strength"])
    negative_prompt = str(d.get("negative_prompt", ""))
    seed_default = d.get("seed", None)
    if isinstance(seed_default, str) and seed_default.isdigit():
        seed_default = int(seed_default)

    pipe_t2i, pipe_i2i, device = build_pipelines(cfg)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        need = {"name", "prompt", "image"}
        miss = need - set(headers)
        if miss:
            raise SystemExit(f"❌ CSV 需要三列：name,prompt,image（缺少: {', '.join(sorted(miss))}）")

        for i, row in enumerate(reader, 1):
            name = (row.get("name") or f"job_{i}").strip()
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                print(f"[SKIP] 第{i}行：prompt 为空")
                continue
            image_path = (row.get("image") or "").strip() or None

            try:
                run_one(
                    pipe_t2i, pipe_i2i, device,
                    prompt=prompt, image_path=image_path, name=name,
                    out_dir=out_dir, width=width, height=height, steps=steps,
                    guidance_scale=guidance_scale, strength=strength,
                    seed=seed_default, negative_prompt=negative_prompt
                )
            except Exception as e:
                print(f"[ERR] {name}: {e}")


# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="SDXL（CSV仅 name,prompt,image；其余参数走 YAML）")
    p.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    # 批处理：只需要传 CSV 路径
    p.add_argument("--batch", default=None, help="CSV 路径（仅三列 name,prompt,image）")
    # 单条（可选）：不批处理时也可单张运行（其余默认从 YAML 取）
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--name", default="job")
    return p.parse_args()


# -------- 入口 --------
if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.batch:
        run_batch(args.batch, cfg)
    else:
        if not args.prompt or not args.prompt.strip():
            raise SystemExit("❌ 需要提供 --prompt 或使用 --batch jobs.csv")
        d = cfg["defaults"]
        pipe_t2i, pipe_i2i, device = build_pipelines(cfg)
        run_one(
            pipe_t2i, pipe_i2i, device,
            prompt=args.prompt.strip(),
            image_path=args.image,
            name=args.name,
            out_dir=d["out_dir"],
            width=int(d["width"]),
            height=int(d["height"]),
            steps=int(d["steps"]),
            guidance_scale=float(d["guidance_scale"]),
            strength=float(d["strength"]),
            seed=d.get("seed", None),
            negative_prompt=str(d.get("negative_prompt", ""))
        )
