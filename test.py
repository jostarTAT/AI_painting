from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch
from PIL import Image
import os
import time

# 加载 Stable Diffusion XL 模型
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")  

# 创建 image2image pipeline
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

# 设置本地路径
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)


creatives = [
{
    "name": "cartoon_happy_walker",
    "image_path": "./input/17.jpg",
    "prompt": "Cartoon-style cheerful character walking with a big smile, swinging arms and bouncing steps, simple round face with expressive eyes, bright colorful outfit, animated and playful vibe, clean background with blue sky or park path, fun and energetic atmosphere"
}



]

# 图像处理和保存
all_init_images = []
all_output_images = []

for creative in creatives:
    print(f"开始处理：{creative['name']}")
    init_image = load_image(creative["image_path"])
    init_image = init_image.resize((512, 512))  # 或更小尺寸

    start = time.time()

    output = pipeline(
        prompt=creative["prompt"],
        image=init_image,
        strength=0.8,
        guidance_scale=10.5
    ).images[0]

    duration = time.time() - start
    print(f"{creative['name']} 生成完成，用时 {duration:.2f} 秒")

    output_path = os.path.join(output_dir, f"{creative['name']}_output.png")
    output.save(output_path)

    all_init_images.append(init_image)
    all_output_images.append(output)

# 生成对比图网格
grid = make_image_grid(
    all_init_images + all_output_images,
    rows=2,
    cols=len(creatives)
)

grid.save(os.path.join(output_dir, "comparison_grid.png"))
