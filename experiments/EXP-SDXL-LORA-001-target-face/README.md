# EXP-SDXL-LORA-001-target-face

状态：进行中（2026-07-18 启动）

用 80 张箭靶照片对 SDXL 1.0 base 做 DreamBooth-LoRA 微调，替代从零训练 GAN 的路线。
本目录自包含：原始 `dataset/` 目录只读，未做任何修改。

## 数据划分

- `data/train/`：72 张（从 `dataset/v1_1024/train` 复制，每 10 张抽走 1 张后剩余部分）
- `data/val/`：8 张（每 10 张抽第 1 张：IMG_4014/4025/4047/4058/4069/4080/4096/4107）
- test：沿用 `dataset/v1_1024/test` 的 20 张，只在最终报告对比时只读引用
- `data/manifest.sha256`：全部副本的校验和

## 方法

- 基座：stabilityai/stable-diffusion-xl-base-1.0（fp16 权重文件，fp32 训练）
- 冻结：VAE、两个文本编码器、UNet 主干全部冻结
- 可训练：仅 LoRA（UNet 全部注意力层的 to_q/to_k/to_v/to_out.0，r=8, alpha=8, dropout 0.1）
- 触发词：`zqx archery target`；训练 prompt 统一为
  `a photo of zqx archery target on a stand outdoors`（v1 简化：单 prompt，未做逐图 caption）
- 数据增强：仅水平翻转（训练集缓存了翻转副本，共 144 个潜变量条目）
- 潜变量与文本嵌入离线预缓存（`outputs/cache/`），训练时不加载 VAE/文本编码器
- 验证损失：8 张 val 图 × 固定时间步 {100,300,500,700,900} × 固定噪声，逐 checkpoint 可比

## 运行

1. Pilot：768px、1000 步（batch 1 × accum 4）、val 每 50 步、ckpt 每 100 步
2. 正式：1024px、1500 步（pilot 通过后）

## 目录

- `scripts/`：precompute_latents.py / train_lora.py / sample.py / val_prompts.txt
- `outputs/cache/`：潜变量与文本嵌入缓存
- `outputs/runs/<run>/`：train_log.csv、val_log.csv、run_config.json、checkpoints/
- `outputs/samples/`：采样对比图（LoRA vs base）
- `outputs/eval/`：损失曲线、记忆化检查

环境：conda env `sdxl`（torch 2.13.0 / diffusers 0.39.0 / peft 0.19.1），
模型权重缓存在 `~/.cache/huggingface/`。训练命令均带 `PYTORCH_ENABLE_MPS_FALLBACK=1` 与 `caffeinate -i`。
