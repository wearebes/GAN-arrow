# GAN-arrow 箭靶安全 ADA 实施计划

> 状态：仅计划，尚未实施
> 编写日期：2026-07-17
> 交接对象：Claude
> 仓库：`/Users/jcy/project/GAN-arrow`
> 核心原则：先证明增强后的图仍然是物理合理的箭靶照片，再允许训练；不得因为某个方法是论文默认项就不加审查地全部启用。

## 1. 本计划要解决的问题

当前仓库已经实现 `AdaBcgAugment`，包括水平翻转、90 度旋转、任意角度旋转、平移、各向同性和各向异性缩放、亮度、对比度、亮度轴反转、色相和饱和度变化。工程测试已通过，但真实图预览暴露出以下问题：

1. 箭靶可能横置或倒置，与当前拍摄场景不符。
2. 靶环可能因各向异性缩放变成椭圆。
3. 蓝、红、黄靶环可能因色相、亮度轴反转或强饱和度变化而失去真实语义。
4. 大幅几何变换配合 reflection padding 可能在边缘制造镜像或重复靶面。
5. 当前 `EXP-GAN-1024-030-ada-v1` 虽完成工程验证，但正式 3000 epoch 尚未开始，不能把“可运行”写成“增强有效”。
6. `030` 以旧的 `005` 为比较对象但计划跑 3000 epoch，真实图像暴露量并不匹配：
   - `005`：约 `100 images × 500 epochs = 50,000` 次真实图像暴露；
   - 当前 `030`：`80 images × 3000 epochs = 240,000` 次真实图像暴露。
   因此原计划不能作为严格的增强效果对照。

本计划把增强改成一个容易解释、可视检查、不会改变箭靶核心语义的 `target_safe_v1` 管线，并建立同数据、同训练预算的配对实验。

## 2. 实施边界

### 2.1 本次必须完成

1. 新增一个明确命名的 `target_safe_v1` 在线 ADA 管线。
2. 保留现有 `AdaController` 的自适应概率机制和 checkpoint 状态恢复能力。
3. 保留旧 `bgc` 代码，仅用于历史复现和诊断；正式新实验不得使用它。
4. 新增增强预览工具，并用真实训练图输出固定种子的对照图。
5. 补齐配置、单元测试、端到端 smoke、实验元数据和文档。
6. 建立无增强对照和安全 ADA 两个同预算实验规格。
7. 将每一阶段的继续条件写成显式 gate。

### 2.2 本次禁止事项

1. 不删除、清理或覆盖用户当前未提交改动。
2. 不修改 `dataset/v1_1024/train`、`val`、`test` 内的 PNG。
3. 不生成离线翻转、重着色或 Copy-Paste 数据集。
4. 不把 test/val 图用于训练、调节 ADA 或挑选 checkpoint。
5. 不启动原 `030` 的 3000 epoch 训练。
6. 不在没有用户明确批准的情况下启动 100 epoch 以上训练。
7. 不加入新的 GAN 架构、损失函数、目标先验、数据裁剪策略或第三方增强库。
8. 不把 pipeline smoke、单批梯度检查或预览图写成生成质量提升证据。

## 3. 锁定的 `target_safe_v1` 增强合同

以下数值是本计划的正式合同。Claude 不得自行扩大范围；如果实现上必须改变，先停下并在报告中说明，不要静默替换。

### 3.1 输入输出合同

- 输入：`NCHW`、RGB、浮点 tensor，训练范围 `[-1, 1]`。
- 输出：shape、dtype、device 与输入一致。
- `p = 0`：必须逐元素精确返回原 tensor，不得发生重采样或 clamp。
- `p = 1`：每张图都执行一次受限管线，但结果仍必须是合理的箭靶照片。
- 随机数必须受 `torch.manual_seed(...)` 控制，固定 seed 可重复。
- 整个管线必须可微；生成器通过增强和判别器反向传播时梯度必须有限。

### 3.2 唯一允许的变换

对被 Bernoulli mask 选中的每张图，执行一次组合几何变换，再执行轻微光度变换：

| 变换 | 锁定范围 | 物理含义 |
|---|---:|---|
| 平移 X | `[-0.02W, +0.02W]` | 模拟相机位置轻微变化 |
| 平移 Y | `[-0.02H, +0.02H]` | 模拟相机位置轻微变化 |
| 旋转 | `[-3°, +3°]` | 模拟轻微相机倾斜 |
| 等比例缩放 | `[0.97, 1.03]` | 模拟轻微距离变化 |
| 亮度 | 在 `[0,1]` 空间增加 `[-0.05, +0.05]` | 模拟自然光照小幅变化 |
| 对比度 | 围绕每张图自身 RGB 空间均值乘 `[0.95, 1.05]` | 模拟轻微曝光差异 |

实现细节：

1. 每张图使用一个 Bernoulli mask，`p` 表示该图是否走完整的安全管线，避免用户需要理解多个独立概率。
2. 几何变换应合成为一次 affine resampling，避免连续插值造成额外模糊。
3. 使用 bilinear sampling。
4. padding 使用 `border`，不得使用会制造镜像重复内容的 `reflection`，也不得用黑色 zero padding。
5. 光度变换前把 `[-1,1]` 映射到 `[0,1]`，完成后 clamp 到 `[0,1]`，再映射回 `[-1,1]`。
6. 缩放矩阵的正逆方向必须用单元测试验证观察到的缩放范围，不能只检查代码参数值。

### 3.3 明确禁止的变换

`target_safe_v1` 中不得出现：

- 水平或垂直翻转；
- 90 度旋转；
- 任意大角度旋转；
- 各向异性缩放；
- perspective、elastic 或 shear；
- luma flip；
- hue shift；
- saturation shift；
- grayscale；
- noise；
- blur/sharpen；
- cutout；
- mosaic、MixUp、Copy-Paste；
- 大幅 crop；
- 任何会覆盖箭杆、落点或改变靶环颜色类别的操作。

## 4. 代码改动设计

### 4.1 `model/ada_augment.py`

1. 保留 `AdaBcgAugment`，在 docstring 中明确标为 `legacy/reference diagnostic only`。
2. 新增 `AdaTargetSafeV1Augment`，只实现第 3 节合同。
3. 保留 `AdaController` 的以下行为不变：
   - `target = 0.6`；
   - `interval = 4`；
   - `speed_kimg = 500`；
   - 根据 real logits 的 sign statistic 上下调节 `p`；
   - state dict 可保存和恢复。
4. 新增一个明确的构建函数，例如：

   ```python
   def build_ada_pipe(name: str):
       ...
   ```

   只接受：

   - `target_safe_v1`：新正式管线；
   - `bgc`：旧诊断管线。

5. 未知管线名必须抛出 `ValueError`，不能退回静默默认值。

### 4.2 `model/train_gan.py`

1. 将 `TrainingConfig.ada_augpipe` 默认值改为 `target_safe_v1`。
2. 配置验证允许 `target_safe_v1` 和 `bgc`，但不得允许任意字符串。
3. CLI `--ada-augpipe` choices 同步为这两个值。
4. 用 `build_ada_pipe(config.ada_augpipe)` 替代硬编码的 `AdaBcgAugment()`。
5. 保持增强插入点不变：
   - real 进入 D 前；
   - detached fake 进入 D 前；
   - G step 的 fake 进入 D 前。
6. `metrics.json` 必须记录：
   - `augmentation.mode`；
   - `augmentation.pipeline`；
   - `ada_target`；
   - `ada_final_p`；
   - `ada_last_rt`；
   - `ada_updates`。
7. checkpoint 必须继续保存 `augmentation_state`，resume 后 `p`、pending counters 和 last `r_t` 不得重置。
8. 如果显式选择旧 `bgc`，训练启动日志必须打印清楚的 legacy diagnostic 提示，但不得中止旧实验复现。

### 4.3 新增 `model/preview_augmentation.py`

实现一个只读预览 CLI，直接调用正式增强类，禁止另外复制一份近似实现。

建议接口：

```bash
/opt/anaconda3/envs/gan/bin/python -m model.preview_augmentation \
  --input dataset/v1_1024/train \
  --pipeline target_safe_v1 \
  --probabilities 0,0.25,0.5,1.0 \
  --num-images 6 \
  --seed 42 \
  --out experiments/EXP-GAN-1024-031-ada-target-safe-v1/augmentation_review/preview_seed42.png
```

要求：

1. 只读取 train，不读取 test/val。
2. 固定选择 6 张训练图，至少覆盖正面、侧视、不同背景和可见箭杆。
3. 每行同一源图，每列为 original、`p=0.25`、`p=0.5`、`p=1.0`。
4. 图中写明源文件名、pipeline、p、seed。
5. 另写一个同名 JSON，记录输入路径、输入 hash、seed、p、输出 hash 和管线合同数值。
6. 不得向 dataset 目录写文件。

### 4.4 实验和文档文件

1. `experiments/EXP-GAN-1024-030-ada-v1/` 保留原样作为旧 BGC 工程验证证据。
2. 只允许在它的 `REPORT.md` 增加一条醒目标记：
   - full BGC 在领域审查中被否决；
   - formal run 从未开始；
   - 被 `target_safe_v1` 计划取代；
   - 不得删除或改写原验证事实。
3. 新建安全 ADA 实验目录：

   `experiments/EXP-GAN-1024-031-ada-target-safe-v1/`

   初始只放：

   - `experiment.yaml`；
   - `config.json`；
   - `REPORT.md`；
   - `augmentation_review/preview_seed42.png`；
   - `augmentation_review/preview_seed42.json`。

4. 在 `README.md` 中把 ADA 示例改为 `target_safe_v1`，同时说明 `bgc` 只保留作历史诊断。
5. 在 `dataset/v1_1024/README.md` 中明确：训练集仍为 real-only，所有随机增强只发生在判别器在线路径。
6. 在 `experiments/EXPERIMENT_SUMMARY.md` 中把 `030` 标为“engineering-only, superseded before formal training”，把 `031` 标为“preview/smoke pending 或 passed”，不得写结果提升。

## 5. 测试计划

### 5.1 `tests/test_train_gan.py`

至少新增以下测试：

1. `test_target_safe_p_zero_is_exact_identity`
2. `test_target_safe_preserves_shape_dtype_and_device`
3. `test_target_safe_is_differentiable_with_finite_gradients`
4. `test_target_safe_seed_is_reproducible`
5. `test_target_safe_p_one_changes_the_image`
6. `test_target_safe_translation_never_exceeds_two_percent`
7. `test_target_safe_rotation_never_exceeds_three_degrees`
8. `test_target_safe_scale_is_isotropic_and_within_three_percent`
9. `test_target_safe_does_not_swap_or_invert_color_channels`
10. `test_target_safe_synthetic_target_keeps_ring_hue_classes`
11. `test_target_safe_synthetic_target_remains_nearly_circular`
12. `test_target_safe_uses_no_reflection_padding`
13. `test_build_ada_pipe_rejects_unknown_pipeline`
14. `test_training_config_defaults_to_target_safe_v1`
15. `test_legacy_bgc_remains_explicitly_selectable`
16. `test_resume_restores_target_safe_ada_probability_and_pending_state`
17. 更新现有 one-step artifact test，使它显式使用 `target_safe_v1`。

测试不能只断言常量；必须对合成箭靶或已知坐标图检查观察到的输出行为。

### 5.2 `tests/test_experiment.py`

1. 将新的正式 spec 默认管线改为 `target_safe_v1`。
2. 验证 `experiment.yaml -> TrainingConfig` 后管线名没有丢失。
3. 保留 unknown/missing field、spec hash、train/test 隔离测试。
4. 增加 `processed_dir=dataset/v1_1024/test` 和 `val` 仍被拒绝的回归测试。
5. 验证旧 `bgc` spec 仍能加载，但不会成为新建实验的默认值。

### 5.3 预览验收测试

预览工具至少应有以下自动测试：

1. 输出 PNG 和 JSON 都存在；
2. JSON 中 input hash 与真实源文件一致；
3. `p=0` 单元格像素与原图一致（排除排版缩放区域，可直接比较增强前 tensor）；
4. 同 seed 二次生成 hash 相同；
5. 不在 train/test/val 下产生新文件。

### 5.4 统一测试命令

```bash
cd /Users/jcy/project/GAN-arrow
/opt/anaconda3/envs/gan/bin/python -m unittest \
  tests.test_train_gan \
  tests.test_experiment \
  tests.test_tracking \
  tests.test_cleanup_experiments
git diff --check
```

通过标准：0 failed、0 errors；warning 必须在交接报告列出，但 deprecation warning 可作为非阻塞项。

## 6. 人工预览 Gate A：代码完成后必须先停

Claude 生成 `preview_seed42.png` 后必须停止，不得继续训练，等待用户查看。

每一张 `p=1.0` 压力预览都必须同时满足：

1. 箭靶仍近似竖直，不得横放或倒置。
2. 靶环仍为圆形，不得出现明显椭圆畸变。
3. 蓝、红、黄颜色类别仍然清楚。
4. 箭杆和落点没有因增强被遮挡或裁掉。
5. 不出现第二个镜像靶面或重复背景主体。
6. 不出现黑边、反射拼接边或明显插值破损。
7. 整体变化能用“轻微机位和光照变化”解释。

任意一条失败：Gate A 不通过。Claude 只能修复实现并重新生成预览，不能通过调低展示概率隐藏问题，因为 `p=1` 是合同压力测试。

## 7. 工程 Smoke Gate B

只有用户批准 Gate A 后才能进行。

### 7.1 Dry check

```bash
/opt/anaconda3/envs/gan/bin/python -m model.run \
  experiments/EXP-GAN-1024-031-ada-target-safe-v1/experiment.yaml \
  --check --mode offline
```

必须确认：

- 只读 `dataset/v1_1024/train`；
- 图像数为 80；
- `augmentation_mode=ada`；
- `ada_augpipe=target_safe_v1`；
- 设备和 G48/D16 配置正确；
- 没有现存 writer 或 resume 状态冲突。

### 7.2 一步 smoke

一步 smoke 必须使用单独的 smoke 实验目录或临时目录，不得把 smoke artifact 写入正式 `031` 目录。为强制执行变换，smoke 可临时使用 `ada_p_initial=1.0` 和 `max_steps=1`，但这两个值不得回写正式 spec。

必须产出并验证：

- finite D/G loss；
- finite G/D gradients；
- `ada_history.csv`；
- `metrics.json` 中 pipeline 为 `target_safe_v1`；
- checkpoint 中有 `augmentation_state`；
- resume smoke 后 ADA state 连续，不从 0 重置。

Smoke 通过仅表示工程链路可用，不表示图像质量改善。

## 8. 科学比较设计

现有 `005` 只能作为历史参考，不能作为 `target_safe_v1` 的唯一严格对照，因为数据预处理、train 数量和增强策略不同。

正式比较必须在 `dataset/v1_1024/train` 上建立配对实验，除增强外所有字段相同：

| 字段 | 无增强对照 | 安全 ADA |
|---|---|---|
| Dataset | `dataset/v1_1024/train` | 相同 |
| Prepared images | 80 | 80 |
| Image size | 1024 | 1024 |
| G/D | G48 / D16 | 相同 |
| Loss | BCEWithLogits | 相同 |
| G LR / D LR | `5e-4 / 1e-4` | 相同 |
| EMA | `0.995` | 相同 |
| Seed | 42 | 相同 |
| Grad accumulation | 16 | 相同 |
| Epochs | 625 | 625 |
| Real-image exposures | 50,000 | 50,000 |
| Augmentation | `none` | `ada + target_safe_v1` |

建议在 Gate A/B 通过后再建立：

- `EXP-GAN-1024-032-v1-noaug-control-625`
- `EXP-GAN-1024-033-v1-ada-target-safe-625`

不得直接从 100 epoch canary 推断正式质量结论，也不得把两个实验放在同一 MPS 设备上并发运行。

## 9. 训练 Gate C 与停止条件

正式配对训练必须另行获得用户明确批准。即使获批，也按以下阶段执行：

1. 先分别运行到 epoch 100，停止并审查；
2. Gate C1 通过后运行到 epoch 300，再停止并审查；
3. Gate C2 通过后才允许完成 epoch 625；
4. 不创建或启动 3000 epoch 版本。

每个审查点必须比较相同 fixed noise、相同 seed 的样本，并报告：

- `ada_p` 和 `r_t` 历史；
- D(real)、D(fake)、G/D loss 的趋势；
- 是否延缓判别器完全分离；
- 靶环清晰度；
- 是否出现可辨识箭杆；
- 是否出现可辨识落点；
- 背景和机位多样性；
- 是否出现增强泄漏：旋转、颜色偏移、边缘复制、椭圆靶环；
- 与无增强对照的同 epoch、同 exposure 对照图。

立即停止条件：

1. 任何生成图出现系统性倒置、异常色环或重复镜像结构；
2. `ada_p` 上升但 D/G 分离没有改善；
3. 生成器 loss 持续恶化且 fixed-noise 视觉没有改善；
4. 只学到平滑靶面，箭杆和落点仍没有进展；
5. NaN/Inf、checkpoint 损坏或 resume 不连续；
6. 数据路径或 spec hash 与批准版本不一致。

## 10. 结果判定规则

“安全 ADA 有效”必须同时满足：

1. 与无增强对照相比，判别器分离明显延后；
2. 同等 50,000 暴露量下视觉结果不差于对照；
3. 箭杆或落点细节至少有可重复的改善，而不是只改善靶环；
4. 没有观察到旋转、色彩、形状或边缘复制泄漏；
5. 多样性没有因增强而进一步下降。

以下情况不能称为成功：

- 测试通过；
- smoke 可运行；
- loss 有限；
- `ada_p` 正常变化；
- 靶环更鲜艳；
- 判别器准确率更高；
- 个别样本偶然出现类似箭杆的纹理。

## 11. Claude 执行顺序

严格按以下顺序执行，不得跳步：

1. 读取本计划、`model/ada_augment.py`、`model/train_gan.py`、`model/experiment.py`、`model/run.py`、相关 tests 和 `030` 报告。
2. 记录 `git status --short`，只识别当前状态，不清理任何文件。
3. 实现 `AdaTargetSafeV1Augment` 和构建函数。
4. 接入配置、CLI、metrics、checkpoint/resume。
5. 实现预览工具。
6. 先写/更新测试，再运行第 5.4 节完整测试。
7. 生成固定预览及 JSON manifest。
8. 更新 `030` 的 superseded 说明，创建 `031` 的计划元数据和报告。
9. 更新 README 和实验摘要，所有措辞保持“engineering validated / formal not run”。
10. 运行 `git diff --check`，检查实际 diff 没有越界。
11. 停在 Gate A，向用户展示预览和测试结果。
12. 未获用户批准，不进行 Gate B/C，不启动长训练，不提交或推送 Git。

## 12. Claude 交接报告模板

Claude 完成 Gate A 后应按此格式回复：

```markdown
## 实施状态
- target_safe_v1：已完成 / 未完成
- 正式训练：未启动

## 实际改动
- 文件：...
- 允许的变换和精确范围：...
- 明确未启用的变换：...

## 验证
- 单元测试：X tests, 0 failed
- p=0 identity：pass/fail
- p=1 physical-safety stress：pass/fail
- gradient：pass/fail
- resume ADA state：pass/fail

## 人工审查
- 预览图：绝对路径
- manifest：绝对路径
- 发现的问题：...

## 证据边界
- 这里只证明增强实现和物理合理性。
- 没有生成质量提升结论。
- 没有启动 100/300/625/3000 epoch 训练。

## 等待用户决定
- 是否批准进入一步 smoke Gate B。
```

## 13. 完成定义

本实施任务只有在以下条件全部满足时才算完成：

- 新管线名称和合同明确；
- 禁止变换在代码路径中不可达；
- p=0、p=1、梯度、颜色、形状、范围和 resume 都有自动测试；
- 真实训练图预览通过 Gate A；
- train/test 隔离保持不变；
- `030` 没有被误报为正式结果；
- 新实验没有偷偷继承 `bgc`；
- 原 3000 epoch 计划已被明确冻结；
- 没有启动任何长训练；
- 没有覆盖、删除或清理用户现有工作。

## 14. 参考依据

- StyleGAN2-ADA 论文：<https://arxiv.org/abs/2006.06676>
- NVIDIA 官方实现：<https://github.com/NVlabs/stylegan2-ada-pytorch>

这些来源支持 ADA 的自适应判别器增强思想，但不替代本仓库对箭靶颜色、方向、几何和箭杆/落点语义的领域审查。
