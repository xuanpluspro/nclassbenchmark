# 多分类植物语义分割对比实验

参考 [vegann-seg-benchmark](https://github.com/xuanpluspro/vegann-seg-benchmark)，将二分类流水线改为任意 C 类互斥语义分割。RiceSEG（6 类）和 GWFSS（4 类）分别训练，分别汇总；不能直接混合两套类别标签。

## 数据与类别

仓库已预建 `data/RiceSegnclass/` 和 `data/GWFSSnclass/`。在 AutoDL 的仓库根目录执行命令，把两个数据集分别上传成：

```text
nclassbenchmark/
  data/
    RiceSegnclass/
    GWFSSnclass/
```

例如仓库位于 `/root/autodl-tmp/nclassbenchmark`，RiceSEG 的完整路径就是 `/root/autodl-tmp/nclassbenchmark/data/RiceSegnclass`。每个数据集目录内包含 JPEGImages、SegmentationClass、ImageSets。Git 只跟踪说明文件和目录占位，上传的数据集不会被提交。


采用已整理好的 VOC 目录：

```text
RiceSegnclass/（GWFSSnclass 同结构）
  JPEGImages/<id>.jpg
  SegmentationClass/<id>.png
  ImageSets/train.txt
  ImageSets/val.txt
  ImageSets/test.txt
```

优先读取 `ImageSets/`，也兼容 `ImageSets/Main/`、`ImageSets/Segmentation/`。列表每行是不含扩展名的文件名。训练、验证、测试 ID 必须互不重叠。建议先对原始图像划分，再切 patch，避免同一原图的 patch 跨集合泄漏；地块/区域/连续拍摄有相关性时按组划分更可靠。

Mask 必须是单通道类别 ID 或 P 模式 PNG：像素值 `0..C-1`；`255` 为忽略像素。P 模式直接读取索引，不转换 RGB，也不除以 255。彩色可视化图不能代替索引 mask。图像用双线性、标签用最近邻 resize 到统一输入尺寸。

- `configs/riceseg.json`：0 背景，1 绿色植被，2 衰老植被，3 稻穗，4 杂草，5 浮萍，依据 RiceSEG 论文标注定义。
- `configs/gwfss.json`：4 类；论文的语义类别是背景、麦穗、茎、叶，但正文没有明确导出 mask 的数值对应顺序。因此保留 `class_1..3` 中性名称，请核对自己的 mask 导出映射后修改名称及 background_index。**不得根据论文表格行顺序猜测像素 ID。** 如果你的 ID 不连续，训练前应正确重映射；这套代码不会猜颜色或重排标签。

本仓库使用你提供的划分，不自动生成或覆盖。RiceSEG 原文采用 8:2 训练/测试，GWFSS 比较随机与区域划分；自定义 8:1:1 与原文实验协议不同，结果不能直接声称严格复现原论文。

## 安装与预训练

先按服务器 CUDA 安装兼容的 PyTorch/torchvision（推荐 PyTorch >=2.3），然后：

```bash
pip install -r requirements.txt
python download_pretrained.py --output pretrained
```

下载慢时可在可联网电脑执行下载脚本，再上传整个 `pretrained/`。其中使用固定版本的四个编码器权重。默认 `--weights local`，加载失败直接报错，不会静默改成随机初始化。也支持 `--weights imagenet` 在线下载、`--weights none` 从头训练。

## 训练

默认输入 512、batch 32、30 epoch、AdamW、学习率 1e-4、余弦衰减；训练 CUDA AMP，评价 FP32。损失为多分类 CrossEntropy + soft Dice，255 不参与损失。Dice 只在当前批次 GT 中出现的类别上取平均，CE 仍约束所有有效像素的全部类别输出。

先检查原始标签与类别像素统计：

```bash
python audit_dataset.py --data-root data/RiceSegnclass --dataset-config configs/riceseg.json
```

RiceSEG 依次训练 experiments.json 中 10 个模型：

```bash
python batch_train.py --dataset-name RiceSEG --run-name pretrained_ep30_bs32 --data-root data/RiceSegnclass --dataset-config configs/riceseg.json --weights local --pretrained-dir pretrained --size 512 --batch-size 32 --epochs 30
```

GWFSS（先核对配置中的标签名称与 ID）：

```bash
python batch_train.py --dataset-name GWFSS --run-name pretrained_ep30_bs32 --data-root data/GWFSSnclass --dataset-config configs/gwfss.json --weights local --pretrained-dir pretrained --size 512 --batch-size 32 --epochs 30
```

模型为 U-Net、U-Net++、MAnet、FPN、PSPNet、DeepLabV3+（ResNet34），U-Net、DeepLabV3+（MobileNetV2），SegFormer-B0/B2。它们是工程对比配置，不等同于两篇论文原始 baseline 的配置。修改 experiments.json 可增加 SMP 支持的组合。

单模型与任意类别数：

```bash
python train.py --data-root /path/to/VOC2012 --dataset-config configs/riceseg.json --arch Unet --encoder resnet34 --weights local --output outputs/rice_unet
python train.py --data-root /path/to/VOC2012 --num-classes 4 --class-names background leaf stem head --weights local --output outputs/custom
```

类别顺序必须对应你自己的像素 ID。显存不足请对所有模型统一减小 batch，或记录实际差异。三次重复可添加 `--seeds 42 43 44`，每次结果单独保存，不能只挑最好的种子。

恢复批量训练：同一条命令追加 `--resume`，保持 run-name 与训练配置一致。重新实验使用不同 run-name。没有 test 划分可设置 `--test-split none`，此时只输出验证结果，不能把它标成测试结果。

## 指标定义与推荐报告方式

累计整个评估集的混淆矩阵（行=真实类别，列=预测类别），然后计算指标；不是先算每张图再平均。

- `IoU_c = TP_c / (TP_c + FP_c + FN_c)`。
- `Acc_c = TP_c / (TP_c + FN_c)`，等于类别召回率；**不使用含大量 TN 的 one-vs-rest accuracy**。
- `Dice_c = 2 TP_c / (2 TP_c + FP_c + FN_c)`。
- 标准 `miou/macc` 包含背景；标准 `dice` 仅平均前景。旧 `mdice` 是包含背景的补充值。
- `PixAcc`：全部有效像素中预测正确的比例，与 mAcc 不同。
- `miou_fg/macc_fg/mdice_fg`：仅从平均中排除背景类别，仍保留背景误判造成的 FP/FN。
- 整个评估集某类 GT 和预测都不存在，IoU/Dice 为 null，不参与均值；GT 不存在但出现误预测，IoU 为 0。GT 不存在时 Acc 为 null。输出参与平均的类别数，审查测试集是否缺类。
- 区域指标和 Boundary F1 保存为 0..1；HD95/ASSD 为像素距离，不能乘以 100。

建议主表：模型、mIoU、前景 Dice、每类 IoU、Params(M)、GFLOPs、FPS。另存逐类别 IoU/Acc/Dice/Precision 和混淆矩阵。少数类性能不能仅靠均值判断，尤其是 GWFSS 的茎、RiceSEG 的衰老植被/杂草/浮萍。报告背景是否参与均值，不能混用两种统计口径。

## 输出

```text
outputs/<dataset>/<run>/
  results.csv
  <model>_seed42/
    config.json
    console.log
    history.csv
    best.pth
    last.pth
    metrics.json
    benchmark.json
    val_per_class.csv
    test_per_class.csv
    val_confusion_matrix.csv
    test_confusion_matrix.csv
```

按验证集 mIoU 选 best；所有 epoch 结束后加载该 checkpoint，在测试集评价一次。测试结果不参与选模型。config 与 checkpoint 记录类别、种子、训练超参数与随机状态。history 只保存标量，逐类最终结果保存在独立 CSV，避免把嵌套数据塞进汇总单元格。

速度测量使用 eager PyTorch FP32、关闭 TF32、batch=1、默认 512x512，50 次预热、200 次 CUDA 计时。只测模型前向，不含读取、预处理与后处理，非 TensorRT。latency 毫秒，FPS=1000/平均 latency。记录硬件和软件版本，只有同设备同输入同精度才能公平比较。CPU 环境不填写 GPU FPS。

FLOPs 使用 fvcore（一次乘加算一个操作）。若存在不支持算子，只输出 `counted_gflops` 下界与 unsupported_ops，完整 `gflops` 留空，避免把漏算值当真实总量。Params 始终记录。

## 验证

```bash
python -m unittest discover -s tests -v
```

包含手算混淆矩阵、缺失类/误预测、忽略标签、跨 batch 累计、P 模式读取与多分类损失反向传播测试。CPU 合成数据验证只能说明工程链路，真实精度和 CUDA 显存/速度需要在服务器实际训练验证。



## 统一指标与结果格式（bimseg-segmentation-v2）

三个仓库使用内容完全一致的 `segmentation_metrics.py`。区域指标先累计整个评估集的混淆矩阵，行=真实类别、列=预测类别；不对逐图 IoU/Dice 求平均。

| 标准字段 | 计算口径 |
|---|---|
| `miou` / `macc` | 全部类别（包含背景）的 IoU / Recall 宏平均；所有任务按验证集 mIoU 选择最佳权重 |
| `foreground_miou` | 仅前景各类别 IoU 的宏平均；二分类即前景 IoU |
| `union_foreground_iou` | 合并所有非背景类别后的 IoU，不能反映前景内部错分 |
| `dice` / `precision` / `recall` | 全数据集统计，前景类别宏平均 |
| `dice_all` | 包含背景的 Dice 宏平均，作为补充，不能与 `dice` 混用 |
| `pixacc` / `fwiou` | 总体像素准确率 / 按真实像素频率加权的 IoU |
| `specificity` | 前景并集的特异度，即背景 Recall |
| `boundary_f1` / `hd95` / `assd` | 逐图逐前景类别计算，先在图内对有效前景类别平均，再对有效图片平均；另存每类结果 |

区域指标分母为零时记录 JSON `null` / CSV 空单元格，宏平均排除该项。边界类别在预测和真实标签中都为空时不参与平均；仅一方为空时 BF1=0、HD95/ASSD=图像对角线。整幅图没有任何有效前景类别对时不参与边界均值；整个数据集都没有时为 null。Boundary F1 默认容差 2 像素，HD95/ASSD 单位为评估尺寸下的像素；其他分数为 0–1。忽略像素不参与统计；将其屏蔽可能在忽略区边缘产生轮廓，比较时必须使用相同有效像素和分辨率。

每次最终评估统一写出：
- `metrics.json`：完整结果、每类指标、混淆矩阵和协议版本。
- `results.csv`：固定字段，每行对应 val 或 test。区域 `precision` 与测速 `precision_mode` 分开，避免覆盖。
- `class_metrics.csv`：每个数据划分、每个类别一行，含 IoU、Dice、Precision、Recall、像素数及边界结果；背景边界列留空。
- `confusion_matrix.csv`：长表 `split,true_class,predicted_class,pixels`。
- 开启复杂度测试时另有 `benchmark.json`：完整网络参数量、GFLOPs、FP32 单图前向延迟/FPS等。

批量仓库的实验根目录 `results.csv` 也使用相同列，每个模型/种子/划分一行，以 `name`、`seed`、`split` 区分，不再使用 `test_` / `val_` 前缀的宽表。模型子目录同样保存上述四个文件。缺失或失败数据留空，失败行记录非零 `status`。

旧 JSON 中的兼容字段暂时保留，标准 CSV 只使用上表字段。`plant-seg-lab` 的 `Foreground IoU` 现在对应 `foreground_miou`，原合并值改称 `Union Foreground IoU`；`nclassbenchmark` 的旧 `mdice` 仍是含背景均值，论文推荐取标准 `dice`（等于 `mdice_fg`）。各仓库的损失函数、模型结构、数据增强和预训练设置不因这次指标统一而改变。

旧权重可以重新评价，无需重新训练；旧 CSV 不会自动变为新口径。使用对应原始训练参数在两个 benchmark 仓库追加 `--eval-only`，或在 plant-seg-lab 使用 `summarize.py` / `evaluate_transfer.py` 重新生成结果。不要混合旧边界指标和 v2 指标。统计代码的验证命令：`python -m unittest discover -s tests -p 'test_metric_*.py' -v`（仅需 NumPy/SciPy，无需 GPU）。


## 已训练模型：批量重新评估与逐类总表

模型已经训练完成时，无需重新训练。新增三个入口：

| 脚本 | 用途 | 是否执行模型推理 |
|---|---|---|
| `batch_evaluate.py` | 自动发现已有 `best.pth`，按模型逐个评估并汇总 | 是；不训练 |
| `evaluate_checkpoint.py` | 单个已有 checkpoint 的独立评估 | 是；不训练 |
| `summarize_results.py` | 只读各模型已有 `metrics.json` 并汇总 | 否；无需 GPU |

### A. 加载十个旧模型，按新口径重新评估

`--runs-root` 填一个数据集的一次实验目录，该目录下面应包含各模型文件夹及其 `best.pth`。兼容旧的扁平输出目录和新的分层目录；不要同时选择多个数据集、重复实验或重复评估目录。

```bash
python batch_evaluate.py \
  --runs-root "outputs/GWFSSnclass/你的原实验目录" \
  --data-root /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --splits val test \
  --batch-size 8 \
  --workers 4 \
  --device cuda \
  --output "evaluation/gwfss_existing_models_v2"
```

只需测试集时改为 `--splits test`。RiceSEG 使用其自己的已训练实验目录及数据根目录：`/root/autodl-tmp/nclassbenchmark/data/RiceSegnclass`。每个数据集分别运行，分别保存汇总。

- 原始架构、编码器、类别数量/名称、输入分辨率和最优轮次直接从 checkpoint 读取，不要求重新填写 epochs、学习率或训练 batch size。
- `--batch-size` 仅是评估 batch size，可以与训练不同；类别和预处理沿用 checkpoint。数据搬家时用 `--data-root` 指定新地址，原路径仍有效时可省略。
- 只读取指定的 val/test 清单及对应图片和掩膜，不要求重新加载训练集。请使用你原实验的独立测试清单；旧 checkpoint 未保存清单 ID，无法追溯验证它是否被改动。本次会记录样本 ID 与清单哈希。
- 直接构造 `encoder_weights=None` 的模型并严格加载完整训练权重；不下载 ImageNet 权重，不依赖原 pretrained 目录，不创建优化器，不进行反向传播、不保存新 pth，也不导出预测掩膜。
- 原始 `best.pth`、`last.pth`、训练历史和结果保留；新输出目录必须不存在或为空。没有指定 `--output` 时自动在原实验目录下新建带时间的 evaluation 目录。
- 每个模型使用独立进程；某个模型失败会记录日志并继续其余模型，最后返回非零退出码，不将失败模型分数记作 0。
- 默认重新测量参数量、GFLOPs 和 FP32 GPU 前向延迟/FPS；`--no-benchmark` 可跳过测速，此时 FPS 留空，不借用旧设备的结果。运算统计不完整时完整 GFLOPs 仍为空，不影响精度结果。
- 只加载自己生成且可信的 checkpoint；原文件包含 NumPy RNG/优化器状态，因此读取使用 `weights_only=False`，模型恢复后不使用其中的训练状态。

### B. 已有完整结果，只重新生成总表

```bash
python summarize_results.py \
  --runs-root "outputs/GWFSSnclass/你的原实验目录" \
  --output "summary/gwfss_existing_results"
```

也可以对 A 生成的 evaluation 目录运行此命令。只读各模型的 `metrics.json`，不加载 pth、不执行推理。旧版 nclass JSON 支持读取 `mdice_fg`、`miou_fg`、逐类 `acc` 等字段；前景 Dice 不会误取含背景的 `mdice`。旧结果中没有的边界指标留空，不会自动补算或伪装为 v2；重新推理才能生成新边界口径。

仅有旧的根目录 `results.csv` 不能恢复未保存的逐类 IoU；请保留各模型的 `metrics.json`，或使用 A 重新评估。汇总器检测到重复模型/种子、不同类别映射、不同数据根目录或不同输入尺寸时会报错，避免把不一致实验拼在一起。多个种子单独成行，不挑选最好种子、不自动跨种子平均。

### C. 输出总表

下列文件保存到 A 的 evaluation 根目录、B 的 output 目录，或新批量训练的实验根目录：

| 文件 | 格式与用途 |
|---|---|
| `results.csv` | 所有模型总体指标，每个模型/种子/划分一行 |
| `all_class_metrics.csv` | 所有模型逐类指标长表，每个模型/种子/划分/类别一行 |
| `test_class_iou.csv` | 测试集论文用宽表，每个模型/种子一行；含 mIoU、前景 Dice、每类 IoU、参数量、FPS |
| `val_class_iou.csv` | 验证集相同格式；只评估 test 时为空表（仅表头） |
| `summary_manifest.json` | 类别映射、来源文件、协议版本及失败模型记录 |
| `evaluation_status.json`、`logs/` | A 的逐模型状态与完整日志 |

A 还会在 `models/` 下保存每个模型的标准 JSON/CSV 结果。新训练完成后，`batch_train.py` 也会自动更新以上总体/逐类总表。

逐类 IoU 列名示例：`iou_0_background`、`iou_1_class_1`、`iou_2_class_2`、`iou_3_class_3`；RiceSEG 会展开成其六个类别。类别名称沿用实际 checkpoint/结果文件，GWFSS 若仍是 class_1..3，请核对自己的标签映射后再用于论文，不根据论文列举顺序自动重命名。数值保存为 0–1，论文制表时区域分数乘 100。

### D. 单模型评估与检查

```bash
python evaluate_checkpoint.py \
  --weights "outputs/某模型目录/best.pth" \
  --data-root /root/autodl-tmp/nclassbenchmark/data/GWFSSnclass \
  --output evaluation/one_model \
  --splits test --device cuda --batch-size 8

python -m unittest discover -s tests -p 'test_result_tables.py' -v
python -m unittest discover -s tests -p 'test_checkpoint_evaluation.py' -v
```

汇总及调度逻辑测试不需要 GPU。实际 checkpoint 加载的 CPU 集成测试在存在 PyTorch/SMP 时执行；缺少依赖时明确跳过。真实训练权重与完整数据集推理需在 AutoDL 上验证。
