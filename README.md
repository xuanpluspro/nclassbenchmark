# 多分类植物语义分割对比实验

参考 [vegann-seg-benchmark](https://github.com/xuanpluspro/vegann-seg-benchmark)，将二分类流水线改为任意 C 类互斥语义分割。RiceSEG（6 类）和 GWFSS（4 类）分别训练，分别汇总；不能直接混合两套类别标签。

## 数据与类别

采用已整理好的 VOC 目录：

```text
VOC2012/
  JPEGImages/<id>.jpg
  SegmentationClass/<id>.png
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
  ImageSets/Main/test.txt
```

也兼容 `ImageSets/Segmentation/`、`ImageSets/`。列表每行是不含扩展名的文件名。训练、验证、测试 ID 必须互不重叠。建议先对原始图像划分，再切 patch，避免同一原图的 patch 跨集合泄漏；地块/区域/连续拍摄有相关性时按组划分更可靠。

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
python audit_dataset.py --data-root /root/autodl-tmp/RiceSEG/VOC2012 --dataset-config configs/riceseg.json
```

RiceSEG 依次训练 experiments.json 中 10 个模型：

```bash
python batch_train.py --dataset-name RiceSEG --run-name pretrained_ep30_bs32 --data-root /root/autodl-tmp/RiceSEG/VOC2012 --dataset-config configs/riceseg.json --weights local --pretrained-dir pretrained --size 512 --batch-size 32 --epochs 30
```

GWFSS（先核对配置中的标签名称与 ID）：

```bash
python batch_train.py --dataset-name GWFSS --run-name pretrained_ep30_bs32 --data-root /root/autodl-tmp/GWFSS/VOC2012 --dataset-config configs/gwfss.json --weights local --pretrained-dir pretrained --size 512 --batch-size 32 --epochs 30
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
- `mIoU/mAcc/mDice`：对类别等权平均，默认含背景。
- `PixAcc`：全部有效像素中预测正确的比例，与 mAcc 不同。
- `miou_fg/macc_fg/mdice_fg`：仅从平均中排除背景类别，仍保留背景误判造成的 FP/FN。
- 整个评估集某类 GT 和预测都不存在，IoU/Dice 为 null，不参与均值；GT 不存在但出现误预测，IoU 为 0。GT 不存在时 Acc 为 null。输出参与平均的类别数，审查测试集是否缺类。
- 所有数值保存为 0..1；论文制表时统一乘以 100。

建议主表：模型、mIoU、mAcc、mDice、PixAcc、Params(M)、GFLOPs、FPS。另存逐类别 IoU/Acc/Dice/Precision 和混淆矩阵。少数类性能不能仅靠均值判断，尤其是 GWFSS 的茎、RiceSEG 的衰老植被/杂草/浮萍。报告背景是否参与均值，不能混用两种统计口径。

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
