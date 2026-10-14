# 数据集存放位置

请将已整理的 VOC 数据分别上传到 RiceSEG/VOC2012/ 和 GWFSS/VOC2012/。

每个 VOC2012/ 内需要 JPEGImages/、SegmentationClass/、ImageSets/，划分列表可位于 ImageSets/Main/、ImageSets/Segmentation/ 或 ImageSets/。

从仓库根目录运行时，分别设置 --data-root data/RiceSEG/VOC2012 或 --data-root data/GWFSS/VOC2012。数据内容被 .gitignore 忽略，不会上传到 GitHub。
