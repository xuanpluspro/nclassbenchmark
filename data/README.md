# 数据集存放位置

实际路径：

```text
data/RiceSegnclass/
  JPEGImages/
  SegmentationClass/
  ImageSets/train.txt
  ImageSets/val.txt
  ImageSets/test.txt
data/GWFSSnclass/
  JPEGImages/
  SegmentationClass/
  ImageSets/train.txt
  ImageSets/val.txt
  ImageSets/test.txt
```

不需要额外套 VOC2012 或 Main 文件夹。分别设置 --data-root data/RiceSegnclass 和 --data-root data/GWFSSnclass。目录名区分大小写。数据内容被 Git 忽略，不会上传到 GitHub。
