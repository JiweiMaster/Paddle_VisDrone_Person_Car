简体中文
# 特色垂类跟踪模型

## 无人机视频下的行人跟踪（Person Tracking)
VisDrone是无人机视角拍摄的数据集，针对VisDrone2019中的数据集进行提取，抽取出class为person的数据，所有序列的帧速率都是30fps。
<div align="center">
  <video width='800' height='600'>
    <source id="mp4" src="./visdrone_person/VisDronePerson.mp4" type="video/mp4">
  </videos>
</div>

## 模型库
### FairMOT在HT-21 Training Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS  |   FP  |   FN   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---: | :----: | :---: | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  67.2 |  70.4  |   9403  |  124840  |  255007  |     -   | [下载链接]() | [配置文件](visdrone_person/config.yml) |

### FairMOT在HT-21 Test Set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |   IDS  |   FP   |   FN   |    FPS   |  下载链接  | 配置文件 |
| :--------------| :------- | :----: | :----: | :----: | :----: | :----: |:-------: | :----: | :----: |
| DLA-34         | 1088x608 |  58.2  |  61.3  |  13166   |  141872  |  197074 |    -     | [下载链接]() | [配置文件]() |

**注意:**
 FairMOT使用2个GPU进行训练，每个GPU上batch size为6，训练30个epoch。

## 快速开始

### 1. 训练
使用2个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
```

### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/{your_model}.pdparams
```

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/{your_model}.pdparams --image_dir=dataset/mot/VisDrone2019_all/images/test/uav0000137_00458_v/img1 --save_videos
```
**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=output/fairmot_dla34_30e_1088x608/{your_model}.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608_headtracking21 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。

## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
@InProceedings{Sundararaman_2021_CVPR,
    author    = {Sundararaman, Ramana and De Almeida Braga, Cedric and Marchand, Eric and Pettre, Julien},
    title     = {Tracking Pedestrian Heads in Dense Crowd},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3865-3875}
}
```
