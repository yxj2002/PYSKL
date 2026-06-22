# Demo

我们目前提供一个离线 GPU 骨架动作识别 demo，以及一个在线 CPU 手势识别 demo。具体说明如下。

## 准备工作

- 在运行骨架动作识别 demo 之前，请确保已经安装 `mmcv-full`、`mmpose` 和 `mmdet`。我们建议直接使用项目提供的 conda 环境，其中已经包含所有必要依赖：

```bash
# 以下命令假设你位于 pyskl 的根目录，也就是 `$PYSKL`
# 该命令在 conda 22.9.0 中可以正常运行；如果你使用较早版本的 conda 并遇到错误，请先尝试更新 conda
conda env create -f pyskl.yaml  # 为该项目创建 conda 环境，环境名为 `pyskl`；如果还未创建，请运行该命令
conda activate pyskl  # 激活 `pyskl` 环境
pip install -e .  # 安装本项目
```

- 在运行手势识别 demo 之前，你需要先安装 `mediapipe`。可以直接通过 `pip install mediapipe` 完成安装。

## 骨架动作识别 Demo（GPU，离线）

提供的骨架动作识别 demo 是离线的，也就是说，它以一个视频片段作为输入，并返回动作识别结果。该 demo 在 GPU 上运行。默认情况下，该 demo 识别 [NTURGB+D 120](https://arxiv.org/abs/1905.04757) 中定义的 120 类动作。

对于人体骨架提取，我们使用 [Faster-RCNN（R50 backbone）](/demo/faster_rcnn_r50_fpn_2x_coco.py) 进行人体检测，并使用 [HRNet_w32](demo/hrnet_w32_coco_256x192.py) 进行人体姿态估计。二者均基于 OpenMMLab 实现。

```bash
# 使用在 NTURGB+D 120 上训练的 PoseC3D（Joint 模态）运行 demo，这是默认选项。
# 输入文件为 demo/ntu_sample.avi，输出文件为 demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4
# 使用在 NTURGB+D 120 上训练的 STGCN++（Joint 模态）运行 demo。
# 输入文件为 demo/ntu_sample.avi，输出文件为 demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
```

请注意，如果要在任意输入视频上运行 demo，你需要一个跟踪器，将每一帧的姿态估计结果组织成多条骨架序列。目前我们使用的是一个基于帧间姿态相似度的[简单跟踪器](https://github.com/kennymckormick/pyskl/blob/4ddb7ac384e231694fd2b4b7774144e5762862ab/demo/demo_skeleton.py#L192)。你也可以尝试编写自己的跟踪器。

## 手势识别 Demo（CPU，实时）

我们提供一个可以在 CPU 上实时运行的在线手势识别 demo。该 demo 以视频流作为输入，并预测当前执行的手势。目前它只支持单手场景。默认情况下，该 demo 识别 [HaGRID](https://github.com/hukenovs/hagrid) 中定义的 15 种手势，包括：Call、Dislike、Fist、Four、Like、Mute、OK、One、Palm、Peace、Rock、Stop、Three [Middle 3 Fingers]、Three [Left 3 Fingers]、Two Up。

对于手部关键点提取，我们使用开源方案 [mediapipe](https://google.github.io/mediapipe/)。对于基于骨架的手势识别，目前我们采用一个轻量版 [ST-GCN++](/demo/stgcnpp_gesture.py) 模型，该模型在 [HaGRID](https://github.com/hukenovs/hagrid) 手势识别数据集上训练。

```bash
# 运行实时基于骨架的手势识别 demo
python demo/demo_gesture.py
```
