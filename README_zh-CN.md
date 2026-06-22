# PYSKL

> 注意：该仓库目前已不再由开发者维护。欢迎自行创建 fork，并基于这份代码继续开发。

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=revisiting-skeleton-based-action-recognition)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dg-stgcn-dynamic-spatial-temporal-modeling/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=dg-stgcn-dynamic-spatial-temporal-modeling)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-kinetics)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-kinetics?p=revisiting-skeleton-based-action-recognition)
[[**报告**]](https://arxiv.org/abs/2205.09443)

PYSKL 是一个基于 **PY**Torch、专注于 **SK**e**L**eton 骨架数据动作识别的工具箱。它支持多种基于骨架的动作识别算法。本项目基于开源项目 [MMAction2](https://github.com/open-mmlab/mmaction2) 构建。

该仓库是 [PoseConv3D](https://arxiv.org/abs/2104.13586) 和 [STGCN++](https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B) 的官方实现。

<div id="wrapper" align="center">
<figure>
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="520px">&emsp;
  <img src="https://user-images.githubusercontent.com/34324155/218010909-ccfc89f0-9ed4-4b04-b38d-af7ffe49d2cd.gif" width="290px"><br>
  <p style="font-size:1.2vw;">左：NTU-RGB+D-120 上的基于骨架的动作识别结果；右：CPU 实时基于骨架的手势识别结果</p>
</figure>
</div>

## 更新日志

- 改进骨架提取脚本（[PR](https://github.com/kennymckormick/pyskl/pull/150)）。现在支持非分布式骨架提取和 k400 风格的数据格式（**2023-03-20**）。
- 支持 PyTorch 2.0：当训练/测试脚本设置 `--compile`，并检测到 `torch.__version__ >= 'v2.0.0'` 时，会在训练/测试前使用 `torch.compile` 编译模型。该功能为实验性特性，不保证性能收益（**2023-03-16**）。
- 提供一个基于 ST-GCN++ 骨架动作识别的实时手势识别 demo，更多细节和说明请查看 [Demo](/demo/demo.md)（**2023-02-10**）。
- 提供用于估计各模型推理速度的[脚本](/examples/inference_speed.ipynb)（**2022-12-30**）。
- 支持 [RGBPoseConv3D](https://arxiv.org/abs/2104.13586)，这是一个基于 RGB 与人体骨架的双流 3D-CNN 动作识别模型。请按照[指南](/configs/rgbpose_conv3d/README.md)在 NTURGB+D 上训练和测试 RGBPoseConv3D（**2022-12-29**）。

## 支持的算法

- [x] [DG-STGCN (Arxiv)](https://arxiv.org/abs/2210.05895) [[模型库](/configs/dgstgcn/README.md)]
- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[模型库](/configs/stgcn/README.md)]
- [x] [ST-GCN++ (ACMMM 2022)](https://arxiv.org/abs/2205.09443) [[模型库](/configs/stgcn++/README.md)]
- [x] [PoseConv3D (CVPR 2022 Oral)](https://arxiv.org/abs/2104.13586) [[模型库](/configs/posec3d/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[模型库](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[模型库](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[模型库](/configs/ctrgcn/README.md)]

## 支持的骨架数据集

- [x] [NTURGB+D (CVPR 2016)](https://arxiv.org/abs/1604.02808) 和 [NTURGB+D 120 (TPAMI 2019)](https://arxiv.org/abs/1905.04757)
- [x] [Kinetics 400 (CVPR 2017)](https://arxiv.org/abs/1705.06950)
- [x] [UCF101 (ArXiv 2012)](https://arxiv.org/pdf/1212.0402.pdf)
- [x] [HMDB51 (ICCV 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126543)
- [x] [FineGYM (CVPR 2020)](https://arxiv.org/abs/2004.06704)
- [x] [Diving48 (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)

## 安装

```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# 该命令在 conda 22.9.0 中可以正常运行；如果你使用较早版本的 conda 并遇到错误，请先尝试更新 conda
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Python 3.10 安装

```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
conda env create -f pyskl_310.yaml
conda activate pyskl
pip install -e .
```

## Demo

请查看 [demo.md](/demo/demo.md)。

## 数据准备

我们为每个受支持的数据集提供 HRNet 2D 骨架，并为 NTURGB+D 和 NTURGB+D 120 数据集提供 Kinect 3D 骨架。要获取人体骨架标注，你可以：

1. 使用我们预处理好的骨架标注：我们直接为所有数据集提供处理好的 pickle 格式骨架数据，可直接用于训练和测试。下载链接和标注格式说明请查看[数据文档](/tools/data/README.md)。
2. 对于 NTURGB+D 3D 骨架，你可以从 https://github.com/shahroudy/NTURGB-D 下载官方标注，并使用我们[提供的脚本](/tools/data/ntu_preproc.py)生成处理后的 pickle 文件。生成的文件与我们提供的 `ntu60_3danno.pkl` 和 `ntu120_3danno.pkl` 相同。详细说明请参考[数据文档](/tools/data/README.md)。
3. 我们还提供了从 RGB 视频中提取 2D HRNet 骨架的脚本。你可以参考 [diving48_example](/examples/extract_diving48_skeleton/diving48_example.ipynb)，从任意 RGB 视频数据集中提取 2D 骨架。

你可以使用 [vis_skeleton](/demo/vis_skeleton.ipynb) 可视化我们提供的骨架数据。

## 训练与测试

你可以使用下面的命令进行训练和测试。基本上，我们支持在单台服务器上使用多张 GPU 进行分布式训练。

```shell
# 训练
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# 测试
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```

具体示例请查看各个受支持算法目录下的 README。

## 引用

如果你在研究中使用 PYSKL，或者希望引用模型库中发布的基线结果，请使用下面的 BibTeX 条目，并同时引用你所使用的具体算法对应的 BibTeX 条目。

```BibTeX
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```

## 贡献

PYSKL 是一个遵循 Apache2 许可证的开源项目。欢迎社区对 PYSKL 做出任何改进贡献。对于**重要贡献**，例如支持一个新的且重要的任务，对应内容会加入我们更新后的技术报告中，贡献者也会被加入作者列表。

任何用户都可以向 PYSKL 提交 PR。PR 会在合并到 master 分支之前接受审查。如果你想向 PYSKL 提交一个**大型 PR**，建议先通过邮件 dhd.efz@gmail.com 联系作者讨论设计，这有助于在代码审查阶段节省大量时间。

## 联系方式

如有任何问题，请联系：dhd.efz@gmail.com
