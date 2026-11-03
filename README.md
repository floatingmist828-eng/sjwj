# 数据挖掘导论 2026 大作业

四项必做任务与拓展一。使用课程原始数据，ZIP 可直接读取。

## 环境与运行

建议使用 Python 3.11。在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.3.1 torchvision==0.18.1
```

将课程数据中的 `Cluster.zip`、`Image_Anomaly_Detection.zip`、`weather.csv` 和 `thyroid.zip` 放入 `DM_2026_Dataset/`，也可以用 `--data` 指定目录。

```bash
python run.py all
python run.py cluster
python run.py image
python run.py weather
python run.py thyroid
python run.py cross-modal
```

图像任务默认使用 torchvision 的 ResNet-50 IMAGENET1K_V2 官方权重，首次运行会下载。离线运行可通过 `--weights /path/to/resnet50-11ad3fa6.pth` 指定权重。若 PyTorch 位于独立环境，通过 `--vision-python /path/to/python` 指定其解释器。特征提取优先使用 CUDA，CPU 也可运行。特征缓存在 `.cache/`，实验结果写入 `outputs/`，可分别用 `--cache` 和 `--output` 更改。

## 实验与输出

| 命令 | 方法与评估 | 主要产物 |
| --- | --- | --- |
| `cluster` | 多层图像特征，比较 K-means、Ward 和谱聚类。已知类别数为 6，类别标签用于事后计算 ARI、NMI 和匹配准确率。 | 聚类分配、列联表、指标、二维投影 |
| `image` | 按类别比较正常图块近邻、使用训练异常样本的全局 SVM、局部与全局分数融合。 | 样本分数、预测、ROC/PR、图块分数叠图 |
| `weather` | 用连续 12 个时刻的全部 21 项指标，预测下一时刻 CO₂。去重后剔除断档及含缺测的窗口，按目标时间作 70%/15%/15% 切分，训练区间拟合标准化，以验证 MAE 选模型。测试为逐步单步预测，历史输入使用截至预测时已观测的记录。 | 时间切分、持久性与回归模型对比、周期特征和损失函数对照、逐时浓度和变化量、残差 |
| `thyroid` | 正常训练样本拟合稳健缩放与正常记忆、高斯混合、隔离森林；混合成分数由正常拟合样本的 BIC 决定。 | 样本分数、预测、ROC/PR、指标 |
| `cross-modal` | 图像转换为局部描述符集合，表格行转换为单描述符集合。共用正常记忆、5 近邻距离、最高 5% 描述符距离聚合及分数校准。 | 原始距离与局部密度修正的逐域对照、等权宏平均、校准分数汇总 |

天气的回归模型学习下一时刻相对最近一次观测的 CO₂ 变化量。周期特征将 12 个历史风向和预测时刻的日内相位编码为正余弦。平方损失下比较周期特征的影响，再用同一组周期特征比较平方损失与绝对损失。预测时刻由采样频率确定。

异常任务把正常训练样本分成拟合和校准两部分。分数越高越异常，经验尾部概率按 `(1 + 校准分数中大于等于当前分数的个数) / (校准样本数 + 1)` 计算。统一判定阈值为 `p_normal <= 0.05`。图像融合使用独立正常子集确定分量尺度。拓展一按域建库并分别校准，合并时用 `1 - p_normal` 排序。图像热图展示局部距离，定量评价采用样本级标签。

密度修正将描述符的平均近邻距离除以局部参考半径的平方根。局部参考半径取这些近邻各自的留一近邻半径均值，以拟合库半径的第 10 百分位作下界。两种分数共用同一数据划分、近邻和聚合规则，各自使用正常校准集确定判定。

仓库地址见 [GitHub](https://github.com/floatingmist828-eng/sjwj)。
