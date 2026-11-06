# v17 —— 论文最终版（2026-11-06）

HighSim I-75 异常驾驶事件检测：φ1–φ4 效用特征 → GRU-AE → GMM 驾驶状态 → 分类控制限（UCL）→ 事件确认。
本文件记录这一版的方法设定、结果和复现方式。代码只有一套，下面的设定都是代码里的默认值。

## 1. 方法设定

| 环节 | 设定 | 依据 / 锚点 |
|---|---|---|
| 原始数据 | `../highsim_download/HighSim_Data_20260601.csv` | — |
| GT（统计意义的异常驾驶事件） | 急减速：min(周荣贵 2024 连续三次曲线 z, 全局单侧 z) ≥ 5，持续 ≥ 1 s（车型分组，不分车道，不扣情境）；同车道 TTC < 1.5 s 持续 ≥ 1 s。验证集 47 个 | 5σ；全局 5σ 下限去掉曲线两端的下凹 |
| 特征 | φ1 速度、φ2 加速度、φ3 横向 / 换道、φ4 碰撞风险；统一标度 z = 5 时 φ = 1 | 3σ 离群、5σ 饱和 |
| 模型 | GRU-AE（64→32，潜变量 D = 16），从头训练，早停 | D = 8 / 16 / 32 对比：D = 16 验证 MSE 最低 |
| 驾驶状态 | GMM（full 协方差）在潜空间聚类；k = 在"每类 ≥ 2% 窗口"约束下 BIC 最小（上限 20）→ k = 6 | 大样本下 BIC 单调下降，k 由 2% 约束决定；k 敏感性表证明结论不受 k 影响 |
| 控制限 | 每类 UCL = max(该类训练帧 SPE 的 q99, 全局 q99)（方案 C）；φ 全为 0 的退化类用全局 q99；类外窗口用各类最小值（= 全局） | GMM 只放宽活跃状态的控制限，不比全局更严 |
| 事件化 | SPE > UCL 连续 ≥ 1 s；同车相邻事件间隔 ≤ 3 s（窗口长度）合并 | 窗口长度 |
| 事件确认 | 事件内至少一帧综合效用强度 √Σ min(z,5)² ≥ 4.62；修复区帧不计；蛇行用 φ3 等价 z = 1 + 4φ3 | 阈值使超阈帧比例 = 单维饱和帧比例（单维饱和是特例） |
| 输出两层 | 模型检测层（统计对比用）：`pred_events_val_model.csv`；确认层（案例展示用）：`pred_events_val.csv` | — |

GRU 的设备问题：Mac GPU（tensorflow-metal）的 GRU 算子只用一组偏置，在 GPU 上训练出的模型两组偏置相等。
`model.zero_recurrent_bias` 按设备处理：Mac GPU 上不改权重；其他设备上把循环偏置置 0，结果与 Mac GPU 一致（差 4e-6）。

## 2. 结果（验证集，47 个 GT）

- 主模型：k = 6；模型检测 255 个事件（47/47 GT）→ 确认 92 个（45/47；漏检的 V1602、V1678 整段在速度修复区）。
- 确认事件复核（`review_v17_final/review_list.csv`）：GT 45、真异常 17、偏异常 18、临界 6、正常 5、数据问题 1 → 异常 87%，明确误报 6.5%。
- 基线对比：`tables/table_baselines_v17.*`（所有神经网络模型用相同设置在 GPU 上训练一次）。稳健的结论：
  - 效用特征：φ 方法 47/47，原始特征方法最多 44/47；
  - 噪声误报（超限帧中 φ 全为 0）：φ 方法 5–10%，原始特征方法 33–63%；
  - GMM：同一 GRU-AE，全局 UCL 447 → GMM 255，召回不变。
  - GRU 与 LSTM 效果相当（不同训练间排序会互换），不声称 GRU 更优。
  - MLP-AE 对训练随机性很敏感（CPU / GPU 两次召回 0.81 / 0.23）；原始特征 GRU-AE + GMM 也有波动（0.91 / 0.81）。
- k 敏感性：`tables/table_k_sensitivity_v17.*`（k ≥ 4 时模型检测 215–255、确认 86–92、GT 不变）。

## 2.5 代码结构

| 文件 | 内容 |
|---|---|
| `pipeline.py` | 分阶段运行入口（data / features / train / detect / baselines） |
| `Data.py` | 原始轨迹 → 平滑速度/加速度、TTC、数据质量标记、GT 事件 |
| `accel_thresholds.py`、`phi1_local.py`、`utility_phi.py` | 加减速阈值曲线、局部车流速度、φ1–φ4 效用特征 |
| `model.py` | GRU-AE 训练、GMM 驾驶状态、分类 UCL、事件化与两层输出 |
| `report_rule.py`、`gmm_outside.py` | 事件确认（综合效用强度）、GMM 类外判断 |
| `baselines.py` | 基线方法与两层主表 |
| `tools/` | k 敏感性、复核图、潜空间图、SPE 柱状图、训练曲线；`relax_anchor.py`、`tail_repair_anchor.py` 为两个常数的数据锚点；`merge_aanomaly.py` 生成公开数据集的标注文件 |
| `tables/`、`figures/` | 论文表与图 |

## 3. 论文图表

| 文件 | 内容 |
|---|---|
| `tables/table_baselines_v17.{csv,tex}` | 基线对比（两层 + 噪声误报列） |
| `tables/table_k_sensitivity_v17.{csv,tex}` | GMM 聚类数敏感性 |
| `figures/fig_latent_clusters_tsne_v17.png` | 潜空间 t-SNE（6 类；降维前加 0.01 SD 扰动打破重复点） |
| `figures/fig_latent_clusters_tsne_inside_v17.png` | 同上，只画本类 99% 稳健椭圆内的点（省略 22.8%，须写进图注） |
| `figures/fig_spe_by_cluster_v17.png` | 各驾驶状态训练帧 SPE（均值 + 1 SD） |
| `figures/fig_latent_dim_*.png` | D = 8 / 16 / 32 训练曲线 |

驾驶状态命名：`runs/v17/cluster_names.json`（Stable Driving、Close Car-Following、Speed-Adaptive Following、Lane Changing、Dense Interactive Driving、Relaxed Following）。重训后类序号会变，须按训练集画像重新对应。

## 4. 复现

```sh
# 数据 + 特征（确定性）
python pipeline.py --workdir runs/v17 --stage data
python pipeline.py --workdir runs/v17 --stage features
# 训练（D = 16，从头训练；v17 实际在 runs/latentD/D16 训练后复制过来）
LATENT_DIM=16 python pipeline.py --workdir runs/v17 --stage train
python pipeline.py --workdir runs/v17 --stage detect
python pipeline.py --workdir runs/v17 --stage baselines          # 基线 + 两层主表
(cd runs/v17 && python ../../tools/k_sensitivity.py 2 12)         # k 敏感性
(cd runs/v17 && python ../../tools/plot_review.py ../../review_v17_final all)
(cd runs/v17 && PROJ=tsne MIXED=4 python ../../tools/plot_latent_clusters.py fig_latent_clusters_tsne.png 400)
(cd runs/v17 && python ../../tools/plot_spe_by_cluster.py)
python tools/plot_train_curves.py runs/latentD figures/fig_latent_dim
```

运行脚本：`runs/v17/v17_chain.sh`、`runs/v17/v17_planB.sh`。`runs/`、`_baseline_results_cache/`、`review_*/` 不进 git（见 `.gitignore`）。

## 5. 注意

- `HighSim_data_with_Anomaly(Final).csv` 是本仓库已发布的标注数据集（Git LFS），其中的 `Anomaly` 标注来自早期版本模型，与 v17 不同；是否用 v17 结果更新，另行决定。
- 训练窗口、阈值、GT 定义的历史决策见代码注释（以 "v16b" 等版本号标注）。
