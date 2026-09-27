# Real10 固定2000步：定位预热后联合训练

执行前固定，2026-09-22。用户同意前1000步纯定位、后1000步原联合目标的对照。

## 设计与已知差异

只新增三个seed的这一固定阶段策略，不改阶段长度、loss系数、网络或数据。
复用已有joint_2000和tip_only_2000作为比较，不覆盖或重训旧结果。

| 设置 | 定位分支更新 | 分类头更新 | 用途 |
|---|---:|---:|---|
| 原联合2000步 | 2000 | 2000 | 已有训练内对照 |
| 纯定位2000步 | 2000 | 0 | 已有拟合能力参照，不是完整响应模型 |
| 1000定位→1000联合 | 2000 | 1000 | 本轮唯一新增策略 |

总优化步数相同，但分类监督量、计算量不同。本轮测试的是“预热并延后分类”的策略，
不是相同分类监督量下的纯顺序互换实验；不声称已隔离梯度冲突。

## 固定条件

- 原fold0：36个训练窗口/9轨迹（20 advance、16 stationary，pos_weight=.8）。
  原9个留出窗口不参与计算或模型选择。原25tip点/5窗/5轨迹全部在训练侧。
- 原SpatialResponse、冻结ResNet18 layer1缓存、ROI224、task/时序、Gaussian sigma1、
  逐窗平均定位CE均不变。尖端坐标只作为target，缺失点仍masked，不新增或修改标注。
- 相同seed20261020/20261120/20261220与旧fold0 initial_state。初始1554参数原样载入。
- 单个AdamW实例，lr=.001、weight_decay=.01、clip1，FP32、确定性CUDA，无AMP/TF32。
- 步1–1000：response head冻结；仅529参数project/location接受 `.1 * localization`。
  复用纯定位的25点batch和窗权重；全部response参数无grad、无AdamW更新或weight decay。
- 步1001–2000：解冻原response head，36窗训练 `response BCE + .1 * localization`。
  不重置共享参数、Adam的step/一阶矩/二阶矩，不重设lr、不增总步数。
- 阶段交界保留checkpoint及完整optimizer状态；最终共享参数step=2000，分类参数step=1000。
  保存1000和2000两处checkpoint仅用于诊断，无最佳step/seed选择。

## 验证与指标

只做本轮必要检查：初始化一致、预热时分类参数未变、第1000步定位CE与旧纯定位
loss曲线同一步一致、切换前后共享Adam状态未重置、最终各参数step计数正确、保存回放
一致及旧文件size/mtime未变。旧模型与runner不改，不新增smoke suite。

记录0、每100步以及1001步的训练loss，以区分阶段切换和最终恢复；中间值不用于选择。
response BCE只为目标分解，不报告新response BA或成功率。预热阶段的BCE是未训练头
的只读诊断，不是优化目标。

定位指标沿用原空间期望/离线argmax的原图px误差、归一化CE、熵和目标两格邻域质量。
同一相机内点→窗平均，然后汇总全部三seed。Side13点/5窗、Top12点/4窗。
两格来自原Gaussian编码宽度，不是合格阈值；px不换算mm。

全部25点保留1000/2000步诊断，共150条新增读数；复用两组旧最终值150条。图版使用
固定9个窗/视角的最早有效tip，全3seed，导出54张新概率图，不挑最好帧。
输出中文450DPI PNG/SVG、真实指标data manifest和只读HTML。

## 执行与判读

新入口 `tools/fit_real10_staged_budget.py`，新out `simulation_output/real10_staged_budget_fit_v1`。
远端4090，沿用project2026-pi；预计约40s联合+3s定位更新，含验证与绘图少于2分钟。
按项目五分钟规则执行。顺序SCP最终入口/协议并回读；大缓存留远端，不传超过100MB文件。
中断时保留目录，`--resume`跳过完整seed，未完成seed从初始状态重跑；不改预算找好结果。

| 观察 | 可得判断 | 不可得判断 |
|---|---|---|
| 相比直接联合改善 | 此预热策略在固定总步数内改善训练拟合 | 改善一定来自顺序而非更少分类更新 |
| 切换后定位恶化 | 加入联合优化后已有拟合未保住 | 未测梯度却断言梯度方向冲突 |
| 最终仍不如纯定位 | 联合优化的定位代价仍在 | 图像/标注无效或必须升级PI05 |

本轮不是泛化、鲁棒性、XAI、contact、world model或实机有效性试验，不采用这些权重。
不改PI05/mixed-head、共享接口、数据/SOFA轨内容；policy_input_allowed=false、deployable=false。
