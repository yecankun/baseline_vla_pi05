# 候选动作选择：文献对齐与最小实施方案

初版：2026-09-16；更新：2026-09-19。范围：算法/VLA；状态：**DP配对数据/训练接口已实现、零步准备通过；下一步用户生成独立32训练+8验证源，尚未训练；暂缓四组几何设计**。
设计、实现、训练与选择比较分开记录；末端值训练/比较见第 17–18 节，新效应差异协议见第 19 节，实际配对数据生成见第 20 节，公平训练入口见第 21 节，验证结果与混杂因素见第 22 节，冻结测试入口与执行口径见第 23 节，最终测试结果与决策见第 24 节，训练/验证输入审查见第 25 节，暂缓的四组设计见第 26 节，DP诊断协议见第 27 节，完成结果见第 28 节。现场工作暂停，保留ACT/real10，无闭环收益结论。

本方案接续用户提供的 `C:/Users/Silence/Downloads/deep-research-report.md`，
用于工程决策，不是论文 Results，也不宣称新 SOTA。当前优先级由
[算法交接](algorithm-track-handoff.md)管理；本方案不授权后续训练、采集或实机动作。

## 1. 收敛后的决策

保留现场 real10 PI0.5 和普通 ACT；研究改为：**固定候选生成器 → 直接任务收益评分 →
有必要时才加入未来预测辅助**。不继续以更低 Dice 为理由扩展 GRU，也不立即做多编码器、
五折真实训练、sim:real 比例搜索或大候选集搜索。

公共 Push-T 先回答“直接监督决策量是否比先预测网格再打分更合适”。真实导丝端只设计兼容接口，
不能把 Push-T 的动作、颜色分割或监督直接迁移过去。已训练的两帧 history 模型保留为预测参考；
旧开发集的 t-1 恢复及 history 排序暂缓，不自动执行原先提议的 1440 步回放。

### 文献中采用什么、不采用什么

- [VGAS v2](https://arxiv.org/html/2602.07399v2)：借鉴冻结策略提案、动作/状态联合评分和
  Best-of-N 的分工。原文结合 proposal-constrained TD 与专家距离几何正则，基础策略为
  SmolVLA-0.5B。几何接近专家是先验，不是未执行动作的真实后果；下述直接效应回归不是 VGAS 复现。
- [WCM v1](https://arxiv.org/html/2607.29613v1)：借鉴预测任务作为价值表征的辅助监督。
  原文主价值头是历史条件的 V，另有动作条件的下一 latent 预测；不能直接称为现成的候选 Q 选择器。
  将其思想加到本项目 scorer 上仍是待验证设计。

只核对与本次决策相关的原文方法，不把 Deep Research 中不可移植的 `turn...` 引用或其整份
结论视为全部已核实。当前没有文献或本项目证据证明“低可见度 + 十条真实数据 + 有动力学偏差的
仿真”这一组合必然有效。

## 2. 已保存候选：瓶颈复核

来源：

- [残差候选 report](../simulation_output/pusht_object_residual_ranking_dev10_v1/report.json)
  与同目录 `analysis.jsonl`；
- [原候选 report](../simulation_output/pusht_object_action_ranking_dev10_v1/report.json)；
- [指标实现](../tools/check_pusht_object_action_ranking.py)的 `analyze` / `aggregate`。

本次只用 PowerShell 读取 JSON 重新汇总，未调用模型或环境。原 140 个已执行候选来自
28 个上下文、10 个 seed；剔除当前视觉无效的 200004/0 后，比较 27 个上下文、135 个候选。
全部属于**反复使用的 development 证据**，不是新 held-out test。

候选固定为 ACT 原始 8 步 + 四个逐步增加的 ±X/±Y 偏移，末步偏移 8 个 native 坐标单位。
它们不是五次独立 ACT 采样，也不能仅因靠近 ACT 就称为策略支持域内。

令 C_k 为保存的第 k 个候选执行 8 步后的 coverage，k=0 为 ACT，s 为选择器输出：

```text
可用上限 H = max_k C_k - C_0
实际选择收益 G = C_s - C_0
未取得的上限 R = max_k C_k - C_s = H - G
```

| 27 个有效上下文的均值 | 数值 |
|---|---:|
| coverage oracle 相对 ACT 上限 H | 0.007416796852 |
| 残差预测器选择收益 G | 0.003460158696 |
| 残差选择 regret R | 0.003956638156 |
| 事后真实物体目标代价 oracle 的收益 | 0.007275351983 |
| 物体目标代价 oracle 距 coverage oracle 的差距 | 0.000141444869 |

G/H = 46.65%，只是这组上下文的“均值之比”，不是逐样本平均、成功率或泛化能力。
coverage 差异超过原容差 1e-6 的仅 7 个上下文，涉及 5 个 seed；其余 20 个没有这个短时指标上的
可分辨差异。**不能据此断言动作相同、长时效果相同，或候选生成器完全无效。**

下面列出全部 7 个可分辨上下文，没有只选有利案例。指标均为 coverage 差，不是百分点成功率。

| seed / anchor | H：最好候选比 ACT 好多少 | G：残差选择比 ACT 好多少 | R：错过多少 |
|---|---:|---:|---:|
| 200000 / 80 | 0.083522925 | 0.083522925 | 0 |
| 200003 / 80 | 0.022591217 | 0.014425865 | 0.008165353 |
| 200006 / 160 | 0.047062413 | -0.041008277 | 0.088070690 |
| 200008 / 0 | 0.024668993 | 0.014075805 | 0.010593188 |
| 200008 / 160 | 0.000007588 | 0.000007588 | 0 |
| 200009 / 80 | 0.002806748 | 0.002806748 | 0 |
| 200009 / 160 | 0.019593632 | 0.019593632 | 0 |

200006/160 占总 regret 的 82.44%；200000/80 占净收益的 89.40%。这说明有明确误选，也有严重的
少数样本主导，不能称为稳定收益。前者的真实 object-goal 最优候选也不是 coverage 最优候选，
但这种代理目标差距远小于本次模型误选损失。残差相较旧预测器平均收益只增加约 2.81e-7。

**行动结论：先固定五候选检验 scorer，不先扩候选数或改偏移。**当前证据够支持这一最小诊断，
不够支持“候选质量已经解决”。若后续独立状态仍没有可用 headroom，应检查候选/时域/评价目标，
而不是继续扩大评分网络。不得专门拟合上述坏案例或根据它们调阈值。

## 3. 最小直接评分基线 DS0（已实现并完成固定训练）

DS 表示 direct scorer，不改用已有项目数据域 D0/R0 的名称。

### 输入、目标与模型

复用 [ObjectData](../tools/pusht_object_dynamics.py)：同一训练源、186/20 整 episode 划分、
21958/2002 有效窗口、原 state/action 归一化与原训练目标图像。目标仍是训练 episode1/frame117，
不是验证或候选未来图像。现有 validation 已多次用于研究决策，应称“既有留出开发集”，不再包装成
未触碰的最终测试。

```text
输入：当前 RGB 导出的 24×24 occupancy、当前可观测 agent XY、
      固定 goal occupancy、有序候选 A[8,2]
输出：S(o,A,g)，一个短时目标代价下降量估计
监督：y = d(current, goal) - d(observed_future_at_t+8, goal)
```

d 复用当前 quadratic soft-Dice。这里的正值表示接近固定视觉目标；不叫成功概率、长期 Q 或真实
导丝前进量。coverage/contact/隐藏物体位姿不进 forward；未来网格仅供训练目标与事后评价。

第一版结构限定为小 MLP：当前/goal 共用 `576 -> 128 -> LayerNorm -> SiLU` 编码；拼接两份
128D 表征、归一化 XY(2D)、按时间顺序展平的动作(16D)，用 `274 -> 128 -> 64 -> 1` 打分。
随机初始化新权重，不载入旧 world-model encoder，以免把预训练收益混入“直接评分”。
不新增 history、Transformer、ensemble 或 ROI 分支。

仅拟合**实际执行的**数据动作及其真实后继。可按训练窗口 y 的均值/标准差标准化目标，统计量仅从
训练 episode 得到并随 checkpoint 保存；退化常量目标时明确报不可学习，不靠造负例解决。
主损失只用 MSE。未执行的噪声动作没有后继，不能套用原后继或自动标成负收益；不同上下文的动作
也不能冒充同状态反事实对。

保持原来的训练窗口均匀抽样，不删除静止/零变化窗口，也不按旧开发案例重加权。准备阶段只统计
训练目标的正/近零/负变化数量（近零沿用目标代价容差 1e-7）；评价同时列出这些分层的误差和整
episode 宏平均，防止大量近零标签掩盖变化窗口失败。分层标签只用于统计，不进入 scorer。

已准备既有固定预算：seed20260915、10000 updates、batch64、AdamW lr3e-4 / wd1e-4、clip1，
同一训练抽样序列、仅 final checkpoint。它是预算匹配的首个诊断，不是最优超参或稳定性实验。
参数量实测 117633；在线延迟尚未评估。已完成的训练命令与结果见
[算法命令文档](algorithm-track-commands.md)。

### 选择与最低限度对照

- 每个上下文复用完全相同的五候选；逐个打分，argmax，精确平局保留 candidate0。
  有效候选边界仍沿用现有 native bounds，不静默裁剪。
- 对照普通 ACT、uniform 解析期望、既有残差“预测后打分”、直接 scorer。
  事后 object-goal oracle 与 coverage oracle 分列，均不是可部署策略。
- 保留既有无效观测处理作为“选择器弃权”。不把回到 ACT 称为实机安全保证。
- 原 28 个上下文不用于训练、筛选 checkpoint 或调参；以后如对其打分，只报告复用开发集诊断。
  它们不能同时成为 scorer 的反事实训练集和最终成绩单。
- 执行形态不变：Push-T 评分整个 8 步前缀，若将来获批闭环，也执行相同前缀。
  不能评分 8 步却仅执行第 1 步，再声称目标已与执行对齐。

DS0 的已知限制是 demonstration-to-policy 动作分布偏移：训练只看实际示范动作，评价却包含
ACT 邻域候选。它是直接效应评分的低成本基线，不自动具备离线 RL 的支持约束或反事实识别能力。
若失败，下一步是区分动作支持不足与表征不足；不是宣称直接 critic 路线已被否定。

## 4. 监督来源：严格分开事实、先验和缺失

| 来源 | 能监督什么 | 不能据此监督什么 |
|---|---|---|
| Push-T 原始轨迹 | 实际动作到实际未来的短时目标变化；DS0 可直接复用 | 同一状态下未执行候选的真实效应 |
| 旧 140 个候选后果 | 事后 headroom、regret、误选与代理目标差距 | 新训练/模型选择后还声称独立测试 |
| 用户最新 real10 | 已记录动作、可观测控制器历史、实际后继；动作跟随可信是用户提供的背景 | 每个未执行候选的优劣、自动成立的 guidance/contact 真值 |
| 专家动作距离/扰动 | VGAS 式局部支持先验，需单独标识 | 真实反事实 reward；离专家近不保证更好 |
| 未来获准的仿真候选执行 | 在该仿真动力学下的候选后果，诊断真值仅在 target sidecar | 无实测对齐时的真实后果或真实成功率 |

缺失标签一律有显式 validity mask；缺失不等于失败/零进展。real10 的 `t/T` 最多是时序弱标签，
暂停、递丝事件、回退都可能破坏它与任务进度的关系。不能照搬“示范末三帧成功”来标定所有真实采集，
尤其不能把失败 calibration pilot 当成功示范。师兄旧数据仍不混训。

未来仿真配对数据必须按原始 episode/source/family 分组切分，然后派生候选与视觉退化；同一快照的
所有候选留在同一 split。目标、sample role、权重、source ID、exact tip/contact/wall/route 不进 scorer。
数据侧负责允许状态与标签来源，算法侧不因 schema 能读就升级数据地位。

## 5. 真实端最小候选合同（只设计，不接入）

已核实 [build_config](../tools/probe_pi05_lerobot_adapter.py) 固定 `chunk_size=1,n_action_steps=1`；
[Real10PI05Policy.predict](../tools/real10_pi05_policy.py) 取单步 xyz，旋转补零，Piper 为独立
state_only head 的 argmax。**当前不是联合 Elite/Piper 多步生成策略。**

候选接口按任务分型，不用 padding 假装与 Push-T 相同：

| 项目 | 公共诊断接口 | 未来真实接口 |
|---|---|---|
| 动作 | `absolute_xy[B,5,8,2]`，native 坐标 | `elite_tcp_delta_6d[B,K,1,6]`，xyz_mm/rpy_rad |
| 候选来源 | 现有 ACT reference + 固定四方向 ramp | 先试 reference + 三次独立采样的 Elite 单步，K=4 是工程起点 |
| Piper | 不存在，不映射伪意图 | `piper_intent_id[B,K,1]`，全部复制这次原 head 的同一个输出 |
| 旋转 | 不适用 | 固定为零，未训练的维度不参与搜索 |
| 状态 | 当前 RGB/可观测 XY | Side/Top、task、原 real10 state_32 与此前控制器快照 |
| 返回 | 选择索引及原候选前缀，禁止自动执行 | 选择索引及原有共享动作包，禁止自动执行 |

candidate0 必须逐值保留原推理输出及其随机流；额外候选不能悄悄改变 reference 的随机状态。
后续实现时先确认当前 PI0.5 API 的独立噪声路径、候选实际多样性与总延迟；这些尚未验证。
不支持显式噪声时应记录实际采样机制，不硬造“独立采样”的声明。
候选有效性依据既有单位/限幅合同，不新增从旧案例拟合的动作阈值。

选择器不混合/平均 `piper_intent_id`，不以 ID 的欧氏距离做几何正则。首版只研究**给定 Piper 输出时
的 Elite 平移选择**，不能声称改善 Piper event boundary；联合 hold/feed 候选需另外定义执行时序和监督。
本轮不扩旋转/retract 搜索，并不撤销项目中有 backend 支持的 retract 能力。

部署输出保持：`observation + task instruction -> elite_tcp_delta_6d + piper_intent_id`。
Controller 继续负责 IK、busy/cooldown、接受/拒绝/实际执行、限幅和安全。两视角都不可用时，
回原策略不保证安全；应走原控制层的缺失观测/暂停处理，不能让 scorer 代替现场安全流程。

## 6. 后续模块与验证顺序

1. **DS0 准备及固定训练已完成**：复用 ObjectData、当前目标函数和候选类型；一次必要的缓存 batch
   检查通过后，另获批准完成 10000 步及 final-only 评价。小文件先 SCP 后远端执行，不追加 smoke。
2. **DS0 开发候选离线评价已完成**：获单独批准后比较了已保存候选的 gain/regret（第 11 节）。
   没有稳定优势，不根据这 7 个 informative context 的细微名次变化决定部署或继续加模块。
3. **新 seed 固定候选对照已完成**：用户另行批准后清点历史并锁定 20 个未用项目评估 seed，
   共用候选后果完成第 12–13 节对照。未发现 DS0 稳定优势；这不是选择器驱动的整 episode 闭环。
   不为凑结果追加回放；该集合现在已经看过，后续改模型时不能继续称其为未触碰最终测试。
4. **只有 DS0 路径值得继续时，做单一 dynamics-aux 对照 DS1**：保留同一 score trunk，增加动作条件的
   t+8 终点表征/网格预测辅助。DS0/DS1 共用初始化、抽样、预算和 score head；两臂可保留同一辅助头，
   分别置辅助权重 0/一个事先固定值。监督仍来自实际后继，不进行长时生成式 rollout。
   辅助权重与尺度在训练协议中预先定下，不在旧坏案例上调优。只有辅助改善独立选择收益才保留。

上述第 4 步是世界模型辅助假设，不是已经验证的创新。当前设计的唯一最小变化是评分目标，而不是
同时变候选、历史、视觉 backbone、Piper head 和多个 loss。

### 指标、效率与停止条件

主指标是相对 base 的实际选择 gain 和 oracle regret；同时报告 harm rate、最坏损失、oracle
headroom、无差异上下文占比、候选拒绝率。pairwise accuracy/top1 只在有真实差异时计算，保留平局数，
不把同一 seed 的 10 个候选对当 10 次独立试验。目标 MAE 与 Dice 属于预测证据，不代替选择收益。

报告 context 均值和 episode/seed 宏平均及逐 seed 值；独立测试有足够 episode 后，按 episode 聚类
重采样估计不确定性，而不是按 frame/candidate 对重采样。新训练仅一个 seed 时明确稳定性未测。
控制周期未知前，不承诺实时性；记录提案、特征、K 候选打分、总耗时的 warm median/p95、峰值显存、
参数量、optimizer steps 和环境步数。最终闭环比较需单独获批，用相同 episode 初始化与预算。

- 若独立候选的 headroom 不足：暂停 scorer 扩容，检查候选覆盖/短时指标/时域。
- 若降低预测误差但 gain/regret 无改善或伤害上升：不升级策略，不继续按 Dice 堆 world model。
- 若直接 scorer 与 dynamics-aux 无可靠选择差异：保留更简单者；不为了“有世界模型”保留辅助。
- 若收益仅来自一个 seed、训练更久或候选更多：结论不确定，补相应最小受控对照，不宣称稳定收益。

## 7. 低可见度与 sim-real 的边界

暂不启动新低对比度实验。未来真实 scorer 优先复用 PI0.5 双视角局部 token，避免仅 mean-pool 丢失细
导丝信息；是否有效仍须实际检查。若引入历史，先用两帧且有同 episode/timestamp/validity，禁止
跨 episode 或未来帧。公开 Push-T 的颜色 occupancy 不作为真实导丝感知方案。

仿真同一已执行轨迹可派生 clean/degraded 配对观测，保留原动作和目标，试验表征一致性；这些是同一个
行为样本，不增加有效独立 episode 数。不能把退化增强叫氧化的真实成像模型，也不把生成式“修复”图像
当作唯一 policy observation。真实不可见的导丝不因训练 loss 下降就变成可观测。

先将 sim 看作候选效应的先验与覆盖来源，而不是默认精确真值。等数据侧允许相应用途后，才比较 real-only
与 sim-pretrain→real-finetune；joint training 是后续选项，不承诺固定 75:25 混合比例。
各臂用相同 real episode、真实样本曝光次数和主干；额外 sim 预算单列。若要归因于数据而非训练更久，
补等总更新步的 real-only 对照。分别报告 sim 与 real，不以总体 loss 掩盖负迁移。

当前 real10 原型已用全部 10 条训练；不能冻结它之后再把其中两条称为全链路 held-out。
未来真实泛化实验需要先固定整 episode 的左右分层拆分，策略/critic/归一化全部遵守同一拆分，
另训独立实验权重；现场原型原封不动保留。五折可在方法有信号后做，不是本次前置工作。

任务终点进入左/右分支与 Elite 引导规范性分开评价。左支自然进入的捷径不能当引导能力；最新十条
“Elite 没掉队”也不等于已提供候选级引导因果标签。没有可信可观测标注时，先不训练 guidance 分数头。

## 8. 方案阶段产物与证据状态（实现前记录）

项目 skill 将本轮限制在算法文档及只读依赖检查；实验规划 skill 用于固定对照、监督来源、允许结论和
尚缺的测试条件。本轮未生成 mock 数据、论文结果章节、图表、训练脚本或新模型，因此没有新增训练命令。
文献输入仅采用上述两篇原文的相关方法段；报告其余推荐不扩展成实现任务。

检查项：旧 JSON 重算与原 summary 一致；单步 PI0.5 由配置/推理源码共同确认；本方案、交接和命令
文档相互链接。新增 optimizer steps / model forwards / environment steps / hardware actions 均为 0。
数据、SOFA、共享合同和周会日志不改；稳定跨轨结论出现后再追加周会事实。

## 9. DS0 实现与训练前检查（2026-09-16）

新增 [评分器](../tools/pusht_direct_action_scorer.py) 和
[运行入口](../tools/run_pusht_direct_action_scorer.py)，未修改原模型、训练器或数据加载器。
SSH 可用，两个最终小文件顺序 SCP 后确认远端入口；`--stage check` 一次通过，内部耗时 0.379 秒。
检查未构造 optimizer，所有参数/缓冲区保持不变，训练输出目录仍不存在。

- 原训练/验证窗口与训练目标不变。目标统计只取 21958 个训练窗口：均值 0.043009500695、
  总体标准差 0.104149264582；正/近零/负变化为 11247/5699/5012，未重采样、删帧或造负例。
- 64 行缓存 batch 的 t+8 索引与原 NumPy goal-cost 计算一致；图像、状态、全部八步动作和网络参数
  均有有限非零梯度。随机初始化的梯度只证明接线，不证明学到了动作效果。
- 初始状态内存往返一致；原五候选接口、无效当前观测保留 ACT、纯选择边界/平局 fixture 通过。
  固定 goal 存为 checkpoint buffer，不把未来观测放进 forward。
- 实现了显式 `--stage train`、固定 final 评价与 `--resume`；这三个执行路径尚未实际运行，
  本轮只验证训练前接口，不宣称训练、恢复或选择收益已验证。

证据：[准备 report](../simulation_output/pusht_direct_scorer_pretrain_v1/report.json)、
[监督对齐图](../simulation_output/pusht_direct_scorer_pretrain_v1/supervision_alignment_zh.png)。
图复用原先固定的训练 episode1/frame55，而非按目标收益挑选；0.377342 - 0.157283 = +0.220059。
agent 检视状态为 `viewed_not_accepted`，原始 report 保留创建时的 `not_viewed`。图只展示监督，
不是预测效果。没有验证集模型推理、旧候选读取、源 RGB 解码、新环境步或实机动作。

下一步仅是获批后执行固定 10000-step DS0 训练；不把本次通过当作 history 回放、候选测试、
dynamics-aux、真实数据重训或现场接入的授权。

## 10. DS0 固定训练与结果回读（2026-09-16）

第 9 节记录的是此前准备阶段；随后用户批准固定运行，远端已完成且未追加训练步数。
seed20260915、10000 updates、batch64、原 split/goal/归一化保持不变。全部 100 个日志区间有限，
optimizer 状态为 10000，final checkpoint 载入后的预测一致；本次未测试断点恢复。
训练和 final 评价计时 16.704 秒（不含启动/SSH），不能当作在线延迟。

2002 窗口 / 20 episode 既有开发验证集：DS0 目标改善量 MAE **0.041895673**；
零变化 0.064914733，单帧残差 0.038677901，双帧历史 0.037660427。
DS0 比零变化低 35.46%，但比残差高 8.32%、比历史高 11.25%；episode 宏平均结论相同。
其效果为负的 348 个窗口 MAE 0.042207383，未胜过零变化 0.039479997。
残差/历史数值读取原 final 报告；数据、预算与最终抽样器状态一致，但模型容量、监督目标与历史
输入并不完全匹配，不能据此宣称直接 scorer 架构本身被否定。

固定验证 episode0/frame76 未换样例：真实改善 +0.031422，DS0 预测 -0.100628。
该错误片段保留展示，不挑好案例。新增回读工具只做一次该样例 forward，不重跑全量验证；
图中当前/未来/目标全为实际观测，DS0 只输出标量。
证据：[训练报告](../simulation_output/pusht_direct_scorer_train_v1/report.json)、
[对比回读](../simulation_output/pusht_direct_scorer_inspection_v1/report.json)、
[固定片段图](../simulation_output/pusht_direct_scorer_inspection_v1/fixed_validation_direct_scorer_zh.png)。
中文图已检视，无缺字或裁切；状态为 `viewed_not_accepted`，原始 report 保留 `not_viewed`。

**决策：保留 ACT 与 real10 原策略，DS0 仅作为诊断基线。**预测误差不等于候选排序收益；
下一步提议是单独获批后对已有候选做离线评分，读取已保存的真实后果，不新增训练、环境回放或
DS1。它仍是复用开发诊断，不是独立测试，也没有真实导丝、低可见度或 sim-real 有效性结论。

## 11. DS0 既有候选离线比较（2026-09-16）

用户批准后已执行，复用 28 个上下文 / 140 个已保存候选后果；同样 27 个上下文有效，
200004/0 仍回到 ACT。一次 DS0 forward 产生 135 个评分，随后才打开未来 target。
原动作、偏移、目标、观测有效性与评价容差不变；残差原报告可精确复现。
新增训练步、环境步、ACT 推理和实机动作均为 0。

| 27 个有效上下文均值 | 残差预测后评分 | DS0 直接评分 |
| --- | ---: | ---: |
| 覆盖率增益，相对 ACT | 0.003460159 | 0.003484408 |
| 距 coverage oracle 的损失 | 0.003956638 | 0.003932389 |
| 比 ACT 差的上下文 | 1/27 | 1/27 |
| 最坏覆盖率增益 | -0.041008277 | -0.025082150 |
| 有差异的 7 处中选中最优 | 4/7 | 2/7 |

覆盖率均值差只有 +0.000024250。逐上下文 2 好 / 22 平 / 3 差，逐 seed 2 好 / 6 平 / 2 差。
DS0 减轻了 200006/160 的旧误选，但没有解决它，也漏掉其他有利动作。
该例仍贡献总 regret 的 67.95%；200000/80 贡献净收益的 82.12%。
仅 5 个 seed、7 个上下文有覆盖率差异，不能把候选对当独立样本，也不能把极小均值变化当成功率提升。
全部上下文、逐 seed、object-goal 与 coverage 指标、uniform 解析期望和两种事后 oracle 均保留。

证据：[离线报告](../simulation_output/pusht_direct_scorer_ranking_dev10_v1/report.json)、
[固定旧案例图](../simulation_output/pusht_direct_scorer_ranking_dev10_v1/saved_candidate_direct_scorer_zh.png)。
图中只有已保存的实际后果网格，DS0 没有生成未来图像。两例在评分前固定，不因本次好坏换例。
仅重绘过一次列标题对齐，未重做推理或统计；图已检视，`viewed_not_accepted`，原报告 `not_viewed`。

**决策：未显示稳定优势，不采用 DS0 替换 ACT/real10，也不据此自动加 DS1 或继续调这 10 个 seed。**
若继续验证，先冻结现有两种权重，按第 6 节在未用 seed 上做固定预算、共用候选后果的比较。
不能按事后 headroom 挑状态或不断补样直到收益变好。本轮没有冻结新测试 seed 或启动环境执行。

## 12. 新 seed 固定对照协议（2026-09-16，执行前锁定）

用户已批准下一步新 seed 对照。工程验证协议如下，不是论文 Results、mock 数据或实机实验。
入口为 `tools/run_pusht_fresh_scorer_pair.py`；先 `--stage prepare` 清点历史 seed 元数据、
写入 `protocol.json` / `seed_inventory.json`，再 `--stage run`。若 seed 发生重叠或 checkpoint
身份改变则停止，不能临时换 seed。历史来源为项目内 Push-T report/started/protocol/manifest，
以及已锁定 benchmark/smoke/development seed 表；并不掌握公开示范数据原始生成 seed。

- 固定 **300000–300019，共 20 个 seed**。不得根据后果增加或替换样本。
- ACT final100000 只负责原始轨迹前缀；DS0、残差均使用现有 final10000，不训练、不选新 checkpoint。
  两者原训练数据/split/目标/归一化一致，架构容量和监督方式不等同，因此不是纯单因素架构消融。
- 每个 seed 固定前缀 **0/80/160** 三个决策点，最多 60 处；沿用 ACT 8 步原动作加四个
  ±X/±Y ramp，末步偏移 8 native 单位。五候选不裁剪、不扩充、不优化。
- 当前观测 RGB/agent XY 和固定训练目标进入评分。两模型选择持久化后，才执行相应后果；
  前缀始终由 ACT 驱动，不由 DS0/残差选择改变。所有方法共用每个候选的一次真实环境后果。
- 用相同 reset + 记录动作重放前缀，不注入隐藏状态。逐步核对重放观测，候选 0 核对原 ACT
  后续观测、覆盖率和 done。只对必要的 reset/checkpoint/既有冻结合同做身份核对，不逐文件新增哈希。
- **最多 29760 环境步**：nominal 3360、前缀重放 24000、候选执行 2400；最多 300 个 8 步候选。
  原环境 episode cap=300。提前 done 立即停止，不在结束后填充动作。
- 主评价群体：当前物体可见、五候选均在原 native bounds 内、五条后果均完整 8 步。
  覆盖率群体不再按未来图像是否可见筛选；物体目标指标额外要求五张终点图像可见，单独列出人数。
  这避免把目标图像缺失混成标签零或悄悄改变主覆盖率群体。全部排除、未到达 anchor 均报告，不替换。
- **主比较**：按 seed 先平均，再比较 DS0−残差的覆盖率增益差。同步报告上下文均值、
  regret、低于 ACT 的比例、最坏损失、oracle headroom、无差异比例、仅有差异时的 top1/pair。
  ACT、uniform 解析期望和相应指标的事后 oracle 为参照；oracle 从不进入模型。
- 原容差不变：coverage 1e-6、object-goal 1e-7。提供逐 seed 值与配对差的样本标准差；
  不将同一 seed 的候选/帧当独立试验，不凭单个训练 seed 宣称训练稳定性或正式成功率提升。
- 固定展示 seed300000 的全部预定 anchor/候选实际 RGB，包含失败、缺失与不可见情况。
  没有新低可见度、sim-real、history、DS1、速度优化或 XAI 贡献，相关能力不作结论。
- 停止条件：完成 20 个 seed 即停止，不按效果延长；技术失败保留 partial 输出、先查原因，
  不自动换 seed、清空目录或重试。评分器不会自动采用为策略。

效率估计依据旧运行：10 个开发 seed、13508 环境步耗时 26.633 秒。本轮预计 1–3 分钟，
在已批准范围内直接执行；记录实际总耗时，但不把它当在线推理延迟。未新增 smoke 套件。
输出为 `simulation_output/pusht_fresh_scorer_pair20_v1/`：协议/seed 清点、逐步动作、预测、
共享后果、实际 RGB、状态及最终报告。主指标只回答新初始化下的短时候选选择，不替代闭环。

## 13. 新 seed 对照结果与停止决策（2026-09-16）

第 12 节协议先锁定后执行，未根据结果换 seed、补样或改变指标。准备扫描 70 个项目元数据文件，
新 seed 与记录无重叠；公开示范原始生成 seed 仍未知，不能声称已验证训练初始状态绝无重叠。
远端一次完成 20 个 seed，内部计时 48.541 秒。环境步 28375/29760，其中 nominal3255、
prefix replay22800、candidate2320；训练步和实机动作均为 0。

实际到达 58/60 个 anchor，执行 290 个完整 8 步后果。300011 在 ACT 第 63 步结束，缺少 80/160；
300006/0 当前图像物体无效，两选择器都回到 ACT，不进入主要比较。因此是 57 个有效上下文 / 20 个
seed；无越界候选、裁剪或未来不可见排除。逐步重放及 candidate0 对原 ACT 的一致性全部通过。

以下增益/损失按 **seed 先平均、再宏平均**，是原始 coverage 单位，不是成功率：

| 方法 | 相对 ACT 增益 | 距 coverage oracle 损失 | 比 ACT 差的上下文 | 最坏选中增益 |
|---|---:|---:|---:|---:|
| ACT | 0 | 0.005262767 | 0/57 | 0 |
| 均匀候选解析期望 | 0.000555665 | 0.004707102 | 期望 6.2/57 | -0.093575498* |
| 残差预测后评分 | 0.003860830 | 0.001401937 | 5/57 | -0.014045329 |
| DS0 直接评分 | 0.003946307 | 0.001316460 | 5/57 | -0.019613539 |
| 事后 coverage oracle | 0.005262767 | 0 | 0/57 | 0 |

*均匀策略的 harm 是候选分布下的期望；最坏值是非零概率支持中的最小增益，不是期望均值的最坏值。

主配对差 DS0−残差为 **+0.000085477**，逐 seed 差的样本标准差 0.002551210；
20 个 seed 为 **3 好 / 13 平 / 4 差**，没有做“显著优于”的推断。
按上下文均值，增益/损失为 0.004154007/0.001385747 对 0.004064032/0.001475723；
逐上下文 DS0 相对残差 3 好 / 48 平 / 6 差。两者相对 ACT 都为 11 好 / 41 平 / 5 差。
17/57 个上下文、12 个 seed 有可分辨覆盖率差异；其余 40 处平局不等于长时动作效果相同。
有差异时选中最优为 DS0 6/17、残差 9/17；pair ordering 为 99/155 对 112/155。
不能把这些候选对当独立 episode。DS0 净收益的 **93.46% 来自 300007/300009 两个 seed**，
取得 headroom 的 74.99% 仅为均值之比；不能据此给出稳定泛化或部署结论。

同 57 个上下文的 object-goal seed 宏平均增益/损失，DS0 为 0.002988999/0.002456060，
残差为 0.003570073/0.001874986；均有 6/57 处恶化。18 个 object-informative 上下文中
top1 为 7/18 对 9/18。直接评分在其训练目标上的选择表现反而弱于残差。

**额外事后诊断，不改主指标：**只读连接已存 `outcomes.jsonl`，每组按真实终点 object-goal cost
最小选择（精确平局取最小 candidate ID），再看该动作的 coverage。真实 object oracle 的 coverage
增益为 0.004736351，距 coverage oracle 仍有 0.000803404；在 300001/160、300007/80、
300010/160 三处甚至比 ACT 差。说明视觉代理目标与实际覆盖率不完全一致，不能把所有错误都归于
预测器容量不足。此处没有新 forward、训练或环境执行，也没有把真实未来/coverage 写入模型输入。
动作分布从示范到 ACT 邻域候选的偏移仍是另一假设，本轮未把它与表征不足做因果区分。

证据：[完整报告](../simulation_output/pusht_fresh_scorer_pair20_v1/report.json)、
[执行前协议](../simulation_output/pusht_fresh_scorer_pair20_v1/protocol.json)、
[固定首个 seed 的全部案例图](../simulation_output/pusht_fresh_scorer_pair20_v1/first_seed_shared_candidates_zh.png)。
本地与远端均保留 58 个预先评分、290 个共用后果、348 张实际 RGB 和执行日志。
图中文字/布局已检查，状态 `viewed_not_accepted`；原报告保留生成时的 `not_viewed`，不冒充用户验收。

**决策：保留普通 ACT / real10；DS0 与残差仅作诊断参考，本轮到预算即停止。**
短时选择存在局部正信号，但直接评分未展示对残差的稳定优势，最坏误选反而更差。
暂缓 DS1，不把“再加世界模型辅助”作为默认下一步；先确定一个能区分目标代理失配与动作支持不足的
最小对照方案，再另行决定实现/训练。不是否定 dynamics-aux 研究假设，也不继续按现有坏例子调参。
本轮不增加 history、低可见度、sim-real、真实模型重训、闭环或实机能力结论，数据/SOFA/共享文档和
周会日志不改。

## 14. 目标对齐审查：目标失配与模型误选同时存在（2026-09-16）

用户批准后只读取现有 57 个有效上下文 / 20 个 seed，未重跑模型、训练、环境或实机。
新增 `tools/audit_pusht_goal_alignment.py`，SSH 通过，最终小文件先 SCP 后远端执行；
一次审查内部耗时 0.209 秒。原 285 个候选终点 RGB 的 24x24 代价重算误差为 0，
原 gain/regret 精确复现；没有重新挑选目标图、样本或候选，没有改写标签。

### 两个目标不是同一个量

- 当前视觉目标：训练 episode1/frame117/global278 的灰色物体可见轮廓，与候选未来图像计算
  quadratic soft-Dice；这是“接近某个示范终点”的代理量。
- 原生 benchmark：已核对远端 `gym_pusht/envs/pusht.py` 的 `_get_coverage`，它计算真实物体几何与
  任务目标几何的交集面积除以目标面积。reward 再按成功阈值缩放并截断，不能把 reward 无条件等同
  未截断 coverage。该几何真值只用于离线诊断，不进入评分器观察。

令 C* 为候选最高覆盖率，Co 为真实视觉代价最小候选的覆盖率，Cs 为模型选中动作的覆盖率：

```text
总覆盖率 regret = (C* - Co) + (Co - Cs)
                 目标代理项    模型偏离视觉最优的有符号项
```

按 seed 宏平均：

| 模型 | 总 regret | 目标代理项 | 有符号模型排序项 |
|---|---:|---:|---:|
| DS0 | 0.001316460 | 0.000763234 | 0.000553226 |
| 残差 | 0.001401937 | 0.000763234 | 0.000638703 |

这是逐项严格相等的描述性分解，**不是因果归因比例**：第二项可以为负，模型把视觉目标预测错，
有时恰好反而选中了覆盖率更好的动作。DS0 该项正/零/负为 8/46/3 处，残差为 6/48/3 处。
因此“更准确地预测当前视觉目标”并不保证 benchmark 指标单调提高。

### 有害选择如何分布

DS0 的 5 处相对 ACT 恶化中，4 处没选对自己的视觉目标，而真实视觉最优不会恶化；
另 1 处（300010/160）选中了真实视觉最优，却损害 coverage。残差对应为 3 处与 2 处。
DS0 最坏例 300005/160 是明确的模型排序错误：视觉与覆盖率最优都选 candidate2，DS0 选了 4。
这否定“只改评分目标就能解释和解决所有误选”的简单归因；并未确定误选来自容量、监督或动作分布。

真实视觉最优的三处伤害，在考察原 1e-7 容差内全部近似最优动作后仍然存在，不是任意平局选择。
原掩码与目标保持不变，仅把事后 Dice 从 pooled24 改为原始 binary96：仍是同三处伤害；
57 处中仅 4 处 oracle 选择改变，平均覆盖率 regret 为 0.000862554，对比原 0.000803404。
因此没有“提高空间分辨率就修好”的证据。该比较没有单独分离目标姿态偏差、遮挡、感知误差和
目标函数形式，也没有训练 96 维版本或把它采用为新评分器。

证据：[审查报告](../simulation_output/pusht_goal_alignment_audit_v1/report.json)、
[全部逐上下文分解](../simulation_output/pusht_goal_alignment_audit_v1/per_context.jsonl)、
[固定诊断案例图](../simulation_output/pusht_goal_alignment_audit_v1/goal_alignment_cases_zh.png)。
图固定为上轮已报告的三个代理伤害与 DS0 最坏伤害，包含不利案例，文字/布局已检视；
状态 `viewed_not_accepted`，原报告保留 `not_viewed`。原模型和旧报告保持不变。

### 下一步建议，不在本轮启动

保留 ACT/real10，暂缓 DS1。先准备 **任务对齐的评分目标与数据接口**，核实原训练数据能提供的
真实已执行动作结果标签；保持网络、候选、训练预算不变，再单独比较监督目标的作用。
公开仿真若有 exact coverage，可以作为隔离的离线训练 target，不是可部署 observation；
没有标签时不能伪造或从截断 reward 无条件逆推，亦不能把其当真实导丝监督。
本轮 57 个已看上下文只能用于诊断，不能训练后仍声称在其上做 untouched test。
动作支持不足仍是待检验假设，不能用本审查宣称已证实，也不继续扩大模型来掩盖目标问题。

## 15. 记录 reward 监督接口：可用性已验证，暂不训练（2026-09-16）

本节保留 v1 创建时的状态。原生几何对应关系与缺失初始值的后续处理已在第 16 节解决，
并未回写、覆盖本节对应的历史工件。

用户批准后，仅审查原始公共数据标签并实现最小算法侧适配器；没有新增轨迹、训练、模型前向或
实机动作。原数据、特征缓存、DS0/残差结构与权重、loss、候选规则、ACT/real10 全部不变。

### 原数据究竟提供什么

固定版本 `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19` 有 25,650 帧 / 206 条轨迹。
原 Parquet 有 `next.reward`、`next.done`、`next.success`，没有 exact coverage 或物体 pose。
reward 范围 0..0.948879719，10,396 个不同值；success 全 false。不能据此把所有示范定性为失败，
也不能把 done 当成功标签。

[官方历史转换器](https://github.com/huggingface/lerobot/blob/8e7d697/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py)
在第 115–149、176–185 行先从原始物体位姿计算裁剪的 `coverage/0.95`，再把结果字段前移一帧，
末尾重复最后一个结果。[其引用的历史环境版本](https://github.com/huggingface/gym-pusht/blob/e0684ff988d223808c0a9dcfaba9dc4991791370/gym_pusht/envs/pusht.py)
定义了相应几何函数。当前快照全部 206 条轨迹都符合末尾重复模式：412 个 done=true 恰好位于
各轨迹最后两行，reward/success 最后两行相等。

这是“官方历史语义 + 快照边界模式一致”的证据，**不是逐帧几何重算**。快照没有标明精确的
原始转换程序/依赖运行版本，也没有保留原始物体位姿，因此尚未证实 recorded reward 与当前
`gym-pusht 0.1.6` 原生 coverage 的严格对应关系。本轮不乘 0.95 制造所谓 exact coverage。

### 固定的最小接口

新增 `tools/pusht_recorded_reward_targets.py` 与准备入口
`tools/prepare_pusht_recorded_reward_targets.py`。前者包装现有 `ObjectData`，不改原加载器。
后者只读数值 Parquet 与已有 object cache；生成独立的 `manifest.json`、`targets.npz`、
`report.json`，位于 `simulation_output/pusht_recorded_reward_targets_v1/`，两台机器均有小文件副本。

| 部分 | 固定定义 |
|---|---|
| 模型输入 | 当前 24×24 RGB 物体占据网格、当前可观测 agent XY、已执行的 8 步绝对 XY 动作 |
| 输出监督 | `terminal_recorded_reward_8step = next.reward[t+7]`，按历史约定对应 t+8 帧结果；越大越好 |
| 输入形状 | `[B,24,24]`、`[B,2]`、`[B,1,8,2]`；监督单独返回 `[B,1]` |
| 不进入输入 | reward、done、success、未来网格、episode/frame ID、旧候选实验结果 |
| 窗口 | 保留全部原 21,958 训练 / 2,002 验证窗口、原 episode split 与 all9-visible 条件 |
| 标签归一化 | 仅训练窗口，均值 0.2984111475702125，总体标准差 0.2741044058242909 |
| 初始 reward 缺失 | frame0 无前一行可恢复；保持 NaN + false mask，不补零、不跨 episode、不自动删窗 |
| 末端 padding | 读取 t+7 行真实转移，最多对应末帧；从不使用重复填充的最后一行作为新转移 |

采用末端 reward 是为了不损失原窗口；若采用 `reward[t+8]-reward[t]`，训练/验证分别只有
21,778 / 1,982 条可构造，180 / 20 条缺少 episode 初始 reward。增量目前只作带有效性掩码的
离线统计，不提供 delta 训练选项。不能把缺失初始 reward 视作 0 或拿 t+1 代替 t。

### 实际验证与决策

最终 15,460 字节代码 SCP 后远端大小回读，单次 CPU 准备内部耗时 **0.277 秒**。
全量源行 episode/frame/state/action 与既有缓存逐值一致；完整窗口不跨轨迹；所有末端标签有效。
一个固定 16 窗口真实 batch 中，三项输入与原 loader 完全相等，目标索引精确一致，标准化有限。
没有解码视频、构造/执行模型、加载权重、执行 optimizer 或环境步；schema-only 改动不增加图片验收。
[报告](../simulation_output/pusht_recorded_reward_targets_v1/report.json)记录了全部计数与 batch 对齐例子。

**接口已完成，但不能立即声称任务对齐训练已就绪。** 下一步最小工作是：

1. 取得或定位原 Push-T 的小型数值 state/episode-boundary 文件，核对记录 reward 的几何与时序
   语义；不需要再下整套视频。本轮没有替用户下载新数据或重算几何。
2. 固定真正匹配的监督对照：原 DS0 是视觉“增量”，当前接口是“末端值”。对同一上下文的精确值，
   二者排序只差与候选无关的常数，但对回归模型并非同一学习问题。建议未来比较
   `-Dice(grid[t+8], fixed_train_goal)` 与 `terminal_recorded_reward`，两边同结构、原 split/window、
   相同步数和候选；或者补回初始 reward 后再做增量对增量。暂不新增训练入口或执行训练。

不要用旧 57 个已看候选结果做监督后仍称其为 untouched test。此接口只适用于当前公开 Push-T
离线诊断，不是导丝任务的新标签，也不证明真实混合训练、低可见度收益、策略成功率或架构创新。
继续保留 ACT/real10，暂缓 DS1；数据/SOFA/共享文档及周会日志不改。

## 16. 全量 reward 几何复算通过，末端监督公平对照已准备（2026-09-16）

### 数值证据消除了什么疑问

从[官方原始数值数据](https://huggingface.co/datasets/lerobot-raw/pusht_raw/tree/f93fc57921866e96d8ae00efbe5bb146a04b088f)
只下载 state、action、episode_ends 的 327 个小文件，共 430,143 字节；没有下载图像。
原始 25,650 行 / 206 条 episode 与既有快照的 agent XY、动作、episode 边界、frame index
逐值完全一致。远端仅增加 `numcodecs==0.13.1`，没有升级 PyTorch、LeRobot 或训练依赖链。

`tools/audit_pusht_reward_geometry.py` 直接调用已安装 `gym-pusht 0.1.6` 的静态覆盖率函数：
先设角度，再设记录的位置，保持真实记录的 body pose；**没有实例化/reset/step 环境**。
比较的是 `clip(coverage/0.95, 0, 1)`，再按 episode 内前移一帧、重复末帧的原转换约定对齐。

| 数值假设 | 对原始 next.reward 的 MAE | float32 完全相等行数 |
|---|---:|---:|
| 保持记录 pose，取下一帧 | 5.970780789e-9 | **25,650 / 25,650** |
| 保持记录 pose，不前移 | 0.008718268 | 15,320 / 25,650 |
| 历史 position-before-angle 重建，取下一帧 | 0.163459673 | 3,924 / 25,650 |

正确重建的最大误差只有 2.978820113e-8，转换 float32 后全部精确相等。
因此本轮确认：**现有 reward 的几何语义和 next-frame 时序是正确的，并没有发现 reward 转换错误**。
历史 position-before-angle 会绕非零重心改变 body position，但它不符合本快照，不能把这一
代码历史差异误报为当前数据 bug。固定后的配对监督可直接使用原始 reward，无须改标签。

该数据记录的最大 coverage 为 0.901435707，reward 最大为 0.948879719；本源没有触发 reward
上限截断。这不让 reward 在一般场景下等于 coverage，也不允许对饱和值无条件逆推。
success 全 false 与保存下来的几何帧一致。[官方采集循环](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/demo_pusht.py#L77)
先存储当前帧与动作，再执行 step，完成后退出；所以没有保存成功后的观测，不足以判定完整示范失败。
此处不重跑示范，不替缺失的最终结果补成功标签。

原始 state 也恢复了全部 episode 起点的静态结果；第 15 节的 180/20 个缺失初始 reward 不必补零。
但本轮不修改 v1 sidecar，不切换为增量监督，继续固定两边都预测末端值，避免增加实验变量。

### 最小公平对照，尚未训练

`tools/prepare_pusht_terminal_score_pair.py` 已生成真实标签和机器可读对照约定：

| 项目 | 固定设置 |
|---|---|
| 视觉组 | `visual_terminal = -Dice(grid[t+8], 原固定训练goal)`，越大越好 |
| 任务组 | `reward_terminal = next.reward[t+7]`，已与 t+8 原生几何 reward 逐值核对 |
| 模型/输入 | 复用原 DirectActionScorer、当前 RGB 网格、可观测 XY 和同一候选动作接口 |
| 不进入输入 | reward、success/done、未来帧、原始物体 pose、exact coverage、episode/frame ID |
| 数据 | 原 21,958 train / 2,002 val 窗口、episode split、可见性条件和归一化全部不变 |
| 初始化/采样 | 两组各重置相同 seed20260915；初始可训练参数、每一步采样的 anchor 顺序相同 |
| 优化 | 各 10,000 步，batch64，AdamW lr3e-4/wd1e-4，clip1；原标准化标量 MSE 形式 |
| 标签归一化 | 只用训练标签；视觉 mean/std=-0.690676155/0.275651602，reward=0.298411148/0.274104406 |
| checkpoint | 各自 fixed final10000，不用验证集挑选中间模型、阈值或网络配置 |
| 评估边界 | 原验证窗口的各自 score MAE/RMSE，分别报告窗口平均与 episode 宏平均；两种单位的 MAE 不直接比大小 |
| 效率/泛化 | 记录同台 4090 的耗时/峰值显存；仅单 seed、单公共任务，不新增鲁棒性、sim-real 或架构结论 |

训练 reward 中零值 3,753 / 21,958，验证零值 385 / 2,002；保留这些标签并报告分布，
不为制造变化而删掉零值或重采样。旧 DS0 预测的是视觉增量，**不能直接冒充本次视觉末端组**。
今后的两组都需要在相同预算下重新训练；本轮没有启动训练，也没有为旧 checkpoint 改名冒用。
候选排序对照须在训练后另行明确范围，保持同一上下文/候选；57 个已看案例仍是开发证据，
不能训练后再称其为 untouched test，更不能把 demonstration 单一动作结果当未执行候选的真值。

### 产物、验证与下一步

- [全量数值审查](../simulation_output/pusht_reward_geometry_audit_v1/report.json)：44.188 秒含小文件下载，
  其中静态几何 6.746 秒；几何数组独立在 `geometry_diagnostic_targets.npz`，仅离线使用。
- [固定对照约定](../simulation_output/pusht_terminal_score_pair_pretrain_v1/comparison_contract.json)与
  [标签准备报告](../simulation_output/pusht_terminal_score_pair_pretrain_v1/report.json)：0.063 秒，
  两组全部窗口保留，reward 标签全部匹配原生重算 float32；配对标签为 `paired_terminal_targets.npz`。
- 实际 optimizer / model-forward / checkpoint-load / environment / hardware 均为 0；没有额外 smoke
  套件、视频解码或逐文件 SHA-256。只是数值/schema 审查，不需要无关的视觉验收。
- 采用 experiment-results-planning skill 的公平性约束，明确目标参数化、预算、基线、单位与声明边界；
  当前是工程实验约定，不写论文 Results、不造 mock 数值，也不扩成一套新 benchmark。

**决策：数据监督语义与公平对照约定已就绪，不再卡在 reward 来源问题。下一步是实现配对训练入口，
在明确授权后按固定预算执行；不是继续改网络或扩大审查。** 当前未实现训练 runner，不能宣称已训练
或已有策略收益。保留 ACT/real10，暂缓 DS1；只更新算法交接/命令/本方案，其他轨与周会日志不改。

## 17. 配对末端评分训练完成，尚未判断选动作收益（2026-09-16）

用户随后批准实现入口并执行固定配对训练。本轮新增 `tools/run_pusht_terminal_score_pair.py`，
本地语法检查后将最终 19,960 字节代码顺序 SCP 到 4090，并回读文件大小再运行。
原 `DirectActionScorer`、loss、候选规则、原始数据/缓存、ACT/real10 与旧 DS0/残差权重均不改。
没有新增环境轨迹、候选 continuation 或硬件动作。

### 实际执行的公平性约束

- 两组各 117,633 参数、固定 seed20260915、10,000 步、batch64、AdamW lr3e-4/wd1e-4、clip1。
- 仍为原 186/20 条 episode、21,958/2,002 个完整训练/验证窗口，不新增筛选或重采样规则。
- 两组初始可训练参数逐值一致；全部 10,000×64 个按序采样的窗口行也逐值一致，保存在独立文件中。
- 当前 RGB 网格、可观测 agent XY、8 步绝对 XY 动作作为输入；固定训练 goal 和 state/action
  normalization 不变。reward、未来网格、原始物体 pose、几何真值从不进入网络输入。
- 只有监督目标及其训练集 mean/std 不同。各自训练到 final10000 才验证，不挑中间 checkpoint。
- 原 preparation contract 中“当时未授权训练”的字段不回写；新的 `run.json` 记录本轮后续授权。

### 固定验证集结果

| 监督组（各自单位） | 窗口 MAE | 窗口 RMSE | episode 宏平均 MAE | 同目标训练均值常数 MAE | 训练耗时 |
|---|---:|---:|---:|---:|---:|
| 视觉末端 `-Dice` | 0.052068737 | 0.074851245 | 0.052103229 | 0.221856971 | 16.436 秒 |
| 任务末端 reward | 0.049758308 | 0.073433905 | 0.049305438 | 0.233537779 | 15.926 秒 |

两组都能在原验证集上比“固定输出训练均值”更好地预测自己的标签。这仅说明监督回归有信号；
**不能把 0.0498 小于 0.0521 解读为 reward 组动作选择更好**，两组学习的是不同量。
也没有用该表证明 action conditioning 真正有效、未执行候选的外推准确或策略成功率提高。

reward 验证集中 385 个零目标窗口 MAE 为 0.020852711；1,617 个非零窗口 MAE 为 0.056640593。
没有删除零目标，完整报告包含两个分层的 episode 宏平均及逐 episode 值。视觉组无零目标验证窗。
本轮只用一个训练种子、已重复使用的原开发验证集，不能作为独立新 test。

### 验证、工件与恢复

实际两组均从 step0 连续完成，无中断：各有 100 行有限 loss/梯度日志，step100 到10000无缺口；
final 中全部 optimizer step 等于10000；重新加载 final 后，两个固定验证窗口的前向输出逐值一致。
保存的 2,002 条预测独立复算 pooled MAE 与报告完全相同。每组 peak allocated CUDA memory
为 130,496,000 字节（不是整张 GPU 的总占用）；验证/重载另耗时 0.027/0.022 秒。
不新增 smoke 套件或逐文件 SHA-256。

远端训练根目录为
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_terminal_score_pair_v1/`，
共 9,029,907 字节。每组保留 final/last、日志、报告与预测；根目录保留初始参数、完整采样顺序及约定。
实现了每千步和受控中断保存、同一步数预算恢复；本轮没有实际触发恢复，不能称恢复演练通过。
入口拒绝覆盖完整结果。准确命令与恢复限制见算法命令文档。

小型证据回传到本地：

- [配对训练报告](../simulation_output/pusht_terminal_score_pair_v1/report.json)。
- [视觉组详细报告](../simulation_output/pusht_terminal_score_pair_v1/visual_terminal/report.json)。
- [reward 组详细报告](../simulation_output/pusht_terminal_score_pair_v1/reward_terminal/report.json)。
- [固定验证预测读回图](../simulation_output/pusht_terminal_score_pair_v1/validation_score_readback_zh.png)：
  固定前六个验证 episode 的中间窗口，显示当前输入、真实后继及预测/真值，未按表现挑例子。
  中文字形、排版及实例已检视，状态 `viewed_not_accepted`；原报告保留 `not_viewed`。
  图不是候选排序结果或策略 rollout。原线性评分头未加限幅，图中可见负 reward 预测及小于-1的
  视觉分数；这是未校准的回归分数，不是概率，本轮不因此改结构或加入裁剪。

**决策：配对训练已完成，下一步应单独比较同一批上下文、同一组候选上的排序与最终选择。**
先复用已有保存结果可避免再跑环境；旧 57 个已看上下文只能作为开发诊断，不能冒充 untouched test，
也没有作为本次训练标签。下一步不隐含新训练、新环境 rollout 或模型晋升授权。
在出现候选选择收益证据前，保留普通 ACT/real10，暂缓 DS1 和更大模型；不改其他轨和周会日志。

## 18. 固定候选对照完成：reward 提高均值，但没有稳定排序优势（2026-09-16）

用户批准后，新入口 `tools/compare_pusht_terminal_score_ranking.py` 复用已保存的上下文与候选后果，
没有新增训练、ACT 推理、环境 reset/step、候选续跑或硬件操作。最终 16,967 字节代码在本地通过
语法检查，顺序 SCP 并回读大小后远端执行，内部耗时 **0.795 秒**，两组各一次模型前向。
原模型/权重/数据/评估代码不改，也不再跑原验证集或下载/解码数据视频。

### 比较口径没有变化

源仍为 `pusht_fresh_scorer_pair20_v1` 的 58 个已保存上下文、290 个完整8步候选后果。
两组均使用其中 **57 个有效上下文、20 个已看 seed、285 个候选**。无效观测300006/0均保留 ACT；
300011未到达的80/160决策点不补选。当前 RGB 重新提取的有效性与原记录一致，候选动作重建逐值
一致，不裁剪、不加门控、不改 goal。真实后果在两组预测写盘后才读取，绝不进入模型输入。

本轮复用原聚合函数，并先完整复算历史报告得到逐值相同的 analysis；新模型只占用重命名后的
评分槽位。旧函数的高分优先/低代价优先入口仅作符号适配，不把 terminal score 当旧 DS0 delta。
沿用 coverage容差1e-6、视觉/预测容差1e-7，报告 seed宏平均、上下文均值、harm、最坏损失、
informative top1与候选对排序。旧文件名含 fresh，但**本轮不是新测试集**。

### 共同任务指标的结果

下表 gain/regret 都是 seed宏平均、原始覆盖率单位，**不是成功率**：

| 方法 | 相对 ACT 覆盖率增益 | 距最好候选的损失 | 比 ACT 更差的上下文 | 最坏增益 | 可分辨场景最优命中 |
|---|---:|---:|---:|---:|---:|
| ACT 原动作 | 0 | 0.005262767 | 0/57 | 0 | 1/17 |
| 视觉末端监督 | 0.000138652 | 0.005124115 | 10/57 | -0.037888898 | 6/17 |
| 任务 reward 末端监督 | 0.003422261 | 0.001840506 | 9/57 | -0.019613539 | 6/17 |
| 实际覆盖率最优（事后） | 0.005262767 | 0 | 0/57 | 0 | 17/17 |

均匀选择的解析期望增益为0.000555665、期望harm为6.2/57；不是另跑的随机策略。
两组 pooled context增益分别为0.000145949/0.003602380，与seed宏平均分开报告。
在同一目标下，reward组的平均损失与最坏损失确实更低，但不能只报这个有利均值：

- 两组 coverage候选对排序都只有 **81/155=52.26%**，最优命中同为6/17。
  57个上下文中只有17处、分属12个seed的覆盖率可区分；候选对不是独立轨迹。
- 两组40/57处选同一个候选；按实际覆盖率比较，reward是3处更好、52处持平、2处更差。
  分seed是3更好/16持平/1更差；reward减visual的均值+0.003283609，样本标准差0.012277375。
- **83.18%的配对均值优势来自seed300009**；其中300009/80由visual的-0.037888898改善为
  reward的+0.125998584，差值+0.163887483。不能把主要由少数场景带来的优势说成普遍可靠。
  去掉该seed后的19个seed差值均值仍为+0.000581211；这是描述性敏感性读回，不换主指标、
  不删掉该样本，也不作显著性或“完全没有其他收益”的结论。
- reward对ACT有7处改善、9处伤害；visual为6处改善、10处伤害。两组仍会主动改坏原动作。
- 旧DS0视觉增量模型在同批上下文的增益0.003946307、harm5/57，优于本轮reward末端模型。
  它不是匹配的terminal对照，只作为历史参照；本轮没有获得新的整体最优选择器。

视觉目标结果也不是全面改善：visual/reward的seed宏平均视觉收益0.002185933/0.003495060，
但视觉候选对排序104/164对96/164，informative top1为8/18对9/18。
285个候选的最大实际coverage为0.847839921，均低于reward饱和阈值0.95；本轮差异不是reward
饱和抹平候选排序造成的。未给原回归输出增加裁剪、sigmoid或概率解释。

### 工件与决定

- [完整报告](../simulation_output/pusht_terminal_score_ranking_dev20_v1/report.json)、
  [逐上下文对照](../simulation_output/pusht_terminal_score_ranking_dev20_v1/per_context.jsonl)、
  predictions.jsonl/npz、started/status已回传本地，完整目录552,950字节。
- [固定案例图](../simulation_output/pusht_terminal_score_ranking_dev20_v1/terminal_score_candidates_zh.png)：
  推理前固定旧300007/80正向DS0例、300005/160最坏伤害例、300010/160代理目标伤害例，包含
  新模型的不利选择。中文、排版与真实RGB已检视；状态`viewed_not_accepted`，原报告不回写。
- 两个模型都没有执行所选动作；所有后果来自同一套已保存候选续跑，没有闭环成功率证据。

**结论：任务对齐监督有局部正向信号，但只换成绝对末端reward没有解决候选排序可靠性。**
保留普通ACT/real10，不晋升任一新选择器，不扩DS1或模型容量，也不继续在这批场景调到好看。
不能将当前错误直接归因为网络容量、动作支持不足或目标参数化；本轮没有把这些原因隔离开。
下一步最小假设是监督“动作带来的变化/收益”而非绝对末端值；可利用已恢复的原始起点结果，
另行固定对照再执行。本轮不准备新标签、不新增训练，更不把这些候选真值当训练数据后又称独立测试。
其他轨、共享契约和周会日志均不改。

## 19. 同起点动作效应对比：接口准备完成，独立训练数据待生成（2026-09-18）

用户离开现场，实机工作暂停，并认可继续研究动作效应对比。本轮只实现最小算法侧接口、
固定实验协议和一次已有缓存上的计算验证，不采集新轨迹、不训练、不连接设备。
使用实验规划 skill 固定公平性、分组和证据口径；这仍是工程实验准备，不是论文 Results。

### 只检验一个机制，不包装成已成立的创新

沿用 `DirectActionScorer`，两个训练臂均为 **117633 参数**：

- pointwise：预测候选真实执行 8 步后的 coverage，使用标准化后的 MSE；
- pointwise_plus_effect_difference：同一个 MSE，加同起点十个无序候选对的差值平方误差的一半，权重固定 1。

两臂看到相同的五个候选及全部实际后果，初始化、batch 顺序、归一化、优化步数完全配对。
原 ACT 候选 0 是第三个无需训练的比较对象。普通评分器也获得全部候选后果，不能把新监督数据量
的优势归因于配对损失。保留真实平局，不把未执行动作标成负例，不跨起点配对。

若 e 是五个候选的标准化预测误差，配对项严格等于 `5/4 * Var(e, population)`。
它重新加权场景内部的误差，不创造新标签或独立样本，也不是新架构、InfoNCE 或已确立的创新。
同一起点减去相同的初始 reward 不改变理想排序；本轮也不靠这种常数变化宣称创新。
后续只有在独立数据上证明决策效应，再考虑可信仿真监督或世界模型辅助。

### 锁定的未来对照（尚未执行）

机器可读协议：[pusht-action-effect-contrast-protocol-v1.json](pusht-action-effect-contrast-protocol-v1.json)。
新 source/reset seed 按训练 410000–410031（32）、验证 420000–420007（8）、测试
430000–430019（20）预留，同一 source 的 0/80/160 三个固定起点和全部候选不能跨 split。
最多 96/24/60 个上下文，480/120/300 个实际 8 步候选后果；缺失、越界或提前结束明确记录，
不补选有利场景，不 padding，不按后果挑起点。当前图像有效性决定输入资格，不按未来图像可见性筛 coverage。
候选仍是 ACT+四个 XY ramp（offset=8），不是五个独立策略采样，也未证明扰动落在策略支持域。

三对训练随机种子 20260918/19/20，每臂各 2000 步、每 batch 16 个完整上下文，AdamW
lr=3e-4/wd=1e-4/clip=1；只用 fixed-final2000，不挑最佳验证 checkpoint、不集成、不搜索 loss 权重。
输入归一化继续使用原 ACT 训练集统计；target mean/std 只能由新的训练上下文计算并供两臂共享。
本轮 schema 检查使用 mean=0/std=1 的机械验证尺度，不从旧开发集拟合训练统计。

主指标为测试 source-seed 宏平均的 paired-minus-pointwise coverage gain，同时完整报告相对 ACT
gain、regret、harm、最坏损失、headroom、informative top1/pair accuracy、平局数和效率。
候选对不是独立试验；置信区间按测试 source seed 聚类，训练重复单独报告。
没有可靠选择收益或伤害增加则不晋升、不扩容、不在看过的测试集上调到好看。
这尚非整 episode 的选择器闭环成功率实验，也不支持真实端/低可见度/sim-real 结论。

### 已实现与实际验证

- `tools/pusht_action_effect_contrast.py`：不改模型结构的两臂 loss、观测/target 分离的 PairPack、
  source 分组检查、训练集统计函数；明确拒绝把 development schema probe 当训练 pack。
- `tools/prepare_pusht_action_effect_contrast.py`：读取既有 dev20 当前 RGB、可观测 XY 和原候选；
  先写 observation 和随机模型 forward，再连接旧实际后果 sidecar；无 optimizer、环境或设备入口。
- `simulation_output/pusht_action_effect_contrast_prepare_v1/report.json`：远端检查内部耗时 **0.138 s**；
  58 个旧上下文中 57 个有效、20 个 source seed、285 个实际候选，570 对中 155 对 coverage 可分辨，
  17 个 informative 上下文。原 300006/0 无效观测仍被排除，没有补选或修改原报告。
- 两臂初始参数/输出相同，loss 梯度有限且不同；配对项的误差方差等价、常数偏移不变性、
  候选同步置换不变性通过。参数没有更新，不保存训练权重；上述数值是接口证据，不是方法收益。
- 新 seed 范围与本项目当前顶层 Push-T run 元数据无交集；这不证明与未知原始数据生成状态独立。
  真正执行前仍需复核 source/reset 身份，不因 seed 不同就声明有效轨迹多样性。

本轮 `optimizer_steps=0, environment_steps=0, hardware_actions=0`。
**training_ready=false**：独立同起点配对训练/验证数据尚不存在，新的公平训练入口与最终测试评价入口
也尚未实现。现有 dev10/dev20 只保留为开发诊断，不改作训练或未触碰测试。
下一步是准备公共 Push-T 的独立同起点候选执行入口，复用原 ACT/原生 reset+prefix replay；
不是修改导丝采集器或等待导丝仿真数据。先交付有预算、输出与中断恢复说明的命令，再执行新数据生成。

## 20. 新训练/验证同起点候选后果已生成，测试组仍未运行（2026-09-18）

用户同意推进第 19 节之后，新增 `tools/collect_pusht_action_effect_pairs.py`，仅服务公共
Push-T 算法实验，不修改导丝采集器、环境、专家、标签或其他轨文档。沿用冻结 ACT、原生环境和
已经验证的 reset/prefix replay/候选执行函数，不实例化旧世界模型或旧评分器，也不训练任何模型。
项目 skill 的范围和传输规则用于限制本轮工作：最终小文件先顺序 SCP、远端读回大小，再执行。

### 原生执行与恢复

固定训练源 410000–410031、验证源 420000–420007 全部完成；不提供测试组执行选项。
每源最多 1488 步（168 原 ACT + 1200 回放 + 120 候选），整批上限 59520 步。
当前图像、可观测 XY、五个候选在对应未来执行前落盘；每个候选从相同 reset 和相同原 ACT
前缀出发，逐步核对当前 RGB/XY 完全一致，候选 0 的后继及 coverage 与原 ACT 一致。
使用 coverage 的位置只有 target/日志/离线汇总；不作为 ACT 或评分模型输入。

按 source 保存 `attempt_NNN/` 和原子 `done.json`。`--resume` 跳过完成源；中断源从其源起点
重做，旧 attempt 保留，不覆盖/拼接不完整轨迹。总预算和日志分别记录成功源步数及所有已记录
尝试的步数。`--stop-after-sources 1` 首先完成410000，实测5.538秒/1488步；随后 `--resume`
从410001继续剩余39源，92.443秒/54504步。40个源各只有一个完成 attempt，没有额外测试源。
这是源边界暂停/恢复证据，不是强制断电恢复测试；突然断电可能丢失最后一段未落盘计数。

第一次收尾 pack 有一个本地变量初始化个数错误（ValueError）。原40源和所有候选日志已保存。
修复后只运行 `--stage pack`，没有重新加载 ACT、reset 环境或重跑源轨迹。
最终完成状态 `completed_train_validation_candidate_pairs`；数据位于两台机器的
`simulation_output/pusht_action_effect_pairs_v1/`，完整回传包1,305,880字节，小于100MB。

### 实际可用数据，不补有利样本

| 分组 | 计划/执行源 | 有效源 | 有效上下文 | 实际执行/有效候选 | 有差异上下文 | 有差异候选对/全部候选对 |
|---|---:|---:|---:|---:|---:|---:|
| train | 32 | 32 | 92 | 460/460 | 40 | 375/920 |
| validation | 8 | 7 | 21 | 120/105 | 8 | 68/210 |

训练源410003、410004、410013、410025在160步起点前已结束，因此96个计划起点只到达92个。
验证源420007的0/80/160三个当前物体提取均无效；其15个候选后果保留在原始日志，但整个上下文
不进入模型样本。已查看该源0步原图，物体在画面下缘被截断；不修改原提取器、增加seed或把缺失
当负标签。验证有效来源因此只有7个，其中5个源包含可分辨的候选后果；有效训练源中24个有差异。
固定首个训练源410000和验证源420000的全部起点候选在8步coverage上均为平局，图中如实保留。
新配对数据不是全都“有用变化”，也不是根据结果挑出的容易样本。

40个初始观测身份互不相同，与已记录项目初始身份没有精确重复，训练/验证source无交叉。
这不证明未知原始示范生成状态无重叠，也不等于充分的几何/行为覆盖。
全部候选对来自113个上下文，不能把1130个候选对当独立轨迹样本。

`train/pack/observations.npz`、`validation/pack/observations.npz`只含current/agent_xy/actions；
对应targets只含terminal_coverage，contexts保存source/anchor路径元数据。PairPack读回及分组检查通过。
目标标准化仅从460个新训练候选计算：mean=0.21085390952974725，std=0.25246952881937657；
验证标签没有参与拟合。原ACT输入统计和训练goal不变，两臂必须共用此份normalization。

### 证据与下一步

实测共55992环境步，optimizer_steps=0，hardware_actions=0，test_sources_executed=0。
这是公共基准的配对数据准备完成，不是导丝正式数据就绪、评分器收益或实机验证。
固定来源对照图 `first_sources_pairs_zh.png` 和上述截断源原图已查看，状态
`viewed_not_accepted`；raw report的not_viewed保留，未将代理检视当用户接受。

`data_ready_for_fixed_scorer_experiment=true`，但 `training_entrypoint_ready=false`。
下一步实现同数据/同初始化/同batch的三对fixed2000训练及固定验证读回；沿用第19节全部参数，
不增大网络、调loss权重或改变候选。最终测试源430000–430019保留到两臂权重固定及评价入口完成后。
不把已查看的训练/验证数据称为最终测试，也不重复生成本批数据。周会日志与其他轨均未修改。

## 21. 普通评分器与效应对比评分器：公平训练入口已实现，暂不执行（2026-09-18）

用户暂停相机示例验证，要求回到主线实现公平训练入口。新增
`tools/train_pusht_action_effect_contrast.py`；按要求本轮不运行测试、训练、验证推理或环境，
也不连接硬件。项目 skill 用于限制算法轨范围、更新最少交接文档及同步最终小文件。
这是实现记录，不是运行通过、训练完成或方法有效的证据。

### 固定公平性与恢复

- 保持第 19 节协议原文和 `DirectActionScorer` 结构不变。普通组 `pair_weight=0`，
  效应差异组 `pair_weight=1`；复用已有 loss 函数，不加入新头、辅助任务或权重搜索。
- 每个训练 seed 只生成一份初始化，两臂 `deepcopy` 全部参数/buffer；保存 `initial_state.pt`。
  同一份 `[2000,16]` 上下文行号表供两臂共用并落盘，两臂在同一步直接消费相同输入和 target tensor。
  所有五个候选、真实平局均参与训练，不能只给对比组更多后果信息。
- 完全沿用 92 个训练上下文、21 个验证上下文及其 source 分组；训练统计不使用验证标签。
  标准化用原 ACT 训练输入统计和本批新训练 coverage 统计，训练 goal 仍为 episode1/frame117。
  保留小型数组快照供恢复时直接比较，不逐文件算 SHA-256。
- 三个训练 seed 固定 20260918/19/20，各两臂 fixed2000，batch16，AdamW3e-4/wd1e-4/clip1。
  总计 12000 个优化器更新；不在命令行提供步数、seed 或 loss 权重调参入口。
- 每 100 个配对更新及正常停止时，原子保存同一 `last.pt` 内的两个模型、两个优化器和共同 step。
  Ctrl+C 等两臂当前更新都结束后保存；突然中断从最近完整配对 checkpoint 恢复，不保存半对。
  已完成 seed 不再优化；最后验证阶段中断只需重跑最终读回。这里是代码提供的恢复路径，尚未实测。

### 最终验证与推理边界

先完成六个 fixed-final2000 权重，再读取验证表现，不选最佳 checkpoint/seed、不集成。
预测原始 terminal coverage，不将输出截断为 `[0,1]`，不改候选、不增加 abstention gate。
并列预测用 `argmax` 首索引，ACT 为 candidate0。旧方法名 `predict_effect()` 在本实验返回的是
terminal coverage score；新模块 `load_final()` 区分该实验与旧 DS0/末端值 checkpoint。

每对保存两臂验证预测后再连接 target 汇总：coverage MAE、相对 ACT gain、oracle regret、harm、
最坏损失、headroom、informative top1、排除真实平局的 pair accuracy 和显式预测平局数，
以及 source 宏平均/上下文平均/每源明细/三个训练重复。预测平局不算排序正确，候选对不算独立轨迹。
Uniform 只算五候选等权期望，oracle 只作事后上界。记录各臂优化计算耗时、固定 warm forward 时间；
warm 时间不包含视觉提取或输入搬运。训练重复标准差不是最终测试 source 聚类置信区间。

所有输出默认位于 SSD：
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_action_effect_contrast_v1/`。
启动与恢复分别为 `--train`、`--train --resume`，完整命令见算法命令文档当前节；不启动旧训练器。
本轮状态：`training_entrypoint_implemented=true`、`training_entrypoint_runtime_verified=false`、
`new_optimizer_steps=0`，没有新权重或评估数字。原采集报告的历史字段不回写，未做运行验证。
测试源 430000–430019 和测试执行入口仍留到六个权重固定之后；当前只完成训练前实现，不晋升模型。
不修改相机示例、数据轨、采集器、实机入口、共享契约或周会日志。

## 22. 三对训练完成：选择收益为正，但固定候选控制几乎追平（2026-09-19）

用户运行了第 21 节入口。本次通过 SSH/SCP 回读原始 report/status、三组日志、保存的批次表与验证预测，
并确认远端六个 final.pt 均存在。每个模型固定 2000 步，共 12000 次保留的优化更新；每组日志 20 行，
从第 100 到第 2000 步，resume_from_step 均为 0。训练器实际完成两个优化器步数校验，并加载全部六个
最终 checkpoint 后才开始验证。没有因某个 seed 表现好而提前选择、增加训练步数或重训。

本轮只运行读取已保存输出的 `tools/inspect_pusht_action_effect_contrast.py`：
新增优化步数、模型推理、环境步数、测试来源与硬件动作均为 0。执行记录不回写原训练/数据报告。
远端根目录位于 SSD；回读包148655字节，不传权重、不涉及大文件手工上传。

### 验证结果：比较三组训练重复，不增加独立场景数

验证集仍是 7 个有效 source / 21 个上下文，其中 8 个有差异上下文、5 个有差异 source、
68 个非平局候选对；142 个真实平局对保留在原数据。被排除的420007不重新补样本。
下面是三个训练 seed 的平均；source 宏平均是主描述口径，当前各 source 都有3个上下文，
所以 coverage/gain 的 source 宏平均恰与上下文平均相同。

| 验证指标 | 普通评分器 | 效应对比评分器 | 对比减普通 |
|---|---:|---:|---:|
| 所选候选8步后 coverage | 0.207096 | 0.218072 | +0.010975（相对+5.30%） |
| 相对 ACT coverage gain | -0.002989 | +0.007986 | +0.010975 |
| Oracle regret | 0.011819 | 0.000843 | -0.010975 |
| 相对 ACT 有害选择率 | 11.11% | 4.76% | -6.35个百分点 |
| Coverage MAE | 0.063451 | 0.062910 | -0.000541（相对-0.85%） |
| Informative top1，上下文平均 | 37.50% | 54.17% | +16.67个百分点 |
| Informative top1，source宏平均 | 30.00% | 46.67% | +16.67个百分点 |
| Pair accuracy，每次pooled后平均 | 65.20% | 64.71% | -0.49个百分点 |
| Pair accuracy，source宏平均 | 53.81% | 54.14% | +0.33个百分点 |

ACT reference coverage=0.210085；uniform期望=0.209099；oracle相对ACT平均headroom=0.008830。
这些仍是固定候选执行8步后的结果，不是整episode成功率。

| 训练 seed | 普通相对 ACT gain | 对比相对 ACT gain | 对比减普通 | 普通/对比有害选择数 |
|---|---:|---:|---:|---:|
| 20260918 | -0.002417 | +0.008111 | +0.010528 | 2/21 -> 1/21 |
| 20260919 | -0.002417 | +0.008080 | +0.010497 | 2/21 -> 1/21 |
| 20260920 | -0.004135 | +0.007768 | +0.011902 | 3/21 -> 1/21 |

配对差均值0.010975，训练seed间样本标准差0.000803。这是训练随机性的描述，不是独立测试
source聚类置信区间，也不能把重复使用的21个上下文当作63个独立验证场景。
保存的预测 argmax、相对ACT增益和MAE与原训练报告一致。

### 两个不能忽略的混杂因素

第一，正向收益高度集中。source420003和420006的三seed平均差分别为+0.051260和+0.029034，
占全部正source差的99.19%；420001为-0.004123，420005仅+0.000656，其余三个source为0。
仅作事后敏感性分析移除上述两个大收益source，剩余平均差为-0.000693。
这不是新的筛除规则，不能修改原验证集或用它调 loss。

第二，事后查看全部5个“固定候选索引”控制，发现固定选候选1几乎一样好：

| 固定索引 | Coverage | 相对 ACT gain | 有害选择率 |
|---|---:|---:|---:|
| 0：原 ACT | 0.210085 | 0 | 0% |
| 1：ACT + native X正向ramp | 0.217884 | +0.007799 | 4.76% |
| 2：ACT + native X负向ramp | 0.202575 | -0.007511 | 23.81% |
| 3：ACT + native Y正向ramp | 0.208888 | -0.001197 | 23.81% |
| 4：ACT + native Y负向ramp | 0.206064 | -0.004022 | 14.29% |

固定索引不是固定绝对坐标动作，底层ACT chunk仍随当前观测变化。对比模型平均coverage只比
固定索引1高0.000188。普通组在三个seed的21次选择中，候选1/4分别是14/7、14/7、2/19；
对比组仅选择1/3，次数为12/9、12/9、16/5。因此“纠正了全局候选方向偏好”仍是合理解释，
目前不能确认学到了超越简单方向偏好的条件化效应评分。上述控制是事后诊断，不能把验证集上
较好的固定候选直接晋升成策略；后续应在触碰测试前写明全部固定索引控制，避免只挑候选1。

### 图像、效率和决策

最终分析：`simulation_output/pusht_action_effect_contrast_inspection_v2/report.json`；
中文图：同目录 `validation_selection_readback_zh.png`。使用首个训练seed20260918，事后选两处最大改善
420003/160、420006/160及最差退步420001/80，实际coverage差为+0.143810、+0.087536、-0.010267。
图像全部来自原始实际候选执行，不是模型生成后继。正例和负例均展示；选择标准写在图上，不作为
随机代表样本或独立统计证据。图已查看，中文无遮挡，状态viewed_not_accepted；用户尚未视觉接受。
v1初次分析保留；v2只增加固定候选诊断与显式警告，没有新模型运行。

每臂2000步的优化计算均值约3.546/3.451秒，不含初始化/保存/分析；warm forward约0.295/0.276ms，
不含视觉提取或搬运。两组结构相同，不将微小计时差包装成速度优势。训练日志loss有限，最终
pointwise MSE约0.0032–0.0040，但两组总loss含义不同，不能直接比较总loss大小判断优劣。

**决策：冻结六个权重，保留当前正向验证信号，但不晋升、不扩容、不重训或搜索loss权重。**
建议下一步准备预留20个测试source430000–430019的固定评价入口，并在执行前将全部固定索引
诊断控制写入单独测试补充协议；保留原主对照，不修改已经完成的训练协议。
若对比组在新source上仍胜普通组，却不能胜固定索引控制，则不能把结果归因于更可靠的条件化评分。
实际测试尚未执行，也尚无本轮测试启动命令；保留普通ACT/real10，不延伸为实机、低可见度或sim-real收益。
按项目skill仅更新算法证据与交接；数据轨、共享接口和周会日志不变。

## 23. 预留20源测试：入口与准备检查完成，等待用户执行（2026-09-19）

用户认可第22节的下一步。本轮新增 `tools/run_pusht_action_effect_test.py` 及独立补充协议
`docs/pusht-action-effect-test-protocol-v1.json`，不修改已完成的训练协议、模型结构、loss、
六个fixed2000权重、普通ACT或real10。项目skill限定算法轨与远端执行；实验规划和统计分析skill
用于在触碰测试前锁定对照、来源分组、统计口径和证据边界，不产生模拟结果或论文收益结论。

### 共用候选后果，先预测再执行

固定测试source430000–430019，沿用原ACT前缀和0/80/160起点、8步后果、原ACT加四个±XY
offset8 ramp。每个source只生成一份原ACT前缀，各评分器和全部控制共用相同的五候选真实后果。
复用原生reset和精确动作前缀回放，不给两臂不同的环境预算或候选集合。

配对数据helper仅新增可选的当前上下文callback，原默认流程不变。六组评分及argmax选择在
相应起点的任何后继动作执行前落盘；模型只读当前RGB提取网格、可观测agent XY、候选动作
和训练goal，不读seed、anchor、未来RGB、coverage或oracle字段。原ACT仍决定后续起点，
所有有效候选均执行，不是由评分器选择驱动的闭环。预测值不裁剪、不增加门控。

上限20源、60上下文、300候选后果、29760原生环境步；当前无效、候选越界、未满8步或提前
结束均按原规则记录，不补seed/起点，不按未来图像质量筛选。metadata只能证明与项目已记录
来源无精确重复，不能证明未知原始数据初态完全独立或场景多样性充分。

### 固定候选控制和来源级统计

主比较保持“效应对比减普通评分器的实际8步coverage”：先在同source内平均，再对source及
三个配对训练重复平均。额外报告全部固定索引0–4、ACT、uniform期望、事后oracle；固定索引
不是固定绝对动作，不能看过测试后只挑一个有利对照或把最优固定索引晋升为策略。

Bootstrap固定10000次、RNG seed20260919，以有效source为整个cluster重采样，同时保留
其全部anchor和三个训练重复。主差值给95%百分位区间；对五个固定索引的五项比较分别给99%
区间，对应名义家族95%的Bonferroni控制，但仍是小样本百分位bootstrap近似。
区间以这三个已训练模型对为条件，不把20源×3重复当60个独立场景；训练重复间SD单独报告。
有效source少于2不给区间；不保证统计功效，不追加有利seed，不搜索显著性或更换主指标。

同时保留gain/regret/harm、headroom、informative top1、pair accuracy及预测平局、MAE、
每源和每训练seed明细、候选选择频次、参数量、原训练耗时及当前评分耗时。
只有普通组差值稳定为正、全部固定控制仍有支持且harm不增加时，才支持进一步研究；
无可靠收益、harm增加或固定方向混杂未解除，均不扩容/调参/晋升。本测试不是整episode成功率，
不外推到导丝、低可见度、sim-real或实机有效性。

### 已完成检查与尚未执行的部分

本地三份Python AST和补充协议JSON检查通过，小文件顺序SCP同步到4090；远端只运行
`--stage prepare`，在既有21个验证上下文上加载六个最终checkpoint。六组预测相对保存数组
的max_abs_error全部为0.0；原metric slots完全不变；零差值的来源bootstrap机械检查通过。
身份绑定使用路径、字节数、mtime及checkpoint内契约，不对每个文件例行计算SHA-256。

证据两端均在 `simulation_output/pusht_action_effect_fresh_test_v1/`：
`preparation_report.json`、冻结的 `protocol.json`、`seed_inventory.json` 和 `status.json`。
本轮共6次既有验证forward，测试source=0、环境步=0、优化步=0、硬件动作=0。
这是准备通过，不是新测试结果，也不是完整执行/恢复路径已验证。

下一步由用户运行 `--stage run`；预计约1–3分钟，依据此前40源约92秒及模型启动开销估计，
并非新入口实测。完整命令在算法命令文档当前节。Ctrl+C或失败后 `--stage run --resume`
跳过完成源，未完成源在新attempt从源起点重做，旧attempt保留；不覆写或拼接残轨迹。
若所有源已完成而汇总失败，只需 `--stage summarize`，不重跑推理或环境。
重试可能超过单轮步数上限，报告分别计完整源和所有已记录attempt；突然断电可能丢失末段计数。

首个测试source430000、全部三个计划起点、首个训练seed已预先固定为图像案例，之后生成
`first_test_source_zh.png`，不按收益挑案例。本轮没有测试轨迹、图像或视觉接受状态可验收；
实际运行后再读报告和直接查看代表图。原训练/验证原始报告不回写；其他轨和周会日志未修改。

## 24. 20源测试完成：验证收益未保持，停止扩展当前效应差异loss（2026-09-19）

用户完成冻结的 `--stage run`。4090原始status为
`completed_frozen_twenty_source_candidate_test`，20个源均完成、每源一个attempt；
环境步数28021，执行耗时45.972秒（到源执行完成，不含最后汇总），优化/硬件动作均为0。
没有因验证结果选seed、改模型或改候选；六个fixed2000权重与预先冻结契约一致。
本次SSH/SCP回读和本地NumPy检查只消费保存的结果，没有重新推理、训练或运行环境。

### 实际样本与一致性

计划60起点中，430000/160和430011/160因原ACT提前结束未到达；其余58起点实际执行290个
候选。430003/160当前物体提取无效，按原规则排除，剩余57上下文/285候选，覆盖全部20源。
没有补样本、改变无效规则或按未来图像质量筛选。18上下文、11个源有可分辨的候选后果；
173/570候选对非平局，397对真实平局仍保留。三个训练重复不是60个独立测试源。

全部20 reset身份互异，与记录中的此前身份无精确交集；这不保证未知原始示范或行为分布独立。
回读逐项连接342组有效模型/上下文预测与原始predictions/outcomes日志，检查argmax、候选后果
和样本顺序一致。由保存NPZ重算主比较及五项控制的全部六个区间，误差在1e-14内；未换统计方法。
完整工件两端均位于 `simulation_output/pusht_action_effect_fresh_test_v1/`；回传archive为
`simulation_output/pusht_action_effect_fresh_test_v1_readback.tar.gz`，758532字节，未传模型权重。

### 主结果：没有稳定的正向选择收益

下表coverage/gain/harm/regret/MAE按source宏平均后再平均三个固定训练重复；
pair accuracy另标其聚合方式。Coverage是固定候选执行8步后的目标覆盖率，不是episode成功率。

| 测试指标 | 普通评分器 | 效应对比评分器 | 对比减普通 |
|---|---:|---:|---:|
| 所选候选coverage | 0.180198 | 0.179368 | -0.000830（相对-0.46%） |
| 相对ACT gain | -0.000885 | -0.001715 | -0.000830 |
| 相对ACT有害选择率 | 12.50% | 19.72% | +7.22个百分点 |
| Oracle regret | 0.008843 | 0.009673 | +0.000830 |
| Coverage MAE | 0.070042 | 0.071523 | +0.001480（相对+2.11%） |
| Informative top1，上下文平均 | 24.07% | 25.93% | +1.85个百分点 |
| Informative top1，source宏平均 | 27.27% | 22.73% | -4.55个百分点 |
| Pair accuracy，每次pooled后平均 | 51.25% | 47.40% | -3.85个百分点 |
| Pair accuracy，source宏平均 | 52.89% | 45.42% | -7.47个百分点 |

主差值=-0.000830194422，按预先约定的10000次source-cluster bootstrap得到95%区间
[-0.002717236858,+0.000978337464]。区间跨0，不能称为统计确证的劣势，更不能称为普遍无效；
但没有可靠正收益且描述性harm增加，已经触发既定“不扩容、不晋升”的决策。
区间以三个已训练模型对为条件，不完整覆盖重新训练过程的不确定性，也不保证统计功效。

| 训练seed | 普通coverage | 对比coverage | 对比减普通 | 普通/对比有害选择数（各57次） |
|---|---:|---:|---:|---:|
| 20260918 | 0.179528 | 0.179907 | +0.000378 | 6 / 11 |
| 20260919 | 0.180549 | 0.179907 | -0.000642 | 7 / 11 |
| 20260920 | 0.180517 | 0.178291 | -0.002227 | 8 / 12 |

训练重复差值的样本SD=0.001313，方向不再一致。ACT reference coverage=0.181083，
高于两组learned selector均值；uniform期望=0.181470，oracle=0.189041。
未取得的平均候选上限不是0（相对ACT约0.007958），不能把本次结果简单归因于“没有好候选”。
反之，这些单次候选后果也不能证明切换后的长时闭环收益。

### 全部固定索引对照：不能事后挑选新策略

| 固定索引 | Coverage | 对比组减该控制 | 预先约定的99%区间 |
|---|---:|---:|---|
| 0：ACT | 0.181083 | -0.001715 | [-0.009634,+0.004784] |
| 1：X正向ramp | 0.179165 | +0.000203 | [-0.001921,+0.002652] |
| 2：X负向ramp | 0.184705 | -0.005337 | [-0.021512,+0.006723] |
| 3：Y正向ramp | 0.181880 | -0.002512 | [-0.017620,+0.008510] |
| 4：Y负向ramp | 0.180517 | -0.001149 | [-0.005212,+0.001756] |

五个区间全部跨0，未证明对任何控制的可靠优势。五项99%百分位区间的名义家族95%控制仍只是
小样本近似。验证中较好的索引1在此不再领先，测试中索引2描述性较高；两者都不能因看过结果
而直接晋升。索引控制的底层ACT动作仍依赖当前观测，不是固定绝对坐标策略。

普通组依旧只选1/4（三seed次数36/21、29/28、7/50），对比组只选1/3（25/32、28/29、41/16），
没有一组选择候选0或2。这个窄方向偏好与此前混杂相符，但尚未做输入干预，不能直接断言网络
完全忽视视觉或动作。当前证据不足以确认观察条件化效应评分的优势。

### 图像与后续边界

已直接查看预先固定的 `first_test_source_zh.png`，中文正常、无明显裁切/遮挡；状态
viewed_not_accepted，未把用户“运行完毕”当作视觉接受。原报告not_viewed字段作为历史输出保留。
首源430000/80、首训练seed的普通组选择4、对比组选择1，真实coverage分别0.727709/0.722663，
均低于ACT0.810008；候选2/3分别可达0.895252/0.894031。起点0全部真实coverage为0，
160步未到达均在图中如实显示；不是看过收益才选的反例，也不代表所有来源。

**决策：冻结这次结果，停止扩展当前effect-difference loss变体，保留普通ACT/real10。**
不增加训练步数、改loss权重、选最佳seed、扩网络或追加测试源来修饰结果；本批测试已被查看，
不再作为未来变体反复调参的“未见测试”，也不把它转入训练。
若继续候选评分方向，建议下一步仅用既有训练/验证集检查状态与候选动作依赖，再确定新假设；
这是待确认的研究步骤，本轮没有实现或运行。结果只限制当前网络/数据/目标/loss配置，
不等于否定所有世界模型、候选评分、低可见度或sim-real方法。
按项目与统计skill记录全部预定比较及不确定性；数据轨、共享契约、硬件入口和周会日志未改。

## 25. 训练/验证输入审查：不是未接入，候选效应在训练集上也未充分学出（2026-09-19）

用户认可第24节后的有界诊断。新增 `tools/audit_pusht_scorer_input_dependence.py`，
使用原六个fixed2000权重、92训练上下文/32源及21验证上下文/7有效源；不打开20源测试数据，
不重新训练、不改变backbone/head/loss/标签/统计、不运行ACT或环境、不连接硬件。
这是失败之后提出的机制诊断，不是预先独立验证的新方法收益。项目skill约束算法轨和远端执行；
实验规划/统计skill用于事先固定输入干预、聚合和无标签边界，不撰写论文Results或宣称新颖性。

### 固定审查，不把错配输入当新训练样本

每个recipient使用同split、同anchor、不同source的全部donor，训练2740对、验证126对。
分别替换当前RGB提取网格、agent XY、二者一起、整个五候选动作bundle；原goal和normalization
始终不变，source/anchor只组织诊断，不进入网络。先平均donor，再平均同source的anchor，
然后source宏平均，最后平均三个训练重复。组合数量不等于独立样本数。

先固定脚本PLAN并在forward前写 `protocol.json`；保存全部score和donor行号，报告所有干预，
不选有利seed。错配后的图像/位置/动作可能不在联合支持域内，**只测评分和排序响应，绝不
沿用recipient或donor的真实标签计算错配输入的MAE、gain或harm**。原始输入才报告真实后果指标。
没有计算p值、挑显著性或据此决定新模型权重；单一固定goal无法识别goal依赖。

### 接线检查与输入响应

全部六模型×两split中，固定候选重排后的score置换误差均为0；五个动作均换为原ACT候选后，
候选score跨度均为0。正常验证预测重现已有保存结果；与原训练数组快照逐项一致。
未发现候选索引接错或动作未接入，但这些检查不保证有用的条件化排序。

以下是验证集source宏平均再平均三次训练重复；“首选变化”是候选索引变化，不是选对率：

| 输入干预 | 普通：原始score变化MAE | 对比：原始score变化MAE | 普通：首选变化 | 对比：首选变化 |
|---|---:|---:|---:|---:|
| 仅当前图像网格 | 0.249638 | 0.250554 | 35.19% | 25.66% |
| 仅agent XY | 0.002693 | 0.002653 | 2.38% | 4.23% |
| 图像网格+agent XY | 0.249605 | 0.250533 | 35.98% | 26.72% |
| 整个候选动作bundle | 0.007755 | 0.008052 | 5.82% | 25.93% |

图像交换后，将每个上下文的五个score减去共同均值，再比较变化，普通/对比的RMS仅
0.000261/0.000136；变化能量约99.9999%以上属于五候选共同平移。这里分母是score变化能量，
不是“99.9999%的决策忽视视觉”或模型参数比例。首选确实会变，不能宣称视觉完全未参与。
小的相对变化仍可能改变原本接近的候选；翻转本身没有正确性含义。
联合state与action的四项交叉差也非0，对比组验证candidate-centered RMS约0.000062，
表明交互存在，但不能只因非0就声称交互充分或能泛化。

### 已有标签上的候选差异拟合

仅在真实候选后果有差异的原始上下文作描述性分层；不改变原正式评价集合。
下列是source宏平均的预测span/真实span之比，并非逐行百分比或新成功率：

| 分组 | 普通预测候选跨度 | 对比预测候选跨度 | 真实候选跨度 | 普通/真实 | 对比/真实 |
|---|---:|---:|---:|---:|---:|
| 训练有差异上下文 | 0.001140 | 0.001072 | 0.037357 | 3.05% | 2.87% |
| 验证有差异上下文 | 0.001112 | 0.000923 | 0.046345 | 2.40% | 1.99% |

训练有差异上下文40个，验证8个。候选score对小范围对称XY ramp近似仿射：在有差异验证
上下文，首阶重建残差/去均值score RMS为普通0.26%、对比0.89%，真实标签对应值33.85%。
这是实际五个候选值的代数诊断，没有拟合新模型。它不证明某一种架构是唯一原因。
**分差压缩本身不必然改变argmax**；需结合排序和此前独立测试的失败，不能仅按幅度判模型无效。

| 原始输入指标 | 普通评分器 | 效应对比评分器 |
|---|---:|---:|
| 训练coverage MAE，source宏平均 | 0.006987 | 0.006694 |
| 验证coverage MAE，source宏平均 | 0.063451 | 0.062910 |
| 训练非平局pair accuracy，每次pooled后平均 | 46.04% | 57.51% |
| 验证非平局pair accuracy，每次pooled后平均 | 65.20% | 64.71% |

共同水平在训练上拟合得较好不等于同状态动作差异学得充分；验证绝对分值误差也明显上升。
对比组训练首选仍仅1/3（55/37、62/30、67/25），普通组主要1/4（首seed另6次选择3）。
不利用本诊断改动任何已经训练的权重或触碰已使用的最终测试。

### 事后补充：方差与剩余误差分解，防止给出错误归因

以下是看到审查结果后，对**已保存原始score和原train/validation target**做的纯NumPy描述性计算；
没有新增forward，不把它写成预先验证假设。等权上下文/候选口径（与上面source宏平均不同）：

```text
Var(y) = Var_i(mean_k y[i,k]) + mean_i Var_k(y[i,k])
MSE(e) = mean_i (mean_k e[i,k])^2 + mean_i Var_k(e[i,k])
e = prediction - actual_target
```

训练target总方差0.063740863，其中同上下文候选项0.000223193，占0.3502%；其余99.6498%
来自上下文均值。验证对应候选项占0.3109%。这解释了为何绝对值拟合可以掩盖较小的动作差异，
但**不等于最终优化误差或梯度仍由共同水平主导**：

| 剩余训练MSE分解（三训练seed均值） | 普通 | 对比 |
|---|---:|---:|
| 共同均值误差项 | 0.000008037 | 0.000006545 |
| 候选相对误差项 | 0.000223808 | 0.000220487 |
| 总MSE | 0.000231845 | 0.000227032 |
| 候选相对项/总MSE | 96.53% | 97.12% |

所以不能只说“动作loss被淹没，继续加权就行”。原pair项只是对同一标签误差重加权：
五候选、pair_weight=1时总目标等于共同均值误差+2.25倍候选误差方差（统一标准化系数略）。
当前配置已经主要剩下动作效应误差，仍可能涉及表示、优化尺度、观察信息不足和样本覆盖；
本次审查不能区分这些因素的唯一因果贡献。

### 工件、执行与下一步

本地AST通过，16506字节入口顺序SCP并在4090读回后执行；324 frozen forward，70818个
context-batch row、354090个candidate score，均为重复诊断输入，不是新增轨迹。
计时段0.497秒，不含进程启动/前置准备/最终写盘；optimizer/environment/hardware均0，
test_data_accessed=false。未修改模型、数据源或旧实验报告，无需新增轨迹图或视觉接受。

两端输出 `simulation_output/pusht_scorer_input_dependence_v1/`，含protocol/report/status及
`scores_and_donor_mappings.npz`；完整小型回传包1488863字节。全部变换、各source、各训练seed
的响应在原report中保留，不把表中均值当新增独立统计证据。数据轨、共享接口和周会日志未变。

保留ACT/real10，冻结当前loss变体和已用测试。下一步建议先设计**显式可观测相对几何/动作交互**
及**相对原ACT的收益建模**的最小方案，保留单因素对照；不要把单纯更换target/head称为已经成立
的创新，也不保证有收益。此处只提出下一研究问题，本轮不改架构、不训练、不扩容或另开测试。

## 26. 相对几何 × ACT相对收益：四组最小开发对照设计（2026-09-19）

**后续状态：本设计暂缓、仍未实现。用户随后认可先检查更强DP底座及候选上限，当前步骤见第27节。**

用户认可第25节后的设计步骤。本节是后续实现的约定草案，**不是实现、训练或新测试结果**。
本轮只读现有代码、已记录证据和4090已安装环境的渲染源码；不打开已用20源测试工件，
不读取新采集数据，不进行forward、优化、环境执行或硬件连接。旧协议、六个scorer权重、
ACT和real10均保持不变。项目skill限定轨道/执行范围；实验规划和统计skill用于固定因素、
比较口径和开发集证据边界。这里不是论文Results，不生成mock成绩或宣称新SOTA。

### 26.1 要回答的问题与四组

第25节提示动作效应在训练集上也拟合不足，但没有定位唯一原因。本轮只比较两个因素：

- **G：显式可观测相对几何特征**。把当前物体、目标、agent和候选指令的关系显式送入评分器；
  保留原网格和绝对动作，不用模拟器物体姿态或接触信息替换视觉。
- **R：ACT锚定的相对收益目标**。优化“这个候选比原ACT好多少”，取消绝对共同水平的回归要求；
  不再叠加旧的全10对effect-difference项，也不搜索pair权重。

| 组名 | G：额外44维几何槽 | R：监督方式 | 用途 |
|---|---|---|---|
| G0R0 | 全零 | 绝对terminal coverage | 重新训练的匹配基线 |
| G1R0 | 当前观测计算值 | 绝对terminal coverage | 单独看几何表示 |
| G0R1 | 全零 | 相对ACT的真实coverage差 | 单独看相对目标 |
| G1R1 | 当前观测计算值 | 相对ACT的真实coverage差 | 看组合是否优于单项 |

四组使用同一模型类和参数形状。主开发比较预定为G1R1减G0R0的选择gain；同时完整报告
G1R0−G0R0、G0R1−G0R0、G1R1−G0R1、G1R1−G1R0，以及交互差
`G1R1−G1R0−G0R1+G0R0`。不把最大的一项事后改称唯一主比较。
G是一个确定性特征块的整体消融，**不能据此区分块内距离、点积、叉积各自的贡献**。

### 26.2 固定44维特征与可观测性

复用当前 `[24,24]` RGB物体网格，使用 `pusht_object_dynamics.grid_centroid` 的加权
cell-center定义：`c = sum(grid[y,x] * ((x+.5)/24,(y+.5)/24)) / sum(grid)`。
同样由已冻结训练goal网格得到g；不用模拟器goal_pose或物体真值中心。
设 `p=agent_xy/512`、`a_t=candidate_absolute_xy[t]/512`、`d=g-c`、`r_t=a_t-c`。

| 槽位，零起始 | 定义 | 维数 |
|---|---|---:|
| 0:2 | `p-c` | 2 |
| 2:4 | `d=g-c` | 2 |
| `4+5t : 4+5(t+1)`，t=0…7 | `r_t.x, r_t.y, dot(r_t,r_t)/2, dot(r_t,d)/2, cross2(r_t,d)/2` | 8×5 |

`cross2(r,d)=r.x*d.y-r.y*d.x`，只是图像平面中的有向几何关系，不是力矩、接触或旋转
标签。除以2是固定解析缩放，不是从验证/测试拟合；不再做每组/每上下文归一化。
这包含候选位置的二次项和状态—动作乘积，原始网格保留T形轮廓；并不保证足以表示真实动力学。

坐标约定为X向右、Y向下。已只读检查4090安装的 `gym_pusht/envs/pusht.py` 的 `_draw`
和 `_get_img`：在512×512画布绘制，经surfarray轴转置得到HWC后resize；
`pymunk_override.py` 默认 `positive_y_is_up=False`。因此 `/512` 对齐归一化画面坐标，
不是对native Y再翻转。24格中心与像素化/遮挡后的可见质心存在离散误差；**c不是刚体质心**。
实现时仅需一次固定训练样本的坐标叠图核对，不重新跑环境，不向真值坐标校准。

当前无效图像继续按原规则处理：训练/验证只用原eligible列表，不能用新几何规则挑选样本；
已列为有效的网格若空/非有限则报数据一致性错误，不填虚构中心。未来在线无效时保留ACT。
目标网格始终来自训练episode1/frame117/global278。候选的8个点是**指令设定点**，
不是实际执行轨迹；最近距离不等于碰撞，目标方向不等于正确推力方向。
不新增PCA角度、历史、未来RGB、seed/anchor、coverage或diagnostic/oracle输入。

### 26.3 统一网络与监督公式

保留现有共享 `576→128 + LayerNorm + SiLU` 网格编码器，保持评分MLP隐藏宽度128/64、
层数和激活；只将44维特征槽接到原274维输入后，成为 `318→128→64→1`。
四组均有相同形状，G0槽为零，G1槽为上表特征；同一训练seed复制完全相同初始化。
保留原训练集的state/action标准化和原始16维有序动作。

设计参数量为**123265**，比旧117633多5632（4.79%，来自44×128输入权重），不是隐藏层
扩宽。四组参数预算一致，但G0的几何列没有输入梯度，不能声称有效表示能力完全相同。
因此旧六个checkpoint只作历史参照，**不拿旧模型直接充当这轮容量匹配基线**；新增G0R0
必须从头按同协议训练。实现时确认实际参数量；本轮未实例化这个新模型。

令 `u_k=f_theta(current, agent_xy, a_k, goal, geometry_slot_k)` 为同一共享网络的标量输出，
`C_k` 为已保存的同起点执行8步后的真实coverage，k=0始终对应原ACT。
全组共享旧训练target统计 `mu=0.21085390952974725`、`sigma=0.25246952881937657`：

```text
t_k = (C_k - mu) / sigma

R0: L_abs = mean_context mean_{k=0..4} (u_k - t_k)^2
R1: L_rel = mean_context mean_{k=1..4} ((u_k-u_0) - (C_k-C_0)/sigma)^2

所有组统一报告 predicted_gain_k = sigma * (u_k-u_0)
selected = argmax_k predicted_gain_k
```

R1对四个非reference候选等权，包含真实平局；不把恒零reference项算进分母，不减小样本权重
掩盖效果。梯度同时通过u_k和u_0，不detach ACT支路；五个分数只需一次bundle forward，
不为每个候选重复计算reference。R0与R1的梯度耦合/损失几何本就不同，这是R因素的内容；
相同步数/学习率不等于相同梯度大小。**本轮不另用较小的gain std放大目标**，避免加第三个
目标尺度因素；若这套固定尺度下失败，不能顺手调尺度后仍称同一次预定对照。

R1不是独立未来预测器：令 `e_k=u_k-t_k`，其loss就是 `mean_{k=1..4}(e_k-e_0)^2`。
它只重新组织原标签和取消共同偏移约束，没有新监督信息，也不单独构成创新证明。
若仅在推理后给旧分数减u_0，argmax完全不变；这里实际改变的是训练目标。

R0可还原绝对coverage预测 `mu+sigma*u_k`；**R1的u存在共同偏移不定性，不能报告它的绝对
coverage MAE，更不能加真实C_0来伪造在线coverage预测**。各组均以predicted_gain比较排序。
ACT参考分数严格为0；完全平局沿用最低索引、优先原ACT，没有新增阈值/保守门控/概率解释。
候选置换时wrapper同时更新reference指针；候选索引不进入网络。输出仍是原候选的8×2
native绝对XY动作，不裁剪，不改ACT，也不触碰导丝 `elite_tcp_delta_6d + piper_intent_id`。

### 26.4 固定数据、预算与未来实施入口边界

- 只用 `pusht_action_effect_pairs_v1/train/pack` 的92上下文/32源训练；原validation/pack
  的21上下文/7有效源仅开发评估。420007仍排除，不补数据、不调整split，不加入错配donor。
- 五候选、offset8、horizon8、anchors0/80/160及已保存全部后果保持不变。所有未来coverage
  只由loss/evaluator读取，不进入forward；无需新增仿真数据或运行ACT。
- 沿用三个训练seed20260918/19/20、每组2000updates、batch16完整上下文、AdamW
  lr3e-4/wd1e-4/clip1、FP32、禁AMP/TF32、原deterministic配置；每seed四组共用同一
  初始化和原先保存的 `[2000,16]` sampled_rows。共12个模型、24000updates，不追加步数。
- 每组fixed-final2000；不按验证选checkpoint、最佳seed或ensemble。12个final全部完成后
  再做验证。相同训练数据/候选/更新数，不声称FLOPs/梯度范数完全相同；记录每组实际耗时，
  warm scoring需包含几何提取和reference差分，报告batch与候选数。
- 实现须使用独立的新入口/新schema/新输出目录，不覆写旧模型和协议；将上述配置在训练前
  写入新protocol。提供prepare/train/summarize和中断恢复，恢复同时绑定四组优化器与batch
  位置；缺组不得仅补跑有利组。具体可执行命令在实现完成前不虚构。

此前每臂2000步约3.5秒纯优化，此轮12臂粗估约一分钟量级纯优化，完整启动/保存/检查另计；
这不是新实现实测，也不授权自动训练。后续先完成最小接口、共享初始化/取样、reference置换/
相同候选、坐标/参数检查，再按项目运行规则交付可执行入口。不要为每个文件新增smoke或hash。

### 26.5 预定报告、判定与混杂因素

每组均报告三seed全部结果和source明细，统一先平均同源anchor，再source宏平均，最后平均
训练重复；三个seed不是三倍独立环境样本。主指标是选择后的实际8步coverage及gain_vs_ACT，
不是score幅度或整episode成功率。

必报：gain、harm（`C_selected<C_0-1e-6`）、regret、headroom、最差gain、全部固定索引0–4、
uniform期望与事后oracle；相对ACT四个非reference候选的gain MAE/RMSE；全部10对的非平局
pair accuracy、真实/预测平局数、informative top1、各候选选择频次。保持target tie1e-6、
prediction tie1e-7；预测平局处理复用原evaluator，不因组名更换。绝对coverage MAE只标R0。
候选span/真实span和仿射残差仅诊断；把分差放大但argmax不改善不算收益。

本轮沿用的7有效验证源已经反复查看，**不是独立确认集**。仅报告配对source差、三个seed
差异与逐一去掉某源后的均值，检查是否被单源驱动；不以该集的p值/区间包装确认性结论。
完整四组表是开发比较，不做小样本“显著”筛选，也不能做有统计功效保证的成功率声明。

预定决策：

1. **没有改善候选差异拟合，或只改变共同分值/幅度**：不继续堆loss、宽度或训练步数。
2. **训练排序改善，验证选择仍无收益/被固定方向匹配/由单源驱动**：记录为机制拟合或泛化
   不确定，不晋升；先明确缺少哪种观察信息或覆盖，再提一个新的有界问题。
3. **G1R1在三个seed的验证均值均优于G0R0且gain_vs_ACT均为正，平均harm不高于G0R0，
   并且优于全部固定索引控制的均值**：仅支持提出后续独立确认。还须报告两个单项是否已经
   达到组合效果；单项相当或更好则没有保留组合复杂度的证据。不自动选最有利组重跑测试。
4. 后续确认需先冻结候选方法和完整协议，再另行授权、预留真正未用source、重新核对身份和
   样本量/检验口径。当前不指定或生成新测试源，更不复用430000–430019、追加到旧测试或转训练。

上述门槛是有限开发数据上的继续研究标准，不是统计确证或部署标准。剩余混杂包括：当前图像
缺少速度/接触历史，可见质心受遮挡影响，24格分辨率有限，五个小ramp覆盖窄，目标仍含固定
训练goal与native真实goal差异。几何块、锚定loss有收益也不能证明它们是唯一因果机制。

**本轮交付到设计为止。** 下一步若认可，先实现四组共用的特征/评分接口与准备入口，不启动
训练。只有选择收益得到独立证据后，才考虑未来表征/世界模型辅助、低可见度或sim-real；
本方案不是通用世界模型，不构成导丝或实机有效性结论。未修改数据轨、共享契约或周会日志。

## 27. 先验证候选是否有救：较强DP底座的原生多采样诊断（2026-09-19）

用户指出ACT整episode成功率22%，同一动作加四个微偏移可能一起失败；认可先检查较强底座
和候选上限，再决定是否训练评分器。这一方向优先于第26节几何方案，后者保留但不实现。
不能仅由22%推出所有候选均坏，也不能保证80%→90%比20%→30%更容易；本轮直接检验
当前候选是否包含更好的动作，不再用改评分器替代这一检查。

使用已有官方预训练Diffusion Policy，历史独立运行100源成功67/100。它的训练数据/预算
与自训ACT不同，不作同条件架构优越性结论；新的公平对照是**同一DP参考与其候选上限**。
不重训底座、不换权重/normalization/DDPM步数，不加载旧ACT评分器作新分布的质量判决。
本轮项目skill限定算法轨/远端/小文件传输；实验规划skill固定候选、后续执行和证据口径。

### 已实现的原生候选接口

- `tools/diffusion_pusht_candidates.py`：同一两帧RGB+agent XY历史下，保留正常
  `select_action`生成的原DP八步chunk为候选0，另用四个固定独立随机流重新运行完整DDPM。
- `tools/probe_diffusion_pusht_candidate_headroom.py`：prepare/run/summarize入口，以及只读
  已保存数值重绘的render-preparation。旧入口、checkpoint和协议均不改。
- 保持n_obs_steps=2、diffusion horizon=16、实际chunk=8、DDPM100、FP32、禁AMP/TF32。
  模型本身原有scheduler配置保留；不额外裁剪native动作、不手工加XY偏移、不消费队列冒充重采样。
- 四个替代候选的初始噪声与每个DDPM去噪步骤的随机数都隔离；退出后正常参考的Torch CPU/CUDA
  RNG及观测/动作队列完全不变。模型只读图像历史和可观测agent XY，不读未来结果/seed/coverage。
  seed仅调度随机生成器，不是网络特征。

### 固定完整诊断：一次干预，而不是事后oracle策略

新诊断source固定500000–500007共8个；prepare使用独立499999，不计入诊断群体。
仅查已记录protocol和seed_inventory：6个metadata文件、199个已有seed、101个已有reset身份，
预留seed无交集。运行时再次检查，并比较实际reset身份；这不是未知示范初态完全独立的证明。
不打开旧20源测试的predictions/outcomes/report，不重用旧测试进行调参或训练。

每个source先执行一条正常DP到首次done或全局step300，在固定80/160步的空动作队列边界
保存候选。起点只由正常DP前缀决定，不按oracle结果挑选；提前结束则记录未到达起点，不补源。
候选先落盘，后执行分支；当前不训练任何选择器，也不按未来后果决定后续名义起点。

每个有效候选均执行：原生reset → 原封不动重放该source的DP动作前缀 → 执行候选8步 →
由**同一个冻结DP**持续运行至首次done/全局step300。使用正常参考采样结束后的同一Torch
随机数状态开始后续策略，所以差异不是任意切换后续随机种子。后续观测随各分支真实变化，
不是继续套用名义轨迹的固定动作；原生重放不注入隐藏物体状态。

所有candidate0分支要求动作、图像、agent状态和成功/终止标志逐步精确重现正常DP后缀；
coverage仅允许1e-12绝对浮点误差（下文修复依据），不改原始数值；其他分支共同前缀同样检查。
无效/越界候选记录但不执行、不裁剪、不替换；非有限值或参考动作无效
直接报告失败。已成功而不足8步的分支立即停止，保留成功，短期指标明确标为
`coverage_at8_or_done`，不填充done之后的伪轨迹。

最多16个上下文、80条完整后缀分支、26400环境步（含前缀重放）、16800个正常策略动作步，
另有最多64个替代chunk采样。主比较只用五候选均合法的共同上下文，其他可用候选集合单列，
不隐藏候选越界造成的可用空间损失。原DP的8条完整episode表现也单独报告，不与条件化起点混算。

### 预定报告与解释

主指标是完整五候选群体中，事后至少一个候选成功减去参考候选成功，先同source内平均起点，
再source宏平均。报告参考失败但其他候选成功的source/起点、全部候选失败、五个固定索引、
uniform期望、8步/提前done覆盖率上限、最终覆盖率上限、各候选实际成功/步数、候选合法性
和两两动作RMSE。索引1–4对应预先固定的随机流，不是可事后挑选晋升的方向策略。

这是**一次候选干预＋一条共同随机流的DP续跑**的有限候选事后上限，不是世界模型已取得的
成功率，也不是反复重排的闭环策略上界。不能将最多80条高度相关分支当80个独立测试场景；
仅8个源的描述性pilot，不给显著性结论，随机续跑的不确定性尚未重复估计。

- 没有成功率上限空间：不训练选择器，先明确候选生成/观察窗口是否需要新假设；不自动加seed。
- 只有一个源出现挽救机会：记录为单源线索，不称稳定空间。
- 多源存在挽救机会：仅支持下一步设计匹配DP候选分布的监督与独立split；仍不是学到的选择收益。
- 之后若研究选择模块，比较同一DP加/不加模块，固定候选和计算预算；不能把底座从ACT换成DP
  的收益算作评分器贡献。此诊断包 `training_allowed=false` / `policy_training_ready=false`。

### 已完成准备与尚未完成部分

SSH正常；沿用已有严格checkpoint loader和原权重，无下载/迁移。两份新Python经本地AST、
顺序SCP、远端大小回读后执行prepare。一次准备耗时69.153秒，48个环境步：16名义、16普通
DP独立重放、8候选0、8后续DP；optimizer/hardware均0，完整诊断source执行数0。
所有前16步参考动作/观测/结果与没有候选接口的普通DP一致，候选0分支也精确重现。
完整300步、多source与中断恢复路径已实现，但尚未实跑，不能把16步准备称为全流程验证。

输入历史为 `[1,2,2]` agent state、`[1,2,1,3,96,96]` images；输出native动作 `[5,8,2]`。
准备样本五个chunk均合法且数值互异，两两action RMSE约1.743–4.691 native单位。
图中候选仍相近：**数值不同不等于行为独立、有更优后果或证明强底座必然带来空间**。
四个替代候选本轮没有执行，因此尚无headroom结果。

两端保存 `simulation_output/diffusion_pusht_candidate_headroom_v1/`，含protocol、seed_inventory、
preparation_report、status、完整短窗日志和 `preparation_candidates_zh.png`。代表图只展示指令
设定点，未把它画成实际运动轨迹。初版重叠编号已改为分面；只由保存的NPZ重绘，没有重跑模型
或环境。准备report原始visual字段保留，实际审阅状态记录为viewed_not_accepted，未获用户接受。
纯重绘曾遇到一条SSH内嵌Python引号错误，未执行forward/环境；随后用render-preparation入口
完成，避免改变已通过准备的数值结果。最终完整小型回传archive为55526字节，不含权重。

**下一步用户运行完整run，预计35–55分钟，建议预留1小时。** 依据旧DP21862步耗时3003.84秒
和本轮预算估算，不是完整新任务实测；提前成功会减少时间。精确命令和恢复方法见算法命令
当前节。Ctrl+C/失败保留attempt，resume跳过完成source、从未完成source开头重新运行；不拼接
残余后缀。所有已记录attempt单列计步，突然kill/断电可能缺少末段计数。不训练评分器、不启动
实机、不修改数据轨/仿真专家/共享契约/周会日志；原ACT、real10和旧六个scorer权重继续保留。

### 用户首次完整运行后的数值比较修复（2026-09-19）

上述“完整诊断尚未启动”为准备阶段记录，现已由用户启动。source500000完成，两处起点
全部候选分支保留；source500001/attempt_001在anchor160候选0的首步报
`reference continuation outcomes differ from original DP`。动作和raw观测检查已通过，
错误来自对整个metrics字典使用`!=`，连连续coverage的浮点尾差也要求完全相同。

对已保存的source500001名义216步动作原生重放3次，不加载模型、不生成新候选，共648环境步。
所有重复的RGB/agent XY一致，success/terminated/truncated/source均与原记录一致；其中一次
49帧coverage有尾差，最大2.7756e-16；原报错step161为0.21366068333047417与
0.21366068333047422，差5.5511e-17。证据为输出根目录
`reference_replay_diagnosis_v1.json`。这说明精确浮点比较不适用于此几何交集指标；
不将这次中断解释为模型、候选或数据质量失败，也不据此调换种子或成功阈值。

最小修复仅针对验证：coverage采用atol=1e-12/rtol=0，其余动作、图像、agent XY及离散
指标继续精确检查；原始coverage不改、不四舍五入，所有结果与随机数/模型协议不变。
同时检查共同前缀metrics，记录实际最大差异，超过容差或离散位变化继续报错并输出具体差异。
新增`verify-reference`只复查失败源的名义轨迹与两个candidate0后缀，不执行替代候选，
不推进正式八源汇总；验证记录保存在`reference_verification/attempt_001/`。

远端回归验证已通过，耗时74.447秒、648环境步：216步名义动作与失败attempt保存值相同；
anchor80和160参考分支分别续跑136和56步至同一终态，动作/图像/agent XY/离散状态严格一致。
coverage最大前缀差2.776e-16、后缀差2.220e-16；成功/终止标志变化、大于容差的coverage、
非有限coverage和来源变化六个反例仍拒绝。没有执行替代候选、优化或实机动作，未恢复完整run。

source500000原来通过的是更严格的逐值检查，保留done.json，不重跑。失败attempt_001完整
保留；验证通过后用户运行`--stage run --resume`，从500001/attempt_002开始。完整试验
仍未结束，不报告headroom/世界模型收益；此修复不授权训练、实机或新的数据扩增。

## 28. DP候选诊断完成：有可挽救动作，但盲目替换会降低成功率（2026-09-19）

用户已完成固定八源run/resume。所有source500000–500007完成，500001使用attempt_002，
此前失败attempt_001保留但不计入比较。500004原DP在step125成功，因此未到160起点；
共有15个决策上下文、75条候选分支，全部候选合法且每组五个数值互异。没有追加种子、
替换失败源、改变起点或重跑旧ACT测试。第27节未完成描述是历史阶段状态，以本节为准。

### 预定主要结果

按同source内平均起点，再对8个source等权平均；不是把75分支视为75独立样本：

| 比较方法 | source宏平均成功率 | 相对原DP绝对变化 |
| --- | ---: | ---: |
| 原DP / 候选0 | 62.50% | — |
| 五个候选均匀随机选择的期望 | 56.25% | -6.25个百分点 |
| 固定起点的事后最优候选 | 87.50% | +25.00个百分点 |

固定索引0/1/2/3/4依次为62.50/56.25/43.75/56.25/62.50%，没有固定替代索引超过原DP。
事后最优是读取全部分支真实后果后的诊断上限，不是训练好的评分器，也不是实际部署算法。
普通原DP完整episode为5/8成功；本轮八源均有可达起点且参考后缀复现原DP，所以上表参考
与完整episode成功率一致，但不是两批独立证据。该小样本不能替代历史100源
67/100基准，也不能把整个模型更换的收益归给评分器。

### 候选里面究竟有什么

- 原DP失败的6个决策点中，4个存在成功替代：500005/80、500006/80、500006/160、500007/80。
  分布在3个source，不是单个好例子。500005/160、500007/160仍然五候选全失败。
- 原DP成功的9个决策点中，6个至少包含一个失败替代。选择器不仅要找挽救动作，还要保留
  原本成功的动作；这解释了为什么随机替换反而降低source宏平均成功率。
- 全部候选失败的source宏平均占比12.5%（原始上下文计数2/15，二者分母不同）。
- 8步/提前结束coverage上限增量0.008097，分支终止时coverage上限增量0.020134。
  分支终态是在各自首次done或全局300步，不能说是同一时间长度的末帧比较。
- 每组五个数值互异，全部两两动作RMSE范围0.5183–12.8560 native单位；这不是轨迹独立性证明。

预定全部source细节：

| source | 已到起点数 | 原DP成功 | 事后最优成功 | 均匀选择期望 | 可挽救起点 |
| --- | ---: | ---: | ---: | ---: | --- |
| 500000 | 2 | 100% | 100% | 80% | 无 |
| 500001 | 2 | 100% | 100% | 70% | 无 |
| 500002 | 2 | 100% | 100% | 70% | 无 |
| 500003 | 2 | 100% | 100% | 40% | 无 |
| 500004 | 1 | 100% | 100% | 100% | 无；125步已成功 |
| 500005 | 2 | 0% | 50% | 30% | 80 |
| 500006 | 2 | 0% | 100% | 50% | 80、160 |
| 500007 | 2 | 0% | 50% | 10% | 80 |

### 完成核验与证据

回传完整小型工件包646626字节，解包约3.4MB，不含模型/数据集大文件。
`simulation_output/diffusion_pusht_candidate_headroom_v1/report.json`为原始汇总。
逐条核对75条保存的branch日志：开头八步与已保存candidate动作一致，step范围/首次done/
success/短期coverage/终止coverage与报告一致；1997个候选0后缀步的动作、离散指标与
名义轨迹一致，coverage最大误差4.441e-16，未超过1e-12。raw图像和agent XY逐步一致性由
运行时检查通过，本轮不重新执行环境来验证。完成attempt共20634环境步；计入失败attempt
为22336步，prepare/修复诊断另计。用户resume这次耗时1546.429秒，没有optimizer或实机动作。

按source/anchor/index排序查看首个挽救案例500005/80的candidate0与2：coverage从
0.9313到0.9562；也查看首个误选伤害500000/80的candidate0与2：从0.9587到0.9117。
四张原生终态图已检视，状态viewed_not_accepted。低分辨率截图仅辅助理解，以环境实际
is_success为成功依据，不把图像相似判成控制等价，也不把agent查看当用户验收。
本轮只做既有数据回读、日志核验和算法文档更新，没有新推理、环境、训练或硬件动作。

### 结论与下一步建议

**现在有证据支持推进候选选择：可挽救动作确实存在，而且盲选有损失；尚无证据说评分器
已经能可靠找到这些动作。** 这排除了“当前DP候选全是坏动作”的解释，但不能倒推旧ACT
评分器失败唯一来自弱底座；底座、训练来源、候选生成方式和续跑时长都不同。

建议下一步直接准备匹配冻结DP原生候选的独立训练/验证数据与普通评分器/效应对比评分器
公平入口，共用同一候选集合、底座、预算和评估口径。不能把旧ACT候选上训练的评分器直接
视为DP匹配基线。先回答“能否学到选择收益”，再决定是否值得加入世界模型辅助，不因存在
25个百分点oracle空间就保证可学性、创新成立或实机收益。

当前8源已查看，冻结为诊断证据，不转成训练集、不用于反复调参，不自动扩seed或追加运行。
下一阶段的数据生成、实现与训练尚未开始；pack仍training_allowed=false。
仅8个source、单条共同续跑随机流支持描述性结论，按统计skill不做事后显著性搜索，保留所有
预定对照与正负案例。按项目skill不改其他轨、共享契约、周会日志和real10现场模型。

## 29. DP匹配配对数据与公平训练入口（2026-09-19）

用户认可第28节下一步。本节交付接口和固定工程协议，不撰写论文Results、不生成模拟性能数据，
也不把换底座或重加已有差分loss视为创新成立。继续使用项目skill的算法轨/远端/最小验证边界，
按实验规划skill提前锁定来源、预算、目标、基线与限制；不改仿真采集主线、数据轨或现场模型。

### 来源与固定任务

协议文件：`docs/diffusion-pusht-action-effect-protocol-v1.json`。新增source按生成前整源划分：
训练510000–510031共32源，验证520000–520007共8源；测试530000–530019仅保留编号，
本入口不支持执行测试。原DP诊断500000–500007冻结，不进入训练/验证，旧ACT配对样本也不混入。
只检查已记录seed/reset身份元数据，不打开旧branch labels或测试report来挑源；不宣称已证明
与公开示范的未知初始状态完全独立。不同seed也不等价于充分物理/行为多样性。

每源原DP到首次done/全局300步，只在原定80/160空队列边界保存五个原生DDPM候选。
复用已经验证的`probe_diffusion_pusht_candidate_headroom.Runtime/nominal/branch`，不重新
实现采样器或偷偷更换原DP动作。候选执行8步，再由同一个冻结DP在共同续跑随机流下执行至
首次done/全局300。动作/观测/离散结果严格回放，coverage容差仅1e-12，原始数值不变。

监督改为该完整后缀的实际terminal coverage，不沿用旧ACT的8步标签；两个新arm都使用同一
目标。实际success单列`evaluation_targets.npz`供离线评估，不作为模型输入或训练loss。
成功提前结束即保留实际结果，不填充done后轨迹、不因为不足8步删除成功样本。当前图像无效
或五候选不全合法的组记录排除原因；未来图像不参与筛样；不得用容易源替换排除/未到达起点。

最多80个上下文、400条分支、132000环境步；预计2–3小时，按已完成八源pilot耗时估算。
全部40源完成才组包，成功/失败样本不按结果重采样。中断保留attempt，resume只重做未完成源，
不拼接残余分支。集合的源、起点、标签定义不因结果而追加或改变。

### 与旧训练核心的兼容

- 新收集/组包入口：`tools/collect_diffusion_pusht_action_effect.py`，prepare/collect/pack。
- 新训练适配器：`tools/train_diffusion_pusht_action_effect.py`，--check-data/--train/--resume。
- 共用`PairPack`、`DirectActionScorer`、`effect_difference_loss`及原配对训练循环。
  仅对旧`train_pusht_action_effect_contrast.py`增加显式schema与reference名称参数，默认ACT
  语义保持；两轨结果不会互相冒充checkpoint，原有权重不改。
- 输入仍为当前RGB可见目标24×24网格、可观测agent XY、五个8×2有序绝对XY动作以及固定
  原训练图像goal。预测前不传seed、anchor、coverage/success、未来RGB或exact object state。
- 保留原训练集input normalization与goal；仅使用新训练集全部候选的terminal coverage
  拟合一份共享target mean/std，不使用validation。成功标签只在预测保存后评估阶段使用。

### 公平预算与评估

两个arm均117633参数，普通标准化coverage MSE与MSE加同起点十对动作差分误差；pair权重
0/1，不删真平局、不合成负例。相同初始权重、相同完整五候选批次与顺序、相同优化步数。
固定三对seed20260918/19/20，AdamW lr3e-4、weight_decay1e-4、batch16、grad_clip1，
每arm2000步，共12000次optimizer更新。只在六个final全部保存后验证，不选最佳seed/步数，
不ensemble，不用验证集调整loss权重。配对checkpoint共同原子保存，恢复时核对数组与批次。

主要开发指标为source宏平均success的效应对比arm减普通arm，同时报告相对原DP的挽救/伤害、
terminal coverage gain/MAE/regret、排序和平局、五个固定索引、uniform/oracle、每源/每训练seed、
训练与评分耗时。先源内平均起点，再源间平均；重复训练seed不是新增独立测试轨迹。
本轮为开发验证，不是fresh final test，更不是每8步反复重排的闭环策略成绩。

仍有明确混杂：原DP消费两帧RGB，而保持不变的评分器只用当前24格图像；评分器未输入剩余
时间预算；续跑只使用每上下文一条共同随机流；coverage回归不等价于直接成功分类；源数量有限。
这些限制先保留以复用原公平框架，不在同轮叠加history/geometry/更大网络。若没有选择收益，
不能仅凭更好的训练loss继续扩展；若有收益，仍需另行固定未见测试，不复用已看八源作验证。

### 实现与准备验证

本地AST/JSON通过，最终小文件顺序SCP并回读后，4090执行prepare通过。复用独立准备源499999
已保存的可观测输入，仅用明确标识的`synthetic_schema_probe`机械标签做接口检查，不读取八源
诊断未来标签。输入形状[1,24,24]/[1,2]/[1,5,8,2]，两组初始化/输出一致，梯度有限且不同，
synthetic success指标及ACT默认指标改名兼容检查通过；合成包训练、未完成数据训练均被拒绝。
这不是训练结果：optimizer/environment/hardware均0，无新DP推理或训练权重。

准备输出`simulation_output/diffusion_pusht_action_effect_pairs_v1/`，回传包15102字节；
data_ready_for_fixed_scorer_experiment=false。未执行新的40源collection、整包归档或DP配对训练；
只验证了接口，不能把synthetic fixture当性能证据。schema-only不新增无关render或视觉验收门槛。

下一步用户运行--stage collect，完成后自动pack；若只有pack失败，单独--stage pack不重跑环境。
完整命令、source断点和未来训练命令见算法命令当前节。训练产物使用SSD新目录
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1`，原ACT、
real10、六个旧scorer与新DP诊断全部保留。先回读新数据再显式--train，本轮未启动训练。
