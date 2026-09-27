# Real10 动作前响应监督接口（2026-09-21）

## 本轮范围与预先固定的边界

实现 `tools/prepare_real10_action_effect_interface.py`，把已有响应识别数据拆成
动作前输入、请求查询、动作后监督和时序审计。仅进行接口/时间边界验证；不训练、不加载
PI05/world model、不修改 backbone、mixed-head、loss、现场权重或共享动作接口。
不改采集器、专家、源标签、数据/SOFA轨文档；不扩大样本或选择“干净且好预测”的子集。

固定使用 `real10_response_baseline_v1/annotation_snapshot.jsonl` 的完整45窗及原10个
整episode LOEO folds。逐项核对最新人工v2修订；如已更新则中止，不静默使用旧标签。
这些folds仅是已反复使用的响应诊断开发划分。原PI05已经训练过全部10条episode，不能
称为干净的PI05 held-out测试，也不能把45个相关窗口视为45次独立采集。

## 核心接口

`锚点及过去可观测历史 + 已选/候选请求 + 预定时域 → 未来窗口运动响应`

| 文件/字段 | 内容 | 是否进入预测输入 |
| --- | --- | --- |
| `queries.jsonl/model_input/history` | 过去1秒内至锚点的双视角图像、原始未归一化state_32/valid_32、相对时间 | 是，图像路径只负责加载像素 |
| `task_instruction` | 原左右分支任务文字 | 是 |
| `nominal_horizon_s` | 固定1.5秒请求时域 | 是；不输入未来实际采样间隔 |
| `candidate_action` | `elite_tcp_delta_6d + piper_intent_id`，附字段有效mask | 是，但本数据Elite缺失 |
| `request_context` | Piper请求burst次数（事件数，不是毫米） | 是 |
| `response_targets.jsonl` | 原人工stationary/advance、证据视角、可见性标签、完整未来图像与实测Elite位姿 | 仅监督，不输入 |
| `timing_audit.jsonl` | 请求后状态、后续Piper/Elite请求、异步完成、真实图像终点、采样原因 | 仅审计，不输入 |
| IDs、episode、split、fold、provenance | 关联及划分 | 不编码为模型特征 |

### 时间与state_32

- 截止时间为锚点side/top/Elite测量的最大host时间；两相机时间偏差保留，不宣称硬件同步。
- 控制器历史只用前一记录，且其post-submit记录时间必须早于当前两张图。锚点同记录的
  busy、accepted、async结果不能作为动作前观测。原始选择的请求本身是查询，不是执行结果。
- 直接复用 `encode_real_state` / `real10_pi05_observable_history_v1`，不新增layout：
  `[0:6]` Elite xyz毫米/rpy弧度；`[6]`历史事件计数；`[14:16]`左右任务；`[16]`历史busy；
  `[27]`历史有效。原缺失/保留槽仍零且mask无效；episode起点history_valid=0。
- 异步开始、完成、事件计数不等于导丝实际位移；不使用depth、exact contact、wall、tip、
  route真值、原始event_id或collector估计占位值，不读取diagnostic_targets。

### 为什么不能填满Elite候选动作

源collector的 `finalize_record(record, next_pose, ...)` 把**下一帧实测位姿差**回填为
`reference_action.elite_tcp_delta_6d`。这可用于原BC模仿目标，但预测动作效果时把它当
当前已知指令，会混入未来观测。`elite_requested_tcp_pose_6d`也不能直接相减代替：它可能
属于很早的、正在执行或尚待异步线程更新的路径指令，且与共享的逐步TCP delta控制不等价。

因此本版本所有真实query均保留 `elite_tcp_delta_6d=null`、六维valid=false，不把缺失
解释为零动作、停止或hold；Piper仍为规范0/1/2，现有标签仅hold=1/feed=2。辅助函数
`complete_action9`只允许显式完整且有效的命令，缺失样本拒绝进入原9维world API。
原 `ActionEffectWorldModel.forward` 没有candidate有效mask，本轮不修改它、不强行接入。

### 响应与效应不能混同

保留整段人工标注与未来图像，不在后续请求处截断后继续沿用整段标签。端点逐视角记录；
审计后续动作时以最后一张响应图的host时间为界，请求时间只知道“观测完成—post-submit”
括区的，边界交叉单列。Elite `elite_path_command_timestamp`记录在发送调用之后，不能
当作精确运动起点、完成时间或锚点新动作；审计同时查看键盘提交事件和发送后时间戳。

即使没有日志中的后续请求，也可能有先前尚未完成的运动、未精确记录的异步执行及操作者
选择偏差。故当前只能称**Piper请求条件下的观测性未来响应监督**，不能称单动作因果效应、
反事实候选优劣、碰壁/触觉标签或策略收益。hold后的advance不自动判为错标。

## 必要验证与执行边界

一次远端短构建集成检查：源观测与事件包逐项对应、输入/响应时间边界、最新人工snapshot
与完整窗口不变、原fold不变、全部图像引用存在；每episode一次未来/当前post-submit字段
扰动不改变query、改变请求仅改变candidate；未知Elite拒绝转9维、原缺失历史编码保持。
只检查图像引用存在，不重复解码整批图像或声称重新视觉验收；schema改动无需新渲染。

新输出目录拒绝覆盖。执行失败保留故障目录，修复/SCP后指定新目录重跑这次短构建；无
optimizer/checkpoint/resume需求。无自动下载或大文件传输。所有policy/formal/deployable
标志保持false，训练及硬件执行标志false。运行命令与实测结果分别写入算法commands和handoff。

## 后续决策

先读回时序/缺失统计，再决定是否做一个固定的“动作前状态/图像＋Piper请求→未来响应”
轻量诊断，和不含请求的控制组比较。这不是本轮已执行内容，也不直接使用既有动作后识别
分数作为world监督收益。完整Elite候选动作world model需要命令类型、目标、下发及可用
时间、执行时域与中途重规划的可信日志，或明确可对齐的其他数据，不能靠回填未来位姿补齐。
