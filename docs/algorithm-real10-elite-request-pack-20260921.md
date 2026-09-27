# Real10 Elite 请求—未来响应监督包（2026-09-21）

## 本轮交付

已实现 `tools/prepare_real10_elite_request_pack.py`，从上一轮审计唯一配回的175次Elite请求
生成独立诊断包：`simulation_output/real10_elite_request_response_pack_v1/`。
完整保留左80/右95个请求和全部10个episode；没有训练、图片重编码、硬件调用或源标签修改。
不是新policy数据，也不兼容现有canonical action9接口。

最终脚本顺序SCP、首行回读后，4090 `project2026-pi` 环境构建主体 **0.431 s**完成。
只复用NumPy状态编码和既有224px图像；未加载神经模型。无需用户补跑或传图。

## 文件和输入边界

| 文件 | 内容 | 是否进入预测输入 |
| --- | --- | --- |
| `observations.jsonl` / `model_observation` | 请求前1秒内双视角历史、state_32及有效mask、过去相对时间、task、固定名义1.5秒 | 是 |
| `requests.jsonl` / `model_request` | 明确的宏动作模式、日志绝对TCP目标、有效mask、原SDK速度参数、Piper请求及burst | 是，仅离线请求条件诊断 |
| `response_targets.jsonl` | 未来图像、实测Elite位姿及相对锚点平移、实际采样时间 | 仅监督 |
| `timing_audit.jsonl` | 原完整命令配对、请求括区、返回时间、后续请求、路径索引、历史截断、旧窗重合信息 | 仅审计 |
| `sample_groups.jsonl`、`folds.json` | episode来源、原train身份、整episode诊断分组 | 仅划分/报告 |
| `manifest.json`、`validation.json` | schema、来源、限制、计数、必要验证结果 | 配置和审计 |

`sample_id`、`sample_index`只用来关联文件。`load_model_inputs(pack)`只读manifest、
observations和requests，返回纯`{"observation": ..., "request": ...}`列表，不返回ID、
未来监督、时序审计或fold。图像path用于加载像素，不编码成文本/类别特征；相对path以
项目根目录为基准，直接指向原 `real10_pi05_compat_v1/images`，没有复制图像。

### 观测和缺失

复用 `real10_pi05_observable_history_v1` 及原 `encode_real_state`，不增减state维度：

- `[0:6]` 当前实测Elite xyz毫米/rpy弧度。
- `[6]` 前一记录Piper事件计数（不是毫米）；`[14:16]`左右任务。
- `[16]`前一记录busy，`[27]`历史有效位；其余原缺失/保留槽保持零及无效mask。
- 前一控制器记录必须早于当前两张图；不使用当前post-submit结果。
- 9个锚点缺历史：count/busy为0且对应mask=false，history_valid=0；**不是已知count=0或idle**。
  state[27]的mask=true，表示“历史缺失”这一状态已知。

所有输入时间均来自过去。为避免旧字段名暗示精确就绪时刻，Elite相对时间明确名为
`elite_pose_query_start_relative_time_s`；请求前性由采集循环顺序确认，非硬件同步或精确发送。
没有使用未来真实时长作为预测输入，名义时域固定1.5秒，不随运动完成/成功与否改变。

### 请求

```text
motion_mode = absolute_tcp_target_to_IK_to_move_joint
elite_target_tcp_pose_6d_xyz_mm_rpy_rad = 原日志明确记录的绝对目标
elite_target_valid_6d = [true, true, true, true, true, true]
elite_path_speed_sdk_parameter = 20
piper_intent_id = 1
piper_burst_count = 1
```

目标是在调用结束后的快照中记录，但它是传给IK/运动调用的**请求参数**，不是后续实测位姿。
valid表示请求值已知，不表示实际执行/到位成功。速度保留SDK参数语义（现装SDK说明为关节
速度百分比），不解释为20mm/s；SDK采集时版本仍未冻结。

不把`目标−当前位姿`填入`elite_tcp_delta_6d`，不增加到policy state_32，也不传给原9维world API。
原IK关节值因单位未确认，仅留原审计；路径索引、目标回写/返回时间、状态、实际执行时长都
不是请求特征。共享接口仍为 `observation + task → elite_tcp_delta_6d + piper_intent_id`。

### 监督和分组

未来5/6条观测逐条保留图像、实测位姿、有效mask和实际采样时间。平移差字段是
`measured_elite_translation_from_anchor_mm`，只作实测响应目标，**不是导丝advance标签**。
全部175条 `joint_motion_response=null`、valid=false；包括唯一与旧UI49重合的窗口，也未
自动迁移其标签。没有将模糊、缺失或未标注样本自动视为stationary，没有读取contact真值。

2个窗口有后续feed请求，保留在监督包并隔离审计，未筛出“无干扰且容易预测”的子集。
旧归位时间戳、8个busy skip、5个终点空请求留在上一轮188事件审计，不伪造为目标请求。

原数据split仍是all-train/no-validation。只继承此前10折LOEO的**heldout episode名称及顺序**，
按175个新事件重建fold索引，旧45窗fold和实验不动。每个事件恰好进入一次测试折，训练/测试
episode互斥；这些是后续可用的**开发诊断分组**，并没有实施拟合或建立独立PI05验证集。
相同左右路径的重复记录也没有因此成为独立geometry family。

## 验证结果

- 175次源记录/审计命令对应，175条保存后输入回读一致，全部175次保留。
- 816条历史观测、985条未来观测；110窗各6条未来帧，65窗各5条。
- 3424个不同图像引用全部存在，未重复解码或重新声明视觉验收。
- 10个episode各一次隔离检查：修改未来实测位姿/BC delta及当前post-submit字段不改变观测；
  修改时序、path index、IK目标等审计字段不改变请求；只改请求目标时仅请求分支改变。
- 9个锚点缺历史的零填充/mask验证通过，10窗历史不足完整1秒；没有因缺历史删除样本。
- 10折整episode覆盖检查与本机回读检查通过；未计算全数据归一化或逐文件SHA。

回读补充：175条请求只有 **38个不同目标xyz、1个目标rpy、1个速度、1个Piper请求值**。
因此最多支持当前任务内Elite平移目标条件的观测性未来响应比较，不能宣称rotation、速度控制、
Piper事件边界、contact/tactile、反事实动作排序或实机策略收益。

## 完成边界与下一步

当前交付到可读取的监督接口；所有policy/formal/real-system标志仍false。没有改变PI05、
mixed-head、loss、控制器、采集器、数据/SOFA文档；没有新增行为渲染或用户视觉验收项。

后续可预先固定一个“小型未来响应预测”对照：持久性/不变预测、仅观测、观测＋Elite目标
请求，使用相同episode分组、目标时域和训练预算。先检查请求条件是否提供额外信息，再讨论
与PI05逐步候选动作语义的连接。未来图像/实测响应可提供监督，当前无需先补标175个窗口；
若要报告导丝advance/stationary准确率，仍需对应窗口的人审，不能由机械臂运动替代。
