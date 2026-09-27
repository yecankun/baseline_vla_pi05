# Real10 现场前就绪检查（算法轨，2026-09-16）

后续状态：用户随后批准的**只显示桥接**已实现并通过跨环境回放，见本文末节和算法命令文档。
以下审查结论保留其检查时点；新增桥接仍没有实机动作下发能力。

## 结论

**训练权重与离线输入/输出已重新实测通过；还不能直接启动模型控制机械臂。**
缺口是现场桥接程序和环境边界，不需要因此重新训练。本次只检测，没有改模型、采集器或控制器，
没有安装依赖、连接相机/机器人、求实机IK、发UDP包或执行任何物理动作。

| 环节 | 本次状态 | 依据/边界 |
|---|---|---|
| 真实10条数据的最终权重 | 通过 | 4090上严格加载Elite3000 / Piper1000，归一化和引用关系一致 |
| 观测到动作 | 通过 | 29条真实记录推理成功，额外state token正确消费；无标签作为输入 |
| 输出单位与schema | 通过 | 6维TCP增量，xyz毫米/rpy弧度；旋转全零；canonical Piper ID |
| 输出到目标TCP的计算 | 通过，纯数值 | 复用现有target_pose_from_delta计算左右各一例，保留当前姿态；未求实机IK |
| 双相机与Elite依赖 | 分散在两个环境 | 推理环境缺Elite/RealSense；采集环境SDK导入通过但没有LeRobot |
| 实时输入→模型→受控执行 | **未接通** | 仓库只有Real10PI05Policy及离线检查的调用者，没有现场客户端/控制桥接 |
| 现场连通性、IK与运动 | 未验证 | 本轮不连接硬件；需操作员到场后分阶段验证 |

## 权重与远端实测

4090同时是训练机和现场推理机，不要再搬大权重。路径相对于`/home/zsw/project_2026`：

```text
simulation_output/real10_pi05_train_v1/elite/final_policy.pt       7,473,655,457 bytes
simulation_output/real10_pi05_train_v1/piper/mixed_head_policy.pt     163,481 bytes
```

调用现有`tools/check_real10_pi05_inference.py`，独立保存本轮结果：
`simulation_output/real10_pi05_onsite_preflight_20260916_v1/`。
完成时间32.746秒，权重加载约29.2秒，首次推理0.289秒、热推理中位0.107秒/p95 0.113秒。
29条输出与9月14日同样本、同seed的结果逐值一致，state编码与训练pack一致，旋转输出全零。
不是重新训练或完整泛化评估，也不是测得了可直接使用的控制频率。

模型原始平移模长在这29例中为0.0118–5.0190 mm，中位0.0894 mm；**原始输出没有执行限幅**。
不要因为多数输出小，就把所有原始动作无条件下发。训练动作对应不规则记录间隔，不能按相机15Hz
或模型约10Hz的计算速度直接重复执行，也不能把位移再乘或除以FPS变成另一种量。

Piper训练内读回仍为：hold2649/feed70，混淆矩阵（真实行、预测列）
`[[2609,40],[59,11]]`。只有11/70个feed被识别，40个hold被预测为feed。
这些数字只说明接口和训练内表现，**不支持首次联调就自动递丝**。

## 固定输入与输出

`tools/real10_pi05_policy.py`的`Real10PI05Policy.predict`接收：

- `side_bgr`、`top_bgr`：真实两路OpenCV uint8 BGR图像；内部INTER_AREA缩至224×224，再转RGB。
  不互换视角，不用单路复制另一视角，不新增裁剪或增强。
- `elite_tcp_pose_6d`：当前测量的`[x_mm,y_mm,z_mm,rx_rad,ry_rad,rz_rad]`。
- `task`：left或right，复用训练时的英文指令和左右one-hot。
- `previous_controller_state`：上一决策的`piper_step_after_command`（事件次数，非毫米）与
  `piper_busy`（bool）。episode开始为None；不得填当前要预测的指令。

state32仍为：0:6当前TCP、6历史次数、14:16任务、16历史busy、27历史有效性，其余既定缺失维填0。
模型输出示例（本轮左任务起点，真实权重输出，未执行）：

```json
{
  "elite_tcp_delta_6d": [-0.0091521, 0.1030427, -0.0043658, 0.0, 0.0, 0.0],
  "piper_intent_id": 1
}
```

实际返回另有独立`diagnostics`，包括三类概率、单位、耗时、`outside_training_intents`和
`hardware_executed=false`。不要将diagnostics当机器人命令。

## 必须补齐的执行映射

### Elite

```text
当前测量TCP + 通过限幅/时效/操作员检查的平移增量
→ 保持当前RPY的目标TCP
→ EC.get_inverse_kinematic(pose=target)
→ 检查IK结果
→ EC.move_joint(target_joint=IK返回值, speed=现场确认的值)
```

已只读检查实际安装的Elite SDK：current_pose调用get_tcp_pose()，默认世界坐标系、当前工具，
姿态默认弧度；get_inverse_kinematic默认弧度。必须确认现场工具/坐标系与采集时一致。
**6维TCP增量不是关节目标。** SDK文档说明move_joint接受8项关节数组，应保留IK返回格式，
不能自行把它截成6项；实机返回长度本轮未查询。其speed是关节速度百分比，不是mm/s。

当前可复用的底层调用在`data/collect/collect_real_shadow_pilot.py`的ElitePathPlayback中，
它服务于预设路径，不是任意策略动作的现场安全控制器。不存在已验证的自动策略限幅/节奏/拒绝路径。
将来持续读取位姿与下发运动应分开连接，避免阻塞move_joint让观测位姿陈旧。

### Piper / 当前替代递丝装置

| 模型canonical ID | 意图 | 原采集控制器的command | 首次联调 |
|---|---|---|---|
| 0 | retract | -1（仅语义对应） | 本模型没有retract训练样本，当前forward-only流程拒绝，不静默改成hold |
| 1 | hold | 0 | 不发送递丝运动 |
| 2 | feed | +1 | 仅在操作者明确允许后请求一次feed，不直接把ID传给submit |

**禁止`executor.submit(result['piper_intent_id'])`。** 旧控制器1表示feed，而模型1表示hold；
这会把hold变成运动。ID2也不是旧控制器的合法标签。旧师兄推理脚本的small_action==1同样
不适用于这个canonical接口，且它使用旧摄像头/Piper arm，不应直接当作real10部署入口。

纯payload构造已读回（没有发送）：
`{"command":"move","parameters":{"action":"forward","value":1}}`。
一个forward包是一次事件，不是固定毫米数。还需要busy/完成状态、cooldown、节奏/次数上限、
异常停止和请求/接受/执行日志；UDP sent/ACK不能自动等同物理递丝成功或运动已结束。

## 当前环境与最新采集配置

| 用途 | Python | 本次只读依赖检查 |
|---|---|---|
| 推理 | /home/zsw/miniconda3/envs/project2026-pi/bin/python，3.10 | Torch/LeRobot/OpenCV有；Elite/pyrealsense2缺失 |
| 既有采集 | /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python，3.12 | Elite/RealSense/OpenCV实际import通过；LeRobot缺失 |

采集工作目录为`/home/zsw/PycharmProjects/real_collection`，不是推理工作目录。
不能把3.12的site-packages直接塞给3.10来混用RealSense二进制扩展。
建议复用已工作的采集环境作为硬件客户端，在同机通过本地通道调用常驻模型进程；这是待实现方案，
不是已经存在的推理服务。或者另行批准在推理环境中安装兼容SDK，但本轮未安装/升级任何包。

核对了全部十条训练episode的manifest，以下配置完全一致：

```text
side RealSense SDK serial: 250122079856
top RealSense SDK serial:  317222072584
color: 1920×1080 @15 FPS, BGR8
feeder destination: 192.168.5.22:8888
feeder source bind: 192.168.5.11:37011
transport: python_socket, wait_response=true, forward_only=true
```

相机SDK序列号不等于udev序列号。旧7月runbook中的side序列号已经不同，不能不核实就照搬。
192.168.5.66仍是已有现场文档的Elite IP参考，本轮没有连通性确认，也没有确认今天接入的相机、
IP、端口占用、急停或工具号。

## 现场前最小下一步

1. 补最小桥接：常驻模型加载一次；采集端发送双图、当前TCP和历史状态，收到动作后只显示/记录。
   默认无运动，不触发采集器原有自动起点移动或预设路径执行。
2. 到场确认两相机视角、位姿单位/工具、时间新鲜度和急停，再运行observation-only。
3. 另行由操作者确认单步Elite执行；初次递丝保留人工触发，不能把当前弱Piper结果直接自动下发。

本轮是审查请求，因此上述桥接、依赖安装和实机执行均没有擅自实施。没有真实IK/运动证据前，
不能把“输出字段可映射到已有SDK”表述为“已经接通机械臂”。

## 后续：用户批准只显示桥接（2026-09-16）

已新增`tools/real10_pi05_bridge.py`，硬件Python3.12客户端通过同机子进程的stdin/stdout调用
常驻Python3.10模型。没有改模型、采集器、控制器或环境；没有打开新服务端口。
原始`piper_intent_id`保持0/1/2，只在`legacy_controller_preview.piper_step_command`做减一：
0→-1/retract，1→0/hold，2→+1/feed。原始和转换后的值都保留在日志，不能再减一次。
这只是兼容映射，不是修改标签、分类头或共享policy输出；canonical0仍无real10训练样本。

用真实最终权重、同一条已保存观测连续请求3次，首次动作与直接推理完全一致；模型只加载一次，
加载29.272秒，推理往返0.316/0.114/0.115秒，退出后模型子进程已结束。三类编号、非法编号拒绝、
PNG无损输入、缺失历史零值/有效位均验证通过。不是三个独立验证样本或实机结果。
证据：`simulation_output/real10_pi05_bridge_replay_20260916_v1/`。

现场入口显示双图、动作与两套递丝编号，但**不包含IK、移动、使能或递丝发送代码**。无真实控制器
快照时保持历史缺失；不会根据预测feed增加计数。真实相机、当前TCP和桌面窗口交互尚待到场核验。
启动与恢复方法见[算法命令](algorithm-track-commands.md#current-real10-display-only-bridge-2026-09-16)。
