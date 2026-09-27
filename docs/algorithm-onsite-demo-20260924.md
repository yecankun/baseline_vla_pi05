# 2026-09-24 周会现场演示准备

更新：2026-09-24。范围：现有 Real10 VLA 的双相机显示、操作者确认后的单步执行，以及独立准备的有界连续入口；世界模型使用已保存的公开 Push-T 离线诊断。此文件不改变算法研究交接、模型权重或数据/仿真/SOFA 轨。

## 到场前状态

- `project4090` SSH 于 2026-09-23 首次连接成功。模型与硬件 Python 分别为 `/home/zsw/miniconda3/envs/project2026-pi/bin/python`（3.10.20）和 `/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python`（3.12.12）；依赖分离保留。
- 原 Elite 权重 `simulation_output/real10_pi05_train_v1/elite/final_policy.pt`（7,473,655,457 B）与 Piper 权重 `.../piper/mixed_head_policy.pt`（163,481 B）均为原路径普通文件。只读 checkpoint 元数据检查：Piper 的 `base_policy_checkpoint` 指向该 Elite 文件，adapter 与 normalization 一致；两者均选 `final_step`。CUDA 可用；GPU 检查瞬时显存 691/24,564 MiB，未列出计算进程。不是明天的占用保证。
- 远端三个入口 `tools/real10_pi05_policy.py`、`tools/real10_pi05_bridge.py`、`tools/run_real10_pi05_once.py` 均存在。采集项目的 UDP 适配器 `/home/zsw/PycharmProjects/real_collection/hardware/feeder_device/udp_controller.py` 存在；不以算法目录旧实现替换。
- 已有回放输入、Side/Top 两张图、中文字体齐全。2026-09-23 用硬件 Python 启动显示桥接的 `--source replay --headless --samples 1`，进程以 0 退出，标准输出显示一次模型预测：Elite TCP delta `[-0.0091521144, 0.1030426621, -0.0043658018, 0, 0, 0]` mm/rad，Piper canonical ID=1 → command=0（hold），`hardware_executed=false`。输出目录：`simulation_output/real10_demo_replay_20260923_v1/`。后续 SSH 只读回读确认 `status=completed`、`predictions=1`、`cleanup_errors=[]`、`worker.log` 0 B、预览图 307,302 B；预览图 SCP 连接超时，尚未查看，visual_status 保持 `not_viewed`。此次没有连接相机/机器人/递丝。
- 9 月 17 日历史记录保留：20 次真实双相机显示推理完成；一次模型动作驱动 Elite 到位读回，Piper 为 hold、递丝包 0。新增 HOME 归位流程只有逻辑测试，没有实际归位证据。以上都不是 9 月 24 日现场结果。
- 9 月 24 日现场操作者报告：采集项目 UDP 适配器的一次前进测试通过。尚未在本文件记录装置实测位移或独立状态读回；此结果不证明模型选择 feed 的路径。

## 最短演示顺序

1. **到场前只读收尾**：回放的状态/数量/空日志已经核对；在 4090 桌面直接查看 `simulation_output/real10_demo_replay_20260923_v1/preview.png`，不重跑同名目录。若不方便查看，仍可使用先前已经查看过的 9 月 17 日历史预览，并明确标注来源日期。
2. **设备交接后，先显示**：在 4090 **桌面终端**运行下方实时命令。9 月 24 日画面核对所得 Side `317222071938`、Top `317222072584`；运行前再次核对 1920×1080/15 FPS/BGR、Elite `192.168.5.66`，并确认其他使用者已交接相机与机器人。该程序只读取当前 TCP、显示原始动作预测，不求 IK、不发送动作或 UDP。屏幕刷新间隔不是控制频率。
3. **仅在操作者确认返回路径与设备状态后，做一次动作**：退出显示桥接，现场先选定单步平移上限，再使用下方命令。现场操作者依次手动输入 `HOME` 和 `EXECUTE`，不得用管道自动填写。HOME 回采集起点而非机械零位，可能远大于选定的模型单步上限；限幅、低速、IK 与到位检查不保证无碰撞。若交接、TCP/tool、回程或导丝状态不明确，就停在显示阶段。
4. **结果与失败处理**：保留每次唯一输出目录；检查 `report.json` 的原始预测、`plan` 中限幅后指令、`homing`、Elite 尝试/应答/到位、Piper 发送/应答、`before.png`/`after.png`。失败不自动重试；先由现场人员检查设备。hold 不发送递丝包；retract 仍拒绝；feed 仅调用采集项目既有 UDP 适配器一次，不从预测伪造递丝历史。
5. **单步确认后，才考虑有界连续入口**：若归位、双相机和一次动作都由现场操作者确认正常，且剩余时间与回程空间允许，可运行下方独立入口。无真实任务成功状态；由人看现场决定何时 Ctrl-C 停止，软件硬上限为 10 次动作和首步下发后 120 秒。异常、超界、无递丝应答即停，不自动重试。若单步失败，不启动连续入口。

回放状态只读查询命令（已执行通过；保留给现场复核，不重跑推理）：

```bash
cd /home/zsw/project_2026 &&
cat simulation_output/real10_demo_replay_20260923_v1/status.json &&
wc -l simulation_output/real10_demo_replay_20260923_v1/predictions.jsonl &&
stat -c '%n %s' simulation_output/real10_demo_replay_20260923_v1/preview.png simulation_output/real10_demo_replay_20260923_v1/worker.log
```

### 实时只显示（设备交接后）

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/real10_pi05_bridge.py \
  --source live --task left --elite-ip 192.168.5.66 \
  --side-serial 317222071938 --top-serial 317222072584 \
  --samples 20 \
  --out "simulation_output/real10_live_demo_$(date +%Y%m%d_%H%M%S)"
```

预留约 30 秒加载模型；20 个显示样本后自动退出，也可按 Q/Esc。若共享设备未交接，直接展示已保存的 `simulation_output/real10_live_display_20260917_120953/preview.png`，明示“2026-09-17 历史现场记录”，不可称当天实时。

9 月 24 日相机只读核对：4090 的 RealSense API 当前枚举两台 D435，`317222072584`（Top）和换线后出现的 `317222071938`（Side）；第三台未在这台主机上枚举，未推断其序列号或用途。Top 画面取自现有 `agp_robot.server` 的在线只读预览；Side 在节点空闲时用 1920×1080/15 FPS/BGR 短暂采集、50 帧自动曝光后复拍。两张画面分别在 `simulation_output/real10_camera_probe_20260924/A_317222072584.jpg` 和 `side_317222071938_warm.jpg`，由代理查看，visual_status=`viewed_not_accepted`，待现场操作者核对。原 Side `250122079856` 此时不在枚举列表。现有 `agp_robot.server`（PID 401722）仍占用 Top，且其 B 相机配置仍指向旧 Side 序列号、处于 stale；在相机使用者正常交接前不要启动本项目双相机桥接，也不要停止该进程。上述画面确认不包含本项目模型推理。

### 独立 Elite 归位（常用操作）

`tools/home_real10_elite.py` 只复用单步入口的归位流程：现场输入 `HOME` 前不连接设备；不启动模型、相机或递丝。目标是记录的采集起点而非机械零位。设备已交接且现场确认 TCP/tool、导丝状态、回程净空、急停后，在 4090 终端运行：

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/home_real10_elite.py --elite-ip 192.168.5.66 \
  --home-speed 5 --home-timeout-s 180 --home-max-distance-mm 500 --execute \
  --out "simulation_output/real10_home_only_$(date +%Y%m%d_%H%M%S)"
```

`report.json` 单独记录连接后的当前 TCP、目标、归位距离、IK、尝试/应答/到位和清理状态。距离上限 500 mm、单关节变化上限 60 度；默认到位超时仍为 60 秒，现场命令显式选 180 秒以适配 5% 低速（可选范围 10–300 秒），异常仍尝试 stop。这些检查不保证路径无碰撞。若之后再运行单步入口，其独立 HOME 门仍需人工确认，已在起点时会跳过重复移动。

到场前对独立入口完成本机 `--help`、取消 HOME 无连接、模拟 HOME 后连接/归位调用/断开与报告检查；这不验证真实归位。9 月 24 日现场两次报告均为 `failed`：10:55:36 伺服未就绪，未尝试运动；10:56:05 的报告显示 308.29 mm 归位目标、Elite 接受移动指令，但 60 秒未到位，程序已尝试并收到 stop。操作者反馈为 5% 速度下 60 秒太短；该原因合理但报告未保存停止时的最终 TCP，重新归位前仍须现场核对实际位置和回程。归位未完成前，不进行单步或连续执行。

### 操作者确认后的一次归位与单步（显示程序退出后）

```bash
read -r -p '现场确认单步平移上限 STEP_MM（毫米，0<值<=10）：' STEP_MM
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_once.py \
  --task left --elite-ip 192.168.5.66 \
  --side-serial 317222071938 --top-serial 317222072584 \
  --feeder-host 192.168.5.10 --feeder-local-host 192.168.5.11 \
  --home-speed 5 --home-timeout-s 180 --max-step-mm "$STEP_MM" --speed 5 --execute \
  --out "simulation_output/real10_once_demo_$(date +%Y%m%d_%H%M%S)"
```

入口使用默认采集起点 XYZ `[-336.181546, 251.652310, 321.987936]` mm / RPY `[3.0661665148309702, 0.03647610536354759, 0.06842115274422467]` rad；若现场标定或 TCP/tool 定义不同，先停下核实，不沿用旧数值直接移动。模型输出是 `elite_tcp_delta_6d`（xyz mm、rpy rad）和 `piper_intent_id`（0 retract、1 hold、2 feed）；旧控制器映射仅一次 `command=id-1`。默认单步上限仍为 1 mm，现场可在 `0<STEP_MM<=10` mm 内显式选择；10 mm 是软件绝对上限，不是推荐值或无碰撞保证。它仅裁剪过大的模型输出，不放大小输出：历史单步模型原始平移只有 0.0188 mm，单纯增大上限不会使那一步更明显。速度不高于 5%、IK/关节变化/到位/异常停止检查保持不变。

### 可选：有界连续执行（单步经现场确认后）

此入口在本机 `tools/run_real10_pi05_continuous.py` 中独立实现，须顺序 SCP、小文件回读与无设备检查后才可考虑现场使用。保留原单步入口。它复用 HOME、TCP 目标/IK、现场所选单步平移上限/5% 速度、异常 stop 与采集项目 UDP 适配器；总 TCP 目标始终在采集起点 10 mm 软件边界内。每次动作完成后等待至少 2 秒，读取新的 Side/Top 与当前 TCP，再推理下一次。不按推理 FPS 下发。原始预测、演示增益后的位移、限幅计划、机器人读回与每步前后双视角分别写入新 `report.json`、`before_XX.png`、`after_XX.png`。递丝历史在无实测 producer 时一直是 missing/validity=0。

2026-09-23 本机最小验证：源码 AST 与 CLI 解析通过；模拟 hold 连续步数上限、两次 feed 串行调用、无递丝应答立即停止且不重发、Q 键在下发前停止，以及取消 HOME 不连接设备的生命周期检查通过。现场上限接口改为可明确选择 `0<STEP_MM<=10`，本机检查裁剪/小输出不放大/retract拒绝通过。模拟器替身不提供真实 SDK/相机/UDP/归位证据。

2026-09-24 现场前代码同步：SSH 连通后，三份小文件按下列顺序 SCP 成功；远端回读大小依次为 20,219 B、17,194 B、12,207 B，硬件 Python 对两个入口的 `--help` 检查通过，关键参数/HOME/EXECUTE 源码标记回读通过。没有连接或操作相机、Elite、递丝装置；连续入口仍只有模拟验证，需待现场操作者确认设备交接和单步结果。同步命令记录：

```powershell
scp tools/run_real10_pi05_once.py project4090:/home/zsw/project_2026/tools/run_real10_pi05_once.py
scp tools/run_real10_pi05_continuous.py project4090:/home/zsw/project_2026/tools/run_real10_pi05_continuous.py
scp docs/algorithm-onsite-demo-20260924.md project4090:/home/zsw/project_2026/docs/algorithm-onsite-demo-20260924.md
ssh project4090 'cd /home/zsw/project_2026 && stat -c %n:%s tools/run_real10_pi05_once.py tools/run_real10_pi05_continuous.py docs/algorithm-onsite-demo-20260924.md && /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/run_real10_pi05_once.py --help >/dev/null && /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/run_real10_pi05_continuous.py --help >/dev/null && echo CLI_OK'
```

以上 `--help` 只解析入口，不连接设备。确认回读再运行下方现场命令；若传输或回读失败，保留原单步/显示方案，不执行新入口。

```bash
read -r -p '现场确认单步平移上限 STEP_MM（毫米，0<值<=10）：' STEP_MM
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_continuous.py \
  --task left --elite-ip 192.168.5.66 \
  --side-serial 317222071938 --top-serial 317222072584 \
  --feeder-host 192.168.5.10 --feeder-local-host 192.168.5.11 \
  --max-actions 3 --max-duration-s 120 --min-interval-s 2 --after-feed stop \
  --translation-gain 50 --max-step-mm "$STEP_MM" --max-distance-from-home-mm 10 --speed 5 --home-speed 5 --home-timeout-s 180 --execute \
  --out "simulation_output/real10_bounded_demo_$(date +%Y%m%d_%H%M%S)"
```

此命令仍先要求人工输入 `HOME`，归位后再输入 `EXECUTE`，不能自动代填。`Ctrl-C` 可在运动中请求中止并由已有移动路径尝试 Elite stop；画面中的 Q/Esc 在动作后和间隔期间生效，物理急停始终由现场人员掌握。120 秒从**首步真正下发**开始计时，用于停止发起下一动作；已发起动作按原 10 秒到位/2 秒递丝等待与异常停止流程结束。50 倍仅在演示控制边界放大模型 TCP 平移 xyz，原始模型输出单独保存、旋转仍为零；5 mm 单步裁剪、10 mm 起点边界、IK 和低速保持，不能把放大后位移说成模型原始预测。为防显示窗口首次打开拖慢下发导致观测超过 2 秒，动作前仅保存双视角并打印计划，动作后显示画面。用户报告递丝装置会阻塞第二次 feed 直到第一次完成；新入口仍串行调用 `feed_once(wait_response=True)`，把该装置行为记为待现场复核，不把 UDP 应答当作物理完成证明。无应答则停止且不重发。最终状态只称人工停止/上限停止/失败，不称任务成功。

### 从当前位置由操作者逐步触发（9 月 24 日追加）

用户确认前 3 步累计约 8 mm 在现场可承受，并要求取消总步数/时间/距起点边界，由其观察终点。新增 `--operator-step-mode` 不自动归位：运行前输入 `RESUME`，当前 TCP 读回后输入 `EXECUTE`，其后**每按一次 Enter 才采集新双视角、推理并执行一步**；在步骤提示输入 Q/其他非空内容或 EOF 结束，`Ctrl-C` 可中断。此模式不使用 `--max-actions`、`--max-duration-s`、`--max-distance-from-home-mm` 的总界，报告相应字段为 null；继续保留每步 5 mm 裁剪、5% 速度、IK/单关节变化检查、2 秒观测新鲜度、递丝一次映射、异常 stop、无重试。它不是自动终点识别，也无碰撞规划；现场操作者必须决定是否触发下一步。50 倍增益为显式演示控制适配，原始模型输出单独保存。

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_continuous.py \
  --operator-step-mode --task left --elite-ip 192.168.5.66 \
  --side-serial 317222071938 --top-serial 317222072584 \
  --feeder-host 192.168.5.10 --feeder-local-host 192.168.5.11 \
  --translation-gain 50 --max-step-mm 5 --speed 5 --min-interval-s 2 \
  --after-feed continue --execute \
  --out "simulation_output/real10_operator_steps_$(date +%Y%m%d_%H%M%S)"
```

此入口仅完成本机无设备模拟；须按顺序同步小文件、远端轻量回读后才现场使用。续跑前核对当前 TCP、导丝/装置状态和后续路径净空。参考采集路径 `path/path1_pose.txt`（left）与 `path/path2_pose.txt`（right）从共同起点均先朝 y 减小，而最近 3 步模型控制的 TCP y 从约 251.65 增至 259.27 mm；这是方向不一致的诊断，不能据此判断当前导丝是否接近终点，应以现场观察决定下一步。

## 9 月 24 日现场结果

- 现场操作者再次运行独立归位：`simulation_output/real10_home_only_20260924_111554/report.json` 为 `completed`，`elite_target_reached=true`，终点 TCP 误差 0.00426 mm、RPY 误差 0.0000223 rad。此前 60 秒超时的运行仍记为失败，不以本次成功回填旧报告。
- 操作者确认无人使用后，正常退出仍占用 Top 相机的 `agp_robot.server` PID 401722；验证进程退出、视频节点无占用。Side `317222071938` / Top `317222072584` 随后用于本项目实时画面。第一次显示推理画面偏暗；相机读取增加 45 帧自动曝光等待后，`simulation_output/real10_live_display_warm_20260924_112629/` 完成 1 次真实双视角显示推理，`status=completed`、`hardware_executed=false`、无清理错误，预览图由代理查看为清晰；visual_status=`viewed_not_accepted`，不代表用户视觉验收。
- 用户现场分别输入 `HOME` 和 `EXECUTE`，选 5 mm 单步平移上限，执行 `simulation_output/real10_once_demo_20260924_112930/report.json`：归位检查 `already_at_start`；模型原始 Elite TCP delta 为 `[0.005820, -0.032701, 0.011925, 0, 0, 0]` mm/rad，裁剪后相同；Piper canonical ID=1/旧命令0，即 hold。Elite `move_joint` 返回并接受，`elite_target_reached=true`，实际终点对目标误差 0.00453 mm / 0.0000186 rad；状态 `completed_one_action_attempt`，无异常和清理错误。递丝包 0，故**本次仍不验证模型递丝**；前后双视角 `before.png`/`after.png` 已保存并由代理查看，visual_status=`viewed_not_accepted`，不宣称导丝任务成功。
- 用户随后告知递丝装置目的 IP 已改为 `192.168.5.10`。上述单步报告中的目的地址仍是旧配置 `192.168.5.6:8888`，但实际递丝包为 0；后续单步/连续命令与入口默认值改为 `192.168.5.10:8888`，本机绑定保持 `192.168.5.11:37011`。新的目的地址尚未由本次模型动作发包验证。
- 第一次 3 步/30 秒连续尝试见 `simulation_output/real10_bounded_demo_20260924_114052/report.json`：`attempted_actions=0`、`completed_actions=0`、递丝包 0。第一步模型输出和计划已打印，但在下发前因画面窗口首次显示耗时、原观测超过 2 秒而拒绝执行；原报告的停止原因合并写作 `max_duration_or_stale_observation_before_dispatch`，根据观测/文件时间戳判定此处为观测过期，不应误说已执行。此后按用户要求将演示时限从首步下发起算，并提供显式 50 倍 xyz 增益；下发前不再阻塞打开预览窗，过期观测仍拒绝。更改已通过无设备模拟：即使首次规划超过所设时限，首步计时仍从下发起点开始；真实新连续执行待现场复核。
- 第二次连续运行见 `simulation_output/real10_bounded_demo_20260924_115218/report.json`：50 倍演示 xyz 增益、每步 5 mm 裁剪、距采集起点 10 mm 边界、最多 3 步/首步下发后 120 秒。实际 `attempted_actions=completed_actions=3`，三步 Elite 均到位，终点误差依次为 0.00838、0.00445、0.00770 mm；第 2 步增益后被裁剪到 5 mm。三次 Piper 都是 hold，递丝包 0；`termination_reason=max_actions_reached`，无异常或清理错误。末步 TCP 距采集起点约 8.14 mm。首步前和末步后双视角已由代理查看，visual_status=`viewed_not_accepted`，不能据此声称导丝到达终点。原有有界入口重跑会先归位，不会从此末步继续。

### 现场固定路径回退演示（待实机运行）

用户观察 Elite 按模型输出运动的方向相反，要求先完成演示，方向原因留待后续研究。操作者的 11 步续跑报告 `simulation_output/real10_operator_steps_20260924_120458/report.json` 记录 11/11 步 Elite 到位，1 次模型 feed 在第 8 步向 `192.168.5.10:8888` 发出并收到 `ok` UDP 回复；它不证明递丝的物理完成，末步 TCP 距采集起点约 32.63 mm。现场判断路径方向时，左任务记录路径 `path/path1_pose.txt` 从起点先向 y 减小；现有模型控制的末步向 y 增大。该方向诊断不代表记录路径必然完成真实导丝任务。

`tools/run_real10_pi05_continuous.py --elite-path-file path/path1_pose.txt` 是**演示回退模式**：HOME 归位到记录采集起点，EXECUTE 后自动重复读取实时双视角、运行现有配对模型并记录原始 `elite_tcp_delta_6d`；实际 Elite 目标改为记录路径插值目标。Piper 仍按模型 canonical hold/feed 意图经采集项目 UDP 适配器执行，retract 仍拒绝。用户确认装置现场正常递丝后，此模式每次 feed 只发 1 包、不请求或等待 ACK、不重发；报告将 `piper_ack_received` 留为 null、`piper_dispatch_status=sent_without_ack_wait`，实际机械完成仍只由现场观察，不能伪造成控制器状态。逐步报告 `plan.elite_action_source=recorded_path_fallback`、`model_elite_tcp_delta_6d_not_executed`、固定路径目标、IK、限幅后位移、动作读回和前后双视角。不能将 Elite 路径动作宣传为模型选择。没有机器可读的真实递丝历史时 validity 仍为 0。

本地无设备检查：路径共 20 个记录 XYZ 点，插值为 76 个有限 Elite 目标；最大名义相邻平移 4.413 mm，小于所选 5 mm；第一个目标向 y 减小，终点与文件末点一致。`py_compile` 与 `--help` 通过。首次启动前 TCP 约 32.63 mm 离起点，HOME 是独立回程，距离可能超过单步 5 mm，仍需要现场人工确认回程净空；固定路径无自动避障。实机任何 IK、位置读回、观测新鲜度或 UDP 发送异常即停止，不自动重试；此模式不会因未收到 ACK 停止。预计 76 步超过 5 分钟，由现场操作者运行和随时 Ctrl-C/急停；到路径末点自动停止，无任务成功判断。

12:17 固定路径现场运行 `simulation_output/real10_path_demo_20260924_121709/report.json`：报告到第 11 步，Elite 第 11 步到位，Piper 模型 feed 指令发包 1 次但 ACK 缺失，故 `termination_reason=feed_ack_missing_no_retry`、`completed_actions=10`、`attempted_actions=11`，无清理错误；第 11 步读回 TCP `[-329.490781, 205.609966, 331.302837, ...]`。不能推断该 feed 没有物理执行，重新运行前必须现场核对递丝状态、实际 TCP 和回程净空，避免重复递丝。该运行的视觉状态 `not_viewed`，未宣称任务完成。用户随后要求 Elite 速度提高 5 倍；新参数 25% 尚未实机验证。

12:22 固定路径 25% 运行 `simulation_output/real10_path_demo_20260924_122245/report.json`：HOME 从上次路径位置回到采集起点，距离 47.45 mm，已到位。Elite 实际发起 8 步且 8 步均到位，位置误差最大 0.0093 mm；前 7 步计为完成，第 8 步模型预测 feed，递丝只发 1 包但 2 秒内未收 ACK，程序停止，故 `attempted_actions=8`、`completed_actions=7`、`termination_reason=feed_ack_missing_no_retry`。第 2、5、7 步的 feed 均收到 ACK；第 7 与第 8 步相邻。12:17 运行也在相邻两个 feed 中后一个无 ACK。用户现场确认递丝装置实际正常，要求不依赖 ACK；现在仅固定路径回退模式改为发包后继续，保持单次发包、无自动重试；原有单步、有界连续和模型续跑模式仍按原逻辑等待 ACK。当前报告末步 TCP `[-330.461592, 218.504293, 329.130337, ...]`，固定路径第 8 步已到位。新行为尚未实机验证。

同步后在 4090 交互式终端运行（**不可用管道填写 HOME/EXECUTE**）：

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_continuous.py \
  --task left --elite-ip 192.168.5.66 \
  --elite-path-file path/path1_pose.txt \
  --side-serial 317222071938 --top-serial 317222072584 \
  --feeder-host 192.168.5.10 --feeder-local-host 192.168.5.11 \
  --max-step-mm 5 --speed 25 --home-speed 25 --home-timeout-s 180 \
  --min-interval-s 2 --after-feed continue --execute \
  --out "simulation_output/real10_path_demo_$(date +%Y%m%d_%H%M%S)"
```

该模式不使用旧的 10 步/120 秒/距起点 10 mm 总边界，因为文件终点是有限边界；每步上限 5 mm、IK/关节跳变和异常停止仍保留。根据用户现场提出的“Elite 调快 5 倍”，仅此固定路径回退模式允许归位和动作速度从 5% 显式调到 25%；其他模式仍限制为 5%，报告分别记录归位和动作速度。25% 已在 12:22 运行中使前 8 步 Elite 到位，不等于全路径通过；速度提高也不保证路径净空。从起点启动要求 HOME 与 EXECUTE；从记录路径中途续跑要求 RESUME 与 EXECUTE。按 Q/Esc 在动作后或间隔期间停止；运动中用 Ctrl-C 触发软件 stop，现场急停由操作者掌握。中断后先查报告和实际 TCP，不会自动续跑。

如现场确认第 8 步后的实际 TCP、导丝状态与后续路径净空，可用 `--path-start-step 9` 从当前位姿续跑。该入口先要求人工输入 `RESUME`，连接后验证 TCP 距第 8 个路径目标不超过 0.5 mm、RPY 差不超过 0.01 rad；再要求独立 `EXECUTE`。任何不匹配都不运动。续跑不会重新发送前 8 步的 feed；模型在第 9 步之后仍可能再次预测 feed，由现场人员观察。若当前位姿或导丝状态不适合续跑，现场操作者应停止，不可跳过位置验证。

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_continuous.py \
  --task left --elite-ip 192.168.5.66 \
  --elite-path-file path/path1_pose.txt --path-start-step 9 \
  --side-serial 317222071938 --top-serial 317222072584 \
  --feeder-host 192.168.5.10 --feeder-local-host 192.168.5.11 \
  --max-step-mm 5 --speed 25 --home-speed 25 --min-interval-s 2 \
  --after-feed continue --execute \
  --out "simulation_output/real10_path_resume_$(date +%Y%m%d_%H%M%S)"
```

## 世界模型与新生讲解材料

- 已保存、可离线打开：`simulation_output/pusht_object_action_ranking_dev10_v1/first_seed_object_prediction_ranking_zh.png`。图中“当前 RGB＋5 组候选动作 → 未来物体网格预测”，实际未来 RGB 仅供事后对照；它是公开 Push-T **开发集离线诊断**，不是 Real10 现场世界模型，也没有接管机械臂。图上同时呈现预测偏差和候选排序局限，不宣称策略提升。
- 讲解接口：VLA `观测＋任务 → 动作意图`；世界模型 `当前观测＋给定动作 → 未来状态/视觉预测`；动作选择与受控执行另属规划/控制环节。现有 Real10 显示与单步入口只使用既有 Elite/Piper 配对权重，不接入头段新诊断或世界模型。
- 若现场无设备交接，使用上述两个**明确标注为历史/离线**的材料讲解，不启动实机入口。演示结果只记录实际发生的推理、动作、回读与视觉状态；成功率、真实递丝闭环和世界模型实机收益均未由此建立。

## 现场结果记录（待填）

| 阶段 | 输出目录 | 实际状态 | 双视角视觉状态与用户反馈 | 动作/递丝事实 |
|---|---|---|---|---|
| 当天实时显示 | `real10_live_display_warm_20260924_112629` | 1 次真实推理完成 | `viewed_not_accepted`，代理查看清晰，待用户反馈 | 显示入口无执行能力 |
| HOME 与一次策略动作 | `real10_home_only_20260924_111554`、`real10_once_demo_20260924_112930` | 归位和 1 次 Elite 到位读回 | `viewed_not_accepted`，前后图由代理查看，待用户反馈 | Piper hold、递丝包 0 |
| 可选有界连续 | `real10_bounded_demo_20260924_114052`、`real10_bounded_demo_20260924_115218` | 首次 0 步；修复后 3 步 Elite 均到位 | `viewed_not_accepted`，首步前/末步后图由代理查看，待用户反馈 | 三次 Piper hold、递丝包 0；第二次因步数上限停止 |

视觉状态仅用 `not_viewed`、`viewed_not_accepted`、`accepted`；`accepted` 须用户明确认可对应工件。单步或有界连续演示均不足以说明完整任务能力或成功率；不在周会现场临时扩成无限循环控制。
