# Commands

Last updated: 2026-09-30

This file keeps copyable commands for the data/simulation/real-collection
track and retains older non-VLA commands for provenance. Current algorithm
innovation commands live in `docs/algorithm-track-commands.md`; the pre-split
Pi-Style/OpenPI/PI05 command block is archived under `docs/archive/`.

## Current: Left Target Reached By Onsite Confirmation (2026-09-30 12:54)

最新 M=`simulation_output/left_magnet_only_20260930_120439/`。
**用户确认“尖端到达目标”，本轮已完成，设备保持停止。不要重放已完成动作。**
M 共九次机械臂动作，总路径 89.963176 mm；12:50:41 递丝一次，`.5:8888` 回复 ok，
画面显示向目标所在左分支推进。12:54:08 STOP/REMOTE、位置未变；本阶段无校准/
清错/重新使能。此前两次退让合计 19.996157 mm 保留为诊断记录。

用户已明确允许模型内沿壁滑行，撤销此前“贴壁必须先解除”的诊断条件，优先到达；
最终发送基于现场约 30 mm 剩余距离。明显折弯、顶住无进展/错误分支/设备异常仍停止
后续动作；没有关闭控制器保护或改动原始基准。固定 **20 mm/forward** 是机构规格，
非尖端位移测量；用户已说明不可改，不再找接收端源码/5 mm 配置。

实况 `http://192.168.5.11:8765/`；最终递丝回放 `http://192.168.5.11:8768/`，
对比 `/review.png`。M/media_feed3，67 帧/6.7 s，全解码/代表画面/HTTP 白名单已核验。
结果 M/endpoint_result.json；总表 M/session_result.json；最后停止核验 M/final_verification。
定位页 `http://192.168.5.11:8770/vbEMVlopVeJCbcJ4wF5jJCmIYc_luARC/` 为本次递丝后固定图。
用户回复已保存/到达，但服务尚未产生新标记文件；**结果按现场确认记录，不计算精确
像素/毫米误差**。任务到达验收不等于全部录像或正式训练数据验收。

`follow_marked_tip_http.py` 默认预览，`--execute` 每次仅一条机械臂命令，原始 W
基准不变，无递丝/归位/自动重试。单段 10 mm/1% 速度、13 mm/3° 监测、姿态 0.5°。
`--corridor-preview` 用有限 IK 轨迹派生包络，默认原基准仍 31 mm/5°。

- `--align-over-tip`：两路近似水平中心误差均减小；M/lateral_alignment_preview
  两段有限走廊 92 mm/12°，已完成。
- `--guidance-review <arm_and_wire_review.json>`：已确认磁响应后向左目标走一步，
  M/guided_left_ik_preview 两段走廊 112 mm/15°，约 80 mm 整体路线从 B/follow2 起算；
  两路估计磁铁与尖端侧向差限制 12.5 mm。仅第一段 step7 执行，本轮结束后不重放第二段。
- `--relieve-wall-contact`：只允许新图像复核净空、明确上壁接触的 +Z 10 mm 单段，
  M/wall_relief_ik_preview、wall_relief2_ik_preview 分别派生 103/106 mm、14° 原基准
  包络，半径 3 mm，已各执行一次。到位观察 5 s；不等于已经解除贴壁。

关键检查 M/corridor_checks、lateral_checks、guided_left_checks、wall_relief_checks
分别 8/9/6/5 项通过。限幅是本次诊断参数，非机械臂硬件最大行程。当前无待执行
命令；新试验须重新读状态/抓图/定位，历史预览不能作为整段控制脚本重放。

以下为**最后已执行退让的参数记录，不要重跑**。目录已存在会拒绝执行；也不要
换输出目录重发它。省略执行开关仅说明工具默认行为，并非当前姿态仍匹配旧输入。

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/follow_marked_tip_http.py \
  --axes-dir simulation_output/left_work_pose_20260930_104551/axes \
  --prior-arm-dir simulation_output/left_magnet_only_20260930_120439/step8_relief \
  --annotation-before simulation_output/left_resume_20260930_100550/annotation/annotations.jsonl \
  --annotation-now simulation_output/left_magnet_only_20260930_120439/annotation_after_guidance/annotations.jsonl \
  --corridor-preview simulation_output/left_magnet_only_20260930_120439/wall_relief2_ik_preview/report.json \
  --relieve-wall-contact --out simulation_output/left_magnet_only_20260930_120439/step9_relief
```

各段 `arm_and_wire_review.json` 分开记录运动、磁响应和接触；step9_relief 的
`physical_review.json` 记录最新允许沿壁后的单次发送依据，`feed_once/report.json`
记录已执行发送，`feed_once/physical_review.json` 记录本次现场到达。唯一 `feed_once/`
已使用，不能删除或换目录补发；未经新任务不得再执行下一条固定 20 mm。

## Earlier: New Feeder IP And Manual-Tip Magnet Following (2026-09-30 11:39)

当前输出 `simulation_output/left_feeder_resume_20260930_111102/`。递丝器现为
**192.168.5.5:8888**，机械臂仍 **192.168.5.66:8055**。11:18:41 首次新地址递丝
收到 ok，用户确认尖端前进；**20 mm/forward 是操作员提供的机构步长**。
本轮按新尖端完成两个 10 mm 磁铁跟随段，再于 11:39:21 递丝一次并收到 ok。
机械臂停止；11:43 新尖端已保存：Side `[897,647]`、Top `[1162.3,880.2]`。剩余直线
像素距离小于上一轮标记间位移，固定 20 mm 可能越过目标，已询问设备端改为 5 mm。
用户已确认约 80 mm 后续磁铁路线与工作间距，**不必重问同一现场确认**。接收端源码
位置无法确认，本机仅找到发送端/旧串口固件；TCP 22/80/443 均拒绝连接。当前仍为
20 mm/forward，未改为 5 mm，保持设备停止。完整左端尚未完成。已完成动作不要重跑。

最新原始运动 `follow1/report.json`、`follow2/report.json`；本次发送
`follow2/feed_once/report.json`；总表 `session_result.json`。实况
**http://192.168.5.11:8765/**；回放 **http://192.168.5.11:8768/**；本次新定位
**http://192.168.5.11:8769/M0nkQiL3yKM0PToZ5yKhqPwDHjIi_Kct/**。

新工具默认只读预览（输出目录必须未存在）：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/follow_marked_tip_http.py \
  --axes-dir simulation_output/left_work_pose_20260930_104551/axes \
  --prior-arm-dir <上一段已完成的机械臂目录> \
  --annotation-before simulation_output/left_resume_20260930_100550/annotation/annotations.jsonl \
  --annotation-now <本次新尖端标记.jsonl> --out <新的预览目录>
```

这是人工尖端 + 局部 Y/Z 响应的诊断动作，不是模型推理或完整血管配准。`--execute`
每次仅走 10 mm，无递丝；原始工作位累计 31 mm/姿态 0.5°/关节 5°/不下降限制不重置。
当前已近累计范围，**不能直接重放上式加执行开关完成全程**。11:52 新只读结果
`remaining_follow_ik_preview/report.json` 给出四段各 10 mm 连续 IK，末段距原工作位
69.622 mm/最大关节差 8.431°，均未执行；后续需要按已确认路线有界调整累计监测，
原始基准不重置。递丝小步配置尚未解决。停止时不自动归位/重试，保留唯一输出目录。

一次递丝的离线证据校验入口（不加执行开关，无网络动作）：

```bash
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/feed_after_verified_arm_http.py \
  --arm-dir <尚未递丝的已完成且物理复核的机械臂目录> --feeder-host 192.168.5.5
```

默认 host 仍为历史 `.13`，当前必须显式指定 `.5`。执行时唯一 `feed_once/` 防止重复；
`--device-recovery-check` 仅用于已有无回复记录、现场报告恢复且 5 分钟内成功网络证据
支持的一次新尝试，保留原记录。W/left_step 的恢复发送和 B/follow2 的首次发送均已用掉。
ACK 与物理推进分别记录；不以收到 ok 自动连续发送。馈送关键测试 4 项已通过。

定位服务新增 `--reference-targets <旧 annotations.jsonl>`，只沿用目标、当前尖端初始
为未审核；服务自行创建 `--out`，不要提前 mkdir。仅标记不触发运动。当前 media、
annotation 仍是固定白名单服务，不暴露控制器接口或目录列表。

## Earlier: Powered-On Feeder Still Unreachable (2026-09-30 11:14)

新只读结果 `simulation_output/left_feeder_resume_20260930_111102/`。用户已打开设备，
但两次检查 `.13` 的 ping/ARP 仍失败；本机网口、路由及 `.66` 机械臂正常。已询问
实际 IPv4、接收绑定地址/端口和网口灯。**本轮没有新增运动或递丝，等网络信息。**
状态总表 `session_result.json`；原机械臂保持上一段 −Y 10 mm 的完成位置。

为网络恢复后的单次新请求准备了只读校验入口（当前失败记录会被拒绝，不连接硬件）：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/feed_after_verified_arm_http.py \
  --arm-dir simulation_output/left_work_pose_20260930_104551/axes/left_step \
  --device-recovery-check <已成功且新鲜的网络检查目录>
```

仅接受原首次未应答、恢复目录的 `network.json` 中 `.13` ping 成功、ARP 有 MAC 且
时间不超过 5 分钟；`--execute` 才有一次发送，使用唯一 `feed_after_device_recovery/`
保留原失败记录、拒绝重复，不移动机械臂。用户已授权设备恢复后的继续工作；需要
成功连通证据，不能重问同一授权或绕过地址问题。4 项测试通过。`one_feed_submitted_no_reply`
只代表 socket 提交无回复。当前尚未执行，不重跑历史发送或删除已有目录。

## Earlier: First Work-Pose Left Step And Failed Connectivity (2026-09-30)

最新结果 **`simulation_output/left_work_pose_20260930_104551/`**。用户现场 30 mm 起始
就位已完成并核验，已在新位置做三轴往返，再沿 **−Y 10 mm** 向左目标引导，实际
9.998610 mm、两路图像运动与预期一致。10:56:01 提交一次前进包但没有回复，随后
ping/ARP 未能确认 `.13` 在线；用户未观察这次物理结果。**当前不重发、不继续运动，
等现场检查递丝器电源/网络/UDP 接收程序，然后只读复查。** 完整左端任务未完成。

实时 **http://192.168.5.11:8765/**；更新后的本轮回放 **http://192.168.5.11:8768/**
（PID 355202；左向引导和一次未应答递丝观察，中间等待未录像）。总表
`session_result.json`，动作 `axes/left_step/report.json`，递丝 `axes/left_step/feed_once/report.json`。
不要重跑已完成命令或删除 `left_step/`、`feed_once/` 后补发。

新单段工具默认预览：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/execute_local_left_step_http.py \
  --axes-dir <本次新鲜的局部轴测量目录> --annotations <现场人工标记文件>
```

每个新 axes 只建立一次预览/执行目录；`--execute` 才发一段 −Y 10 mm，无递丝。
必须从已确认工作位进行有效 Y 方向测量且仍在原位，双相机方向均吻合人工目标，
否则拒绝；不是模型推理或整条左血管自动导航。已通过 3 项关键拒绝/单位检查。

`CameraRecording` 新增 `save_original_frames=True` 可同时保留每帧两路原始 JPEG，
本轮递丝前后 86 对原图已保存；默认存储/运动行为不变。本轮脚本的
`one_feed_sent` 状态只表示 socket 提交，不代表 UDP 到达、ACK 或物理执行。

## Earlier Today: Calibration Ready And Initial Axis Round Trips (2026-09-30)

用户已现场校准并回 REMOTE，实际回读 precision=1。仅重新使能一次伺服后，已于
10:31 完成三个 10 mm 往返（6 次机械臂动作），最终回起点，STOP / REMOTE、servo/
sync=true。**不要重跑已完成的启动、校准或往返命令。** 今日递丝 0，完整左端任务
尚未完成；局部相机响应不是完整场景配准。磁铁工作目标仍为表面在尖端上方 30 mm。

结果目录 `simulation_output/left_resume_20260930_100550/`；总表 `session_result.json`，
轴运动 `axes/report.json`、`axes/summary.json`、`axes/axis_motion_review.png`。
实时 **http://192.168.5.11:8765/**；今日往返回放 **http://192.168.5.11:8768/**
（PID 345341；图片为起点及 Y +10 mm，最终已返回）。
今日定位 **http://192.168.5.11:8769/S0K2_A6UKaJ-M6Sh0Ju4XMnWBygFoK1L/**
（PID 346777；`current_for_annotation/` 固定图、`annotation/` 记录）。今天标记已保存
并复查，网址后加 `review.png` 可查看。**正在等现场将磁铁表面放到尖端上方约 30 mm、
停止并切回 REMOTE；此间不发运动或递丝。** 就位后重新只读核验，不盲目长距离靠近。

若需要当前状态，只读工具用新目录：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/inspect_elite_readonly.py \
  --elite-ip 192.168.5.66 --out simulation_output/elite_left_current_readonly
```

已完成的轴映射工具默认只做 IK 预览，不能把历史 ROI 用作新姿态的当前观测：

```bash
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/probe_elite_camera_axes.py \
  --rois simulation_output/left_resume_20260930_100550/axis_tool_rois.json \
  --out simulation_output/elite_axis_ik_preview_new
```

`--execute` 才执行三轴各 +10 mm 后返回；1% 关节速度，整轮共用起点范围 TCP 13 mm、
总姿态角 0.5°、关节 3°；失去相机、就绪或位姿范围则停止，不递丝。4 项关键测试通过。
`initialize_elite_http.py` 默认只读，4 项初始化检查通过；今天启动已完成。
`restore_elite_precision_http.py` 的两次远程尝试均被停止，随后由现场恢复，不能继续
串联或重新运行。10 项校准检查通过；历史失败数值详见数据 handoff。

## Historical Paused Checkpoint And Manual Target Review (2026-09-29)

用户已要求暂停、明天继续。恢复入口为 `docs/data-track-handoff.md` 顶部断点；
**不要自动运行历史运动或递丝命令**。当前阶段只完成手工标记、被动局部跟踪复核、
状态查询及三个 10 mm 候选点逆解，没有新增运动或递丝。

实时相机：**http://192.168.5.11:8765/**。
标记页：**http://192.168.5.11:8767/fNxhoGegadHiJkMAvy2f8KQHMkW7nrZ_/**；
其后加 `review.png` 可查看放大的中文标记复核图。固定图不代表明天现场状态。
标记服务 PID 187177；元数据在
`simulation_output/left_full_route_assessment_20260929/annotation/server.json`。
`serve_left_target_annotation.py --resume` 只在相同 capture-dir 下恢复 URL 和已保存
记录；`--review-image` 仅发布指定 PNG。浏览器刷新恢复最新标记，不触发设备动作。

离线复核可复用既有照片，必须指定新的输出目录：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
MPLCONFIGDIR=/tmp/project2026-mpl /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/review_left_target_tracking.py \
  --annotations simulation_output/left_full_route_assessment_20260929/annotation/annotations.jsonl \
  --capture simulation_output/left_full_route_assessment_20260929/marked_passive_capture \
  --out simulation_output/left_marked_review_recheck
```

已有结果位于 `left_full_route_assessment_20260929/marked_review/`。该工具只复用
局部图像跟踪，不能把静止点输出当成移动尖端验证或像素—机器人标定。

## Latest Feedback And Media Review (2026-09-29)

用户对最新 10 mm 轮次仍反馈“机械臂仍未见移动”。物理动作记录已标为存在争议，
不自动重跑运动或递丝。下节数值仍为控制器/电机证据，不是现场接受的物理结果。

相机录像/原图对照页：**http://192.168.5.11:8766/**（PID 180401，
`tools/serve_hardware_review_media.py`）；实时页 **http://192.168.5.11:8765/**。
对照服务只提供固定的相机 JPEG 和录像白名单，诊断 JSON/日志不对外提供；媒体
路径 200、报告与路径穿越 404 已核验。服务信息保存在本轮 `media_server.json`。

## Completed Precision Recovery And Staged Hardware Verification (2026-09-29 17:59)

用户提供校准失败全文并确认 500 mm 工具范围、60 mm 净空、人员离开和按此范围重试。
17:55:00 重试恢复精确状态 1；17:56:58 机械臂完成一次 10 mm 诊断动作（TCP 回读
10.002412 mm，画面及电机/编码器配对支持可见运动）；17:58:39 递丝一包返回 ok。
**本轮动作已完成，不要重跑已完成命令。** 下方 17:49 失败是已恢复的历史记录。

本轮证据：`simulation_output/left_authorized_20260929/index.html`；机械臂录像
`arm10/motion.mp4`、动作报告 `arm10/report.json`、视觉和电机复核
`arm10/physical_review.json`、递丝报告 `arm10/feed_once/report.json`。

后续同类分阶段检查使用 `execute_left_coordination_http.py --fixed-step-mm 10
--arm-only --execute --out <新的输出目录>`；需要重新取得当前现场/相机和控制器状态。
它只移动一次机械臂并记录双相机和原始反馈，不递丝。独立核验物理动作后，将证据
来源写入该目录的 `physical_review.json`，才可调用
`feed_after_verified_arm_http.py --arm-dir <该目录> --execute`，只发送一包、不移动
机械臂；缺少执行开关时只检查记录。每个机械臂结果只能创建一次 `feed_once/`，
即使失败也不删除该目录后重跑，以免重复发送。

校准工具的 `--onsite-clearance-60mm` 只适用于已确认工具长度/空间条件的本次方案；
`--recover-confirmed-calibration-failure` 只清除一次已核对的 7000-C，不能用作通用
清错。默认监测范围不变，扩大模式参数详见数据 handoff；不是厂商运动范围保证。
9 项校准、8 项协同、3 项分阶段递丝检查通过。当前已精确，不需要再次校准。

## Authorized Calibration Attempt Stopped (2026-09-29 17:49)

用户已明确授权校准和机械臂/递丝，17:47:35 校准已实际调用一次。命令回复 true，
随后 RPY 变化超出监测程序 0.02 rad 阈值，程序发送 stop 并获得确认。复查控制器
为 ERROR、精确状态 0，六轴电机速度为 0；未发送路径运动或递丝，不自动重试。
本次已不受下面历史审批拒绝阻塞；需先核对报警全文和校准正常运动范围。

结果目录 `simulation_output/left_authorized_20260929/` 包含预检、校准录像与日志、
停止后只读复查和 `review.json`。最近五条报警字符串为
`[0-7000-C],[0-E030-1],[0-E030-1],[0-E030-1],[0-E030-1]`，含义尚未确认。
已有完整记录；不要为重新取得记录而再次执行校准命令。

## Offline Left Route Reference (2026-09-29)

提取原仿真视频的连续左分支轨迹，按弧长每 10 mm 生成参考点；无硬件接口。
必须使用新的输出目录，已有结果不会覆盖。

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
MPLCONFIGDIR=/tmp/project2026-mpl /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/prepare_sim_left_route.py \
  --demo simulation_output/simulation_demo_20260929 --spacing-mm 10 \
  --out "simulation_output/left_route_reference_$(date +%Y%m%d_%H%M%S)"
```

已完成输出为 `simulation_output/left_route_prepared_20260929/`：28 段参考、原路径
275.874479 mm；CSV、报告和中文预览图已检查。仿真世界系尚未配准到当前实机，
不能把 CSV 当作机器人命令；10 mm 是弧长取样间隔，最后一段小于 10 mm。
本工具不调用下面的校准、机械臂运动或递丝命令。

## Monitored Precision Recovery Preparation (2026-09-29 17:35)

编码器精确校准可能产生关节运动并持久改变控制器校准状态。自动审批审核已拒绝
本次实际调用：当前只有继续左分支的授权，缺少该具体操作的明确授权。
**未执行校准；不要改用间接调用绕过拒绝。** 已准备代码、6 项离线检查和只读录像预览。

默认只读预览命令（不调用校准或运动）：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/restore_elite_precision_http.py \
  --elite-ip 192.168.5.66 \
  --out "simulation_output/elite_precision_preview_$(date +%Y%m%d_%H%M%S)"
```

已完成预览为 `simulation_output/left_continue_20260929_172936/calibration_preview/`，
包含 `report.json`、`cameras.avi`、`camera_times.jsonl` 和两路首末图。
用户明确授权一次受监测校准后，方可加 `--execute-calibration`，使用新的输出目录。
实际调用只发一次校准，不递丝、不归位、不重试；15 s 内持续查询，变化超出 TCP
5 mm / RPY 0.02 rad / 单关节 3° 或遇到故障则尝试停止。发生失败后先查看报告和现场，
不把再次启动当作恢复。该采样监测不能替代现场空间与人员检查。

## Read-only EC Identity And Physical-Motion Diagnosis (2026-09-29 17:26)

用户报告机械臂肉眼未移动；此前 10.000398 mm 为控制器 TCP 差，物理动作尚未独立确认。
IP `.66`、实际 socket peer `.66:8055`、EC66 型号和现场示教器角度已核对一致。
最新编码器精确状态接口及 M472 均为 0，执行入口现要求精确状态为 1，不会自动校准。
此问题需先核对处理，不能仅凭伺服/同步正常而继续放大或循环发送动作。

只读诊断入口，不移动、不递丝、不修改设备设置：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/inspect_elite_readonly.py \
  --elite-ip 192.168.5.66 \
  --out "simulation_output/elite_readonly_$(date +%Y%m%d_%H%M%S)"
```

已验证输出 `simulation_output/left_route_diagnosis_20260929_171840/ip_verified/report.json`。
目前不自动执行下面的运动命令；应先恢复精确状态，再完成可见物理动作验证。
厂家精确模式说明见 https://www.elibot.com/service/articles/list/193 。

## One Left Hardware Action And Coordination (2026-09-29)

相机直播保持运行，从当前 TCP 做一次左任务动作，不采用旧入口的默认采集起点归位。
以下执行命令会实际移动；每次使用新目录，重新运行意味着新增一次动作，不是恢复旧动作。

用户新授权的固定 10 mm 模式：模型提供方向，用户指定目标位移范数，目标姿态不变。
在当前场景确认运动范围后，使用下面入口执行一次机械臂动作，再递丝 forward 一步：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u tools/execute_left_coordination_http.py \
  --fixed-step-mm 10 --execute \
  --out "simulation_output/real_left_fixed10_$(date +%Y%m%d_%H%M%S)"
```

去掉 `--execute` 只读取模型和 IK，不发送动作。新参数接受 `(0,10]` mm；
此模式固定起终点 TCP 距离，采用 `move_joint`，不保证笛卡尔直线路径。
关节变化检查为最多 5°、速度 1%、位置到位容差 0.02 mm，其余状态和失败停止检查保留。
旧 gain=40 / cap=3 mm 模式仅在不指定 `--fixed-step-mm` 时使用。
递丝 forward 的 `value` 不控制长度，每包仍一步，实际距离尚未标定。
这是人工指定幅度的诊断协同，不是原生模型动作或正式训练标签。

已完成 5 项离线检查和 17:07 的实机只读预览：
`simulation_output/real_left_10mm_20260929_165933/preview/`。
10 mm 目标逆解通过，最大关节变化 2.147067°；该预览目标未直接执行。
执行命令会使用当时的新观测重新推理，预览旧目标不作为可直接重放的命令。

用户确认当前空间无障碍、人员离开后，17:11 已用新观测执行一轮，输出同一父目录
下 `execution/`：实际控制器回读位移 10.000398 mm、到位误差 0.002445 mm，
1 条机械臂命令后递丝 forward 1 包并收到 ok，无重试。用户另确认本轮导丝实际前进，
见 `onsite_feedback.json`；17:12 独立回读仍 STOP、无漂移。`comparison_confirmed.png`
为三阶段对照图。到位后画面显示有人调整相机且 Top 模糊/遮挡，后续动作应重新检查
现场和固定相机；不把本轮协同记录表述为已完成左分支导航。

已完成的小步范围：模型原幅度，最多 1 mm、1% 关节速度、固定姿态、递丝保持。

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u tools/execute_left_model_step_http.py \
  --execute --out "simulation_output/real_left_step_$(date +%Y%m%d_%H%M%S)"
```

实际已执行结果为 `simulation_output/real_left_once_20260929_163532/execution/`：
1 条运动命令，回读平移 0.073434 mm、到位误差 0.001181 mm、递丝 0 包。

用户新指定的一轮协同联调：模型方向 ×40，最多 3 mm、1% 关节速度，
机械臂到位并取得新图后，递丝 forward 一步。递丝是人工指定诊断动作；
保存原始模型意图与实际指令，不作为模型自主协同成功或正式训练标签。
现场确认相机固定、血管入镜且人员离开运动范围并复查后，已于 16:49 执行一轮。
以下为复现入口，每次启动是新的动作轮次：

```bash
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u tools/execute_left_coordination_http.py \
  --execute --out "simulation_output/real_left_coordination_$(date +%Y%m%d_%H%M%S)"
```

两入口去掉 `--execute` 仅作模型/IK 预览。预计约 30–40 秒，失败保留报告。
若发生动作后错误或 UDP 无回执，先核对 `report.json` 和实际设备状态，不自动重发。
协同 UDP 目标为 `192.168.5.13:8888`，源 `192.168.5.11:37011`；
收到字节回执不等于物理推进已确认。首次预检
`simulation_output/real_left_coordination_20260929_164308/` 当时因现场调整未执行；
就绪后的实际执行输出为
`simulation_output/real_left_coordination_ready_20260929_164847/execution/`：
机械臂实测 0.727881 mm、到位误差 0.003588 mm，随后发送 1 包 forward 并收到 ok。
用户现场确认“看到导丝实际前进”，单独保存 `onsite_feedback.json`；未量化推进距离，
未判定左分支导航成功。`comparison_confirmed.png` 为包含现场反馈的三阶段对照图。

## Live Model Prediction Preview (2026-09-29)

保持相机直播服务运行，在宿主机读取两路最新画面和机械臂 TCP，分别展示左右任务
模型输出、1 mm 限幅结果及逆解。只做预测和查询，没有执行参数或递丝连接：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u tools/preview_real10_from_camera_http.py \
  --out "simulation_output/real_model_preview_$(date +%Y%m%d_%H%M%S)"
```

本机 GPU 实测约 31 秒（模型加载约 30 秒）。默认使用既有 Real10 配对权重、
`192.168.5.66` 位姿、`http://192.168.5.11:8765` 双相机 JPEG；不会改动直播占用。
拒绝覆盖旧目录；失败后保留 `report.json` / `worker.log` 并在新目录重试。
实际输出 `simulation_output/real_model_preview_20260929_1625/` 包含原图、
`prediction_preview.png`、`report.json`；两分支预测完成，未发送任何动作。
当前两候选均为小于 1 mm 的平移和递丝保持。用户要求先展示再确认执行；
预测文件是观察记录，不能跳过新观测和状态检查直接当作运动命令。

## Live Camera Preview And Feeder Address (2026-09-29)

同一局域网访问 `http://192.168.5.11:8765/`，双相机页面自动刷新。
最新原图：`http://192.168.5.11:8765/side.jpg`、
`http://192.168.5.11:8765/top.jpg`；状态：`/status.json`。
当前服务 PID 为 `126099`，输出 `simulation_output/camera_live_20260929_reconnect/`。
这只是相机预览，未连接机械臂或向递丝装置发送动作。

相机空闲后，在有 USB 权限的宿主机终端启动（默认使用新的输出目录）：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u tools/serve_camera_preview.py \
  --host 192.168.5.11 --port 8765 \
  --side-serial 317222071938 --top-serial 317222072584
```

前台按 Ctrl+C 停止并释放相机。停止已有后台服务前，先检查命令和输出目录身份：

```bash
ps -p 126099 -o pid,args
```

确认仍为上述预览服务后执行 `kill -INT 126099`；重启后的 PID 以新目录
`server.json` 为准。后续真实采集前先停止预览，避免同时打开相机。

递丝地址按用户确认更新为 `192.168.5.13:8888`。JSON 格式：
`{"command":"move","parameters":{"action":"forward","value":1}}`。
`forward` / `backward` 每包一步；`turn_left` / `turn_right` 的 `value` 为旋转度数。
本轮仅验证新 IP 网络可达和本地 JSON 构造，未发送 UDP 动作包。

## Camera Position Preview Only (2026-09-29)

在两路相机空闲／完成交接后，从可访问 USB 的宿主机终端运行：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B tools/capture_camera_preview.py \
  --side-serial 317222071938 --top-serial 317222072584
```

只拍照，无模型推理、机械臂连接或递丝调用；60 帧预热约 4 秒。
默认生成新的 `simulation_output/camera_preview_时间/`：两张全分辨率 PNG、
`preview.png`、`report.json`。序列号绑定的 Side/Top 仅为配置名称，需根据实际构图
确认视角。发现相机占用时先由使用者交接，不同时启动另一个采集流。
不要覆盖旧结果；中断后保留报告，使用新目录重试。此入口已完成宿主机双相机实拍，
60 帧预热后于 16:01:38 保存新图并释放设备，输出
`simulation_output/camera_preview_20260929_takeover/`。先前展示的 15:51:48 图来自
另一只读会话；用户明确授权后已停止该会话完成接管。后续每次刷新仍使用新的输出目录。

## Linux Diagnostic Simulation Demo (2026-09-29)

本机独立演示入口；规则专家驱动，不连接机器人或加载 PI05。机器人全景与导丝特写
并排输出到 MP4，另有 HTML 播放页、起止帧、`report.json` 和 `states.jsonl`。
场景是诊断配置，未作实机标定。已有 `.venv` 已安装，直接运行：

```bash
cd /media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026
LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/python -B -u tools/run_simulation_demo.py
```

默认运行左右分支，各最多 320 步，每 4 步渲染一次；输出目录自动使用当前时间。
打开终端所示输出目录内的 `index.html` 或 `demo.mp4`。如仅需左分支，增加
`--tasks left`。如指定 `--out`，必须是新目录；中断后保留该目录，在新目录重新运行，
此短演示不提供断点续跑。视频按 15 FPS 播放，不代表实机控制频率。

若需重建本机环境，基于已存在的算法 Python 建立项目内虚拟环境：

```bash
/home/zsw/miniconda3/envs/project2026-pi/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir yourdfpy==0.0.60 'trimesh[easy]==5.1.0'
```

系统另需 `ffmpeg` 与 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`；本机已具备。
继承环境实际版本：MuJoCo 3.8.1、Gymnasium 1.3.0、NumPy 2.2.6、OpenCV 4.12.0.88、
SciPy 1.15.3、Pillow 12.3.0。软件渲染的 8 步短跑连同场景加载耗时约 19 秒。
完整两分支实际耗时约 139 秒，输出 126 帧、8.4 秒视频；结果目录为
`simulation_output/simulation_demo_20260929/`。此目录已经完成，复现时使用默认新目录。

## Conventions

Run commands from the repository root:

```powershell
cd D:\PycharmProjects\project_2026
```

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Current mainline defaults:

```text
collector = simulation.collect_formal_tip_line_guidance
guidewire = tip-centric / hard-elastic
wire visual = continuous line
action = Piper signed feed/hold intent + Elite TCP delta / IK execution
```

The older collection entrypoints are now legacy/diagnostic only:

```text
simulation.collect_tip_guided_wire
simulation.collect_mujoco_physical_guidance
```

Do not use legacy collection scripts for new mainline data unless the task is
explicitly a comparison or regression against an older route.

## Visual Route Debugging

Adjust the visual guidewire route:

```powershell
.\.venv\Scripts\python.exe tools\adjust_wire_visual_outlet.py `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --task left `
  --camera side `
  --start-fraction 0.58
```

Use this when the rendered guidewire exits from the wrong side or the tail path
does not follow the intended vessel-entry route. The tool edits
`wire_visual_piper_exit_point`, `wire_visual_entry_point`,
`wire_visual_entry_progress`, and optional `wire_visual_route_points` in the
route config; it does not change expert control, policy labels, or rollout
semantics.

When `wire_visual_route_points` are present, the tip-centric renderer treats
them as a fixed visual entry prefix. The final route point is used as the
attach marker for where the renderer should resume the dynamic route-to-tip
tail; earlier route points are smoothed with centripetal Catmull-Rom
interpolation. Optional `wire_visual_dynamic_route_points` then shape only the
post-attach visual segment directly before it reaches the current tip. These
controls are visual-only.

Controls:

```text
e: cycle active point (outlet -> entry -> route points)
n: add a manual route point between entry and tip
g: add a dynamic route point after the prefix attach marker
x: delete active route point
j/l: move x, i/k: move y, u/o: move z
[/]: change movement step
1/2/3: side/top/overview camera
h: show/hide robots
s: save, q: quit
```

## Current Mainline Collection

Small visual/logic pilot:

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\formal_tip_line_visual_route_v1_small_check `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 3 `
  --start-fraction-min 0.42 `
  --start-fraction-max 0.54 `
  --max-steps 700 `
  --sample-every 10 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-segments 240 `
  --formal-data `
  --progress-log-every 100
```

Current accepted visual-route baseline:

```text
simulation_output\formal_tip_line_visual_route_v1_small_fix1
accepted_episodes: 20
attempts: 20
samples: 567
left/right success: 10/10 and 10/10
max_contact_strength: 0.0
contact_p95 max: 0.0
min_tip_distance_to_wall: ~1.386 mm
visual review: accepted by user
```

Scale from the accepted visual route:

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\formal_tip_line_visual_route_v1_dataset `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --start-fraction-min 0.42 `
  --start-fraction-max 0.54 `
  --max-steps 700 `
  --sample-every 10 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-segments 240 `
  --formal-data `
  --progress-log-every 100
```

This mainline collector hides tool markers and path tubes by construction. It
uses the current fixed formal settings:

```text
route_plan_step=0.15
start_fraction_min=0.42
start_fraction_max=0.54
elite_target_semantics=front_up_magnetic_target
elite_ahead=0.010
elite_height_offset≈0.00924 with the current vessel_scale=0.077
piper_cmd=0.70
piper_command_period=40
piper_command_width=20
route_plan_command_phase_lock=true
wire_visual_mode=line
wire_visual_radius=0.0008
wire_visual_rgb="0.02 0.02 0.018"
wire_tip_visual_rgb="0.78 0.04 0.02"
wire_tip_visual_segments=8
wire_tip_visual_radius_scale=2.2
wire_tip_marker_radius=0.0020
wire_tip_marker_alpha=0.95
wire_visual_offset=(0.0, 0.0, 0.0)
wire_segments=240
tip_tail_decay_segments=18
```

Formal visual-distance estimator collection with the accepted side-camera
coverage config and fixed rendered-image estimator:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --camera-config simulation\camera_configs\mujoco_camera_side_coverage_v1.json `
  --out simulation_output\formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --visual-distance-estimator `
  --progress-log-every 100
```

Do not use `formal_tip_line_black_red_head_sidecam_v1_dataset_v2` estimator
fields for training. That dataset collected successfully, but visual review
found that the older projection estimator could draw wall markers at visibly
wrong top-camera locations. The fixed estimator measures the rendered image and
emits missing/low-confidence estimates when the red guidewire head is occluded
or too close to the image boundary.

Current result:

```text
simulation_output\formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4495
missing image refs: 0
provenance audit: real_direct_plus_visual_contact passes
visual check: docs\_formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1_visual_check
visual review: accepted by user
side estimator visible: 3915/4495
top estimator visible: 0/4495
estimated_contact_flag: 0/4495
estimated_image_distance_px p50/p95: ~6.54 / 7.79 px
piper_step_command feed/hold: 2355/2140
```

Treat this as the current accepted formal visual-distance estimator candidate.
Use continuous `estimated_image_distance_px` plus
`contact_estimator_confidence`; the all-zero binary contact flag is not yet a
strong training signal.

Paper-aligned estimated 3D tip collection adds a synthetic calibrated RGB-D
estimator on top of the accepted rendered-image red-tip detection. This emits
`estimated_tip_pos_3d`, `estimated_tip_heading_3d`,
`tip_estimator_visible`, `tip_estimator_confidence`, and
`tip_estimator_latency_steps` with observation provenance. The implementation
uses red-tip image visibility plus a declared synthetic calibration/noise
wrapper; it does not expose clean MuJoCo `tip_pos` as a policy input.

Use the manually adjusted top-oblique camera config when collecting the next
estimated-tip dataset. It keeps the accepted side camera unchanged, while the
top camera looks diagonally downward from a less occluded angle so the red
guidewire head remains visible.

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --out simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --visual-distance-estimator `
  --estimated-tip-3d `
  --progress-log-every 100
```

Audit a collected estimated-tip dataset before training:

```powershell
.\.venv\Scripts\python.exe tools\audit_observation_provenance.py `
  simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1 `
  --observation-schema real_direct_plus_estimated_tip_contact `
  --sample-limit 200 `
  --out docs\_formal_tip_line_estimated_tip_3d_top_manual_dataset_v1_audit
```

Use this before any new model training that consumes visual-distance/contact
estimator fields. After collection, inspect:

```text
manifest success and rejected attempts
missing image refs
side/top estimator visibility counts
estimated_image_distance_px and contact_estimator_confidence
estimated_contact_flag distribution
contact/wall/tip_to_magnetic metrics
sample side/top frames
provenance audit with real_direct_plus_visual_contact
```

## Git And GitHub Workflow

This repository is for code, durable docs, configs, and lightweight reference
assets. Do not use GitHub as the storage place for `simulation_output/`,
`branchs/`, local virtual environments, or model/video artifacts.

Use `main` directly when:

- the change is small and low-risk, such as docs, commands, comments, or a
  localized bug fix;
- the change only clarifies current project state or handoff information;
- you have already checked that it does not change action semantics or the
  agreed simulation route.

Use a branch first when:

- changing environment dynamics, action semantics, expert logic, rollout
  execution limits, or dataset acceptance rules;
- running parallel ideas that may be discarded, such as alternative control
  formulations or refactors;
- the change will need side-by-side comparison before merge;
- the user may want to keep the current `main` as the stable reference.

Recommended branch names:

```text
feat/<topic>
fix/<topic>
exp/<topic>
docs/<topic>
```

GitHub is mainly a stable rollback/sync point, not a destination for every small
local tweak. Local commits can be finer grained; pushes should mark meaningful
milestones.

Good push points:

- after a confirmed new mainline entrypoint;
- after changing formal data validity rules;
- after a durable project-direction decision or accepted experiment conclusion;
- after a validated dataset/model/control milestone;
- before a long training or collection run only when the run depends on a
  confirmed code state worth preserving as a rollback point;
- before broad cleanup, renaming, or other repo-wide edits;
- at the end of a work session only if the current state is stable enough for a
  future conversation or another machine to resume.

Do not push immediately for small parameter tweaks, smoke-only experiments,
wording-only edits, or intermediate trial-and-error changes. Batch them into the
next meaningful milestone commit/push once the result is accepted.

Typical small-update flow on `main`:

```powershell
git status
git add <changed files>
git commit -m "docs: update handoff after accel-limit review"
git push
```

Typical experiment flow on a branch:

```powershell
git switch -c exp/magnet-attached-envsuccess-v2
git status
git add <changed files>
git commit -m "exp: adjust corrected-semantics collection acceptance"
git push -u origin exp/magnet-attached-envsuccess-v2
```

Before pushing, check that you are not accidentally including large or local
files:

```powershell
git status --short
git diff --stat
```

If the work changed durable project direction, update and commit these together
with the code when appropriate:

```text
docs/project-state.md
docs/handoff.md
docs/experiment-registry.md
docs/decision-log.md
```

## Smoke-Test Data Collection

Use this before a long collection run:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --out simulation_output\mujoco_physical_feed_action_dataset_v2_smoke `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 3 `
  --seed 8601 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --progress-deltas 14 20 28 40 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 80
```

## Diagnostic Oracle Data Collection

Current MuJoCo expert collection is diagnostic/oracle data, not validated formal
sim-to-real data. The expert uses simulation-only state. Do not scale it as
formal data unless the expert/control stack is revised to satisfy
`docs/simulation-expert-validity-audit.md`.

Current diagnostic dataset command:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --out simulation_output\mujoco_physical_feed_action_dataset_v2 `
  --tasks left right `
  --episodes-per-task 40 `
  --max-attempts-per-episode 4 `
  --seed 8601 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --progress-deltas 14 20 28 40 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 80
```

After collection, inspect:

```text
simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json
```

Check accepted/rejected episodes, sample count, `min_segment_distance_to_wall`,
contact, and progress gain.

Formal validity guard:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --formal-data
```

This intentionally fails with the current oracle expert. A successful
`--formal-data` run should only become possible after the expert/control stack
uses real-implementable inputs.

## Tip-Centric Guidewire Data Collection

Use this for the parallel tip-centric abstraction. It keeps the external
`piper_feed_elite_joint` action schema and corrected Elite-attached magnetic
semantics, but collects from `simulation.tip_guided_wire_env` instead of the
full polyline physical guidewire.

Current tip-centric collection is also diagnostic/oracle data because the expert
uses exact simulated tip, vessel frame, lookahead, wall/contact, and IK.

Short script smoke:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\tip_guided_wire_dataset_smoke_v1 `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 2 `
  --seed 9801 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 120
```

If the smoke is clean, collect a moderate dataset:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\tip_guided_wire_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9801 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 120
```

After collection, inspect:

```text
simulation_output\tip_guided_wire_dataset_v1\manifest.json
```

Check accepted/rejected episodes, sample count, `env_success`,
`stall_fraction`, `min_tip_distance_to_wall`, `max_contact_strength`, and
`tip_to_magnetic_median/p95`.

Formal validity guard:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --formal-data
```

This intentionally fails with the current tip-centric oracle expert.

Formal route-plan probe:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_smoke_v1 `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 2 `
  --seed 9901 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.62 `
  --success-mode env_success `
  --max-steps 500 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --progress-log-every 100 `
  --keep-failures
```

This is not expected to match the oracle expert immediately. Judge it by
progress gain, wall/contact behavior, `tip_to_magnetic`, and whether any
episodes reach `env_success`.

Current result:

```text
simulation_output\formal_route_plan_tip_smoke_v1
formal_valid: yes
accepted episodes: 4/4
left/right env_success: 2/2 and 2/2
samples: 150
steps median: 183.5
stall_fraction: 0.0
min_tip_distance_to_wall min: 0.000546 m
tip_to_magnetic median: ~50-59 mm
contact_p95: ~0.568-0.601
```

This proves route-plan is a formal-valid baseline candidate, but it is not
clean enough to scale. Next, test smaller planned Elite lead and slower route
advance before training:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$variants = @(
  @{name='step024_ahead010'; step='0.24'; ahead='0.010'},
  @{name='step020_ahead010'; step='0.20'; ahead='0.010'},
  @{name='step024_ahead006'; step='0.24'; ahead='0.006'},
  @{name='step020_ahead006'; step='0.20'; ahead='0.006'}
)
foreach ($v in $variants) {
  .\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
    --out "simulation_output\formal_route_plan_tip_sweep_$($v.name)" `
    --tasks left right `
    --episodes-per-task 2 `
    --max-attempts-per-episode 2 `
    --seed 9911 `
    --start-fraction-min 0.58 `
    --start-fraction-max 0.62 `
    --success-mode env_success `
    --max-steps 500 `
    --sample-every 5 `
    --image-size 224 `
    --render-width 960 `
    --render-height 720 `
    --robot-visual-mode kinematic `
    --action-mode piper_feed_elite_joint `
    --hide-tool-markers `
    --hide-path-tubes `
    --expert-mode route_plan `
    --formal-data `
    --route-plan-step $v.step `
    --elite-ahead $v.ahead `
    --progress-log-every 100 `
    --keep-failures
}
```

Judge the sweep by keeping `env_success` while reducing `contact_p95` and
`tip_to_magnetic_median/p95`.

Current sweep result:

```text
all variants: formal_valid and env_success 4/4

step020_ahead010:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001548 m
  tip_to_magnetic median: ~3.6-4.7 mm
  tip_to_magnetic p95: ~6.1-10.6 mm
  steps median: 197.5

step020_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.002541 m
  tip_to_magnetic median: ~4.3-5.4 mm
  steps median: 200.5

step024_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001494 m
  tip_to_magnetic median: ~5.5-5.9 mm
  steps median: 185.5

step024_ahead010:
  contact_p95: 0.0
  max_contact_strength: 0.008
  min_tip_distance_to_wall min: 0.001375 m
  tip_to_magnetic median: ~9.4-9.8 mm
  steps median: 183.5
```

Recommended moderate formal route-plan collection:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9921 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.20 `
  --elite-ahead 0.010 `
  --progress-log-every 100 `
  --keep-failures
```

Moderate collection for the current real-action-aligned signed-Piper route-plan
expert. This uses the v8 settings: lower-frequency feed/hold labels, explicit
feed magnitude, and route-plan progress locked to feed phases with a reduced
nominal route step.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_signedpiper_dataset_v8 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9961 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.24 `
  --elite-ahead 0.010 `
  --piper-cmd 0.70 `
  --piper-command-period 40 `
  --piper-command-width 20 `
  --route-plan-command-phase-lock `
  --progress-log-every 100 `
  --keep-failures
```

Elite front-up guidance audit:

```powershell
.\.venv\Scripts\python.exe -B tools\audit_elite_frontup_guidance.py `
  simulation_output\formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1 `
  --out simulation_output\formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1\elite_frontup_audit.json
```

Interpretation:

- `actual_forward_m` is `(magnetic_pose - tip_pos) dot path_tangent`; it should
  usually be positive if the Elite-attached magnet is actually in front of the
  guidewire head.
- `desired_forward_m` is available for datasets collected after the diagnostic
  fields were added and checks the route-plan expert's original desired target.
- `formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1` should not be used
  as the accepted front-up source: its audit found episode median actual
  forward projection around `-2.9 cm`, with about 94-95% negative samples. The
  cause was route-plan progress lag from `--route-plan-step 0.15`.
- A short `--route-plan-step 0.24` smoke under
  `simulation_output\_smoke_frontup_routeplan_step024` restored positive
  desired and actual forward projections for both branches.

Previous result to avoid repeating:

```text
formal_route_plan_tip_signedpiper_smoke_v3 used period=125,width=5. It passed
the formal-data guard and produced sparse labels, but failed all 4 attempts
because the feed duty cycle was too low: accepted_episodes=0/4,
piper_step_command counts feed/hold=20/460, transition_fraction ~=0.0795,
tip_to_magnetic median about 149-168 mm, and contact max about 0.51-0.55.

formal_route_plan_tip_signedpiper_smoke_v4 used period=250,width=125. It passed
formal/env success 2/2, but both short episodes finished before the first hold
phase, so every saved command was feed. It also used full feed magnitude and
degraded expert quality: contact p95 about 0.54-0.59 and tip_to_magnetic median
about 40-42 mm, far worse than the clean v2 millimeter-scale, zero-contact
expert.

formal_route_plan_tip_signedpiper_smoke_v5 used piper_cmd=0.70 and
period=40,width=20. It passed formal/env success 2/2 and produced real feed/hold
labels (feed/hold=53/48, transition_fraction ~=0.24), but quality was still not
clean enough: left contact_p95 ~=0.594, right contact_p95 ~=0.00088, and
tip_to_magnetic median stayed around 20-21 mm. This supports a route-plan/Piper
phase mismatch hypothesis rather than a pure label-frequency issue.

formal_route_plan_tip_signedpiper_smoke_v6 added route-plan-command-phase-lock
with route_plan_step=0.20. It kept formal/env success 2/2 and the same feed/hold
labels, but quality worsened: contact_p95 max ~=0.605 and tip_to_magnetic median
rose to about 29-32 mm. The likely issue is that phase-lock compensated by duty
cycle, so the feed-phase route step became 0.40.

formal_route_plan_tip_signedpiper_smoke_v8 used route_plan_step=0.15 with the
same phase-lock and Piper cadence. It restored clean expert quality:
formal/env success 2/2, zero contact, complete images, feed/hold=55/52, and
tip_to_magnetic median about 2.2-2.8 mm with p95 about 7.5-7.8 mm. A small
stress run, formal_route_plan_tip_signedpiper_smoke_v8_stress4, accepted 8/8
episodes with zero contact, complete images, feed/hold=192/176, and sample
tip_to_magnetic p95 about 7.7 mm. This is the current signed-Piper formal
expert setting to scale.
```

Current moderate signed-Piper dataset result:

```text
simulation_output\formal_route_plan_tip_signedpiper_dataset_v8
formal_valid: yes
accepted/env_success: 40/40
rejected attempts: 0
samples: 1807
image refs: 3614/3614
contact_p95 max: 0.0
sample contact p95: 0.0
tip_to_magnetic episode-p95 median: ~7.6-7.7 mm
sample tip_to_magnetic p95: ~7.7 mm
piper_step_command feed/hold: 951/856
piper_feed values: 0.7 / 0.0
transition_fraction median: ~0.244
```

Legacy continuous-Piper training command. This was useful as a first diagnostic,
but the resulting rollout failed left/right because `piper_step_command`
feed/hold was regressed into mid-valued continuous `piper_feed` outputs rather
than preserved as a discrete real command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600
```

Piper-classification training command. It keeps the external
`piper_feed_elite_joint` rollout action but trains Piper as a three-class
`piper_step_command` head (`-1=retract, 0=hold, 1=feed`) and maps feed back to
`piper_feed=0.7` during rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Result: the model trained and saved correctly, but open-loop diagnostics showed
this is not yet a usable signed-Piper baseline. Piper feed/hold accuracy was
only about `61.8%` (`hold->feed 289/856`, `feed->hold 402/951`), and Elite
absolute target quality was poor (`elite_linf` median/p95 about `0.122/0.206`,
predicted target-step p95 about `0.112` versus expert `0.022`). Do not keep
rerunning this exact training as the next default.

Real-direct observation-schema diagnostic. This removes exact simulator-only
state such as `tip_pos`, `path_progress`, wall distance, local route frame, and
exact contact from the structured policy input. Use this as an observation
alignment check, not as proof that the small BC should become the main path:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_real_direct_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema real_direct `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Senior-Piper-like observation-schema diagnostic. This is closer to the inherited
real Piper input contract: image plus Elite TCP 6D pose, Piper step, and task id.
The simulator now writes `elite_tcp_pose_6d`; old datasets fall back to
`elirobot_pose * 1000 + zero orientation`, so prefer newly collected data for a
true 6D comparison when available:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_seniorlike_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Expected data-schema check:

```text
Each piper_feed_elite_joint action should keep piper_feed and additionally
write piper_step_command (-1=retract, 0=hold, 1=feed) plus piper_command_label.
This preserves current continuous execution while making the candidate real
Piper command label explicit.
```

Current formal route-plan dataset result:

```text
simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1
formal_valid: yes
accepted episodes: 40/40
left/right: 20/20
rejected attempts: 0
samples: 1360
image refs: 2720/2720 exist
env_success: 40/40
steps median/p95/max: 164.5 / 205 / 212
min_tip_distance_to_wall min/median: 1.386 / 1.906 mm
max_contact_strength max: 0.0
contact_p95 max: 0.0
stall_fraction max: 0.0
slow_fraction max: 0.0
tip_to_magnetic median: 3.47 mm
tip_to_magnetic episode-p95 median/max: 6.66 / 11.08 mm
sample tip_to_magnetic p95/max: 8.55 / 13.26 mm
elite action-step episode-p95 median/max: 0.01577 / 0.01701
```

Train a small route-plan baseline:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600
```

Current training result:

```text
model: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1
dataset: simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1
samples: 1360
action mode: piper_feed_elite_joint
elite action representation: absolute
elite_smoothness_weight: 0.0
best val_loss: 0.001086 at epoch 8
epoch 8 val_piper_loss: 0.000423
epoch 8 val_elite_loss: 0.000664
```

After training, run a no-video formal rollout in the same tip environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100
```

Do not add diagnostic rollout helpers such as `--piper-feed-phase-guard`,
`--elite-tip-anchor`, scripted Piper, or tactile safety to a formal rollout.

Current formal rollout result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_rollout_formal_r1
formal_valid: yes
left/right success: True / True
left/right steps: 310 / 281
max_contact_strength: ~0.605 / 0.574
contact median: ~0.365 / 0.403
stall fraction: ~0.332 / 0.374
slow fraction: ~0.452 / 0.480
tip_to_magnetic median: ~25 / 19 mm
tip_to_magnetic p95: ~92 / 127 mm
elite target step p95: ~0.093 / 0.063
```

This is a negative loop-closure result despite numerical success. Do not render
video for this checkpoint. Run an open-loop diagnostic first:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out docs\_baseline_formal_route_plan_tip_step020_ahead010_v1_open_loop_diag `
  --batch-size 64
```

Current open-loop result:

```text
piper_abs median/p95: ~0.026 / 0.049
piper sign mismatch: 0.0
elite_linf median/p95: ~0.051 / 0.088
expert Elite target-step p95: ~0.0156
predicted Elite target-step p95: ~0.113
expert Elite delta-from-current p95: ~0.0062
predicted Elite delta-from-current p95: ~0.0872
```

Next model-side check: train the same formal route-plan dataset with Elite
delta targets, then diagnose open-loop before rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --elite-action-representation delta
```

After training, diagnose before rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out docs\_baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_open_loop_diag `
  --batch-size 64
```

Current Elite-delta open-loop result:

```text
model: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta
best val_loss: 0.000147 at epoch 8
piper_abs median/p95: ~0.011 / 0.027
piper sign mismatch: 0.0
elite_linf median/p95: ~0.0136 / 0.0212
expert Elite target-step p95: ~0.0156
predicted Elite target-step p95: ~0.0312
expert Elite delta-from-current p95: ~0.0062
predicted Elite delta-from-current p95: ~0.0219
```

The no-video formal rollout has been run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100
```

Current result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_rollout_formal_r1
formal_valid: yes
left/right success: False / False
left/right steps: 600 / 600
distance_to_target: ~0.0617 / 0.0773 m
max_contact_strength: ~0.606 / 0.552
contact median: ~0.590 / 0.525
slow-progress fraction: ~0.683 / 0.700
tip_to_magnetic median: ~309 / 434 mm
tip_to_magnetic p95: ~503 / 657 mm
```

This is a negative closed-loop result. Do not render video and do not continue
naive delta as the main model formulation.

Route-plan-anchor check:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0020_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --elite-route-plan-anchor `
  --elite-route-plan-anchor-step 0.20 `
  --elite-route-plan-anchor-ahead 0.010 `
  --elite-route-plan-anchor-joint-limit 0.020
```

This is intended as a formal-safe bounded-target check because the anchor uses a
registered route plan and scheduled progress, not current tip/contact/wall
oracle feedback. Keep `--formal-eval`; if the validity guard rejects it, treat
that as a bug in the audit logic rather than bypassing the guard.

Current `0.020` result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0020_rollout_formal_r1
formal_valid: yes
metadata elite_route_plan_anchor_joint_limit: 0.020
left/right success: True / True
left/right steps: 277 / 276
max_contact_strength: ~0.606 / 0.574
tip_to_magnetic median: ~34 / 32 mm
tip_to_magnetic p95: ~50 / 54 mm
magnetic/Elite pose step p95: ~4.1 / 3.7 mm
progress zero fraction: ~33% / 34%
progress slow fraction: ~42% / 44%
```

This is a real improvement over the unanchored Elite-delta rollout, but it is
not physically clean enough to render video or treat as solved. This directory
has been overwritten once, so future comparisons should use a distinct output
directory and trust `meta.json` over the directory name. The next no-video check
should tighten the route-plan anchor joint window:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --elite-route-plan-anchor `
  --elite-route-plan-anchor-step 0.20 `
  --elite-route-plan-anchor-ahead 0.010 `
  --elite-route-plan-anchor-joint-limit 0.010
```

Current `0.010` result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1
formal_valid: yes
metadata elite_route_plan_anchor_joint_limit: 0.010
left/right success: True / True
left/right steps: 292 / 284
max_contact_strength: ~0.606 / 0.573
tip_to_magnetic median: ~34 / 29 mm
tip_to_magnetic p95: ~57 / 57 mm
contact p95: ~0.602 / 0.565
elite_target_jump_p95: ~0.0168 / 0.0154
executed_joint_jump_p95: ~0.0077 / 0.0069
magnetic/Elite pose step p95: ~3.5 / 2.6 mm
```

Interpretation:

```text
The 0.010 anchor improves command smoothness but not physical quality.
Do not render video or keep shrinking the anchor joint window as the default
next step. Diagnose schedule/Piper/tip progress mismatch first.
```

Diagnose route-plan anchor phase against expert data:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_route_plan_anchor_phase.py `
  --dataset simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1 `
  --rollout simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1 `
  --out docs\_route_plan_anchor_phase_diag
```

Current diagnostic result:

```text
expert plan_minus_actual median/p95:
  left  -2.567 / -0.428
  right -2.467 / -0.428
rollout plan_minus_actual median/p95:
  left   4.808 / 11.640
  right  3.354 / 10.987
rollout corr(plan_lead, contact):
  left  0.820
  right 0.866
```

Interpretation: the scheduled route-plan anchor is out of phase with actual
tip/Piper progress during BC rollout. In the expert data, planned progress is
slightly behind actual tip progress at sampled states; in rollout, planned
progress runs far ahead and contact rises with that lead.

Current moderate dataset result:

```text
simulation_output\tip_guided_wire_dataset_v1
accepted episodes: 40/40
left/right: 20/20
rejected: 0
samples: 3231
image refs: 6462/6462 exist
env_success: 40/40
wall penetration: none in episode/sample checks
stall_fraction: 0.0 on all episodes
```

Train a small baseline on this dataset:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\tip_guided_wire_dataset_v1\manifest.json `
  --out simulation_output\baseline_tip_guided_wire_v1 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700
```

Roll it out in the same tip-centric environment, without video first:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --out simulation_output\baseline_tip_guided_wire_v1_rollout `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard
```

Do not omit `--env-type tip` for this rollout. The default rollout environment
is still the full polyline guidewire for backward compatibility.

Formal rollout validity guard:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --out simulation_output\formal_eval_probe `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical
```

`--formal-eval` rejects diagnostic/oracle rollout interventions such as
`--elite-tip-anchor`, `--tactile-safety`, `--scripted-piper-feed`, and
`--piper-feed-phase-guard`.

## Simulation Validity Audit

Audit existing datasets and rollouts before treating them as mainline evidence:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_simulation_validity.py `
  simulation_output\tip_guided_wire_dataset_v1 `
  simulation_output\mujoco_physical_feed_action_dataset_magnet_attached_envsuccess_v1 `
  simulation_output\baseline_tip_guided_wire_v1_elitedelta_anchor0020_rollout_r1 `
  --out docs\_simulation_validity_audit_current
```

Interpretation:

```text
formal_valid: no known oracle/diagnostic mechanism detected
diagnostic_oracle: useful for diagnostics, not formal data-generation evidence
unknown_legacy: insufficient metadata; do not treat as formal without review
```

Compare formal route-plan expert datasets against real `branchs` motion/Piper
anchors without involving BC rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\compare_formal_expert_real_alignment.py `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_stress_wide_v1\manifest.json `
  --out docs\_formal_expert_real_alignment
```

Interpretation:

```text
Use real branchs pose only as Elite/magnetic-arm motion anchor, not guidewire truth.
The current report finds the largest confirmed gap in Piper semantics:
real binary 0=stop/hold, 1=step_forward labels versus continuous simulated
positive/negative piper_feed. This does not mean retract should be removed; it
means retract needs an explicit real Piper command and label.
```

Diagnose the trained tip-centric model on expert states before another rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --manifest simulation_output\tip_guided_wire_dataset_v1\manifest.json `
  --out simulation_output\baseline_tip_guided_wire_v1_open_loop_diag `
  --batch-size 64
```

Current diagnostic result:

```text
piper_abs_error median/p95: ~0.019 / 0.059
piper_sign_mismatch_fraction: 0.0
elite_linf_error median/p95: ~0.051 / 0.071
expert Elite target step p95: ~0.020
predicted Elite target step p95: ~0.049
```

Do not render another video if this diagnostic still shows large Elite target
error and target-step jitter on expert states.

## Baseline Training

Train first without high Piper negative weighting:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700
```

If Piper rollback is insufficient, try a conservative weighted variant:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_piperw2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --piper-feed-negative-loss-weight 2.0
```

Avoid defaulting to:

```text
--piper-feed-negative-loss-weight 4.0
```

because it previously amplified Elite joint-output instability.

## Baseline Training With Elite Smoothness

Use this when closed-loop rollout shows systematic Elite joint target jitter.
This keeps the external action schema as `piper_feed_elite_joint` and adds an
adjacent-sample Elite joint delta loss during training:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_smooth4 `
  --epochs 8 `
  --batch-size 32 `
  --temporal-batch-size 16 `
  --image-size 224 `
  --max-steps 700 `
  --elite-smoothness-weight 4.0
```

After training, compare rollout target jitter against:

```text
baseline_mujoco_physical_feed_action_v2_rollout:
left/right target_jump_p95 ~= 0.0549 / 0.0529

expert labels in mujoco_physical_feed_action_dataset_v2:
elite joint step p95 ~= 0.00561
```

## Baseline Training With Elite Delta Actions

Use this after rollout-time rate limiting reduces flashes but video still shows
Elite jitter. This keeps the external rollout action schema as
`piper_feed_elite_joint`, but trains the model to predict Elite joint deltas from
the current Elite joint state:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --elite-action-representation delta
```

Roll it out first without video:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard
```

## Rollout With Video

Use this after training:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

For a different checkpoint, change only:

```text
--checkpoint
--out
```

unless the experiment intentionally changes environment parameters.

## Rollout With Elite Acceleration Limit

Use this to test a real-bandwidth execution layer without retraining. It keeps
the external action schema as `piper_feed_elite_joint`, treats the model Elite
output as the desired six-joint target, and limits how fast the executed Elite
joint delta can change between environment steps:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002
```

If it preserves success but jerk remains high, try a tighter value:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel001_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.001
```

If the logged tool pose is smooth but the video still shows mechanical-arm
twitching, keep the `0.010` velocity limit and `0.002` acceleration limit, then
add a jerk limit:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002 `
  --elite-joint-jerk-limit 0.001
```

Start without video. If it preserves success and lowers executed joint jerk,
render a video with the same limits.

Then compare against real bandwidth:

```powershell
.\.venv\Scripts\python.exe -B tools\compare_rollout_control_bandwidth.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel001_r1 `
  --real-summary simulation_output\real_control_bandwidth\real_control_bandwidth_summary.json `
  --out simulation_output\rollout_control_bandwidth_accel_limit
```

Judge this first without video. Only render video after success, drift, step,
acceleration, jerk, and direction-reversal metrics look promising.

## Elite Jump Diagnostics

Run after rollout:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --no-plots
```

Key outputs:

```text
elite_diagnostics_summary_all.json
left\elite_diagnostics_summary.json
right\elite_diagnostics_summary.json
```

Key metrics:

```text
elite_target_joint_step_linf
elite_executed_joint_step_linf
elite_target_to_executed_joint_l2
elite_tool_step
magnetic_step
tip_to_elite_tool
tip_to_magnetic
```

## Real Branch Data Inspection

Use this to summarize the real data under `branchs/`:

```powershell
.\.venv\Scripts\python.exe tools\inspect_branch_real_data.py `
  --root branchs `
  --out simulation_output\real_branch_data_inspection
```

Outputs:

```text
real_branch_summary.json
real_branch_paths.json
real_branch_paths.csv
```

Use this as a reality check for MuJoCo tuning. Current real-data summary:

```text
real paths: 24
pose/image frames: 4039 / 4039
position_step_p95: roughly 5 mm
rotation_step_linf_p95: roughly 1.1e-5
piper labels: binary 0/1
```

## Real Control Bandwidth Inspection

Use this to summarize real Elite/magnetic-arm motion bandwidth from `branchs/`.
This is the preferred anchor for deciding whether a rollout-time rate limit or
control layer is physically plausible:

```powershell
.\.venv\Scripts\python.exe tools\inspect_real_control_bandwidth.py `
  --root branchs `
  --out simulation_output\real_control_bandwidth
```

Outputs:

```text
real_control_bandwidth.md
real_control_bandwidth_summary.json
real_control_bandwidth_paths.json
real_control_bandwidth_paths.csv
```

Current summary:

```text
paths: 24
pose frames: 4039
position step p95 median/max: 5.072 / 5.564 mm/frame
position acceleration p95 median: 2.982 mm/frame^2
position jerk p95 median: 2.835 mm/frame^3
direction reversal fraction median: 0.0000
zero step fraction median: 0.4131
piper labels: {"0": 1849, "1": 2190}
```

If the real sampling rate becomes known, re-run with `--fps`:

```powershell
.\.venv\Scripts\python.exe tools\inspect_real_control_bandwidth.py `
  --root branchs `
  --out simulation_output\real_control_bandwidth_fps30 `
  --fps 30
```

## Rollout-vs-Real Control Bandwidth Comparison

Use this after one or more rollouts exist. It compares simulated
`elirobot_pose`, `magnetic_pose`, and sim-only `tip_pos` against the real
Elite/magnetic-arm bandwidth anchor:

```powershell
.\.venv\Scripts\python.exe -B tools\compare_rollout_control_bandwidth.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_smooth025_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta_rollout `
  --real-summary simulation_output\real_control_bandwidth\real_control_bandwidth_summary.json `
  --out simulation_output\rollout_control_bandwidth_comparison
```

Outputs:

```text
rollout_control_bandwidth_comparison.json
rollout_control_bandwidth_comparison.csv
rollout_control_bandwidth_comparison.md
```

If the current environment refuses writes under `simulation_output`, use a
temporary writable output such as:

```powershell
--out docs\_rollout_control_bandwidth_check
```

Initial check result:

```text
v2 rollout: elirobot step_p95 10.065 mm, accel_p95 10.507, jerk_p95 16.144
rate010_r1: elirobot step_p95 5.777 mm, accel_p95 6.013, jerk_p95 7.947
rate010_smooth025_r1: elirobot step_p95 4.369 mm, accel_p95 4.439, jerk_p95 5.766
elitedelta rollout: elirobot step_p95 11.021 mm, accel_p95 7.171, jerk_p95 10.068
real anchor: step_p95 5.072 mm, accel_p95 2.982, jerk_p95 2.835
```

Interpretation:

`rate010_smooth025_r1` matches the real step scale reasonably well, but its jerk
and direction reversals are still high. This explains why the video can still
look jittery even when step p95 is close to real.

## Elite Visual Jitter Source Diagnosis

Use this when numeric bandwidth metrics look good but the video still appears
jittery. It separates policy target jitter, executed joint jerk, logged tool
pose smoothness, pose consistency, and coarse video frame-difference spikes:

```powershell
.\.venv\Scripts\python.exe -B tools\analyze_elite_visual_jitter_source.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1_video `
  --out docs\_elite_visual_jitter_source_accel002_video_check
```

Outputs:

```text
elite_visual_jitter_source.json
elite_visual_jitter_source.csv
elite_visual_jitter_source.md
```

Current `rate010_accel002_r1_video` diagnosis:

```text
left:  target_step_p95=0.05348, executed_accel_p95=0.00200,
       executed_jerk_p95=0.00400, tool_jerk_p95=2.039 mm
right: target_step_p95=0.04561, executed_accel_p95=0.00200,
       executed_jerk_p95=0.00400, tool_jerk_p95=2.024 mm
diagnosis: policy_target_jitter_remains, executed_joint_jerk_high,
           logged_tool_pose_smooth
```

Interpretation:

If the video still jitters here, do not treat it as a simple tool-position
bandwidth problem. The logged tool pose is already smooth; remaining jitter is
more likely from policy target jumps, joint-chain motion, or robot visual/link
mapping.

## Diagnostic Video With Robot Visuals Hidden

Use this only to isolate whether the visible robot meshes are the source of the
video jitter. Keep `--robot-visual-mode kinematic` so policy side/top camera
inputs stay in-distribution; `--diagnostic-video-hide-robots` hides robot geoms
only in the saved video frames. Add `--diagnostic-video-overlay` when the
magnetic point should remain visible after robot geoms are hidden:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_diag_hide_robots_overlay_video `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002 `
  --robot-visual-mode kinematic `
  --video `
  --video-camera overview `
  --render-every 2 `
  --diagnostic-video-hide-robots `
  --diagnostic-video-overlay
```

Do not use `--robot-visual-mode none` for this diagnostic. That also changes
policy inputs and caused the model to fail with sustained Piper rollback.

## Sim-vs-Real Alignment Table

Use this after real-data inspection and after a representative MuJoCo dataset or
rollout exists:

```powershell
.\.venv\Scripts\python.exe tools\compare_sim_real_alignment.py `
  --real-paths simulation_output\real_branch_data_inspection\real_branch_paths.json `
  --sim-manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --rollout simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --out simulation_output\sim_vs_real_alignment
```

Outputs:

```text
simulation_output\sim_vs_real_alignment\sim_vs_real_alignment.json
simulation_output\sim_vs_real_alignment\sim_vs_real_alignment.md
docs\sim-vs-real-alignment.md
```

Important semantic constraint:

```text
Real branch pose = Elite/magnetic-arm trajectory, not guidewire trajectory.
Compare it with simulated elirobot_pose/magnetic_pose, not with tip_pos as real ground truth.
```

## Controlled Checkpoint Comparison

Use this when comparing two models in the same environment. Keep all parameters
identical except `--checkpoint` and `--out`.

Unweighted reference:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v1\best_model.pt `
  --out simulation_output\compare_unweighted_newenv_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

Weighted reference:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_weighted_v1\best_model.pt `
  --out simulation_output\compare_weighted_newenv_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

Then diagnose both:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py simulation_output\compare_unweighted_newenv_rollout --no-plots
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py simulation_output\compare_weighted_newenv_rollout --no-plots
```

## TCP-Delta Elite Control Path

Use this path after the TCP-delta action implementation. It matches the
inherited Elite control semantics more closely: model predicts Elite TCP delta,
rollout converts TCP target to joints through IK.

Collect a fresh formal dataset with TCP action labels:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9971 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.24 `
  --elite-ahead 0.010 `
  --piper-cmd 0.70 `
  --piper-command-period 40 `
  --piper-command-width 20 `
  --route-plan-command-phase-lock `
  --progress-log-every 100 `
  --keep-failures
```

Train the senior-like TCP-delta baseline:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Run open-loop diagnostics before any rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls_open_loop_diag `
  --batch-size 64
```

Current v10 open-loop interpretation:

```text
piper_sign_mismatch_fraction ~= 0.397
elite_linf median/p95 ~= 0.947 / 1.676 in TCP pose-delta metric space
expert/predicted Elite target-step p95 ~= 7.20 / 7.01
```

Elite TCP-delta scale is now close to the expert target scale, so do not
immediately roll out or tune Elite smoothing. Piper remains weak, and the error
tracks the scheduled route-plan command phase: `step % 40` is feed for the
first half of the cycle and hold for the second half, but
`senior_piper_real_like` does not expose that controller phase. This is a label
/ observation-closure issue, not evidence that more BC rollout tuning is the
right next step.

Diagnostic phase-state training. This keeps the same formal dataset and
real-style TCP-delta Elite action, but adds explicit controller phase
`sin/cos(step modulo 40)` to the senior-like state. It is a diagnostic for
scheduled Piper labels, not a new oracle simulator input:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like_with_phase `
  --piper-command-period-for-state 40 `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Then diagnose before any rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls_open_loop_diag `
  --batch-size 64
```

Current phase-state diagnostic result:

```text
piper_sign_mismatch_fraction ~= 0.371
elite_linf median/p95 ~= 0.910 / 1.577 in TCP pose-delta metric space
expert/predicted Elite target-step p95 ~= 7.20 / 7.62
```

Interpretation: adding explicit controller phase only slightly improves Piper
classification (`0.397 -> 0.371`) and does not justify rollout as the next
default. Treat this as evidence that scheduled Piper feed/hold should become an
explicit real controller/state-machine decision, or be kept outside the BC
target, rather than another small-BC tuning target.

## Estimated Tip / Contact Interface Training

Use this only after the dataset passes the provenance audit for
`real_direct_plus_estimated_tip_contact`. This is a small interface check for
estimated 3D tip/contact observations plus real-style Elite TCP delta and Piper
step classification. Diagnose open-loop before any rollout.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1\manifest.json `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema real_direct_plus_estimated_tip_contact `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Then run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1\manifest.json `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_open_loop_diag `
  --batch-size 64 `
  --image-size 224 `
  --max-steps 700
```

If the open-loop diagnostic is acceptable, rollout must use the same camera and
formal guidewire visual semantics as the estimated-tip dataset. The checkpoint
schema enables the policy estimator automatically.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 5 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --progress-log-every 100
```

If rollout reaches the target by nearly continuous Piper feeding and shows
visible contact, test the real-style Piper step controller. This keeps the
policy output as feed/hold intent but executes at most one Piper primitive per
cooldown window, matching the senior `piper.step_forward(...)` style more
closely than continuous cached feed.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_piperctrl_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 5 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --piper-step-controller `
  --piper-step-cooldown 10 `
  --progress-log-every 100
```

## Observation Provenance Audit

Use this before treating a dataset/model run as formal sim-to-real evidence.
It checks the selected `--observation-schema`, not merely which diagnostic
fields are stored in samples. Storing MuJoCo truth for diagnostics is allowed;
feeding it to the policy is not.

Current v12 senior-like schema should pass:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_observation_provenance.py `
  simulation_output\formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta `
  --observation-schema senior_piper_real_like `
  --sample-limit 50 `
  --out docs\_observation_provenance_v12_seniorlike_check
```

Legacy/full simulator state should fail this formal gate because it consumes
privileged simulator fields:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_observation_provenance.py `
  simulation_output\formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta `
  --observation-schema full_sim_state `
  --sample-limit 50 `
  --out docs\_observation_provenance_v12_fullsim_check
```

If future data uses estimator/tactile fields, the manifest should include
`observation_provenance`; see:

```text
docs/estimator-tactile-interface-spec.md
docs/real-observable-interface-audit.md
```

## Current Registered-Geometry Wallguard Rollout

Use this after training a
`real_direct_plus_estimated_tip_registered_geometry` checkpoint. This is the
currently accepted controller candidate: it does not hold Piper and does not
add active away-wall TCP motion. It only removes the predicted Elite TCP-delta
component that points toward the estimated wall normal.

Do not reuse the rejected strong wallguard setting that held Piper and added
active away-wall correction; it broke tip-magnet coupling and caused the left
branch to fail.

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 8 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --wire-segments 240 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --registered-geometry-wall-safety-guard `
  --registered-geometry-wall-risk-threshold 0.50 `
  --registered-geometry-wall-margin-threshold-m 0.0015 `
  --registered-geometry-wall-pull-threshold-m 0.0025 `
  --no-registered-geometry-wall-hold-piper `
  --registered-geometry-wall-away-delta-mm 0.0 `
  --registered-geometry-wall-max-correction-mm 0.4 `
  --formal-eval `
  --progress-log-every 100
```

## Real Pilot Data Collection

Use this when collecting a small real-system pilot for shadow-mode validation.
This replaces the old hard-coded `data_collect_mode1.py` /
`data_collect_mode2.py` path for new real pilot captures.

The collector is safe by default: it records operator Piper labels, images,
Elite TCP pose, Piper state, and controller logs, but it does not command Piper
and does not advance `piper_step` unless a real execution mode is selected.

Local mock smoke:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out simulation_output\_smoke_real_shadow_pilot_mock `
  --task left `
  --side-source mock `
  --top-source duplicate_side `
  --pose-source mock `
  --max-frames 3 `
  --no-preview `
  --sample-period 0.01
```

Lab-side log-only pilot template:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300
```

In this pure log-only mode, the guidewire will not move unless the operator or
another program controls Piper outside this script. For a real moving pilot
where Piper is controlled externally, add:

```powershell
--external-piper-control
```

This still does not call Piper hardware from the collector, but it treats the
keyboard feed/retract labels as externally executed commands for `piper_step`
accounting.

Replacement UDP feeder device mode:

```text
transport: UDP JSON
default target: 192.168.5.22:8888
forward/backward semantics: one packet = one feeder step
nominal feed scale: 12 mm/step, onsite observed range about 11-13 mm/step
status: experimental fallback only; repeated onsite trials were not good enough
        for current mainline data collection
```

The collector records this as `feeder_device_control_enabled`, not as the old
PiperRobot path. `piper_step` remains the integer feed/retract step count, while
`piper_insertion_length` is the nominal millimeter estimate from
`--feeder-step-mm`.

As of 2026-07-10, do not use this UDP feeder mode as the default real-data
collection route. It is kept for protocol/debug reuse, but the device showed
enough practical delivery problems in repeated onsite tests that data collected
with it should be treated as hardware-diagnostic or fallback data, not clean
expert demonstration data.

If the real entry S-shaped bend has been removed, re-test feed-only insertion
before permanently rejecting Piper or the feeder. The current hypothesis is
that the earlier push failure may have been dominated by the real mechanical
S-bend blockage rather than by actuator control alone. Treat this as a smoke
test first: watch whether the guidewire advances smoothly, then validate the
records before using the data.

Standalone one-step feeder probe on Ubuntu:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device \
  --action forward \
  --host 192.168.5.22 \
  --port 8888 \
  --execute
```

Standalone five-step feeder probe:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device \
  --action forward \
  --count 5 \
  --interval-s 1.0 \
  --host 192.168.5.22 \
  --port 8888 \
  --execute
```

Lab-side UDP-feeder collection template on Ubuntu, using the current observed
camera IDs from the 2026-07-07 setup (`10=side`, `4=top`; re-check if cameras
were unplugged):

```bash
python data/collect/collect_real_shadow_pilot.py \
  --out real_pilot_YYYYMMDD_left_udp_feeder_001 \
  --task left \
  --side-source opencv \
  --side-camera-id 10 \
  --top-source opencv \
  --top-camera-id 4 \
  --pose-source elite \
  --elite-ip 192.168.5.66 \
  --warmup-frames 30 \
  --sample-period 0.2 \
  --max-frames 300 \
  --collection-mode mode1_auto_piper_manual_elite \
  --piper-label-source auto_periodic \
  --piper-auto-feed-period 0.8 \
  --piper-auto-feed-count 14 \
  --stop-after-auto-feed-count \
  --enable-feeder-device-control \
  --feeder-host 192.168.5.22 \
  --feeder-port 8888 \
  --feeder-step-mm 12.0 \
  --piper-control-source feeder_device \
  --elite-control-source human \
  --session-note "UDP feeder: auto periodic feed, human/manual Elite guidance"
```

Validate after collection:

```bash
python tools/validate_real_shadow_pilot.py \
  real_pilot_YYYYMMDD_left_udp_feeder_001 \
  --check-readable-images
```

Keyboard labels during preview:

```text
0 or h: hold
1 or f: feed
r or b: retract label
q or esc: stop
```

Output layout:

```text
manifest.json
records.jsonl
frames/side/*.png
frames/top/*.png
logs/elite_tcp_pose.csv
logs/piper_state.csv
logs/controller_commands.csv
summary.json
```

Validate a collected pilot immediately:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_20260704_left_001 `
  --check-readable-images
```

Render collected PNG frames back into a video:

```powershell
.\.venv\Scripts\python.exe tools\render_real_shadow_video.py `
  collected_data\real_align_20260707_left_elite_step_calib_001 `
  --camera hstack
```

The real collection format stores frame images under `frames/side/*.png` and
`frames/top/*.png`, with paths and timestamps listed in `records.jsonl`.
`--camera hstack` renders side and top views side by side. Use `--camera side`
or `--camera top` for a single view. The script infers FPS from record
timestamps unless `--fps` is provided.

Before collecting motion data, run a short camera/Elite-pose smoke that does
not move Piper:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_smoke_left_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 30 `
  --collection-mode shadow_static
```

Then validate:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_smoke_left_001 `
  --check-readable-images
```

Mode1-style pilot: Piper advances periodically while the operator controls
Elite/magnetic guidance. This is the closest replacement for senior
`data_collect_mode1.py`, but it writes the current `records.jsonl` schema.
Only use this after physical setup/supervision is confirmed, because the
collector commands Piper:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_mode1_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --collection-mode mode1_auto_piper_manual_elite `
  --piper-label-source auto_periodic `
  --piper-auto-feed-period 0.8 `
  --piper-auto-feed-count 14 `
  --stop-after-auto-feed-count `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source human `
  --session-note "mode1: collector periodic Piper feed, human/manual Elite guidance"
```

Mode2-style pilot: Elite is moved by the operator or a senior-style external
path script while this collector commands Piper from keyboard labels. This is
the closest replacement for senior `data_collect_mode2.py` when using the new
schema:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_mode2_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --collection-mode mode2_auto_elite_manual_piper `
  --piper-label-source keyboard `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source senior_script `
  --session-note "mode2: external/senior Elite path, keyboard Piper feed labels"
```

If another program controls Piper and the collector should only record, use:

```powershell
--external-piper-control
```

With external Piper control, also set `--piper-control-source human` or
`--piper-control-source senior_script` so the dataset records who actually
executed the motion.

## Real-System Shadow-Mode Adapter

Use this before commanding real robots. It runs the current dual-camera/state
policy on real-system-style image/state records and writes raw policy outputs
plus the conservative Elite-only wallguard output. It never calls Elite or Piper
hardware APIs.

To convert the senior `branchs/` data into this format:

```powershell
.\.venv\Scripts\python.exe tools\build_branchs_shadow_input.py `
  --branch-root branchs `
  --out simulation_output\branchs_shadow_input_visualcontact_full.jsonl `
  --visual-contact-estimator
```

This conversion is diagnostic only. `branchs/` contains one camera, Elite 6D
pose, and Piper 0/1 labels, but no top camera, estimated 3D tip, or registered
wall-geometry estimator fields. The converter therefore duplicates the single
image as both side/top input and writes low-confidence estimator placeholders.
With `--visual-contact-estimator`, it also runs a simplified HSV red-tip plus
image-edge-distance estimator that fills `estimated_contact_flag`,
`contact_estimator_confidence`, and `estimated_image_distance_px`. This is a
temporary 2D contact/tactile-like signal, not an estimated 3D tip replacement.

Input JSONL schema, one record per frame:

```json
{
  "task": "left",
  "step": 0,
  "side_image": "path/to/side.png",
  "top_image": "path/to/top.png",
  "state": {
    "elite_tcp_pose_6d": [0, 0, 0, 0, 0, 0],
    "piper_step": 0,
    "piper_insertion_length": 0.0,
    "estimated_tip_pos_3d": [0, 0, 0],
    "estimated_tip_heading_3d": [0, 0, 0],
    "tip_estimator_confidence": 0.0,
    "tip_estimator_visible": 0,
    "estimated_contact_flag": 0,
    "contact_estimator_confidence": 0.0,
    "estimated_image_distance_px": 0.0,
    "estimated_wall_margin": 0.0,
    "estimated_wall_margin_fraction": 0.0,
    "route_estimator_confidence": 0.0,
    "estimated_magnet_wall_pull": 0.0,
    "estimated_wall_side_risk": 0.0,
    "estimated_wall_normal_3d": [0, 0, 0],
    "estimated_route_tangent_3d": [0, 0, 0]
  }
}
```

Run shadow inference:

```powershell
.\.venv\Scripts\python.exe tools\real_shadow_policy_adapter.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls\best_model.pt `
  --input-jsonl path\to\real_shadow_input.jsonl `
  --image-base-dir . `
  --out simulation_output\real_shadow_predictions.jsonl
```

The output records:

```text
raw_policy_action          # model output before safety filtering
shadow_controller_action   # action after Elite-only wallguard
wallguard                  # active/corrected/reason/correction metadata
```

Use `raw_policy_action` to judge model ability and
`shadow_controller_action` to judge real-system execution behavior. A high
wallguard intervention rate means the model is not yet reliable, even if the
guarded action looks safer.

## Branchs-Native Real-Data Baseline

Use this to test whether the model/training pipeline can learn the senior real
data format itself. This is not a final VLA model and not a complete
sim-to-real interface; it is a real-data diagnostic baseline.

Build the manifest:

```powershell
.\.venv\Scripts\python.exe tools\build_branchs_training_manifest.py `
  --branch-root branchs `
  --out simulation_output\branchs_training_manifest_full.json
```

The generated labels are:

```text
input:  duplicated single camera image + Elite TCP 6D pose + Piper step + task
output: Elite TCP delta from pose[t+1] - pose[t] + Piper hold/feed label
```

Train:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification
```

Diagnose before any rollout or real control:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

## Single-Camera Senior-Like Sim Alignment

Use this to test whether a sim-trained model still fails on `branchs` after
matching the single-camera senior-like input shape more closely. This is a
schema/domain diagnostic, not a rollout-tuning step.

Build a sim manifest that duplicates one selected camera as both side/top:

```powershell
.\.venv\Scripts\python.exe tools\build_single_camera_seniorlike_manifest.py `
  --manifest simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1\manifest.json `
  --out simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --camera side
```

Train:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification
```

Diagnose on sim same-schema data:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

Diagnose the same sim-trained checkpoint on senior real `branchs` data:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_on_branchs_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

Current result:

```text
sim same-schema open-loop:
  Piper mismatch 21.17%, Elite TCP-delta linf median/p95 0.44/1.69
branchs open-loop:
  Piper mismatch 45.73%, all hold frames predicted as feed,
  Elite TCP-delta linf median/p95 5.32/9.71
branchs shadow:
  feed=4039, hold=0, mismatch=1849/4039=45.78%
```

Interpretation:

```text
Single-camera schema matching alone does not close the sim-to-real gap.
Continue toward visual/domain alignment or real-data collection rather than
treating the missing second camera as the only blocker.
```

## Sim/Real Visual-Domain Audit

Use this after a sim-to-real shadow or branchs open-loop mismatch to check
whether the image distributions are already visibly different before tuning the
policy.

```powershell
.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out docs\_sim_real_visual_domain_audit_singlecam_v1 `
  --camera side `
  --samples-per-task 12
```

Current result:

```text
sample pairs: 24
missing inputs: 0
luminance_mean: sim 184.38 vs real 120.49
saturation_mean: sim 0.278 vs real 0.388
bright_ratio: sim 0.324 vs real 0.002
edge_density: sim 0.034 vs real 0.026
contact sheet: docs/_sim_real_visual_domain_audit_singlecam_v1/contact_sheet.png
report: docs/_sim_real_visual_domain_audit_singlecam_v1/report.md
```

Manual read:

```text
The sim side images are bright, clean, pale-rendered scenes with translucent
vessel geometry and synthetic tabletop/background. The real branchs images are
darker, green-background physical camera frames with real robot/fixture
appearance, stronger crop/perspective, and real occlusion. This is an obvious
visual-domain gap and supports fixing rendering/domain alignment or collecting
new aligned real data before another BC rollout-tuning loop.
```

## Branchs-Like MuJoCo Render Preset

Use this for the current A route: keep MuJoCo, but render formal images closer
to the senior `branchs` side-camera distribution before trying more model
tuning. This is a renderer/domain diagnostic, not a policy-quality result.

Short smoke:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\_smoke_branchs_like_render_preset_v1 `
  --tasks left right `
  --episodes-per-task 1 `
  --max-attempts-per-episode 1 `
  --max-steps 320 `
  --sample-every 20 `
  --progress-log-every 100 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --render-domain-preset branchs_like_v1 `
  --wire-segments 240 `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 8 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --formal-data
```

Audit against `branchs`:

```powershell
.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\_smoke_branchs_like_render_preset_v1\manifest.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out docs\_sim_real_visual_domain_audit_branchs_like_v1 `
  --camera side `
  --samples-per-task 11
```

Current smoke/audit result:

```text
left/right env_success: 2/2
contact_p95/max: 0.0 / 0.0
luminance_mean: sim 118.60 vs real 120.76
saturation_mean: sim 0.407 vs real 0.392
bright_ratio: sim 0.000 vs real 0.003
edge_density: sim 0.027 vs real 0.026
contact sheet: docs/_sim_real_visual_domain_audit_branchs_like_v1/contact_sheet.png
```

User visual review:

```text
Accepted as usable for the next step. Remaining major visual gaps are real
glass-vessel reflections and real-camera blur; physical fixture/crop/occlusion
differences also remain. Treat this as good enough to collect a small
branchs-like dataset before training, not as final visual realism.
```

Plan C renderer-only Isaac spike:

```text
Current machine does not need Isaac Sim installed. Export a portable package
locally, copy it to the Isaac host, and build USD stages there. This is not a
physics/control migration.
```

Export package locally:

```powershell
.\.venv\Scripts\python.exe tools\export_isaac_renderer_spike.py `
  --sim-manifest simulation_output\formal_tip_line_branchs_like_v1_full\manifest.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\isaac_renderer_spike_branchs_like_v1_pkg `
  --camera side `
  --samples-per-task 3 `
  --copy-reference-images
```

Copy the whole output directory to the Isaac host:

```text
simulation_output\isaac_renderer_spike_branchs_like_v1_pkg
```

Build USD stages on the Isaac host from inside the copied package:

```powershell
<ISAAC_SIM_ROOT>\python.bat isaac_build_stage.py --package package_manifest.json --headless
```

Linux equivalent:

```bash
<ISAAC_SIM_ROOT>/python.sh isaac_build_stage.py --package package_manifest.json --headless
```

Expected package contents:

```text
package_manifest.json
samples.jsonl
README.md
isaac_build_stage.py
assets/
reference_images/
```

Expected Isaac-side output:

```text
isaac_assets/vessel_0422.usd
isaac_stages/<frame_id>.usd
```

Acceptance criterion:

```text
First open the generated USD files in Isaac and visually compare them against
reference_images/sim and reference_images/real. Only add automated Isaac
rendering after the USD scene loads correctly on that Isaac version.
```

## Algorithm / VLA Commands

Current algorithm commands live in:

```text
docs/algorithm-track-commands.md
```

The pre-split Pi-Style/OpenPI/PI05 command history is preserved at:

```text
docs/archive/algorithm-command-history-before-track-split-2026-07-17.md
```

## Documentation Updates After Experiments

After meaningful data/simulation/real-collection experiments, update:

```text
docs/data-track-handoff.md
docs/experiment-registry.md
```

Algorithm experiments update:

```text
docs/algorithm-track-handoff.md
docs/algorithm-track-commands.md  # only for reusable command changes
```

Update shared `docs/project-state.md` or `docs/handoff.md` only when the
cross-track interface or routing changes. Group-meeting facts from either track
go into the shared append-only `docs/weekly-meeting-log.md`.

If a decision changes, update:

```text
docs/decision-log.md
```

If a new recurring failure is discovered, update:

```text
docs/troubleshooting.md
```
