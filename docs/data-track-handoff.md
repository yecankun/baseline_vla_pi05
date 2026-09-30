# Data Track Handoff

Last updated: 2026-09-30

This document is for the agent continuing the unresolved data problems:
simulation data quality and real data collection. Algorithm innovation and
PI05/OpenPI model changes may be handled by a separate agent. Keep those two
threads connected through the data/action interface, but do not let model
experiments redefine data semantics silently.

Algorithm state and commands live in `docs/algorithm-track-handoff.md` and
`docs/algorithm-track-commands.md`. Do not mirror algorithm experiment history
here; keep only the shared data/action interface needed by the data track.

## Current: Left Target Reached — Operator Confirmed, Devices Stopped (2026-09-30 12:54)

当前 **`simulation_output/left_magnet_only_20260930_120439/`**，简称 M。
**用户明确确认“已保存，尖端到达目标”，本轮左分支任务按现场结果完成。**
12:50:41.335 仅递丝一次，设备返回 `ok`，两路画面可见导丝向目标所在左分支推进；
12:54:08 只读核验机械臂 STOP/REMOTE、位置与最后一次动作结束一致，未再动臂/递丝。
到达结果记录于 **M/endpoint_result.json**，总表 **M/session_result.json**。

用户后续明确允许模型内**沿壁滑行**、优先到达目标；此前“贴壁必须先解除”的诊断
条件已被这一指令取代，不能继续作为当前任务的绝对禁行规则。明显折弯、顶住无进展、
错误分支或设备异常仍需停止后续动作。机构 `.5:8888` 为用户给定 **20 mm/forward**，
固定步长不可改；不再找接收源码/尝试改为 5 mm。本阶段最终 **递丝 1 次**。

**结果边界**：用户说“已保存”，但截至最终核验，新页仍未生成 `annotations.jsonl`。
因此到达是明确的**现场确认结果**，并非两路像素误差/毫米误差测量；不补写虚构尖端。
终点原图在 M/final_verification，任务结果 `accepted` 仅限本次现场到达确认；
录像为 `viewed_not_accepted`，不代表整个数据集或 PI05 策略通过验收。

工作位原始基准始终为 `left_work_pose_20260930_104551/axes/`（W），没有重置。
已确认的约 80 mm 后续左向路线从 B/follow2 起算，不要每段重新起算或重复询问同一
范围净空。当前工作是人工定位/局部图像响应诊断，**非 PI05 输出、非已验收训练数据**。

- **step1–4，12:07–12:11**：四段约 10 mm 左上跟随，共 **39.984682 mm**。
  用户在前三段期间说“暂时没有看到导丝变化”；第四段也未确认响应。随后用户说明
  人工测试磁铁能吸引当前导丝，并要求修正位置继续尝试。旧拟合维持了原始偏移，
  不能将其当作已经准确对准尖端，不再将导丝材料/磁铁类型当作当前阻塞。
- **step5，12:24:58–12:25:04，9.994335 mm**；**step6，12:25:38–12:25:44，9.993446 mm**：
  两段恒高 `[0,-10,0]` mm，合计 **19.987781 mm**。磁铁 ROI 水平中心相对旧尖端
  偏差 Side +62 / Top −30.8 px 降为 −4 / +0.2 px。这是近似图像对中，不是磁极/
  工具 TCP 标定，也没有独立测量 30 mm 表面间距。第二段 Side 可见向上弯曲，现场
  明确回复 **“看到导丝随磁铁偏转”**，仅确认本段磁响应，非轴向插入/无接触/最终到达。
- **12:29:43 新标记**：Side tip `[877.2,609.7]`、Top `[1162.3,862.6]`；目标仍
  Side `[768.7,663.5]`、Top `[1207.3,891.1]`。相对偏转前 tip 变化 Side `[-19.8,-37.3]`、
  Top `[0,-17.6]` px，主要是抬起。不能把机器人位移或图像像素当导丝插入毫米数。
- **step7，12:34:10 起，恒高左移 9.994556 mm**：磁铁可见运动，但尖端未确认继续
  左移。随后用户明确说明 **“现在导丝紧贴上侧血管壁，无法确定能否偏转；距离紫色
  目标大约还有30mm”**。因此没有执行第二个左向 IK 目标，也没有发送固定 20 mm。
- **step8_relief，12:40:43–12:40:50**：固定姿态向上远离血管 **10.001071 mm**，
  停留观察 5 s，用户回复 **“仍贴着上壁”**。
- **step9_relief，12:43:42–12:43:48**：再向上 **9.995086 mm**，观察 5 s；累计退让
  **19.996157 mm**。两个相机直接复核退让运动。用户随后确认仍贴壁，但贴壁位置向
  目标偏移，并明确要求允许贴壁继续完成任务；这一反馈替代了当时等待解除贴壁的步骤。
- **12:50:41.335，step9_relief/feed_once**：在磁铁当前位置保持不动，依据现场剩余
  约 30 mm 与允许沿壁的指令，向 `.5:8888` 发送 **一次 forward** 并收到 `ok`。
  没有重试/第二次发送。M/feed3_response_recording 保存 **67 对原图/6.7 s**；
  图像可见导丝沿目标分支推进。用户随后明确确认**尖端到达目标**，结束本轮。
  20 mm 是机构单步规格，非独立测得尖端走了 20 mm；不存在精确误差测量。

当前 STOP/REMOTE、precision=1、servo/sync=true，TCP
`[-328.159038,81.164634,370.661748,3.140243,-0.007524,-1.594430]`。
M 共 **9 次机械臂动作、89.963176 mm 路径、递丝 1 次、880 对 1920×1080 原图**，
无校准/清错/重新使能。今日累计远程路径动作 **24**、递丝尝试 **4**（旧 .13 一次无
应答，新 .5 三次应答）。12:49:49 发送前网络/当前画面/停稳复查通过。
`session_result.json`、各段 `arm_and_wire_review.json` 分开保留运动/磁响应/接触证据；
step9_relief/physical_review.json 记录最新用户允许沿壁后的一次发送依据，
其 feed_once/physical_review.json 保存本次最终现场到达反馈。单次输出目录已用掉。

`tools/follow_marked_tip_http.py` 新模式与关键验证：

- `--corridor-preview`：从已核对有限轨迹派生包络，原始 W 基准不变，默认仍 31 mm/5°。
  旧四段左上走廊 72 mm/9°；两段水平对中走廊 92 mm/12°，二者均已完成。
- `--align-over-tip`：恒高 −Y 10 mm，要求两路近似水平中心误差都减小。
- `--guidance-review`：实际磁响应确认后，按人工目标走一次 −Y 10 mm。原基准上限
  112 mm/15°，从 B/follow2 的确认路线 ≤80 mm；两路估计磁铁与尖端侧向差 ≤12.5 mm。
  这是本次诊断程序的试验界限，非硬件规格。其第二段有 IK 但未执行；本轮结束后不重放。
- `--relieve-wall-contact`：已观测上壁接触、且新相机复核向上净空后，仅 +Z 10 mm。
  两次分别派生 103/106 mm 与 14° 原基准界限，走廊半径均 3 mm。始终保留单段
  10 mm、1% 速度、13 mm/3° 监测、姿态 0.5° 与原始高度下限，无自动重试/递丝。
- M/corridor_checks.json（8 项）、lateral_checks.json（9 项）、guided_left_checks.json
  （6 项）、wall_relief_checks.json（5 项）通过；涵盖基准不重置、有限轨迹、方向、
  越界、无磁响应/无接触证据拒绝、尖端无进展不能连续超前、退让方向不能朝向血管。

实时 **http://192.168.5.11:8765/**；当前最终递丝回放 **http://192.168.5.11:8768/**，
对比图 `/review.png`。M/media_feed3，67 帧/6.7 s，视频全解码与代表画面复核通过，
`viewed_not_accepted`；当前媒体 PID **396159**；页面/图/视频 200、私有报告 404。
录像标题描述的是录制时待核验，网页与 manifest 记录随后收到的现场到达确认。
历史 M/media（step1–4）、media_lateral（step5–6）、media_guided（step5–7）、
media_relief（两段退让）均保留。
最终固定定位页 **http://192.168.5.11:8770/vbEMVlopVeJCbcJ4wF5jJCmIYc_luARC/**，
PID **394618**，M/annotation_after_feed3；原目标沿用，尚未收到新的像素标记文件。
旧 8770 URL 已替换，旧标记保留于 M/annotation_after_guidance；旧 8769 页更早。

**当前无后续动作待执行。** 保持终点状态，不自动归位/重放/递丝。若开展新试验，先
重新读设备状态、抓图和定位；不能将当前到达确认当作另一轮运动的逐项验证。

## Earlier: Feeder Recovered, Two Tip-Following Moves And A Second Feed (2026-09-30 11:39)

当前阶段 **`simulation_output/left_feeder_resume_20260930_111102/`**，简称 B；工作位
基准仍为 `simulation_output/left_work_pose_20260930_104551/axes/`，简称 W。
**完整左端送丝仍未完成。机械臂已停；新标记和剩余路线现场确认已收到，当前缺递丝接收端源码/步长设置入口。**

- 用户告知“IP 变成 5.5”，实际新地址 **192.168.5.5:8888**；11:17 ping 2/2、ARP
  MAC `30:83:98:8e:38:84`。旧 `.13` 故障已由地址变更解释，不能再当作当前阻塞。
  所有当前递丝命令须显式传 `--feeder-host 192.168.5.5`，保留历史默认不代表仍用 `.13`。
- 11:18:41 向新地址发送一次 forward 并收到 `ok`，原始报告保留于
  `W/left_step/feed_after_device_recovery/report.json`。用户明确确认“导丝尖端实际前进了”；
  每个 forward **20 mm** 是用户提供的送丝机构步长，不等于独立测得尖端走了 20 mm。
  `physical_review.json` 单独保存现场反馈，原 `.13` 无回复记录不改、不自动补发。
- 11:23 新人工尖端：Side `[1041.7,673.3]`、Top `[1085.6,883.5]`；目标仍为 Side
  `[768.7,663.5]`、Top `[1207.3,891.1]`。B/annotation_after_feed 第 2 版与第 1 版相同。
  先前局部图块跟踪几乎不动，未跟住人工确认的尖端，**不可作为移动尖端跟踪输入**。
- 新 `tools/follow_marked_tip_http.py` 使用人工尖端变化和 W 的局部图像响应，仅拟合 Y/Z，
  固定姿态、每段 10 mm、1% 速度，无递丝/归位/自动重试，**不是 PI05 模型输出**。
  同一原始 W 基准下累计 TCP ≤31 mm、姿态 ≤0.5°、关节 ≤5°、不下降；单段原有
  13 mm / 3° 监测仍在。只读预览检查当前位姿、图像一致性和 IK，再显式执行。
  关键离线单位/方向/累计范围检查通过；局部响应不是全场三维配准，未独立测量 30 mm 间距。
- B/follow1：11:30，一段 `[0,-9.583169,+2.857074]` mm，实测 **9.995234 mm**；
  Side `[-31,-10]` px/NCC .935、Top `[15,-4]`/.796，54 帧。
  B/follow2：11:37:49–11:37:54，一段 `[0,-9.537642,+3.005558]` mm，实测 **9.996867 mm**；
  Side `[-31,-11]`/.821、Top `[15,-4]`/.546，56 帧。第二段局部拟合剩余约 8.55 mm，
  采用 10 mm 后略领先拟合位置，不声称精确对齐。两路原图可见磁铁移动；Top 模板置信
  下降，不能据此声称精确定位/间距。两段 `physical_review.json` 均为助手复核，
  `viewed_not_accepted`，不是用户验收。
- **11:39:21.786**：B/follow2/feed_once 仅向 `.5:8888` 再发一次 forward，收到 `ok`；
  机械臂保持不动。B/feed2_response_recording 保存 **52 对原图、5.2 s**；本次尖端
  是否前进尚待现场反馈，不能沿用 11:18 那次的确认。
- **11:43:43 新标记已收到**：Side 尖端 `[897,647]`、Top `[1162.3,880.2]`，目标不变。
  相对上一轮变化 Side `[-144.7,-26.3]`、Top `[76.7,-3.3]` px；剩余直线像素距离
  129.36 / 46.30 px，分别为刚才标记间位移的 .88 / .60。人工标记支持继续接近目标，
  但像素距离不是血管弧长/毫米，也未独立证明本次 20 mm 真实插入量或无接触。
  `annotation_after_feed2/progress_review.png` 已直接复核，`viewed_not_accepted`。
  **不再发固定 20 mm**；已询问设备接收端能否改为 5 mm。现有 UDP 客户端只能发
  固定 forward/backward，value 仅控制旋转，不能靠改 Python 的 value 缩短前进。

最终 STOP / REMOTE、precision=1、servo/sync=true，TCP
`[-328.148480,150.819230,345.782031,3.140238,-0.007601,-1.594423]`。
B 阶段新增机械臂 2 次、递丝尝试 2 次均回复；今日累计路径动作 **15** 次，递丝尝试
**3** 次（旧 .13 一次无回复、新 .5 两次回复）。本阶段无校准、清错、重新使能。

实时 **http://192.168.5.11:8765/**；本轮两段跟随与递丝回放 **http://192.168.5.11:8768/**。
新固定定位页 **http://192.168.5.11:8769/M0nkQiL3yKM0PToZ5yKhqPwDHjIi_Kct/**，
B/annotation_after_feed2；目标沿用，当前两路尖端已保存第 1 版。
旧 8769 URL 已替换，旧图和标记文件保留。当前累计距原工作位约 29.7 mm，已接近
31 mm 包络；`remaining_follow_estimate.json` 仅离线估计：跟随当前新尖端尚需
Y −43.48 / Z +4.60 mm，拟合残差最大 9.44 px；直接外推至最终目标约 Y −80.10 /
Z −3.21 mm，距原工作位约 109 mm。**这些是局部模型外推，不是可直接执行的路径**。
用户随后明确回复 **“整段路径已核实，满足条件”**，对应整段约 80 mm 后续路线的
关节/长杆/磁铁净空和约 30 mm 工作间距；**这项现场确认已完成，后续不要重复询问同一项**。
11:52:10 `remaining_follow_ik_preview/report.json` 只读预览跟随当前尖端的四个 10 mm
目标：每段 `[0,-9.944457,+1.052507]` mm；段间最大关节差 1.215–1.337°，末段相对
原基准 TCP 69.622 mm、最大关节差 8.431°。四段均有连续逆解，**全部尚未执行**；
现有 `follow_marked_tip_http.py` 31 mm/5° 原始包络尚未修改，不能直接用默认执行器
重放预览。若继续，应根据已确认路线和实际逐段画面制定有界扩展，保留原始基准；
不能重置基准掩盖累计位移、把插入量当尖端位移或连续盲目递丝。
本轮回放为 54+56+52=162 帧/16.2 s，视频全解码和三段中文代表画面通过检查，
媒体页/图/视频 200、私有报告 404；当前媒体 PID 365191，原上一段媒体保留于 B/media。
定位服务已用 `--resume --review-image .../progress_review.png` 加入进展对比图，
原 URL 和标记字节保留，当前 PID 366684；同 URL 后加 `review.png` 可看复核图。

**接收端检查结果（11:53）**：用户允许检查并修改接收程序，但回答“无法确认源码位置”。
已搜索当前工程、`/home/zsw/PycharmProjects/real_collection`、Downloads 和 Desktop：
匹配 `turn_left/turn_right` 协议的仅 Python **发送端**；现有 raw STM32 是旧串口
单字节命令及固定电机时序，不含当前 UDP JSON 接收代码。Downloads 的
`v2_ethernet_verified_20260909` 是其他七路线圈控制协议，不应改刷到递丝器。
`.5` 的 TCP 22/80/443 均拒绝连接，仅建立连接探测，未登录、未发 UDP、未改设备。
当前 **20 mm/forward 仍有效，未改为 5 mm**；需要当前设备的接收/固件工程或实际
步长配置入口，不能修改客户端 value 假装已经缩短，也不能把旧固件时序直接等同毫米。
此检查期间新增机械臂和递丝指令均为 0；整段路线已确认这一事实不会因源码缺失而失效。

## Earlier: Feeder Reported Powered On; Still No Network Resolution (2026-09-30 11:14)

用户“递丝装置已打开，继续”授权继续任务。新检查目录
**`simulation_output/left_feeder_resume_20260930_111102/`**，总表 `session_result.json`。
11:11:04 首次 ping 两包无响应；11:14:34 给开机联网时间后再次一包仍无响应，ARP
`INCOMPLETE`，期间也读到 `FAILED`。本机 enp5s0 UP、carrier=1、`.11/24`、目标路由
和源地址正确，UDP 本地端口 37011 未占用；`.66` 控制器查询成功。因此目前问题在
递丝器地址/链路可达性，尚不能归因为 JSON、动作字段或程序协议。

控制器仍 STOP / REMOTE、precision=1、servo/sync=true，TCP 与上一段左向 10 mm
完成位置相同；两路新原图已查看，`viewed_not_accepted`。**本次新增机械臂、使能、
校准和递丝指令均为 0**。已询问现场开机后的实际 IPv4、UDP 接收绑定地址/端口及
网口灯，等待信息；不能因为已通电而记成已联网，也不向无应答地址盲目补发。

已准备 `feed_after_verified_arm_http.py --device-recovery-check <目录>` 处理用户
明确恢复设备后的一次新请求：保留原 `feed_once/` 不动，要求首次确为未应答、
新的 `.13` ping/ARP 成功且在 5 分钟内；另用唯一 `feed_after_device_recovery/`
防止重复。实际发送前仍查当前精确/停止状态、原机械臂位姿和新鲜图像；不移动机械臂。
原默认单次规则不变，不自动重试。新增无回复状态 `one_feed_submitted_no_reply`，
不把 socket 提交误记成设备响应。4 项关键测试通过，当前失败网络记录确实被拒绝，
**未执行恢复递丝、未创建恢复发送目录**。网络未通，不应传执行开关尝试碰运气。

当前任务仍为完整左分支送丝，上一段机械臂已完成，导丝推进未确认。收到实际地址或
网络恢复信息后先重新查连通和画面，再完成单次设备响应核验。实况仍为
**http://192.168.5.11:8765/**，最新动作回放 **http://192.168.5.11:8768/**（PID 355202）。

## Earlier Work Pose: 10 mm Guidance And Unacknowledged Feed (2026-09-30 10:56)

用户已回复“已完成，继续”，确认现场建立磁铁表面在尖端上方约 30 mm 的起始工作位，
并停止、回 REMOTE。新输出 **`simulation_output/left_work_pose_20260930_104551/`**，
总表 `session_result.json`。此前等待起始就位的状态已解除，完整左端送丝仍未完成。

- 10:45:52 只读核验 STOP / REMOTE、precision=1、sync=true、servo=false。TCP
  `[-328.148887,179.930352,339.913869,3.140231,-0.007633,-1.594412]`；TOOL3 仍全零。
  相机确认磁铁已靠近血管上方、没有可见人员；30 mm 是现场给定位置，不是相机独立
  测距。10:48:28 仅上伺服一次，无清错/同步/校准。
- `axes/` 10:48:42–10:49:19：在新工作位置三轴各 10 mm 往返共 6 次，全部到位，
  无停止/重试，373 帧。新的局部图像矩阵（行 Side u/v、Top u/v，列 X/Y/Z）为
  `[[0.3,3.3,0.1],[-0.2,0.2,-2.9],[-0.3,-1.6,0],[-0.2,0,-1.7]]` px/mm。
  条件数 24.043，不适合当作完整三维配准使用。Y 出程模板置信 Side 0.952、Top 0.925，
  对应 `[33,2]` / `[-16,0]` px；两个视角的 **−Y** 水平方向均朝人工左目标。
- 新 `tools/execute_local_left_step_http.py` 默认只读预览，`--execute` 才执行一段
  **−Y 10 mm**，保持高度/姿态。使用人工目标和局部 Y 响应，**不是 PI05 模型动作**，
  不是整条血管轨迹；双视角方向、模板置信、当前位姿和逆解检查后复用受限单段执行。
  每个 axes 目录仅可创建一次 `left_step/`，不自动重试/归位/递丝。3 项关键测试通过。
- `axes/left_step/`：10:54 实际接受并完成一段，TCP 位移 **9.998610 mm**；Side
  `[-33,-2]` px、NCC 0.934，Top `[16,0]` px、NCC 0.940，与局部预期一致。55 帧。
  原图已直接复查并结合编码器/控制器回读记录 `physical_review.json`，机械臂物理
  动作可见；`viewed_not_accepted`，不等于用户验收。
- `axes/left_step/feed_once/`：10:56:01.126 仅一次 `forward,value=1` send 调用，
  目标 `.13:8888`，绑定 `.11:37011`，等待 2 s **未收到回复**。脚本 `one_feed_sent`
  只表示已向 socket 提交一包，不证明网络到达或设备执行。无补发。机器人前后均
  STOP / REMOTE、precision=1、servo/sync=true。
- 网络只读核查：ping `.13` 两包无应答，ARP `INCOMPLETE`，路由为 enp5s0/source
  `.11`。当前无法确认该地址在线，具体是供电、连接、地址还是程序故障待现场检查。
  用户回复 **“刚才没有观察到”**，含义为没有观察结果，不能记成确认设备未动。
- `feed_response_recording/` 保存 86 对全分辨率 JPEG（新增可选
  `CameraRecording(..., save_original_frames=True)`，默认行为不变）及录像。被动
  局部对应变化 Side `[0.011,0.074]` px、Top `[-0.018,-0.011]` px；对照图已查看，
  尚不能确认尖端前进。局部块可能跟随管壁/背景，不用这个结果断言真实导丝未动。

最终机械臂 TCP `[-328.146218,169.931744,339.919362,3.140233,-0.007615,-1.594415]`。
本阶段远程机械臂 **7** 次（6 次轴往返、1 次左向引导）；今天累计远程路径动作
13 次，不含现场手动定位/早期原生校准；今天递丝发送尝试 **1** 次、回复 **0**、物理
推进仍未知。**保持后续机械臂运动及递丝停止，等待递丝器电源/网线/接收程序检查。**
恢复后先查网络及新状态/画面；未解决首次发送和导丝观察前不自动重发，也不继续把
磁铁带远。用户要求完整左端送丝仍有效，不以本次 10 mm 检查声称完成。

当前回放 **http://192.168.5.11:8768/**（PID **355202**）已替换为本段左向引导与
递丝观察：55+86=141 帧 / 14.1 s，明确标记中间间隔未录像，4 张前后原图；mp4
全解码及两段中文代表画面已检查，媒体 200、私有报告 404。仅固定媒体白名单。
实时 **http://192.168.5.11:8765/**。旧 axes_media 和原现场标记记录均保留。

## Earlier Today: Calibration And Initial Axis Checks (2026-09-30)

用户已回复“已现场完成校准，机械臂停止并切回 REMOTE”。10:22 只读回读确认
STOP / REMOTE、precision=1、sync=true，servo=false；不能继续把下节历史 ERROR
当成当前状态。现场操作改变了机械臂位置，新 TCP 的 Y 约 -3.79 mm，旧失败停止时
约 -119.30 mm。10:30 仅使能一次伺服，未再次清错、同步或校准。

本轮目录 `simulation_output/left_resume_20260930_100550/`：

- `onsite_recovered/report.json`、`current_after_onsite/`：现场恢复后的状态和原图。
- `axes_preview/`：三个 +10 mm 目标先逆解，最大关节差 X 1.419° / Y 1.116° /
  Z 1.606°。`servo_after_onsite/` 记录单次使能。
- `axes/`：10:31:03–10:31:42 执行 X、Y、Z 各 +10 mm 后返回起点，共 **6 次**
  `move_joint`，速度 1%，全部接受并到位。TCP 回读出程距离分别 9.99881、9.99974、
  10.00033 mm；最终回读与起点相同。局部监测相对同一起点：TCP 13 mm、总角
  0.5°、各关节 3°，相机失效/状态异常/超时/超范围会 stop；无错误或补发。
  全过程 380 帧，`axes_media/motion.mp4` 可完整解码。相机可见磁铁实际位移，
  不仅依据 ACK 或控制器坐标；用户尚未验收该录像。
- 图像差分（Side/Top 原图像素）：X `[-3,-6]/[-7,+1]`，Y `[+22,+1]/[-19,0]`，
  Z `[+1,-22]/[+1,-19]`。`axes/summary.json` 和 `axis_motion_review.png` 已复查，
  `viewed_not_accepted`。Top Z 模板匹配仅 0.553，背景与磁铁重叠；这是磁铁当前位置
  附近的整数像素局部响应，不是完整相机外参，也不能外推成整条血管的机器人路径。
- 最终 STOP / REMOTE、precision=1、servo/sync=true，TCP
  `[-514.878509,-3.790118,465.922930,3.138705,0.055285,-1.594439]`。
  TOOL3 偏置仍全零，TCP 是法兰，磁铁末端偏置未验证。

**今日递丝包仍为 0，完整左端送丝未完成。** 用户指定的几何仍为磁铁表面在尖端
上方 **30 mm**；当前位置与该工作位置尚未建立测量对应。当前相机和工具的局部
运动响应不能证明远距离靠近路径或整条血管的净空。下一步先获得今天的新尖端/目标
标记，再建立可核验的 30 mm 起始工作位及移动导丝观察，不能重放仿真或累计盲目递丝。

当前服务（仅媒体/标记，无运动接口）：

- 实时双相机 **http://192.168.5.11:8765/**。
- 今日三轴运动回放 **http://192.168.5.11:8768/**，PID **345341**，已替换旧启动/
  校准展示页。左右对比原图是起点与 **Y +10 mm 处**，不是最终停留位置；页面明确
  说明最后返回。旧校准媒体仍在 `media/`，旧服务信息另存历史文件。
- 今天固定定位页 **http://192.168.5.11:8769/S0K2_A6UKaJ-M6Sh0Ju4XMnWBygFoK1L/**，
  PID **346777**；10:37:27 原图在 `current_for_annotation/`，标记在 `annotation/`。
  用户已保存第 1 版两路标记；同网址后加 `review.png` 查看中文复核图。重启保留
  原网址和全部标记字节，昨天 Top 构图不可复用。

今天 10:39:32 用户标记：Side 尖端 `[1130.5,690.9]`、目标 `[768.7,663.5]`；
Top 尖端 `[1037.3,897.7]`、目标 `[1207.3,891.1]`。新 5 对被动图像局部对应
各 5/5，最大漂移 Side 0.187 px、Top 1.147 px；只说明静态外观对应，不是移动导丝
语义跟踪验证。`marked_review/` 已查看，`viewed_not_accepted`。

**当时等待现场建立起始工作位（已由后续回复解除）**：已请熟悉本机的操作员将磁铁朝向导丝的表面放在
当前尖端正上方约 30 mm，相机/血管不动，随后停止、切回 REMOTE；若现场无法就位，
先返回定位方案。这是建立可测量起点的外部步骤，不能把之前“校准完成”当作“30 mm
工作位已就绪”。现场操作期间远程运动和递丝均保持停止。当前悬空磁铁离尖端较远，
10 mm 局部图像响应不能核验整段靠近路线/真实间距。收到就位回复后先重新只读核验
状态和画面，再做工作区局部对应与增量引导，不要求重复校准或重放历史动作。

新 `tools/probe_elite_camera_axes.py` 默认只读 IK 预览，执行开关仅控制三轴各 10 mm
往返，不递丝、不归位、不校准、不自动恢复；4 项关键离线检查通过。默认命令不能
重跑已完成动作；后续需从当前实际状态规划。媒体页新增标签和裁切参数以正确展示
今日位置，HTTP 页面/图片 200、私有报告 404。总表 `session_result.json` 已更新。

## Earlier Today: Startup And Two Interrupted Calibrations (2026-09-30 10:17)

用户已说“继续左分支任务”，撤销昨天的暂停。磁引导几何按用户最新说明更新为：
**磁铁朝向导丝的表面在尖端上方 30 mm**（原话“磁铁表面到尖端 30mm 位于上分”，
按上方理解）；此前 20 mm 被后续明确值替代。这是操作员给定的工作目标，未测得
当前真实间距，也不代表已建立机器人坐标。

本轮输出 `simulation_output/left_resume_20260930_100550/`。10:05 两路相机和历史
服务仍可用，但 Top 已重新摆放且起初看不到磁铁，昨天 Top 像素标记已失效。
10:06 EC66 `.66:8055` 只读状态为 STOP / REMOTE、M473=1、servo=false、sync=false、
precision=0，计算关节全零、TCP `[816,33,-2,...]`；未同步的计算坐标不能用于运动。
报警历史为空，抱闸全未释放，电机反馈静止。用户随后再次确认今天的相机调整、
工具最远端 ≤500 mm、周围 ≥60 mm 净空、人员离开。10:10 最新原图复查两路均能
看到磁铁和血管，未见人员进入；不能将该局部净空扩展为整条血管轨迹净空。

- `startup/`：新工具 `tools/initialize_elite_http.py` 按厂家启动流程执行一次
  clearAlarm 释放启动抱闸、一次同步、一次上伺服，10:10:49 成功。每阶段单次发送、
  状态回读、相机新鲜度及电机变化监测，失败停止，无 SDK 自动重试循环。
  同步前不使用全零计算关节/TCP 作为物理位置；电机反馈处理 ±360° 表示跳变。
  本轮最大电机反馈变化 0.03202°，M473=0、servo/sync=true，precision 仍为 0。
  默认只读，显式执行还需要当天现场条件；4 项离线检查通过。
- `precision/`：10:11:16 一次原生编码器校准，true 仅为接受；约 0.4 s 后旧软件
  单关节 3° 阈值被 J5 3.75193° 触发，发送 stop，进入 ERROR / precision=0。
  采样 TCP 最大 3.81672 mm、总旋转 3.65971°、500 mm 工具点估计上界 35.74829 mm。
  报警历史 `[],[],[],[],[0-7000-C]`。不是自行判定编码器硬件坏。
- `precision_recovery/`：核验静止后，一次已知校准失败清错及一次恢复尝试。
  新 `--bounded-j5-recovery-from` 仅给 J5 5°，其余轴仍 3°；TCP 5 mm、姿态总角
  5°、工具点估计 50 mm 均相对**第一次校准原始位置**，不在重试时重置基准。
  10:15:53 再次停止：TCP 5.75562 mm / 5 mm，J4 3.52623° / 3°；同一采样总角
  5.14455°、工具点上界 50.63521 mm 也超过设定值。没有第三次自动重试。
  参数入口拒绝以一次恢复报告再串联恢复；默认校准限制不变。报错现列出超限轴。
  处理仅一个非空 7000-C、前方空历史槽的表示，拒绝未知非空报警混入；10 项校准
  离线检查通过，包括原始基准不重置和其他关节限制保留。
- `after_recovery_stop/`：10:17:11 三次回读六轴速度全零，ERROR / REMOTE，
  servo/sync=true、precision=0。最终 TCP `[-514.890854,-119.298941,465.920892,
  3.123007,0.093941,-1.595931]`。相对首次校准前 TCP 5.81475 mm、姿态 5.19142°、
  工具点估计上界 51.10294 mm，说明采样停止阈值不是硬件停止距离保证。

用户核对最新示教器全文：**“编码器标定失败，请重试或者手动移动每个轴完成标定”**。
已请求熟悉本机的现场操作员结合实际长杆扫掠空间，按示教器/厂家流程完成精确校准，
随后保持停止并切回 REMOTE，或说明现场无法完成。**当时等这一外部状态，期间保持
远程运动停止，不清错、不再次放大限制、不使用 SDK jog 代替现场步骤。** 用户当时报告
报警文字只是提供诊断信息，不能写成已经校准成功。恢复后需只读复查实际状态和新图。

截至 10:17 路径运动 **0**、递丝包 **0**；启动和两次校准包含真实硬件操作，不可写成完全
无动作。完整左分支仍未完成，30 mm 目标间距、当前磁铁 TCP、相机—机器人对应及
移动尖端追踪均未验证。`session_result.json` 保存工作目标、条件、次数和待办。
原纯媒体复核页（已被最新三轴回放替换），旧 PID **338043**：启动及两次校准
共 94 帧 / 9.4 s 拼接录像，明确标注间隔未录像；四张原图，固定媒体白名单，无硬件
接口、日志或目录发布。中文代表画面和原图已查看，`viewed_not_accepted`。
实时相机仍为 **http://192.168.5.11:8765/**；昨天定位页保留作历史，不能当作新画面。

## Historical Pause; Resume From Marked Target And Magnetic/Camera Alignment (2026-09-29)

用户最后明确说：**“先暂停吧，明天继续，提醒我进行到哪里了”**。任务暂停；
不得自动继续标定、运动或递丝。完整实机左端送丝仍未完成。明天用户恢复后先读取本节，
复查当前相机、控制器和现场位置；今天的图片、位姿与净空不能直接当成明天的现况。

本次已完成：用户认为 Side 更清楚，并在固定原图上标记尖端和目标，两路均有完整点。
`simulation_output/left_full_route_assessment_20260929/annotation/annotations.jsonl`
第 3 版（18:14:15）为最新；Side 尖端 `[1149.1,689.8]`、目标 `[755.5,670.1]`，
Top 尖端 `[736.9,905.4]`、目标 `[866.2,897.7]`。这些是用户选定的二维位置，
不是已标定机器人坐标、管腔中心线或已验证出口的三维位置。

18:16:46–18:16:52 被动读取 20 对原图；新工具
`tools/review_left_target_tracking.py` 复用已有 LK 探针，两路局部点均输出 20/20，
相对人工初值最大漂移 Side 0.227 px / Top 0.059 px。仅证明静止局部外观对应，
不证明运动时能追踪真实导丝；标记原图与这段采集间还有未观测时间间隔。
中文复核图已查看，`viewed_not_accepted`。结果在 `marked_passive_capture/`、
`marked_review/`；没有将像素距离换算成毫米或把尖端—目标直线当作血管路径。

18:17 控制器只读复查仍为 STOP / REMOTE、精确状态 1；活动工具 TOOL3 偏置全零，
当前 TCP 代表法兰，磁铁偏置未验证。18:20 只读求解 X/Y/Z 各 +10 mm 的候选点，
最大单关节差分别 1.384° / 1.146° / 1.604°；设想每轴到点后返回起点，共六段，
**只求逆解，没有执行**，也未证明整段扫掠空间或完成相机标定。
记录 `marked_controller/report.json`、`local_axis_plan.json`。
本次完整路线评估阶段机械臂动作 **0**、递丝 **0**；勿与 17:56/17:58 历史动作混合。

暂停前最后一个未回答的问题是：**磁铁在实际实验中应距导丝头多少毫米、位于血管
哪一侧，才能有效引导？若从未验证，明确记录未验证。** 用户的暂停回复不是参数答案。
后续还需建立当前相机/工具/机器人位置对应，验证运动中的尖端观察，才能规划和执行
分段协同送丝。不能靠放大原模型小向量、重放未配准仿真路径或累计盲目递丝完成。

定位页保留原网址 **http://192.168.5.11:8767/fNxhoGegadHiJkMAvy2f8KQHMkW7nrZ_/**，
当前 PID **187177**，仅固定原图、标记及复核 PNG；刷新会恢复第 3 版标记。
复核图在该网址后加 `review.png`。重启保留了全部标记字节和原 URL，图片 200、
报告与路径越界请求 404 已检查。实时相机 **http://192.168.5.11:8765/** 与历史
动作媒体 **http://192.168.5.11:8766/** 保持运行；这些网页没有运动控制接口。
断点总表 `assessment.json` 已标 `paused_by_user`。

## Full Left-End Delivery Requested; Initial Tip/Target And Registration Gaps (2026-09-29 18:06)

用户随后说“可以，但动的也太少了吧，接下来可以完成完整的导丝送到血管左端吗”，
将任务明确为到达左端，不能以再做一次固定 10 mm 诊断代替完整导航。
18:06:36 只读复查 STOP / REMOTE、精确状态 1，TCP 与上轮一致；新增机械臂动作 0、
递丝包 0。两路最新原图已查看：血管和磁铁可见，但代理不能可靠定位导丝头，也不能
从“左端”唯一确定实际出口。已向用户询问可见导丝头的相机/位置及目标出口。

输出 `simulation_output/left_full_route_assessment_20260929/`，含 `current/` 原图、
控制器快照及 `assessment.json`。已有场景 scale/alignment 工具用于历史仿真对齐，
未找到适用于当前重摆相机/血管的相机或血管到机器人变换；原仿真 demo 的配置明确为
诊断场景。旧 path1/path2 的当前距离检查保存在 assessment，不直接归位或重放。
当前磁铁 TCP 未核验，递丝一包的实际毫米数未知。原 60 mm 净空确认适用于此前局部
校准，不能当成整条血管路径净空证明。

完整送丝仍需：先确定可观察的真实导丝头和目标出口，建立当前场景路径/工具坐标
对应，再按可见推进分段执行，丢失导丝头、不推进或到达目标即停止。未声称可以靠
放大模型小向量或预设重复次数完成左端任务；等待定位信息期间不发送盲目递丝。
最新“可以、动得太少”保留为新的任务反馈，不自动改写前一轮具体争议记录。

## Onsite Feedback Disputes Latest Arm Motion; Media Available (2026-09-29)

用户对 17:56:53–17:56:58 的新轮次再次反馈“机械臂仍未见移动”。因此下面代理的
视觉判断不能当成已获现场确认的物理成功。保留原控制器、电机/编码器数据与前后
原图；`arm10/physical_review.json` 的 `arm_physical_execution_confirmed` 改为 null、
`physical_motion_disputed=true`，保留此前代理判断及来源。现场原话保存在
`arm10/onsite_feedback.json`；`session_result.json` 和本地 `index.html` 同步纠正。
反馈后没有再次发送校准、机械臂运动或递丝，递丝物理推进也没有本轮确认。

只读图像复核 `arm10/image_motion_diagnostic.json`：局部灰度模板匹配，Side 磁铁
区域估计位移 `[+9,+3]` px（NCC 0.867），窗口/支架两处固定背景均 `[0,0]` px；
Top 工具区域 `[-1,0]` px、NCC 0.718 较低，两处背景均 `[0,0]` px。
该证据支持 Side 局部变化，但没有像素到毫米的标定，不能消除现场反馈矛盾或证明
有效磁引导；不据此自动放大/重复动作。

已发布仅相机媒体的局域网对照页 **http://192.168.5.11:8766/**：6.4 s 双相机录像、
四张运动前后原图及 Side 白色磁铁局部切换。服务入口
`tools/serve_hardware_review_media.py`，PID **180401**，持久信息在本轮目录
`media_server.json`，没有硬件接口。实时相机仍为 **http://192.168.5.11:8765/**。
曾尝试发布整个诊断目录，被自动审核拒绝（超出图像链接请求，会暴露校准/状态
文件）；随后缩小为固定媒体白名单并获准。已验证照片/录像 200，报告、校准、
路径穿越请求 404；未暴露完整诊断目录。此发布问题已解决，无待批准动作。

当前结论：校准已恢复，机械臂命令与一次递丝已发出，但物理运动效果仍有争议，
未完成完整左分支；下一步需结合该录像核对现场所观察的末端与有效位移。

## Precision Restored; Initial Agent Visual Review And One Feed (2026-09-29 17:59)

本节替代下面 17:49 的未恢复状态。用户提供示教器全文“编码器标定失败，请重试或
手动移动每个轴完成标定”，随后明确确认：工具最远端距法兰不超过 500 mm、机械臂
与末端周围至少 60 mm 无障碍空间、人员离开，并允许按该范围重试。
授权/现场条件存于 `simulation_output/left_authorized_20260929/retry_readiness.json`。

`restore_elite_precision_http.py` 新增显式 `--onsite-clearance-60mm` 和
`--recover-confirmed-calibration-failure`：只对静止、REMOTE、精确状态 0 且最近报警
以 `[0-7000-C]` 开头的已核对失败清错一次；要求恢复 STOP 后才校准一次。
原默认阈值不变，现场确认模式保留 TCP 5 mm / 单关节 3°，姿态总旋转角和各 RPY
最多 5°，另按 500 mm 工具半径计算工具点位移上界，超过 50 mm 停止。
这些是现场范围内的诊断监测参数，**不是厂家给定的正常校准范围或硬件停止保证**。
增加停止后的录像/状态/报警查询。9 项相关离线检查通过。

17:54:57 清错一次成功，校准重试一次；17:55:00 精确状态 **1**、STOP / REMOTE、
伺服/同步正常，校准成功，无再次停止。采样最大 TCP 变化 0.067553 mm、总姿态旋转
2.23185°、单关节变化 2.26659°、500 mm 工具点位移上界 19.51144 mm。
结果 `calibration_retry/`，35 帧录像可解码、代表首末画面已查看。

随后完成一次分阶段实机协同：

- `arm10/`：17:56:53–17:56:58，一次 `move_joint`、1% 速度，最新左任务 PI05
  平移方向归一化到 10 mm；原模型仍为 hold，原平移范数 0.04930074 mm。
  目标 ΔXYZ `[-0.521250,+9.911148,-1.223702]` mm，保持姿态。
  TCP 回读 ΔXYZ `[-0.519070,+9.913754,-1.223229]` mm，范数 **10.002412 mm**，
  目标误差 0.003431 mm。J1/J6 电机反馈分别约 −1.13094° / −1.12616°，
  原始编码器对应 +4379 / +2919 counts；全轴配对数据保存在报告。
- Side 前后原图中磁铁相对固定背景有可见位移，Top 同样可见末端移动；代理视觉复核
  和电机/编码器变化共同支持本轮实际运动。`physical_review.json` 明确来源，不能
  将此代理复核写成现场用户确认，也不能将 10.002 mm 写成相机标定测量。
- 新增 `--arm-only` 以先完成机械臂录像/回读、再独立复核；8 项协同检查通过。
  64 帧双相机录像可解码，已转为 `arm10/motion.mp4` 便于查看。
  本轮控制台 planned 事件的 `manual_feed_once=true` 为旧日志字段错误；该进程实际
  递丝包 0，原日志保留；代码已修正后续事件字段，无重跑运动。
- `arm10/feed_once/`：17:58:39，在上述复核后，通过新工具
  `feed_after_verified_arm_http.py` 发送 forward 一包到 `.13:8888`，收到匹配 ok。
  发送前后机器人 STOP、精确状态 1，位置未漂移；没有再驱动机械臂。
  新工具要求机械臂物理动作复核、精确状态及位置一致，输出固定在该机械臂结果的
  `feed_once/`，目录已存在即拒绝，避免模糊回执导致重发；3 项离线检查通过。

本轮总计：校准尝试 2 次（首次软件停止、现场确认后重试成功），清错 1 次、路径
运动 1 次、递丝 1 包。当前授权恢复步骤已完成。`index.html`、`session_result.json`
汇总本轮；图像 `viewed_not_accepted`，递丝物理推进仍待本轮现场反馈。已向用户询问
机械臂/导丝观察结果，明确不会触发下一轮。**没有执行完整左分支轨迹**；仿真世界系
到当前实机场景配准、磁铁 TCP 和导丝头闭环引导仍未验证。

## Calibration Authorized And Attempted; Software Stop, Controller ERROR (2026-09-29 17:49)

用户明确授权“允许编码器校准，驱动机械臂和递丝”。此前审批缺少授权的问题已解决，
本次校准命令获准运行；不要继续把当前阻塞描述为缺少用户授权。
输出 `simulation_output/left_authorized_20260929/`：

- `precheck/`：17:47:02 只读复查 STOP / REMOTE、精确状态 0；两路画面已查看，
  机械臂/工具与血管可见、未见人员靠近。
- `calibration/`：17:47:35 发送一次 `calibrate_encoder_zero_position`，回复 true。
  约 0.28 s 后软件采样发现 RPY 最大变化 0.02360 rad 超过代码阈值 0.02 rad，
  TCP 位置变化仅 0.0253 mm；程序发送一次 stop，回复 true。没有重试。
  **是监测程序主动中止，尚不能判定该姿态变化是硬件故障或正常校准动作。**
- `post_stop/` 和 `post_stop_diagnostics.json`：17:47:56 / 17:48:58 复查
  `RobotState.ERROR`、精确状态和 M472 仍为 0；三次六轴电机速度均为 0，TCP 稳定。
  SDK 最近五条报警返回 `[0-7000-C],[0-E030-1],[0-E030-1],[0-E030-1],[0-E030-1]`；
  不是五项已证实的新故障，含义仍待现场示教器文字核对。
- 停止后与校准前相比，控制器 TCP 平移 0.029248 mm、最大 RPY 变化 1.39210°；
  J5 计算角变化 +1.42091°，电机位置反馈 +1.27932°，原始编码器变化 −3316 counts。
  有电机/编码器配对变化证据，但尚无本次现场可见物理运动确认。
- 新增路径运动命令 **0**、递丝包 **0**；校准未完成、左分支未执行。
  校准 AVI 共 6 帧，已全部解码检查；首末及停止后代表画面已查看，
  `review.json` 为 `viewed_not_accepted`。

已向用户询问示教器报警全文及本次是否看到小幅转动。等待该信息并核对校准正常运动
范围后再决定恢复；没有清错、放宽阈值或重新发送校准。官方启动流程要求等精确状态
恢复为 1，SDK 的 true 只表示接受校准，不代表校准成功。当前阻塞为设备 ERROR 和
校准中断，需要诊断，不是审批。原仿真路径的场景配准问题仍未解决。

## Left Simulation Route Prepared; Hardware Remains Unverified (2026-09-29)

用户回复“先忽略，继续左分支任务”，未明确授权下面被拒绝的编码器校准；未重试校准、
移除精确状态检查或发送新运动/递丝指令。继续完成离线轨迹准备：
`tools/prepare_sim_left_route.py` 从原演示的 248 个左分支样本提取仿真世界系 TCP，
按原路径弧长每 10 mm 取样，输出 29 点 / 28 段参考，原路径长 275.874479 mm，
最后一段弧长 5.874479 mm。这是取样间隔，不是实机单次运动至少 10 mm 的保证。

输出 `simulation_output/left_route_prepared_20260929/`：`left_sim_waypoints.csv`、
`report.json`、`left_route_preview.png`、`review.json`。端点和段长检查通过，中文图已
查看、无缺字或裁切，状态 `viewed_not_accepted`。工具没有硬件或递丝接口；
`executable_on_robot=false`，没有生成实机坐标或递丝时序。重采样未保留原始姿态与时序。
仍需恢复设备精确状态、验证可见物理动作，并完成当前场景配准和磁铁 TCP 核验，
才能将该参考转为可执行连续路径。未完成实机左分支。

## Continue Requested; Monitored Calibration Prepared, Automatic Review Rejected Execution (2026-09-29 17:35)

用户再次要求继续左分支。17:29 和 17:34 复查仍为 STOP / REMOTE、精确状态 0，
TCP 与前次一致，伺服/同步正常。双相机已固定于当前构图，末端和血管可见，所查看
画面中未见人员靠近。新增独立 `tools/restore_elite_precision_http.py`：厂家启动
流程的一次编码器精确校准，保留原运动入口的精确状态要求；不把校准并入普通动作。
默认只读，显式 `--execute-calibration` 才会调用一次 `calibrate_encoder_zero_position`。
校准可能引起关节运动并持久改变控制器校准状态。记录双相机录像和 TCP/电机/编码器
采样；监测 TCP 5 mm、RPY 0.02 rad、单关节 3° 的变化，异常/超时/相机失效停止，
不归位、不切换工具、不递丝、不重试。监测是采样检查，不是硬件级路径限制。

`tools/test_restore_elite_precision.py` 的 6 项离线检查通过：一次恢复、已精确不操作、
初始状态拒绝、回执异常停止、变化越界停止、超时/相机故障停止。
只读端到端预览通过，保存 8 帧双相机 AVI，画面已查看，`viewed_not_accepted`。
输出 `simulation_output/left_continue_20260929_172936/calibration_preview/`。
17:35 尝试执行该明确的恢复步骤时，**自动审批审核拒绝创建进程**，理由是用户只授权
继续左分支，尚未明确授权可能引起关节运动并持久改变校准状态的编码器零位校准。
因此没有发送校准、路径运动或递丝指令；执行目录未创建。不改用间接调用或移除检查。
代码、预览及必要检查已完成；下一步需用户明确授权这一次受监测校准，或由现场完成
厂商启动校准。该批准只覆盖恢复步骤，不代表左分支轨迹已配准或完成。

## Physical Arm Motion Disputed; IP Verified, Precision State Not Ready (2026-09-29 17:26)

用户要求像 `simulation_output/simulation_demo_20260929/` 视频一样执行连续左分支，
并明确反馈“机械臂肉眼完全没有移动”。因此下面 17:11 的 10.000398 mm 只可作为
控制器 TCP 回读差，不能记为独立验证的物理运动或有效磁引导；递丝现场确认仍有效。
用户再确认机械臂 IP 为 `192.168.5.66`，并核对示教器 J1≈−179.32°、J2≈−88.66°、
J6≈182.03° 与查询基本一致，排查转向硬件就绪/物理证据，不重复猜测目标 IP。

只读检查输出 `simulation_output/left_route_diagnosis_20260929_171840/`：

- 实际 socket peer 为 `192.168.5.66:8055`，本机源 `.11`，接口 `enp5s0`；
  型号 EC66，控制软件 V3.6.3。旧协同入口默认 IP 也为 `.66`。
- STOP / REMOTE、servo=true、sync=true、estop=0，六轴抱闸均返回已释放。
  电机输入端反馈与软件关节位置基本一致，采样时速度为零；没有动作前后的原始
  电机/编码器配对证据，不能用本次静止查询证明前次物理运动。
- `get_servo_precise_position_status()` 和 M472 均为 **0**；M473=0。
  [厂家启动流程](https://www.elibot.com/service/articles/list/193) 将 M472=0 定义为
  未校准模式、1 为精确模式。这是已确认的就绪问题，但尚未证明它就是“不动”的原因。
- 全局运行速度为 5。当前运行/示教工具号均为 3，工具 3 偏置全零，当前 TCP 为法兰
  基准；工具 0 的偏置为 `[1.553,4.472,350.837,-2.912152,0.106221,-0.529934]`。
  未更改工具号、偏置、速度、伺服、抱闸或校准；工具偏置差不能解释为已证实的无运动原因。
- 本次诊断新增运动命令 0、递丝包 0；相机可见磁铁工具及血管，Side 仍有人出现在场景内。

新增 `tools/inspect_elite_readonly.py`，只读、要求显式 IP、新输出目录；实际运行通过，
`ip_verified/report.json` 无查询错误。共享 `robot_ready()` 增加精确状态必须为 1 的
检查，不自动校准；7 项离线检查通过，覆盖精确状态 0/未知时拒绝、正常状态通过和原
协同顺序/限幅。协同报告增加目标 IP、真实 socket peer、控制器测量来源及独立物理
确认字段。当前精确状态下后续执行会在发送运动前拒绝。

仿真差异也已查清：视频用 `MuJoCoRoutePlanGuideExpert`，左分支 247 步、TCP 累计
275.874 mm、净位移 262.252 mm、最大关节范围约 37.99°；`TipGuidedWireEnv` 将
导丝头限制在预设路线附近，是简化诊断模型。原实机仅一次 PI05 方向的固定 10 mm
动作，未执行该连续路线。仿真世界系没有当前实机场景配准；旧 path1/path2 起点距
当前 TCP 441.283 mm，不能直接归位重放。已导出 `sim_left_reference.csv` 和
`route_comparison.json`，仅仿真参考，不能作为实机命令。

当前待办：现场核对并恢复编码器精确状态，固定相机且清空运动范围；用同一轮录像、
电机/编码器前后值与 TCP 共同核验真实动作，再解决磁铁 TCP、血管路径到机器人坐标的
对应关系后规划连续左分支。不得跳过当前问题循环归一化 PI05 小向量，也不把仿真轨迹
平移到当前 TCP 后直接发送。没有执行编码器校零或任何新运动。用户现场反馈与纠正
记录保存在诊断目录 `onsite_feedback.json`；图像仍为 `viewed_not_accepted`。

## Fixed 10 mm Command Cycle And Controller Readback (2026-09-29 17:11)

用户在只读预览后明确回复“已现场确认空间无障碍，人员已离开”，见
`simulation_output/real_left_10mm_20260929_165933/onsite_readiness.json`。
已用新观测执行一轮，输出该目录下 `execution/`，耗时 36.29 s。
17:11:08 新模型原始平移范数 0.038916 mm，固定距离目标 ΔXYZ
`[+0.783307,+9.750987,-2.074773]` mm（没有重放下文 17:07 的预览目标）；
IK 最大关节变化 1.104039°。1% 关节速度，17:11:13 到位，控制器回读增量
`[+0.781254,+9.751786,-2.073713]` mm，范数 **10.000398 mm**，
目标位置误差 0.002445 mm、最大姿态误差 0.000012162 rad。
机械臂命令 1 次，随后向 `192.168.5.13:8888` 发送 forward 1 包并收到匹配的 ok，
没有重发或清理错误。17:12:10 独立复核 STOP / REMOTE、TCP 无漂移，两相机正常更新。

用户随后确认“看到本轮导丝实际前进”，反馈保存在 `execution/onsite_feedback.json`；
物理推进确认来源为现场观察，距离未知。原始 `report.json` 保持运行时的
`feeder_physical_execution_confirmed=null`，后续确认见反馈文件及 `review.json`。
原始模型仍预测 hold，实际递丝来自用户指定诊断协同，不改写模型标签。
三阶段图像及中文对照图已查看，`viewed_not_accepted`。到位/递丝阶段画面出现人员
在设备附近调整相机，Top 被遮挡且模糊，不能用这些图像判定导丝路径或导航成功。
现场反馈只确认本轮实际前进，不代表图像或数据验收；本轮已结束，不自动触发下一轮。
后续动作前需重新检查现场和相机固定状态。

用户明确授权调整代码，要求机械臂单次运动至少 10 mm。协同入口新增
`--fixed-step-mm 10`：取最新左任务模型的非零平移方向，归一化后生成范数为
10 mm 的诊断目标，目标姿态不变。原始模型输出独立保留；固定距离来自用户，
不能将其描述为模型原生 10 mm 动作或左分支导航验证。此参数定义起终点 TCP
位移，底层仍为已验证的 `move_joint`，不是笛卡尔直线轨迹保证。
未指定新参数时保留原来的 gain=40 / cap=3 mm 行为；原执行器总上限仍为 10 mm。
固定距离模式的 IK 最大关节变化检查从原协同模式的 1° 调为 5°，仍保留检查。
关节速度保持 1%，位置到位容差 0.02 mm，实测位移须回读，不能保证恰好 10.000 mm。
继续保留新鲜帧、TCP 漂移、STOP/REMOTE/伺服/同步/急停状态、IK 和失败停止检查。
递丝仍在机械臂到位并取得新图后发送一次 forward；未标定每包实际长度，不能用
`value=10` 声称递丝 10 mm，不能把既往用户的推进确认扩展到新轮次。

`tools/test_left_coordination.py` 的 5 项离线检查（含多个子用例）通过：固定距离和
方向/姿态、旧模式保留、异常或越界目标拦截、机械臂/图像/递丝顺序、错误后不递丝及
无回执不重发。实机只读模型/IK 预览已完成，输出
`simulation_output/real_left_10mm_20260929_165933/preview/`，耗时 30.92 s。
17:07:36 机器人仍 STOP / REMOTE，伺服和同步正常，estop=0；原始平移范数
0.012242 mm，固定距离目标 ΔXYZ `[+6.404588,+1.740003,-7.480216]` mm，
等效方向缩放约 816.85，IK 最大关节变化 2.147067°，原始递丝意图 hold。
motion_command_attempts=0、feeder_packets=0、清理无错误；未运行新模式的实际动作。

17:07 两路新图已查看，血管可见，但 Top 构图较上一轮发生改变、机械臂末端在画面外，
Side 也未完整覆盖末端及其周围空间。当时等待当前扩大步长的现场路径确认，
后已收到上述现场确认并执行；不是缺少用户对代码调整或左分支任务的授权。该预览仅为审查证据，
执行时必须重新采图、推理和检查，不能直接重放其旧目标。图片为
`viewed_not_accepted`；实现、只读验证和一次 10 mm 实际动作均已完成。

## One Left Coordination Executed And Physically Confirmed (2026-09-29 16:49)

用户回复“已就绪，可以复查并执行”，16:48:47 已重新查看两路新图：血管重新入镜，
未见人员靠近机械臂；机器人仍 STOP / REMOTE，伺服和同步正常，estop=0。
按既定范围执行了一轮，输出
`simulation_output/real_left_coordination_ready_20260929_164847/execution/`。
总耗时 32.63 s，机械臂 1 条运动命令、递丝 1 包，无重发、无清理错误。

16:49:50 新左任务模型 ΔXYZ `[+0.004814,+0.017211,-0.003249]` mm，
×40 后目标增量 `[+0.192546,+0.688450,-0.129974]` mm，未触发 3 mm 上限。
16:49:51 以 1% 关节速度完成；位姿回读增量
`[+0.190577,+0.690726,-0.128020]` mm，范数 **0.727881 mm**（上一小步 0.073434 mm）。
到位位置误差 0.003588 mm、最大姿态误差 0.000011760 rad。
确认到位、停止且取得新图后，向 `192.168.5.13:8888` 发一包 forward，
收到 `ok {"command":"move","parameters":{"action":"forward","value":1}}`。

用户随后明确反馈“看到导丝实际前进”，保存为 `onsite_feedback.json`，
物理推进确认来源是现场用户观察，未量化推进距离。程序原始 `report.json` 中
`feeder_physical_execution_confirmed=null` 保留运行时语义；后续确认见反馈文件和
`review.json`，不能把用户观察伪装成编码器测量或视觉算法输出。
原始模型仍为 hold，这次递丝来源是用户指定协同联调；没有改变模型原始标签。
16:50:37 独立查询机器人仍 STOP、最终 TCP 无漂移，双相机持续更新。
三阶段图片与中文对照图已保存、查看，`viewed_not_accepted`；用户确认的是导丝
前进这一物理事件，不是全部图像/数据/导航结果验收。此一轮已完成，不自动触发下一轮，
也不表述为完成左分支导航。

## Larger Left Coordination Prepared; Onsite Adjustment In Progress (2026-09-29 16:45)

用户在上一小步完成后要求“继续执行左分支，让递丝装置和机械臂协同工作，动作幅度大点”。
已明确下一轮范围：左任务最新模型方向 ×40、范数最多 3 mm、关节速度 1%、
机械臂到位后递丝 forward 一步，先一轮。递丝是用户指定的诊断动作，独立记录
为人工指令，不把模型 hold 改写为模型 feed 或正式训练标签。
独立入口 `tools/execute_left_coordination_http.py` 已准备，保留旧的 10 mm 上限；
基于现有相机 HTTP、ModelProcess、限幅、IK、到位/失败停止助手和现场 UDP 适配器。
UDP 目标显式 `192.168.5.13:8888`，源 `192.168.5.11:37011`，无回执不重发。
5 项无硬件检查通过：正常到位只发一次/无 ACK 不重发、机械臂失败不递丝、
动作后相机失败不递丝、到位后位姿漂移不递丝、超过 3 mm 不运动或递丝。

16:43:08 预检机械臂 STOP / REMOTE、伺服同步正常；随后查看图像发现有人正在
机械臂附近调整支架和相机。16:45:37 复查仍在调整，Side 转向机械臂上方且不见血管，
Top 背景和构图也在变化。已要求现场固定两相机、血管入镜并离开运动范围后回复就绪。
输出 `simulation_output/real_left_coordination_20260929_164308/` 包含预检和两次图像、
`readiness.json`。协同执行尚未启动，motion_commands=0、feeder_packets=0；
用户任务授权有效，等待的是现场就绪信息，不应重复索要任务授权。

## One Left Model Step Executed (2026-09-29 16:39)

用户随后要求“照你的方法执行左分支”，已按此前明确的 ≤1 mm / 1% 速度 / 固定姿态 /
递丝保持范围执行一次。新增 `tools/execute_left_model_step_http.py`，从当前位姿执行，
不归位，不改变原执行器限幅；相机 HTTP 直播一直保留。动作前重新采图、读取位姿和
模型预测，拒绝非 hold、陈旧帧、位姿漂移及超过 1° 的 IK 关节跳变。5 项无硬件检查
通过：单次执行、递丝意图变化拦截、旧帧拦截、TCP 漂移拦截、SDK 拒绝后停止且不重发。

实际输出 `simulation_output/real_left_once_20260929_163532/execution/`：总 32.01 s，
16:39:18 下发 1 条运动命令且得到 true 回应；16:39:19 完成，状态 STOP / REMOTE。
新模型 ΔXYZ `[-0.004857, +0.073662, -0.005572]` mm，未放大、未触发限幅。
位姿回读 ΔXYZ `[-0.005295, +0.072962, -0.006416]` mm，范数 0.073434 mm；
目标位置误差 0.001181 mm、姿态最大误差 0.000004526 rad；递丝 0 包。
动作后两路图像时间晚于完成时间；清理无错误。16:39:51 独立只读复核仍 STOP，
TCP 与完成读数相同，直播正常。该结果证明一次动作执行链路，不证明左分支导航成功。
原图、`report.json`、`final_readback.json` 和前后对照 `comparison.png` 已保存。
对照图已查看，`viewed_not_accepted`；照片不能独立证明该微小位移或导丝推进。

## Requested Left Step Above Existing Limit (2026-09-29 16:31)

用户确认左分支，但将单次移动要求改为大于 10 mm。已授权该任务方向，
不应误记为尚未选择分支；本次未执行的原因是现有执行边界与路径证据不足。
`tools/run_real10_pi05_once.py` 的 `MAX_DEMO_STEP_MM=10.0` 及
`action_plan` 检查拒绝更大的单步上限。离线用 11 mm 参数检查，得到
`Demo translation norm limit must be in (0, 10] mm.`，未修改或绕过限幅，
未以多次小步替代用户要求的单次大步，也未回退执行旧的 1 mm 方案。

16:31:35 只读检查仍为 REMOTE / STOP、伺服开启、同步正常、estop=0，
TCP 与前次预测一致；双相机正常。输出
`simulation_output/real_left_large_step_20260929_163135/` 保存 `preflight.json`、
`decision.json` 和两张新图。图像已查看，`viewed_not_accepted`；
motion_commands=0、feeder_packets=0。相机图像未提供经过标定的碰撞净距证明，
现有 IK 结果仅覆盖原小步长；不能作为放大约 100 倍后路径可执行的证据。
更大步长须先明确具体目标并完成现场路径验证；当前直播继续运行。

## Live Model Prediction Preview (2026-09-29 16:25)

用户选择“模型验证：先展示模型动作预测，再确认执行”。新增独立只读入口
`tools/preview_real10_from_camera_http.py`，复用已有 ModelProcess、1 mm 动作限幅
和 IK 检查，从运行中的相机 HTTP 服务取图，不中断直播、不直接打开 USB。
使用现有 Real10 PI05 / state_only Piper 配对权重，不修改模型、训练或旧执行入口。
同一组实拍图和实测 TCP 分别预测 left / right；无递丝历史，沿用 validity=0，
JPEG 输入与当前视角未经训练分布一致性确认。递丝头只用状态，不看图像。

输出 `simulation_output/real_model_preview_20260929_1625/`：加载 29.94 s，
总耗时 31.29 s；拍摄于 16:24:52，图像到位姿读取结束帧龄 34/43 ms。
左任务 ΔXYZ `[-0.008407, +0.100205, -0.006598]` mm，范数 0.100773 mm；
右任务 `[-0.049355, +0.225339, -0.011361]` mm，范数 0.230960 mm。
均未触发 1 mm 限幅、旋转为零、逆解通过；均为 canonical intent=1 → hold，
hold 分类概率约 0.929（不是物理安全概率）。两个采样输出不构成任务因果性验证。

实机前后 REMOTE / STOP、伺服开启；TCP 平移读数变化 0 mm。
motion_commands=0、feeder_packets=0，清理无错误。中文预测图已查看，
`viewed_not_accepted`。实现和只读模型链路已验证；实际运动尚未获本轮确认。
后续候选范围：选定分支后单次平移 ≤1 mm、1% 关节速度、保持姿态且递丝保持，
需获取新观测并重新核查；若新模型输出改为递丝或设备状态改变，应停止重新展示。
不直接调用旧入口的默认长距离归位。命令见 `docs/commands.md`。

## Real Validation Preparation (2026-09-29 16:21)

用户要求进行实机验证，准备阶段询问此次范围（单步联调 / 模型预测后执行 / 仅递丝），
随后选择模型预测后确认执行，见上节。16:19:56 只读检查 Elite `192.168.5.66` 为 REMOTE / STOP，
servo=true、sync=true、estop=0；本会话未执行清报警、使能或状态切换。
两相机帧龄约 47/55 ms，无错误；递丝 `192.168.5.13` ping 成功。
前后语境中的 ERROR 是较早查询结果，不代表此次状态。

独立输出 `simulation_output/real_validation_20260929_161956/` 保存
`preflight.json`、两路当前图与 `single_step_plan.json`。候选最小联调方案为
当前 TCP 基坐标 Z +1 mm / 原姿态 / 1% 关节速度，再返回；只做了 IK，
最大关节变化 0.1603°，执行前必须重新读状态并计算。递丝候选为 forward、
backward 各一包，无旋转、不重发，实际每步长度未知。未采用旧采集起点归位。
这些仅是待选定范围的计划；本轮当前 motion_commands=0、feeder_packets=0，
不能记作动作验证通过。图像已查看，`viewed_not_accepted`；直播继续运行。

## Live Camera Preview And Feeder Address (2026-09-29)

用户确认递丝设备为 `192.168.5.13:8888`，使用 UDP JSON：
`{"command":"move","parameters":{"action":"forward","value":1}}`。
`forward` / `backward` 每包一步；`turn_left` / `turn_right` 按 `value` 度旋转。
现有协议实现一致，已更新本仓库适配器、探测入口、影子采集和两处实机执行入口的
默认地址。新 IP ping 成功且邻居可达；仅验证局域网连通，未发送 UDP 动作包，
未验证设备执行或回执。下方旧 IP 检查保留为历史记录。

新增 `tools/serve_camera_preview.py`，提供只读双相机网页和最新 JPEG。
16:10:28 在宿主机启动，PID `123582`，`http://192.168.5.11:8765/`；
`/side.jpg`、`/top.jpg` 为最新原图，`/status.json` 为帧号、时间和错误状态。
相机输出 1920×1080、15 FPS，网页每次完成请求后约 400 ms 刷新；两路使用各自
主机到达时间，非硬件同步。超过 2 秒的旧帧返回错误，网页标记未更新。
无机器人连接或递丝接口。服务持续占用双相机；后续采集前需先停止此预览。
输出 `simulation_output/camera_live_20260929/`，日志为同名 `.log`。
HTTP 检查两图均正常解码且帧号、时间持续增加，见 `http_verification.json`。
初次 HTTP 图像已查看；随后 Side 出现 USB 断连，旧帧被接口拒绝，Top 仍正常。
增加独立相机采集线程的断连重试（每次重新预热 45 帧），停止旧服务后重启；
当前 PID 为 `126099`，输出 `simulation_output/camera_live_20260929_reconnect/`，
网址不变。16:16:12 再次验证两路 JPEG 及时间、帧号持续更新，无相机错误；
自动重试逻辑未通过人为拔插另行验证。新图已查看：Side 已能看清血管模型，
Top 为绿色背景的侧向血管视图。`viewed_not_accepted`，用户仍在调整位置。
运行和停止方法见 `docs/commands.md`。

## Read-Only Hardware / Camera Check (2026-09-29)

用户要求先确定硬件准备情况并查看相机图像以调整位置；未授权本轮运动、使能或递丝。
宿主机只读检查识别两台 USB3.2 D435：`317222071938`、`317222072584`。
本会话沙箱内无 `/dev/video*` 或 NVIDIA 节点，但宿主机实际 RTX4090 可用；不要将
沙箱设备不可见误写成宿主机驱动故障。

15:56:59 Elite `192.168.5.66` SDK 查询：REMOTE、ERROR、servo=false、sync=true。
未清报警、使能、求 IK 或运动。递丝目的 `192.168.5.10` ping 无响应、邻居表 FAILED；
未发 UDP 控制包，物理状态与当前 IP 待现场核对，不能据此断言设备故障。

两相机被 PID 121874 的另一只读审核会话占用，其输出为
`simulation_output/real10_reasoning_live_readonly_20260929_v3`。先保留该会话，
展示并另存其 15:51:48 原图：Side 严重模糊；Top 是桌面侧向全景、血管靠下，
不是可直接采用的俯视构图。快照和独立查询记录保存在
`simulation_output/hardware_camera_check_20260929/`，图像时间不等于状态查询时间。
图像已查看，`viewed_not_accepted`。

新增 `tools/capture_camera_preview.py` 仅打开双相机 RGB 流，60 帧预热后保存
1920×1080 原图、带序列号/时间的拼图和报告，然后释放相机。没有机器人、递丝或
模型导入。用户随后明确要求接管两个相机；核对 PID 121874 的命令身份后发 SIGINT，
该只读进程正常退出，`fuser` 确认无占用。拍照入口于 16:01:38 取得两路新图，
1920×1080、60 帧预热、host 到达时间差约 40.9 ms，完成后释放相机，无清理错误。
新图在 `simulation_output/camera_preview_20260929_takeover/`；Side 仍严重模糊，
Top 已有绿色背景但仍为桌面侧向构图。拼图已查看，`viewed_not_accepted`，
等待用户调整位置。命令见 `docs/commands.md`；这不构成实机运动准备完成。

## Local Simulation Demo (2026-09-29)

用户选择先运行“机器人与导丝运动”的仿真演示。当前 Linux 工作副本没有
`simulation_output/robot_scene_mvp/scene_config.json`，且会话没有 NVIDIA 设备。
新增 `tools/run_simulation_demo.py`，复用 `TipGuidedWireEnv` 和
`MuJoCoRoutePlanGuideExpert`，通过 EGL/Mesa 软件渲染生成双画面 MP4、起止帧、
HTML 播放页及逐步 JSONL。未连接硬件、加载 PI05 或启动训练。

`simulation/scene_configs/mujoco_scene_demo_v1.json` 是独立诊断配置：机器人基座
沿用环境默认值，血管比例 0.077 来自已有命令文档，关节初值来自保存的仿真记录。
它不是丢失标定文件的恢复，也不是实机标定结果；规则专家结果不能作为 VLA 指标。
原始路线、机器人资源和模型均保留。每次输出使用新目录，拒绝覆盖已有结果。

环境为项目 `.venv`，继承本机 `project2026-pi` 的软件包；新增依赖只安装在 `.venv`。
已验证 MuJoCo 3.8.1、Gymnasium 1.3.0、trimesh 5.1.0、yourdfpy 0.0.60 可加载场景。
首次短跑因专家构造参数不匹配失败，记录保存在 `simulation_demo_check_20260929`；
修正后 `simulation_demo_check_20260929_v2` 完成 8 步、导丝头移动约 16.9 mm，
代表图已查看，`viewed_not_accepted`。完整左右分支演示输出在
`simulation_output/simulation_demo_20260929/`：耗时 138.64 秒，左右分支分别在
247/246 步触发现有仿真成功条件，距目标仍为 40.8/42.1 mm（该环境的终止容差），
不表示准确到达真实目标。导丝头净位移 253.4/254.8 mm，Elite 最大关节变化
0.663/0.738 rad。126 帧 H.264 视频（1280×552、15 FPS、8.4 秒）全部解码通过，
495 条状态记录与步数一致；起止拼图已查看，中文与两视图可见。
运行记录为 `report.json`；后续视觉核查为 `review.json`，状态
`viewed_not_accepted`，用户视觉接受待定。实现与运行核查完成。
这是独立演示，不改变原正式数据验收或真实采集结论。复现命令见 `docs/commands.md`。

## Project Priority

The project priority is still:

```text
simulation realism and real-system correspondence > BC rollout success
```

Treat all current VLA/OpenPI/PI05 results as algorithm feasibility only. They
are useful for checking that the interface can train, but they are not
real-system validation.

The current policy-facing target contract is:

```text
observation + task instruction -> Elite TCP delta + Piper discrete intent
```

Execution remains controller-owned:

- Elite TCP delta -> IK -> joint execution.
- `piper_intent_id` remains the compatibility name for guidewire feed intent;
  current real execution uses the replacement feed device.
- Controller owns timing, cooldown, bounds, safety checks, and executed-command
  logs.

The 32D `action_32` tensor is only a framework compatibility surface:

```text
action_32 dims 0:6  = Elite TCP delta
action_32 dims 6:9  = Piper intent one-hot compatibility slice
action_32 dims 9:32 = padding
```

Do not treat the Piper one-hot compatibility slice as the final preferred
action head. Algorithm evidence for that decision lives in
`docs/algorithm-track-handoff.md`.

## Machine Split

Use the Windows workstation for simulation, export, data inspection, docs, and
small smoke checks:

```text
local project root: D:\PycharmProjects\project_2026
local Python: .\.venv\Scripts\python.exe
```

The Ubuntu 4090 is the algorithm track's LeRobot/OpenPI execution host. The data
track may transfer an accepted export there when the two tracks coordinate, but
should not run or diagnose model training from this handoff:

```text
remote host: cn-hk-bgp-4.ofalias.net
remote ssh port: 27455
remote user: zsw
remote project root: /home/zsw/project_2026
remote conda env: project2026-pi
```

SSH from Windows:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 27455 zsw@cn-hk-bgp-4.ofalias.net
```

Copy one file to the 4090:

```powershell
scp -P 27455 -i $env:USERPROFILE\.ssh\id_ed25519 `
  path\to\local_file.py `
  zsw@cn-hk-bgp-4.ofalias.net:/home/zsw/project_2026/path/to/local_file.py
```

Copy one remote output back:

```powershell
scp -P 27455 -i $env:USERPROFILE\.ssh\id_ed25519 `
  zsw@cn-hk-bgp-4.ofalias.net:/home/zsw/project_2026/path/to/output.json `
  path\to\local_output.json
```

Current algorithm execution commands belong in
`docs/algorithm-track-commands.md`.

## Algorithm Interface Needed By Data Track

Detailed model status belongs in `docs/algorithm-track-handoff.md`. The data
track only needs to preserve the agreed export boundary:

- Keep exporting explicit `elite_tcp_delta_6d` and `piper_intent_id`.
- Do not hide Piper intent inside a continuous action-only interpretation.
- Keep side/top images, state, task, and estimator/tactile provenance explicit.
- Keep `action_32` only as an optional framework compatibility tensor.
- Report interface or provenance problems to both tracks instead of changing
  model targets inside the data pipeline.

## Simulation Data Line

Current simulation mainline:

```text
simulation/mujoco_guided_wire_env.py
simulation.collect_formal_tip_line_guidance
```

Current guidewire assumption:

```text
tip-centric / hard-elastic guidewire with continuous line-shaped rendering
```

Current formal start range:

```text
--start-fraction-min 0.42
--start-fraction-max 0.54
```

Current accepted simulation direction:

- MuJoCo-first, not Isaac-first.
- Elite magnetic target should be front-up relative to the guidewire tip or
  registered route point, not directly overhead.
- Guidewire visual route should begin near the feeder/Piper outlet and pass
  through the vessel entry, including the S-bend-like entrance geometry.
- Background, lighting, camera visibility, line thickness, and blur/reflection
  should be compared against real side/top captures.
- Formal data must not depend on hidden MuJoCo truth unless a matching real
  observable path exists.

Important existing artifacts:

```text
simulation_output/formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1
simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1
simulation_output/lerobot_project2026_frontup_step024_v1
docs/_formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1_audit/
```

Important simulation tools:

```text
tools/audit_elite_frontup_guidance.py
tools/audit_pi_style_dataset.py
tools/prepare_pi_style_training_pack.py
tools/prepare_openpi_compat_pack.py
tools/prepare_openpi_temporal_view.py
tools/validate_openpi_compat_pack.py
tools/export_openpi_compat_to_lerobot.py
tools/audit_openpi_tactile_signal.py
tools/relabel_openpi_tactile_threshold.py
tools/build_contact_supervision_sidecar.py
tools/train_contact_estimator_oof.py
```

Current simulation data tasks:

1. Keep improving correspondence to real captures:
   - guidewire entry geometry;
   - camera domain;
   - Piper feed event scale;
   - Elite motion scale;
   - observable contact/tip estimator fields.
2. Keep tactile/contact fields explicit and provenance-aware.
3. Do not use exact MuJoCo contact/wall/tip truth as policy input unless it is
   only a diagnostic and clearly marked as such.
4. Before new formal synthetic data is accepted, run formal/provenance audits.
5. Use BC or PI-style training only as an interface regression check after data
   changes.

Recommended simulation acceptance checks:

```text
formal validity passes
complete side/top images
front-up Elite target audit positive
reasonable tip-to-magnet coupling
reasonable wall/contact visual review
Piper intent distribution is not degenerate
observation provenance contains no hidden oracle-only fields
```

### Current Temporal Test View

The first 4936-record algorithm export passed schema checks but was too
temporally redundant for a clean learning comparison. The data-track follow-up
created a no-relabel, uniform-stride-5 view:

```text
simulation_output/openpi_compat_pack_frontup_step024_temporal_stride5_v1
docs/simulation-training-data-temporal-audit-20260717.md
```

Current evidence:

```text
1005 samples / 40 complete episodes
34 train episodes / 6 validation episodes, task-stratified
previous Piper label accuracy: 90.36% -> 51.09%
previous Elite translation MAE: 0.2517 -> 0.9601
source Piper transitions retained: 472 / 472
complete image validation: 2010 checked / 0 missing
image references: Windows/Linux portable forward-slash form
self-contained transfer size: 83.48 MB
```

This view is accepted for a bounded Elite-translation plus Piper hold/feed
algorithm test. It does not solve visual-domain diversity, all-zero Elite
rotation, missing Piper retract, missing positive contact examples, or the
teacher-student observability decision. Do not use it for a tactile-benefit or
three-class-control claim.

### Contact Estimator Supervision Layer

The current formal dataset now has a separate offline-only contact supervision
sidecar:

```text
simulation_output/contact_supervision_frontup_step024_v1
docs/contact-supervision-sidecar-audit-20260718.md
```

It contains 4936 image-pair references and 411 diagnostic MuJoCo contact
positives across five whole-episode folds. All 9872 side/top references pass.
Exact contact stays under `diagnostic_target` with
`policy_input_allowed=false`; it is not exported into `state_32`.

The old RGB edge-distance estimator has zero recall/F1 against the sidecar.
The replacement RGB-local estimator completed full five-fold, whole-episode
OOF evaluation on all 4936 records:

```text
output: simulation_output/contact_estimator_oof_full_v1
RGB-local AP / F1: 0.99138 / 0.95848
precision / recall: 0.93519 / 0.98297
tip-position-only AP / F1: 0.82569 / 0.68216
left AP / F1: 0.99308 / 0.96732
right AP / F1: 0.98365 / 0.87179
all five fold AP values: 0.97521-0.99972
```

All 59 truth contact runs were detected; onset delay median/P95/max was
`0/0/1` sampled frames. There were 35 frame errors, 68.6% within one frame of
a truth transition. This passes the simulation estimator feasibility gate and
allows only a separate diagnostic estimator-driven contact-probe expert. It
does not make the source data suitable for a tactile-benefit claim, because the
source expert still lacks a clean contact-conditioned action change.

Real OOD audit output:

```text
simulation_output/contact_estimator_real_pilot_ood_v1
300 failed-pilot records / 5-model ensemble
predicted positives at 0.5: 0
probability P95 / max: 4.16e-43 / 1.26e-33
real contact truth: unavailable
policy_input_allowed: false
```

This is saturated negative domain-transfer behavior, not real accuracy
evidence. The real pilot remains calibration-only. Do not publish an all-zero
real `estimated_contact_flag`; keep it invalid until a manually annotated real
calibration set supports detector validation and recalibration.

### Diagnostic Contact-Probe Collector

A separate estimator-driven probe collector now exists:

```text
simulation/collect_estimated_contact_probe.py
docs/estimated-contact-probe-audit-20260718.md
```

It preserves explicit `elite_tcp_delta_6d` and `piper_intent_id`, keeps Piper
on the original feed/hold schedule, and only removes the Elite component toward
the planned wall side after an estimated-contact confirmation delay. It rejects
`--formal-data` and does not modify the mainline route-plan expert.

The left `2.0x`-radius smoke generated 11 exact-contact steps and 45 Elite
response steps, but the source-trained estimator had `TP/FP/FN = 0/41/11` on
that new trajectory. Initial right `n1` smokes produced no exact contact. A
dedicated geometry audit then found a usable right `n2/-1` route:

```text
start 0.42: radius fraction 0.95, 27/60 contact steps
start 0.48: radius fraction 1.10, 24/60 contact steps
start 0.54: radius fraction 0.95, 5/60 contact steps
```

The actual right dual-camera collector confirmed 27 exact-contact steps in 100
steps at `0.42/n2/-1/0.95`, but estimator recall was still zero (`FN=27`). The
top view was occluded during a reviewed contact frame. Therefore the collector
is currently useful only for generating additional offline contact
supervision. Its estimator fields are marked
`policy_input_allowed=false`, and the dataset must not yet be handed to the
algorithm track as contact-conditioned expert data.

The first long left supervision bucket is complete:

```text
simulation_output/probe_left_start042_v1
3 episodes / 540 samples / 0 missing images
exact contact: 153
estimator TP/FP/FN per episode: 4/40/47, 4/41/47, 4/41/47
```

This confirms the collector can produce dense offline contact supervision, but
also confirms that augmented estimator OOF retraining is required before any
policy-facing export.

The right 0.42 bucket is complete as well:

```text
simulation_output/probe_right_start042_v1
3 episodes / 540 samples / 0 missing images
exact contact: 81
estimator TP/FP/FN per episode: 0/0/27 for all three episodes
```

It is a hard-positive supervision bucket only; the current RGB estimator misses
all of its contact frames.

The right 0.48 bucket is complete:

```text
simulation_output/probe_right_start048_v1
3 episodes / 540 samples / 0 missing images
exact contact: 153
estimator TP/FP/FN per episode: 0/0/51 for all three episodes
```

This provides a denser right-side positive segment, but remains offline
supervision only.

The right 0.54 bucket is complete:

```text
simulation_output/probe_right_start054_v1
3 episodes / 540 samples / 0 missing images
exact contact: 15
estimator TP/FP/FN per episode: 0/0/5 for all three episodes
```

All four planned buckets are now ready for merge and augmented OOF training.

The merge and sidecar stages are complete:

```text
simulation_output/probe_supervision_merged_v1/manifest.json
simulation_output/contact_supervision_probe_augmented_v1/manifest.json
52 episodes / 7096 samples / 14192 checked images / 0 missing
exact-contact positives: 813 (left 530, right 283)
```

The next required artifact is `contact_estimator_oof_probe_augmented_v1`; keep
all estimator fields policy-disabled until its probe-held-out metrics pass.

The augmented OOF artifact now exists, but promotion is rejected:

```text
overall RGB AP/F1: 0.9917/0.9269
right 0.54 AP/F1: 0.5377/0.2500
right 0.54 precision/recall: 0.1429/1.0000
augmented real pilot positives: 0/300
```

The right 0.54 errors are temporally structured: truth is steps `44-48`, while
the model predicts `19-52` plus step `54` in every episode. Do not duplicate
that bucket or silently change the threshold. Define a separate risk flag or a
boundary-aware temporal estimator before another OOF/OOD run. Do not set
`policy_input_allowed=true` from the aggregate score.

The temporal audit is complete at
`simulation_output/contact_estimator_oof_probe_augmented_v1/temporal_risk_audit.json`.
It confirms right 0.54 behaves as an early risk warning (median lead 25 frames,
three pure risk runs), while the other probe buckets align with exact contact.
The next interface decision is whether to add a separate
`estimated_contact_risk_flag`; do not overload `estimated_contact_flag`.

The policy-safe diagnostic pack is now materialized at:

```text
simulation_output/contact_risk_diagnostic_pack_v1/manifest.json
simulation_output/contact_risk_diagnostic_pack_v1/samples.jsonl
simulation_output/contact_risk_diagnostic_pack_v1/diagnostic_targets.jsonl
```

It contains 7096 samples from 52 complete episodes, 14192 checked side/top
image references with no missing files, and 813 exact-contact positives in the
separate diagnostic target file. `samples.jsonl` contains only observable
state, images, the optional early-warning risk flag/probability, and the
explicit `elite_tcp_delta_6d + piper_intent_id` target. Legacy feed/hold labels
are normalized to `retract=0`, `hold=1`, `feed=2`; no action semantics changed.
Exact contact, wall distance, and risk conditioning all remain
`policy_input_allowed=false`. This pack is suitable for interface/diagnostic
inspection only, not for a tactile-benefit or real-system readiness claim.

## Real Collection Line

Current real data entrypoint:

```text
data/collect/collect_real_shadow_pilot.py
```

Current onsite execution environment, confirmed 2026-07-21:

```text
SSH: zsw@192.168.5.11
real collection root: /home/zsw/PycharmProjects/real_collection
real collection Python: /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python
algorithm/LeRobot root: /home/zsw/project_2026
```

Run real collection from `real_collection`, whose interpreter contains the
Elite SDK. Keep `/home/zsw/project_2026` for the LeRobot/algorithm environment.
Synchronize the project-facing collector and hardware adapter into both roots
when those files change; do not assume the two interpreters are interchangeable.

The onsite operator runs collection directly on the lab computer. SSH is only
for file inspection and synchronization, not for keyboard control or preview.
The current linked-pilot workflow is fully manual: `n` advances Elite by one
preset trajectory point and `f` sends one forward feeder event. These actions
are independently timed by the operator; do not enable
`--piper-feed-on-elite-path-step`, Elite auto playback, or periodic auto feed.
The feeder backend is forward-only by default and blocks retract keys.
The linked pilot uses `--start-recording-on-first-action`: post-approach preview
frames are not saved until the first valid operator action, and that action is
preserved as sample `0`.

The checked `path1/path2` files do not match the latest observed Elite pose:
their first point is about `346 mm` away and their nearest checked point remains
more than `234 mm` away. Repeated collection now uses the explicit
`--elite-path-approach-start` startup phase: Elite moves directly to the first
selected path point before recording begins, the reset motion is excluded from
training records, and the first operator `n` command advances to the following
point. The initial pose/distance and final approach error remain logged. Without
this option, execution still fail-closes above
`--elite-path-max-start-distance-mm` (default `30 mm`).

The first automatic approach attempt
`real_pilot_20260721_180414_left_s_bend_manual_linked` was stopped before the
first path point because the collector used `move_joint` for the long reset,
which produced a twisted joint-space TCP path. It contains only a manifest and
is not data. The corrected startup primitive is Elite `move_line` with explicit
Cartesian speed semantics (`speed_type=0`); subsequent short trajectory steps
retain the inherited `move_joint` behavior.

Current real validation/render tools:

```text
tools/validate_real_shadow_pilot.py
tools/render_real_shadow_video.py
```

Known real system details:

```text
Elite IP used onsite: 192.168.5.66
Piper interface: can0
Previously observed camera IDs: side=10, top=4
Camera IDs can change after unplug/replug
Camera warmup should be about 20-30 frames
```

The current camera update binds side/top RealSense devices by SDK serial
`317222072584` / `317222071938`, requests `1920x1080@15` color plus
`1280x720@15` native depth, aligns depth to color, and stores raw `uint16` depth
PNG paths in optional `side_depth` / `top_depth` record fields. Runtime stream
profiles, intrinsics, depth-to-color extrinsics, and depth scale belong in the
manifest. Depth through the transparent vessel is auxiliary calibration data,
not an exact contact label.

Do not pass the udev `ID_SERIAL_SHORT` values `318123025778` / `318123027596`
to `pyrealsense2.config.enable_device`; those identify the UVC shell rather than
the RealSense SDK devices and produce `RuntimeError: No device connected`.

The corrected dual-camera RGB-D smoke is accepted at
`real_diag_20260721_192500_dual_rgbd_1080p_warmup100`: 100 unique warmup frames
followed by two records, side/top color `1920x1080`, aligned side/top depth
`1920x1080 uint16`, all four image files readable, and zero validator
errors/warnings. Background latest-frame capture reduced the side/top
host-arrival timestamp delta from the serial-read smoke's `57.18 ms` to
`25.75 ms`. After the real warmup, nonzero aligned-depth coverage was about
`81.3%` for side and `73.9-74.3%` for top. This validates the collection
interface and startup stabilization, not transparent-vessel depth accuracy or
contact inference.

Known real hardware status:

- The JSONL/image/pose recording path can produce synchronized records when the
  hardware behaves.
- Physical guidewire delivery remains the main blocker.
- Removing the physical S-bend helped, but did not fully solve delivery.
- The UDP feeder protocol works. After an onsite mechanical adjustment on
  2026-07-21, the replacement feed device completed a basic usable feed test.
  Stable S-bend delivery, repeatability, and synchronized capture are not yet
  verified, so it remains a diagnostic candidate rather than accepted expert
  collection hardware.
- The collector's Python `socket.sendto` path returned without error onsite but
  did not move the device, while the same JSON sent through Bash `/dev/udp`
  worked. Use the explicit `--feeder-udp-transport bash_dev_udp` diagnostic
  mode until the Python-socket discrepancy is resolved.
- The exact Bash `printf` transport completed a physically observed single-step
  movement in `real_diag_20260721_s_bend_feeder_printf_single_004`. Its 40
  records validated with zero errors/warnings; the executed log contains one
  `feed_feeder_bash_dev_udp_sent`, side/top sync P95 is 4.07 ms, and
  image/action sync P95 is 14.20 ms. This closes the single-step synchronized
  diagnostic gate, not the repeatability or expert-data gate.
- The first five-step repeatability run
  `real_diag_20260721_s_bend_feeder_printf_repeat5_005` validated 60 records
  with five `feed_feeder_bash_dev_udp_sent` events and no errors/warnings. All
  five commands produced physical forward movement, but one movement was
  visibly shorter. Side/top sync P95 was 30.33 ms and image/action sync P95 was
  33.99 ms. Repeated two-second-period trials ruled out command cooldown as the
  main cause: some initializations delivered normally, while others produced
  almost no effective progress. The current onsite interpretation is subtle
  initial guidewire-state variation causing local impingement in the vessel
  model, not proven feeder-output variability. The nominal `60 mm` summary is
  command-derived and is not measured displacement. After the substantial
  onsite device adjustment, the old `12 mm/step`, `11-13 mm` range, and all
  derived `60 mm` values are invalidated. New captures must leave insertion
  millimeters null. Fixed millimeters per feed are not a valid calibration
  target for the curved, initialization-sensitive vessel setup; physical
  progress must remain image/observation derived.
- Current July 2026 real captures are calibration/hardware-diagnostic data
  unless explicitly reclassified as clean demonstrations.
- `real_pilot_20260721_192749_left_s_bend_rgbd` is the first physically
  successful S-bend linked feeder-plus-Elite episode: 350 complete RGB-D
  records and nine executed feeder events. It remains a physical-success and
  visual-reference candidate rather than accepted training data because the
  shared Elite SDK connection blocked pose polling during `move_joint`;
  `148/350` records (`42.3%`) used stale poses and image-to-pose P95 was
  `1.76 s`. The next capture must use
  `--elite-path-separate-connection` and pass the pose-sync audit before more
  successful episodes are collected.
- `real_pilot_20260721_200628_left_s_bend_rgbd` is the first clean successful
  candidate under the corrected connection mode: 310 complete `1920x1080`
  RGB-D records, eight operator-confirmed feeder events, zero stale Elite
  poses, image/pose P95 `81.23 ms`, and image/action P95 `87.92 ms`. The
  validator found zero errors and five isolated synchronization warnings.
  Retain it for detailed visual/action audit before algorithm handoff.
- The immediately following `200813` and `201018` episodes were physically
  unsuccessful after seven and two logged feeder events, respectively, while
  their data interfaces and Elite pose synchronization remained valid. The
  initial onsite hypothesis was feeder thermal degradation. Later inspection
  found that the hot-melt-adhesive fixture had loosened after heating and could
  no longer clamp the guidewire consistently. Stop the current dynamic
  collection session rather than chase another success under changing hardware
  state; future sessions should begin cold and record powered-on/cooldown time.
- The final right-route reference is
  `real_pilot_20260721_203449_right_s_bend_rgbd`: 183 complete RGB-D records,
  zero stale Elite poses, and final Elite path index `16/19`. The operator
  confirmed that the guidewire passed the bifurcation before the last several
  feed commands stopped producing physical motion. Treat it as a valuable
  right-route prefix plus actuator-failure suffix, not a complete expert
  success. The full session classification is in
  `docs/real-collection-onsite-audit-20260721.md`.

Do first onsite:

1. Verify camera IDs and warmup.
2. Verify image synchronization.
3. Verify Elite connection and servo state.
4. Verify CAN/Piper bring-up before assuming Python-side failure.
5. Test guidewire physical delivery before collecting long data.
6. Stop early if the guidewire mechanically cannot pass.

Useful camera probe pattern:

```bash
python - <<'PY'
import cv2
for i in range(12):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    print(i, ok, None if frame is None else frame.shape)
    cap.release()
PY
```

Common CAN/Piper issue:

```text
can0 may be STOPPED, missing, or ERROR-ACTIVE after reboot/replug.
Bring up can0 before running collection with Piper control.
```

Use existing command notes in `docs/commands.md` for current real collection
templates. After every real collection, run validation immediately and render a
quick video if the dataset is intended for review.

Real dataset format:

```text
records.jsonl
frames/side/*.png
frames/top/*.png
manifest or summary json
Elite TCP pose/state
Piper state/action label
timestamps and timing diagnostics
```

Real data interpretation:

- `branchs/` is senior-provided real image and Elite/magnetic-arm motion
  reference data.
- Do not use `branchs/` as guidewire centerline or guidewire trajectory truth.
- Use current onsite captures mainly for calibration/alignment and failure
  analysis until the physical delivery path is reliable.

## Data Interface Checklist

For both simulation and real data, the preferred VLA-facing sample should
eventually expose:

```text
observation:
  side image
  top image
  Elite TCP pose/state
  Piper insertion/state/action history
  task / branch instruction
  image-derived tactile/contact fields with confidence/provenance

target:
  elite_tcp_delta_6d
  piper_intent_id

optional compatibility:
  state_32
  action_32
```

Keep these fields explicit:

```text
elite_tcp_delta_6d: continuous target
piper_intent_id: discrete target
estimated_contact_flag: image-derived or null/invalid
estimated_contact_risk_flag: optional early near-wall/contact warning; not a strict contact label
estimated_image_distance_px: image-derived or null/invalid
tactile/contact confidence: explicit confidence, not hidden truth
```

Do not silently create:

```text
centerline labels from visual wall clicks
guidewire path truth from branchs pose
real contact labels from MuJoCo exact contact
real retract labels unless real retract was commanded/logged
```

## Recommended Next Work For This Agent

Simulation track:

1. Use the audited right `n2/-1` geometry with explicit start buckets; do not
   collapse `0.42/0.48/0.54` to one global radius fraction.
2. Use the four bucket commands in `docs/commands.md` to create additional
   offline supervision, then merge the original formal source with probe
   manifests using `tools/merge_probe_manifests.py`.
3. Retrain with whole probe episodes held out and report source/probe metrics
   separately.
4. Require nonzero held-out probe-contact recall and temporal onset performance
   before setting probe estimator fields `policy_input_allowed=true` or sending
   contact-conditioned expert data to the algorithm track.
5. Keep the behavior Elite-only at contact. Do not add exact-contact feedback
   or strong Piper hold/retract recovery.
6. Re-audit the current front-up dataset against real visual anchors.
7. Identify which sim-to-real gaps are still visible:
   - glass reflection and blur;
   - guidewire thickness and brightness;
   - camera placement;
   - Piper feed event scale;
   - Elite motion scale;
   - contact/risk estimator realism.
8. Make small simulator/data-export changes only when they improve
   real-system correspondence.
9. Re-export packs only after a meaningful simulator/data-interface change.

Real collection track:

1. Stop feed-period tuning. Standardize and record the initial guidewire state:
   axial reference mark, entry angle, visible slack/curvature, and pre-feed
   side/top frames. Use a unique output directory for every trial.
2. Retain both clear-progress and locally blocked feeder-only trials. Compare
   their initial frames and suspected blockage location; treat `piper_step` as
   executed-command count, not physical insertion truth.
3. Do not spend further onsite time on a Piper-only comparison. Use the
   replacement feed device as the real feed backend and proceed to linked
   feeder-plus-Elite guidance trials after the initial-state procedure is
   stable.
4. For the linked pilot, let the operator independently trigger Elite next-point
   and feeder-forward events from the local preview. Use the unrecorded automatic
   Elite start approach on every episode; never auto-couple path steps and feed.
5. Keep every failed or partial run validated and video-rendered immediately;
   treat it as calibration/failure-analysis data, not clean expert data.

Cross-track:

1. Use real captures to calibrate simulation, not to claim policy success.
2. Keep algorithm handoff fields stable: `elite_tcp_delta_6d`,
   `piper_intent_id`, side/top images, and contact/tactile metadata.
3. Record durable findings in `docs/weekly-meeting-log.md`; do not bury them
   only in chat.

## Files To Read First

For a fresh data-track agent:

```text
AGENTS.md
.codex/skills/project-2026-guidewire-mujoco/SKILL.md
docs/project-state.md
docs/handoff.md
docs/data-track-handoff.md
docs/commands.md
docs/real-data-collection-checklist.md
docs/real-alignment-20260707-calibration-audit.md
docs/vla-target-interface.md
docs/control-layer-contract.md
```

Open archived docs only when a specific historical result is needed.
