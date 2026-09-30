# Algorithm Track Commands

Current execution state and active experiment selection are owned by
[algorithm handoff](algorithm-track-handoff.md). Select its matching command
section here; dated results and commands below retain their original scope.
This command file does not independently authorize a new training/evaluation run.

Last updated: 2026-09-29

This file contains current commands for the algorithm innovation / VLA track.
Simulation collection, real collection, and hardware commands remain in
`docs/commands.md` and the data-track documents.

## Current: Reasoning Review V1 Completed Offline (2026-09-29)

当前项目本机根目录：`/media/zsw/SSD1T/robotic_llm/baseline_vla_pi05/project_2026`。
本轮使用系统`python3`，不需要模型环境、API密钥或设备连接。
已有结果：`simulation_output/real10_reasoning_review_v1/index.html`和`summary.png`。
完整结论见`docs/algorithm-real10-reasoning-review-results-20260929.md`。

已执行：

```bash
python3 -B tools/review_real10_reasoning.py prepare
# 当前会话查看packets.jsonl和before图，真实提交reviews.jsonl后才执行score。
python3 -B tools/review_real10_reasoning.py score
python3 -B tools/report_real10_reasoning_review.py
python3 -B -m unittest discover -s tools -p test_real10_reasoning_review.py -v
```

16候选已完整审核和统计，3项必要测试通过。prepare拒绝覆盖现有输出，score拒绝
覆盖已完成统计。若另做研究，必须使用独立`--out`；不得复制本次建议冒充新模型
评估。若准备后中断，可直接继续完成同目录reviews.jsonl；score前须一次覆盖全部
16条且顺序一致。该入口仅汇总实际代理建议，不含自动LLM/API调用或硬件执行。
无需用户补跑当前实验；下一步缺少新片段和独立任务标签，见协议的证据边界。

## Deferred: Frozen Temporal Dependence Probe Completed (2026-09-23)

固定入口tools/probe_real10_region_temporal_dependence.py与协议
docs/algorithm-real10-region-temporal-protocol-20260923.md。源为
simulation_output/real10_region_response_pair_v1全部60份checkpoint；新输出
simulation_output/real10_region_temporal_dependence_v1。不是训练入口，optimizer updates=0。
原45窗/10轨迹/3seed，真实序列/首帧重复/末帧重复三条件；模型与归一化全部冻结。

代码23,592B、协议5,199B已顺序SCP并远端尺寸/help回读。以下已执行，**不重跑**：

~~~bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_real10_region_temporal_dependence.py --stage probe --resume
~~~

完成行：Complete: 60 frozen checkpoints, 810 diagnostic predictions, zero optimizer updates.
主体0.954s，new_units=60/skipped_units=0，峰值allocated746.32MiB。真实回放差0，
重复帧原始时序差分0，参数/buffer/旧工件未变。完整report存在时--resume不再推理；
若曾中断，每个unit为同checkpoint三条件的原子JSON，完整项跳过，未完整项原样重做。
不要并发写同out、删除旧结果或改变snapshot绕过恢复检查。

simulation_output/real10_region_temporal_dependence_v1_readback.tar.gz为240,722B，
已回传本机并解压至simulation_output/；只含小JSON/代码/协议，不复制任何大缓存、
checkpoint或原图。远端保留原source与冻结缓存，无需用户再次传图或权重。

以下本机只读渲染已执行；需要重绘时才再次运行：

~~~powershell
.\.venv\Scripts\python.exe -B tools/probe_real10_region_temporal_dependence.py --stage render
~~~

全部810条predictions、60条unit_checks、18组seed_reports、12组重复条件相对真实
impacts，以及9组region-global差齐全。逐条核对真实回放、原标签/轨迹归属与指标重算
通过。4张PNG/SVG、index.html、hardcase_readback.json、data-manifest.md和review.json
完整；visual_status=viewed_not_accepted，用户接受pending。

区域BA真实/首帧/末帧为62.00/55.33/45.33%；全局为50.33/43.50/46.50%。
首帧重复仍有约11.83pp区域-全局优势，不能把区域收益解释成已学到可靠动力学；
末帧重复改变锚点且可能OOD。下一步静态参照训练仅为建议，**尚未实施或启动**。
无待用户运行命令。当前结果页为
simulation_output/real10_region_temporal_dependence_v1/index.html。

## Historical: Regional Response Pair Completed (2026-09-23)

固定协议docs/algorithm-real10-region-response-protocol-20260923.md；独立模型
tools/real10_region_response.py，训练入口tools/train_real10_region_response_pair.py。
原45窗/10轨迹/3seed、同25帧定位监督来源、同8481响应参数与200步。两组仅比较
global_broadcast与regional_4x4；旧定位器按对应seed/fold冻结复用，不新增定位更新。

最终小文件已顺序SCP并远端尺寸/help回读。已执行以下命令，仅供复现定位，**不重跑**：

~~~bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_region_response_pair.py --stage profile
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_region_response_pair.py --stage train --resume
~~~

profile通过、40步临时权重丢弃且不计算留出性能，完整保守估时105.9s。实际60拟合
12000步全部完成，主体12.103s、无跳过；report.json存在，--resume直接退出不再优化。
BA均值50.33%→62.00%，但推进召回64.00%→57.33%，详见handoff；不是部署结论。
没有待用户运行任务，不再改网格/参数/seed/阈值试跑。

输出simulation_output/real10_region_response_pair_v1/；原始source为
simulation_output/real10_spatial_aux_pair_v1，localizers为
simulation_output/real10_head_response_pair_v1。旧461MiB特征缓存继续留4090。
小归档simulation_output/real10_region_response_pair_v1_readback.tar.gz为4,559,866B，
包含60份小checkpoint、配对协议快照、270条OOF响应、60条fit及36条区域记录。
已回传解压至本机simulation_output/并完成逐项验证；60份checkpoint/30对初始化和
对应旧定位器一致，六组预测指标重算最大差1.11e-16，缓存/保存重载最大差0。

已执行以下独立绘图入口，只读结果、无训练或模型推理；需要重绘时才再次执行：

~~~powershell
.\.venv\Scripts\python.exe -B tools/report_real10_region_response_pair.py
~~~

固定输出中文450DPI PNG/SVG：三seed指标、全部六难例分数，以及最小seed的六例
Side/Top首/中/末帧4×4质量图；附index.html、hardcase_readback.json、data-manifest.md。
8张PNG/SVG全部已渲染并查看；review.json保存验证与视觉状态。当前
visual_status=viewed_not_accepted，用户接受pending；网格质量不是接触概率或精确分割。
可直接打开simulation_output/real10_region_response_pair_v1/index.html和
figures/region_pair_summary_zh.png；图版和原图留本机，远端保留训练工件。
中断恢复机制保留：仅未完成fit从固定初始化重做，完整fit跳过；不并发启动或删除旧结果。

## Historical: Frozen Head-vs-Tip Response Pair Completed And Read Back (2026-09-23)

用户已执行上一小节固定训练；simulation_output/real10_head_response_pair_v1/包含
60份checkpoint、270条OOF响应、150条留出几何/概率图。主体136.908s、132000次更新，
没有中断恢复或跳过项。**无待用户运行命令，不重跑训练，不改seed/步数/阈值。**
完整配对结果BA均值49.33%→49.50%，三seed差-4.5/+5.5/-0.5pp，不采用本轮识别器。
准确率、混淆矩阵、定位和难例的完整解释在algorithm-track-handoff当前小节。

已通过SSH打包并SCP回本机的小归档：

- 远端：/home/zsw/project_2026/simulation_output/real10_head_response_pair_v1_readback.tar.gz
- 本机：simulation_output/real10_head_response_pair_v1_readback.tar.gz
- 大小4,590,987B；包含本轮小checkpoint/元数据/概率图，不包含旧大特征缓存或原图。
- 解压到本机simulation_output，保留原训练结果，不覆盖原数据/现场权重。

本机已执行只读结果渲染，无模型推理或优化：

~~~powershell
.\.venv\Scripts\python.exe -B tools/report_real10_head_response_pair.py
~~~

产物index.html、hardcase_readback.json、figures/data-manifest.md及7张中文450DPI
PNG/SVG完整。30对初始状态一致，全部配对和指标重算通过；review.json记录验证、
源报告两机时间/路径差异和viewed_not_accepted状态。可直接打开：

- simulation_output/real10_head_response_pair_v1/figures/response_pair_summary_zh.png
- simulation_output/real10_head_response_pair_v1/figures/response_pair_hardcases_zh.png

图版/原图留本机；远端保留训练产物。后续仅需再阅读现有结果，无需执行历史profile、
train或summarize命令。若将来确需重绘，复用同一绘图命令，不改冻结实验。

## Historical: Frozen Head-vs-Tip Response Pair — Ready For User Run (2026-09-23)

固定协议docs/algorithm-real10-head-response-pair-protocol-20260923.md；训练入口
tools/train_real10_head_response_pair.py；独立绘图入口
tools/report_real10_head_response_pair.py。已完成本机逐折归属检查、最终小文件顺序SCP
和远端尺寸回读。默认参数已冻结，不需要修改或重跑profile：

- source: simulation_output/real10_spatial_aux_pair_v1
- labels: simulation_output/real10_head_segment_annotation_audit_v1
- out: simulation_output/real10_head_response_pair_v1
- 原45窗/10轨迹/10折LOEO、3seed、两组同25帧定位监督；#4响应照常参与。
- 每组2000定位步后冻结，再200响应步；最终阈值0.5，不选模型或阈值。

已执行的短计时（不是完整实验）：

~~~bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_head_response_pair.py --stage profile
~~~

profile.json记录440次临时更新且权重丢弃；两阶段冻结正确，缓存/完整forward
logit最大差0。没有留出推理或正式checkpoint。完整逻辑任务估时330.998s（约5.5分钟），
超过五分钟，**以下完整训练交用户在4090运行，目前尚未运行**：

~~~bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_head_response_pair.py --stage train --resume
~~~

预期完成行：Complete: 60 fixed fits, 270 OOF responses, 150 heldout geometry records.
中断后重跑同一命令；已完成seed/fold/arm checkpoint跳过，仅未完成项从固定初始化
重做，保持协议与输出目录。先确认旧进程已退出，不并发重复启动；writer lock阻止重复写。
完整report已存在时--resume不会重新优化。不要删除输出或修改seed/步数规避恢复检查。

完成后读取report.json、OOF响应、留出几何、热图和逐项checkpoint读回检查，比较全部
3seed的BA/accuracy/轨迹宏准确率与六个固定难例。小结果回传本机后才执行绘图：

~~~powershell
.\.venv\Scripts\python.exe -B tools/report_real10_head_response_pair.py
~~~

绘图将生成中文450DPI PNG/SVG、data-manifest.md、hardcase_readback.json、index.html；
当前没有完整结果，不应先运行绘图或宣称收益。旧大缓存留远端，单个>100MB文件交用户
传输，小脚本和结果由agent处理。原标注、原模型、现场权重和共享动作接口保持不变。

## Historical: Head-Region Training Fit And Render Completed (2026-09-23)

入口 tools/fit_real10_head_region.py；预先固定协议
docs/algorithm-real10-head-region-fit-protocol-20260923.md。已完成3seed×2000步的25帧
训练拟合及全部图版，**无待用户执行命令，不重跑或启动完整LOEO**。#4仍在旧fold0留出侧，
未评估。仅更新原529参数定位分支，不训练响应、PI05、mixed-head或world model。

默认source/labels/pure/out分别为：

- simulation_output/real10_spatial_aux_pair_v1
- simulation_output/real10_head_segment_annotation_audit_v1
- simulation_output/real10_tip_only_fit_v1
- simulation_output/real10_head_region_fit_v1

最终训练代码28,423B与协议4,994B已顺序SCP并回读，真人审计快照远端已存在。已执行：

~~~bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/fit_real10_head_region.py --stage fit
~~~

实际8.048s。保持原场景ROI和缓存，框/线仅进入目标；不下载权重或重编码图像。
独立out含protocol、训练几何快照、450条frame_diagnostics、loss_curves、
model_readback、heatmaps.npz、report及六份checkpoint。原大缓存和3份seed恢复包留远端。
约3.5MiB simulation_output/real10_head_region_fit_v1_readback.tar.gz 已回传并解压到本机
同名out；下载包排除seed恢复包及锁文件，不含大缓存。

本机已执行：

~~~powershell
.\.venv\Scripts\python.exe -B tools/fit_real10_head_region.py --stage render
~~~

7张450DPI中文PNG及SVG、真实数据manifest与index.html齐全。查看发现#77标题重叠后，
只改render顶部留白和色标说明并重渲染；最终入口28,522B已再SCP并回读，**没有重训**。
训练执行快照entrypoint_snapshot.py仍保留原始版本；最终代码除render外的AST完全一致，
render_snapshot.py是最终版本，不覆盖训练快照以伪造版本一致。六份checkpoint回放差0，
人工source快照字节相同，450条数字重算差0。全部7图viewed_not_accepted，用户接受pending。

仅真正中断恢复时，使用该次冻结训练入口和原协议（本次已完成，无需执行）：

~~~bash
cd /home/zsw/project_2026
env PYTHONPATH=tools OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  simulation_output/real10_head_region_fit_v1/entrypoint_snapshot.py --stage fit --resume
~~~

完成seed会跳过，未完成seed从原初始化重做，不调步数/seed或覆盖out。冻结入口恢复避免
仅render修改触发整文件版本保护；report存在时不优化。下一步响应对照尚未实现/授权启动。
不能跨fold直接复用本轮定位checkpoint造成轨迹泄漏，也不能把粗区域概率当成功率。

## Historical: Six-Window Head Annotation Audit Completed — No Training (2026-09-23)

真人标注已读回：19次保存、6窗36/36帧位已审阅；29个几何（15完整/13部分/1不确定），
6不可见、1无法可靠圈定，后7项保持null。22条两点短线与7个partial粗框只作粗定位线索。
本轮没有训练或标签改写，**无待用户执行命令，也不需要继续补画缺失项**。

新增 `tools/audit_real10_head_segment_annotations.py`，本机已执行：

```powershell
.\.venv\Scripts\python.exe -B tools/audit_real10_head_segment_annotations.py
```

默认输入 `simulation_output/real10_head_segment_annotation_v1`、原图根 `collected_data`，
输出 `simulation_output/real10_head_segment_annotation_audit_v1`。默认out已存在，不要删除
或重复执行；以后确有新revision需要回读时，明确指定新的 `--out`，保留本轮快照。
该入口只读真人包，拒绝fixture；使用原UI的schema读取，校验revision/来源/null语义，
不产生预测或自动标签。元数据独立审计可用 `--metadata-only`，不需要原图/GPU/LeRobot。

本轮远端命令已完成，输入为单独上传的真人元数据快照（不是测试fixture）：

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  tools/audit_real10_head_segment_annotations.py \
  --pack simulation_output/real10_head_segment_annotation_audit_input_20260923_v1 \
  --metadata-only
```

远端执行前已顺序SCP最终脚本及manifest/真人JSONL，尺寸回读为11,272/43,572/22,331字节。
未传图像、未启动远端标注页。远端report已下载为本机out的 `remote_readback_report.json`，
与本机report除来源路径和生成时间外一致；旧点/响应来源与UI准备快照逐字节不变。

本机 `index.html` 与 `figures/head_annotation_*.png` 六图可直接打开，图像/中文已查看，
`review.json`记录最小核查及逐窗注意事项。新图版visual_status为viewed_not_accepted，
用户接受pending；不要把人工框、后续帧几何决定的展示裁剪或diagnostic字段传给policy。
下一步弱定位实验尚未实现/启动。原8795页可继续查看；需要恢复服务时使用下节原命令，
不要重复prepare、清空真人保存历史或把fixture复制过来。

## Historical: Head-Segment Coarse Annotation UI — Human Review Only (2026-09-23)

入口已实现、已核查，不启动训练。正式页面 `http://127.0.0.1:8795/`，原六窗Side/Top首中末
36个帧位；新包 `simulation_output/real10_head_segment_annotation_v1`。说明与schema见
`docs/algorithm-real10-head-segment-annotation-guide-20260923.md`。旧点、原图和响应只读。

本机已执行过一次准备，**不要重复执行 `--prepare`，也不要删除目录重建**：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --prepare
```

页面服务已在本机后台启动；如果服务仍可访问，直接使用，不要再开第二个写同一包的进程。
日后关闭后恢复，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --port 8795
```

已保存revision自动读取；Ctrl+C只停止服务，不删除已保存标注。未保存草稿不保证刷新后恢复。
进度只读检查：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --summary
```

新几何只追加 `head_annotations.jsonl`，首次人工保存前该文件不存在，36帧均unreviewed。
粗框或短折线可逐帧切换，遮挡处使用“另起一段”；不可靠时保持null，不补全25mm。
只需填标注人、选择完整性/缺失状态、保存各窗口，不用重新标旧响应，也不需要再运行模型。

实现同步与远端版本回读已完成（中途SSH曾超时，恢复后最终成功）：

```powershell
scp -C -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/review_real10_head_segment_annotation.py project4090:/home/zsw/project_2026/tools/
scp -C -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/real10_head_segment_annotation.html project4090:/home/zsw/project_2026/tools/
```

```bash
cd /home/zsw/project_2026
wc -c tools/review_real10_head_segment_annotation.py tools/real10_head_segment_annotation.html
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/review_real10_head_segment_annotation.py --help
```

版本尺寸Python14,507B、HTML19,075B。只上传小脚本/页面/元数据；远端原图未补传，不在
远端启动真人标注页。4090隔离fixture验证API追加2次保存、重新加载、小数坐标、不相连折线、
null缺失，拒绝越界/缺失带几何/旧revision。结果见正式本机包中的 `verification_remote.json`。
本机另外验证原图服务及浏览器框选/线段/完整性/独立缺失/草稿换窗保护/保存刷新。

本机测试包 `simulation_output/real10_head_segment_annotation_ui_fixture_v1`，远端测试包
`simulation_output/real10_head_segment_annotation_remote_fixture_v1`。测试标注均带
`annotation_source=ui_fixture_not_human`，不可复制到真人包。不要将8796测试页用于人工标注。
正式页中文及原图已查看，用户视觉接受pending。无新的模型输入、训练、性能或接触结论。

## Historical: Head-Segment Read-Only Review Completed — No Training (2026-09-23)

用户同意后，固定原六窗的36张首中末图已导出并全部查看。入口
`tools/review_real10_head_segment.py`；out为`simulation_output/real10_head_segment_review_v1`。
没有检测、分割或新标签，原稀疏点和窗口响应不变；不执行上一轮LOEO训练建议。

新入口已顺序SCP到4090，9,014字节/help回读通过。远端当前项目路径没有这批原图，
因此不运行完整图像导出、不上传大图；本次只是无需LeRobot/GPU的本机只读PNG/HTML整理。
已在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment.py
```

默认只读输入：`simulation_output/real10_wire_correspondence_v1`、`collected_data`和
`simulation_output/real10_spatial_aux_pair_v1/source/annotation_snapshot.jsonl`。
中文字体使用本机微软雅黑；远端可用Noto CJK，但这不代表远端原图齐全。`--raw-root`或
`--font`仅在明确对应原路径/字体时指定，不能改案例、点位或图像增强寻找有利结果。

**已完成，无待用户运行命令。** 重复执行默认out会保留旧目录并报错，不删除结果重跑。
约2.44s导出`index.html`、`figures/head_review_*.png`六图、source快照、report和入口快照；
人工点仍29 visible/1 ambiguous/6 unreviewed。这是原点状态，不是头段可见性统计。
inspection-notes/review记录定性观察和查看范围；全部六图viewed_not_accepted，用户接受pending。

本地文件方式打开页面可查看36个原图链接。若HTTP托管，应从项目根目录保持相对路径，
新包并未复制全部原图。展示窗可使用后续人工点，只作离线阅读，不能接入policy或监督。
源标注字节未变；约25mm不映射成像素标尺。没有新的可见率、分割指标或模型效果。
下一步候选为原六窗头段粗框/短折线人审入口，尚未实现，也未创建新的标注任务。

## Historical: Staged 1000+1000 Fit Completed — Training-Only (2026-09-22)

用户已批准并完成3seed的固定阶段策略。入口 `tools/fit_real10_staged_budget.py`，协议
`docs/algorithm-real10-staged-budget-protocol-20260922.md`；默认路径为：

- source: `simulation_output/real10_spatial_aux_pair_v1`
- pure: `simulation_output/real10_tip_only_fit_v1`
- joint: `simulation_output/real10_joint_budget_fit_v1`
- out: `simulation_output/real10_staged_budget_fit_v1`

只训练原fold0；1000步纯定位→1000步原联合，原AdamW状态保留，不重置共享分支。
最终共享step=2000、response step=1000。总更新数相同，但分类更新量与直接联合不同。
**fit和render均已完成，无待用户执行命令，不要重跑或启动10折。**

最终入口和协议已顺序SCP至远端tools/docs，尺寸/help回读后执行过：

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/fit_real10_staged_budget.py --stage fit
```

只有真正中断时同一命令加 `--resume`：依原配置/入口快照保留完整seed，未完成seed从
旧fold0初始化重跑；不修改step/seed/out寻找有利结果。完整report存在时resume不再训练。
1000/2000步checkpoint含model与optimizer状态；seed*_complete.pt为整seed恢复结果。
旧对照不重训、不覆盖，诊断包不含461MiB特征缓存和seed恢复包。

799KiB readback归档已下载并检查tar清单后解压。本机先用 `--stage render`生成图版，
发现案例色条标题出界，最终使用如下完整命令设置tight边界后重新导出，无需torch：

```powershell
@'
import runpy
import sys
import matplotlib
matplotlib.rcParams['savefig.bbox'] = 'tight'
sys.path.insert(0, 'tools')
sys.argv = ['tools/fit_real10_staged_budget.py', '--stage', 'render']
runpy.run_path(sys.argv[0], run_name='__main__')
'@ | .\.venv\Scripts\python.exe -B -
```

此设置只影响导出边界，未改训练入口/执行快照、数值或概率图。render需本机旧pure/joint
包中的heatmaps.npz及原ROI224图像；不能把新out单独复制过去就假定依赖完整。输出
`figures/staged_budget_fit_zh.png/.svg`、9组 `staged_case_*_zh.png/.svg`和`index.html`。
最终summary/#65Top/#62Side已查看、用户接受pending，其他7图未查看；图版仅在本机。

主体43.417s，3×2000新更新，6份保存模型回放差0，切换前后共享Adam状态未变。
150条旧对照逐字段相同，150条新增读数、54张新概率图、66条loss曲线全部回读；
readback_verification/review保存数值和查看范围。空间均值训练误差Side139.32/Top127.90px，
相对直接联合降低46.82%/40.75%；但Top峰值从预热14.24px升至115.29px，尚不能称定位稳定。
详细判读见当前handoff，不使用这些权重作策略或触觉输入。

建议的原LOEO泛化比较尚未实现/执行。下一轮需先固定同总预算配方、留出评估及恢复入口，
按超过五分钟的完整逻辑任务交用户运行；不要把本轮fold0训练点改善当作响应BA或实机收益。

## Historical: Equal-Budget Joint Control — Fixed 2000 Steps (2026-09-22)

用户已同意并完成原联合目标2000步对照。入口 `tools/fit_real10_joint_budget.py`，默认新out
`simulation_output/real10_joint_budget_fit_v1/`，默认source为原spatial_aux_pair，pure为
上一轮tip_only_fit。**仅训练fold0的三个seed，不能据此启动10折或重训纯定位。**
协议、计算估时和证据边界见 `docs/algorithm-real10-joint-budget-protocol-20260922.md`。

最终入口与协议顺序SCP到远端tools/docs，尺寸/help回读后执行：

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/fit_real10_joint_budget.py --stage fit
```

真正中断时在相同命令后加 `--resume`，已完成seed依据 `seed*_complete.pt`保留；未完成
seed从原初始化重跑，不使用缺少optimizer状态的旧200步权重继续训练。完整report存在
时resume不会再训练。不要为寻找有利结果改seed、step或out。

输出3份最终checkpoint、3份seed恢复包、全部300行比较读数（仅75行为新增推理）、
27张新概率图和report/model_readback；旧比较数据指向已完成pure包。只回传小型产物，
大特征缓存和seed恢复包留远端。本机复用旧pure热图和ROI224绘图：

```powershell
.\.venv\Scripts\python.exe -B tools/fit_real10_joint_budget.py --stage render
```

**fit与render均已完成，不需重跑。** 主体79.176s、3×2000新更新、3份最终checkpoint；
旧联合/纯定位没有重训或覆盖。仅424KiB诊断包回传，完整seed恢复包留远端。联合2000步
Side/Top空间均值误差262.00/215.86px，纯定位同预算为115.16/99.29px，全部训练点。
小文件传输有两次连接超时，重传成功并回读最终版后才执行；没有混用不完整版本。

`figures/joint_budget_fit_zh.png/.svg`与9组 `joint_case_*_zh.png/.svg`为本轮图版，
`index.html`可本机查看；render需要原pure包中的heatmaps.npz及原ROI224图像，不能仅
复制新out到另一台机器就假定依赖齐全。225条旧对照逐字段未变，300项窗宏汇总重算差0，
3份保存模型回放最大差0。readback_verification/review记录数值与视觉检查范围。
summary/#65Top/#62Side已查看、用户接受pending，其他7图未查看；图版仅在本机。

执行结果与解释以当前handoff为准；同预算是同更新数，不是同FLOPs或同wall-clock。
不计算留出指标、不改空间均值读出、不部署新checkpoint。建议的1000定位预热+1000联合
阶段对照尚未实现或执行，当前没有待用户运行命令。

## Historical: Pure Tip Fitting Completed — Training-Only Diagnostic (2026-09-22)

用户同意后已完成 `tools/fit_real10_tip_only.py --stage fit`。新输出
`simulation_output/real10_tip_only_fit_v1/`；原source为
`simulation_output/real10_spatial_aux_pair_v1/`。**3seed×2000步已完成，无需重跑。**
原模型/数据不改，25点全部用于训练内诊断，非泛化、响应识别或策略训练。
预算与比较口径见 `docs/algorithm-real10-tip-only-fit-protocol-20260922.md`。

已执行过程（记录用途，两个SCP均需完成后再执行）：

```powershell
scp -C -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/fit_real10_tip_only.py project4090:/home/zsw/project_2026/tools/
scp -C -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 docs/algorithm-real10-tip-only-fit-protocol-20260922.md project4090:/home/zsw/project_2026/docs/
```

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/fit_real10_tip_only.py --stage fit
```

只有真正中断且未完整生成report时使用相同命令加 `--resume`：已完成seed依据
`seed*_complete.pt`保留，未完成seed从原初始化重跑；不追加其旧loss序列。已完整输出的
目录在resume下只提示完成，不再优化。不删除旧目录，不另换seed/out反复寻找有利结果。

输出protocol/source代码快照、6份200/2000步checkpoint、3份完整seed结果、report、
point_diagnostics.jsonl（300项，只有25个不同人工点）、loss_curves、model_readback和
heatmaps.npz。完整seed结果留远端供恢复；小型readback归档不含它们和原461MiB缓存。
本机绘图不需要torch或重新推理：

```powershell
.\.venv\Scripts\python.exe -B tools/fit_real10_tip_only.py --stage render
```

fit主体7.307s、峰值allocated616.9MiB；300项逐点读数的窗宏指标本机重算与report差0，
全部概率图归一化偏差最多2.38e-7，6份checkpoint重载差0。只回传约3.4MiB归档，完整
seed恢复包和大特征缓存留远端。render已完成，不需再次运行；同一入口源码未变化。
`figures/tip_only_fit_zh.png/.svg`为汇总，`tip_fit_case_*_zh.png/.svg`为9组固定案例，
`index.html`为本机只读浏览页。summary/#65Top/#62Side查看状态为viewed_not_accepted，
其他7图未查看；用户尚未接受。readback_verification/review保存数值及视觉检查范围。

空间误差与解释见当前handoff。不得把2000步结果直接当成对旧200步联合训练的公平
胜出；本轮不评估留出，不计算新的response BA，不部署这些权重。下一步建议的同2000步
联合训练对照尚未实现/执行；当前没有待用户运行命令。

## Historical: Saved Spatial Fit Diagnosis Completed — No Retraining (2026-09-22)

用户已授权并完成既有60份权重的只读定位诊断，入口
`tools/diagnose_real10_spatial_aux_fit.py`。默认source为原
`simulation_output/real10_spatial_aux_pair_v1`，新out为
`simulation_output/real10_spatial_aux_fit_diagnostic_v1`。
**infer已完成，不需重复运行，不启动train。** 主体1.521s、0优化步，既有权重/参数未改，
原留出输出最大回放差0。训练点本身尚未可靠定位，详情见当前handoff。

已执行的远端过程（记录用途）：

```powershell
scp -C -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/diagnose_real10_spatial_aux_fit.py project4090:/home/zsw/project_2026/tools/
```

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/diagnose_real10_spatial_aux_fit.py --stage infer
```

输出protocol/report、entrypoint_snapshot、point_diagnostics.jsonl（1350训练/150留出）、
model_readback.jsonl、visual_heatmaps.npz（108张56×56图、约1.21MiB）及索引。
仅复用旧空间特征；没有下载模型或SCP461MiB特征缓存。约1.5MiB诊断归档已完整回传，
tar列表核验后解压至新本机目录，不覆盖旧配对结果。

本机重画已有诊断，无模型执行：

```powershell
.\.venv\Scripts\python.exe -B tools/diagnose_real10_spatial_aux_fit.py --stage render
```

figures包含spatial_fit_diagnosis_zh及9组distribution_case_*，各有PNG/SVG；index.html
直接本机打开。图中固定全部三seed；训练可视化fold0、每窗/视角最早有效tip；不是最好
训练模型。定位均值/峰值/均匀期望比较和目标两格概率质量仅诊断，不改模型接口或阈值。
图版完整在本机，未传远端。助手查看summary/#65Top/#62Side，其他图未视觉检查；
review.json记录用户接受pending。图版按实际读出，没有新增点/缺失帧填充或隐式标签。

渲染首次查看后只调整轴范围/色条边距，未重跑infer。entrypoint_snapshot是执行推理版，
render_snapshot是最终纯绘图版；两者不要手动覆盖成相同版本。infer拒绝既有out；真正
失败保留目录与错误，不以换out反复重试已完成工作。当前没有待用户运行命令。

下一步纯定位拟合能力试验尚未实现/启动；不能把该建议当作重训授权或世界模型改动。

## Historical: Spatial Auxiliary Pair Completed — Readback Only (2026-09-22)

用户已执行完上一节训练命令，**不要再次启动train/profile/prepare**。远端
`simulation_output/real10_spatial_aux_pair_v1/`已完整60份200步checkpoint，训练主体
180.152s，270条OOF响应/150条留出tip预测与report齐全。两组平均BA48.33%→50.83%，
配对变化+12.5/+7.0/−12.0pp，未达到一致性条件；不选最佳seed接入其他模型。

本轮小文件已回传：report、fit_records、oof_predictions、oof_tip_locations、train.log、
last_train_invocation、原source与模型/runner快照。**未拉回461MiB特征缓存或权重**。
若以后需重新获取，仅指定这些小文件/目录，不对整个实验目录递归SCP。

新增本机只读回读与画图入口，无torch/LeRobot推理；沿用项目.venv及Microsoft YaHei：

```powershell
.\.venv\Scripts\python.exe -B tools/report_real10_spatial_aux_pair.py
```

此命令已成功执行：逐行回算六组指标，核对训练预算/点数量/class weight/保存重载记录、
原10折和当前人工响应快照；不修改原report或checkpoint，不训练、不搜索阈值，不新建标签。
同名派生readback/图版可重建；review.json若已存在会保留，重新生成图不自动获得验收。

本机输出：

- `simulation_output/real10_spatial_aux_pair_v1/readback.json`：数值/来源核对及逐seed变化窗；
- `.../figures/spatial_aux_results_zh.png`和.svg：完整三seed配对指标；
- `.../figures/tip_case_{39,62,65,77,100}_zh.png`和.svg：全部25个tip目标×3seed×2组读出；
- `.../figures/paired_metrics.csv`、`data-manifest.md`、`index.html`和`review.json`。

图版仅本机保存，助手已查看全部6张PNG，中文/标记可读；状态viewed_not_accepted，用户
接受pending。模型输出坐标是空间分布的期望，不是heatmap峰值/物理追踪轨迹。#4材料点
不用于tip诊断；#77缺失Side首帧不回填。没有单独新增某个有利seed或只画最佳结果。

旧训练入口和冻结协议保持不变。下一步若获准，仅用已有模型诊断训练点拟合与留出定位，
本轮没有启动这项推理、额外训练、扩标或PI05/world model接入。

## Historical: Fixed Spatial Auxiliary Pair — User-Run Training Ready (2026-09-22)

入口`tools/train_real10_spatial_aux_pair.py`；预先协议
`docs/algorithm-real10-spatial-aux-protocol-20260922.md`。新远端目录
`simulation_output/real10_spatial_aux_pair_v1/`已准备，不覆盖旧工件。45窗/原10折/三个
固定seed/两组/各200步，只有tip辅助权重0/0.1不同；合计60次拟合。**正式训练未启动**。

已完成：本机AST，顺序SCP与远端help/文件大小回读，prepare1.918s；profile使用最大
训练折每组20步（临时权重丢弃），两组初始化一致。完整任务估计约7.3分钟，交用户运行。
两组0.691/0.316s的短跑耗时含首组CUDA预热差，不能当效率对比结论。
602张图空间特征缓存483381096字节，仅保留4090，不需要用户重新传图，也不要自动SCP。

**待运行，在4090终端执行一次：**

```bash
cd /home/zsw/project_2026
set -o pipefail
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_spatial_aux_pair.py --stage train --resume \
  2>&1 | tee -a simulation_output/real10_spatial_aux_pair_v1/train.log
```

预期逐项输出`1/60`至`60/60`，末尾`Complete: 60 final fits`与配对BA差。`--resume`
也可用于首次运行：只跳过验证为200步完成的checkpoint。终端/SSH中断时先确认旧进程
是否仍在运行，然后重复同一命令；文件锁拒绝并发写，未完成的当前组从原seed重跑。
不要删除整个out、重跑prepare/profile、修改代码快照或重新选择seed。若出现真实异常，
保留日志与`.partial`并检查原因；不要为凑齐60项跳过失败折。

输出：

- `checkpoints/seed*_fold*_<arm>.pt`：60份最终200步权重，初值与重载输出一致性记录；
- `oof_predictions.jsonl`：45窗×3seed×2组=270行，不是新增独立数据；
- `oof_tip_locations.jsonl`：25个留出tip点×3×2=150行，分视角原图px；
- `fit_records.jsonl`、`report.json`、`last_train_invocation.json`、`train.log`。

完成后先回读结果再解释效果。报告含各seed accuracy/BA/混淆矩阵/整轨迹宏准确率、
配对变化及跨seed均值/样本标准差；无tip标注的折定位指标null，不记0。未来帧仍仅限
动作后响应识别；不生成部署权重、伪标签或PI05/world model改进结论。完整结果图待回读
真实结果后生成，当前没有新的模型性能图或视觉验收。

如果全部60项已完成、仅汇总中断，可单独重建报告（不优化模型）：

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_real10_spatial_aux_pair.py --stage summarize
```

以下是已完成的小文件同步和计时记录，**不需用户重复执行**：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/train_real10_spatial_aux_pair.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 docs/algorithm-real10-spatial-aux-protocol-20260922.md project4090:/home/zsw/project_2026/docs/
```

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/train_real10_spatial_aux_pair.py --stage prepare
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/train_real10_spatial_aux_pair.py --stage profile
```

## Historical: Spatial Tip Auxiliary Preparation Completed — Zero Updates (2026-09-22)

模块`tools/real10_spatial_response.py`及入口`tools/prepare_real10_spatial_aux.py`；协议
`docs/algorithm-real10-spatial-aux-protocol-20260922.md`。复用旧ROI224、45窗/10折及最新版
人工点，输出`simulation_output/real10_spatial_aux_v1/`。**已完成准备与零步检查，不需补跑；
当前没有train选项或训练runner。**

最终文件先顺序SCP、读取入口/help，然后4090执行；以下均是已完成记录：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/real10_spatial_response.py tools/prepare_real10_spatial_aux.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 docs/algorithm-real10-spatial-aux-protocol-20260922.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 6 tools/real10_spatial_response.py tools/prepare_real10_spatial_aux.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/prepare_real10_spatial_aux.py --help"
```

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/prepare_real10_spatial_aux.py --stage build
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/prepare_real10_spatial_aux.py --stage preflight
```

build0.035s：602个旧图像引用，25个尖端点/5窗，4个材料点排除；不解码/改标。
preflight主体0.728s：仅编码六窗80张图、前向及backward；两臂1554参数且初值输出一致，
heldout/无效点辅助梯度0，参数未更新。只用缓存resnet18-f37072fd.pth，不自动下载。
随机初始loss只供数值检查，不是训练结果或模型效果。

输出model_inputs/supervision/point_audit、原annotation_snapshot/folds、四点审计快照、
protocol/report/preflight、代码/设计快照。小工件已回传本机；无权重或大型特征缓存需要传输。
本机图版命令（不重新推理）：

```powershell
.\.venv\Scripts\python.exe -B tools/prepare_real10_spatial_aux.py --stage render
```

`figures/spatial_supervision_zh.png`及.svg来自人工点，来源表在figures/data-manifest.md；
index.html可直接打开。图版助手已查看，review.json记录用户接受pending及独立回读结果。

恢复：build拒绝既有out，preflight拒绝既有preflight.json。SSH不确定时先读回标记，不能
因传输中断重跑已完成计算。真正build失败保留目录并选新out；preflight未完成时先确认
进程已退出、核对原因再决定复跑，不能把零步输出当已训练checkpoint。后续训练入口与
完整预算另行实现，不直接恢复已停止的global-pool/追踪配置。

## Historical: Human Response Interface Refresh Completed — No Training (2026-09-22)

沿用`tools/prepare_real10_action_effect_interface.py`，不修改代码或默认值。旧默认reference
属于修订前历史实验；本轮显式使用`real10_window_response_mil_v1`中的最新45窗人审快照，
导出新目录`simulation_output/real10_action_effect_interface_human_v2/`，不覆盖旧接口。
以下为**已完成记录，不需用户补跑，也不自动启动后续训练**。

最终入口顺序SCP并回读后，在4090运行：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/prepare_real10_action_effect_interface.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 7 tools/prepare_real10_action_effect_interface.py"
```

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_real10_action_effect_interface.py \
  --reference simulation_output/real10_window_response_mil_v1 \
  --out simulation_output/real10_action_effect_interface_human_v2
```

构建0.333s；45窗/10折、914图像引用、208历史观测，原时间隔离/缺失动作检查通过。
两端已保存queries、response_targets、timing_audit、annotation_snapshot、folds、protocol、
report共7个小文件。独立回读确认query/folds/audit与旧包字节相同、未来观测未变、人审与
当前源一致；响应类别仅按既有人审修订更新#62/#100。未训练、下载或运行图像编码器。

该入口无模型/checkpoint；输出已存在时拒绝覆盖。若构建真正中断，先保留故障目录，
修复原因并按授权范围选择新`--out`重做接口导出，不借此改变窗口或标签。下一次若人审
源变化，原snapshot检查会中止，需显式确定新版监督reference，不能关闭检查或回用旧标签。
全动作候选仍缺失，不能把此包直接传入要求完整action9的世界模型。下方多seed实验已结束，
不追加seed/阈值搜索。

## Historical: Fixed Multi-Seed Response Retest Completed (2026-09-22)

入口`tools/retest_real10_window_response_seeds.py`，调用原`run_real10_window_response.train`
新增的显式`base_seed`参数，原CLI默认行为保持。固定协议
`docs/algorithm-real10-window-response-seed-protocol-20260922.md`。**已完成，不需用户补跑**。
新增20261020/20261120/20261220，原20260920只读；没有重新编码图像、改标签或训练PI05。

输出`simulation_output/real10_window_response_seed_retest_v1/`：

```text
protocol.json / design_protocol.md / entrypoint_snapshot.py / training_entrypoint_snapshot.py
annotation_snapshot.jsonl / folds.json
seed_20261020/ / seed_20261120/ / seed_20261220/
  protocol.json / features.npz / prepared.json / annotation_snapshot.jsonl / folds.json
  fold_models/ / fold_log.jsonl / oof_predictions.jsonl / report.json
report.json / window_stability.jsonl / validation.json / readback_validation.json
summary.md / interpretation.md / review.json / index.html
seed_comparison_zh.png / case_stability_zh.png
```

每seed20个小模型×200步，新增总60模型/12000步，拟合与检查14.515s。新增配对BA差为
+4.0/−3.5/−0.5pp，均值0；不再追加seed、不挑最好模型，不部署或接入世界模型。
两端保存小报告/图版；新权重与复制特征留4090，未回传本机。原v1工件完整保留。

以下仅为已完成命令记录，运行前按顺序同步最终入口/协议并回读：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/run_real10_window_response.py tools/retest_real10_window_response_seeds.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 docs/algorithm-real10-window-response-seed-protocol-20260922.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/retest_real10_window_response_seeds.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/retest_real10_window_response_seeds.py --help"
```

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/retest_real10_window_response_seeds.py --stage run
```

入口先只读重放旧20模型，再按固定顺序跑完全部3seed；仅顶层`report.json`的completed
表示整项完成。输出已存在时拒绝，不能选择性恢复/替换较差seed。若真正中断，保留失败
目录并在授权范围内以新`--out ..._v2`完整重跑；不改参数、不把中断当选seed机会。

本轮小报告归档为129,002 bytes，排除权重/NPZ（不是拆分大文件规避100MB规则）：

```bash
tar --exclude=features.npz --exclude='*.pt' --exclude='*.npz' \
  -czf simulation_output/real10_window_response_seed_retest_v1_reports.tar.gz \
  -C simulation_output real10_window_response_seed_retest_v1
```

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090:/home/zsw/project_2026/simulation_output/real10_window_response_seed_retest_v1_reports.tar.gz simulation_output/
# 仅在目标目录不存在、传输完整时解压；当前v1已存在，不再执行覆盖
tar -xzf simulation_output/real10_window_response_seed_retest_v1_reports.tar.gz -C simulation_output
.\.venv\Scripts\python.exe -B tools/retest_real10_window_response_seeds.py --stage render
```

render只读保存报告，不加载模型/拟合；两张图助手已查看，用户接受pending。
已独立重算4次混淆矩阵与新增seed汇总，旧OOF/新模型回放差均0、人工源未变。
原控制器参照结果跨seed完全相同。没有新增硬件、数据/SOFA轨或周会日志变更。

## Historical: Human-Window Response Pooling Pair Completed (2026-09-22)

入口`tools/run_real10_window_response.py`；固定协议
`docs/algorithm-real10-window-response-protocol-20260922.md`。45个人工响应窗、10个原LOEO折，
只训练轻量动作后响应编码器；**本轮已完成，不需要用户补跑，不训练PI05或调用硬件**。
输出`simulation_output/real10_window_response_mil_v1/`，两端保留协议/代码/标注/折快照、
features、prepared、20个神经折模型+10个控制器ridge模型、fold_log、OOF、report，以及
本机生成的summary/interpretation/review/validation、两张中文PNG和index.html。

结果：mean准确率60%/BA58.5%；logmeanexp准确率68.89%/BA67%，4窗改对/0改错，轨迹3胜7平。
控制器参照BA也是67%，新模块尚不采用/部署；两组训练准确率均100%。详见handoff。

以下为已完成命令记录；先SCP最终代码/协议，再回读与一次合成汇聚检查：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/run_real10_window_response.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 docs/algorithm-real10-window-response-protocol-20260922.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/run_real10_window_response.py && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_window_response.py --stage check"
```

完整逻辑任务预计1–3分钟，实际准备2.428s、20个模型拟合与评价4.548s（均不含SSH/导入等）：

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_window_response.py --stage all
```

复用本机/远端已有ROI图与缓存ImageNet权重，不下载新文件。`all`=prepare+train；训练后
最后写入`report.json`且status=completed才算完成。现有v1不可重跑/覆盖，不把恢复当调参。
中断保留目录，未来经授权用新的`--out ..._v2 --stage all`完整运行；仅prepare完成且尚无
fold_models时可对该out执行`--stage train`。本轮无断点续训/最优轮次选择。

回传的训练工件归档（不含后来本机图版）大小1,715,046 bytes，小于100MB：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090:/home/zsw/project_2026/simulation_output/real10_window_response_mil_v1_readback.tar.gz simulation_output/
# 确认传输完整且本机目标目录不存在后才解压；现有v1不要再解压覆盖
tar -xzf simulation_output/real10_window_response_mil_v1_readback.tar.gz -C simulation_output
.\.venv\Scripts\python.exe -B tools/run_real10_window_response.py --stage render
```

`render`只读保存预测并写图/HTML，不拟合；训练后的代码小改仅限表头对齐、显示比例/措辞，
冻结训练代码不变。绘图输入默认本机已有`real10_response_roi_images_v1`。
已从保存OOF独立重算指标、核对20模型重载差0、原人审源不变；没有逐文件SHA。
两图助手已看，用户接受pending；所有输出只作开发诊断。下一项多seed复核尚未启动。

## Historical: Foreground / Fixed-Identity Six-Window Probe Completed (2026-09-22)

入口`tools/probe_real10_foreground_correspondence.py`；本轮已完成，不需要用户跑长任务或训练。
仅算法轨离线诊断；没有新权重/依赖/图片包下载。输入复用
`simulation_output/real10_bootstapir_tracking_v1/inference_inputs/`，未来人工点所在
`evaluation_source/`仍留本机供预测后的评估，不进入4090推理。

输出`simulation_output/real10_foreground_correspondence_v1/`包括：

```text
entrypoint_snapshot.py / input_manifest.json / protocol.json
inference_report.json                    # 4090计算4.176s，不是在线端到端时延
predictions_raw_template.jsonl           # 候选与门控输出分开
predictions_foreground_anchor.jsonl
foreground_support.jsonl / diagnostic_maps.npz
evaluation_raw_template.jsonl / evaluation_foreground_anchor.jsonl
report.json / README.md / interpretation.md / numerical_score_check.json
review.json / index.html / human_*.png    # 6张中文图版；用户接受pending
```

普通模板输出4/9末点、仍错误放行#77 Top；前景臂0/9末点输出，8个候选平均偏差69.791px
（同子集BootsTAPIR26.502px）。**不采用，不放松门限，不自动续跑或接入世界模型**。
前景使用全窗后续图像，不是在线因果方法；拒判不是静止/不可见/碰壁。

以下为已完成命令记录；`infer`拒绝已有输出目录、`report`拒绝已有report。不要覆盖v1：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/probe_real10_foreground_correspondence.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && head -n 7 tools/probe_real10_foreground_correspondence.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/probe_real10_foreground_correspondence.py --stage check"
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/probe_real10_foreground_correspondence.py --stage infer"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r project4090:/home/zsw/project_2026/simulation_output/real10_foreground_correspondence_v1 simulation_output/
.\.venv\Scripts\python.exe -B tools/probe_real10_foreground_correspondence.py --stage report
```

`--stage check`只验证合成100px跳跃、无信息时null、无种子时不初始化；不证明真实精度。
检查中发现并修复零方差残差块的有限伪NCC峰，最终版本同步/回读后才进行真实推理。
真实候选分数随后经独立float64直接公式只读核对，结果记录在`numerical_score_check.json`；
没有再次执行推理或按真实标签改阈值。5份标注源快照字节仍相同。

若仅传输中断，先检查目标文件长度，只重传缺失/截断文件，不重跑推理。若将来经授权修复
实际实现问题，保留v1，用`--out simulation_output/real10_foreground_correspondence_v2`
新目录（infer/report使用同一个out）；当前入口无断点续跑，不拼接不同版本结果。
本轮已查看6张图版，`report.json`/`review.json`仅更新查看状态，不改预测和评价数值。

## Historical: BootsTAPIR Six-Window Inference Completed (2026-09-22)

用户已下载官方权重，严格state_dict加载通过。6窗推理/评估已完成，**无需再下载、重跑或训练**。
结果在`simulation_output/real10_bootstapir_tracking_v1/result_v1/`：

```text
inference_report.json   # 4090推理完成；循环1.032s，不含权重加载/传输
predictions.jsonl       # 人工未来点不进入模型，候选与输出分开
protocol.json           # 原先冻结协议，未调参
evaluation.jsonl
report.json
README.md
interpretation.md      # 逐窗解释、证据限制与不采用决策
review.json            # 助手已看图；用户结果接受仍pending
index.html
human_004.png / human_039.png / human_062.png
human_065.png / human_077.png / human_100.png
```

9个可评末帧的候选平均点偏差：不动27.915、旧LK27.110、新24.141px。新输出9/9，但没有
拒判任何已初始化序列；不能把覆盖率当成功率。#65 Top偏差1.630px是主要局部改善，#77 Top
仍76.377px且模型分数0.9982。不要把此结果自动当监督或接入世界模型；旧LK与新模型输入/
时间信息口径不同。完整解释见当前handoff和interpretation.md。

以下仅为**已完成命令记录，现有result_v1禁止覆盖**。本轮没有代码修改或新增smoke：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/probe_real10_bootstapir_tracking.py --stage infer"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r project4090:/home/zsw/project_2026/simulation_output/real10_bootstapir_tracking_v1/result_v1 simulation_output/real10_bootstapir_tracking_v1/
.\.venv\Scripts\python.exe -B tools/probe_real10_bootstapir_tracking.py --stage report
```

原始1920×1080图像与evaluation_source仍仅在本机使用；4090推理使用上轮上传的66张原像素
固定裁剪，不是全原图再次上传。5份输入快照字节不变。6张中文图版已查看，未宣称用户已接受。
后续若用户批准新方法，使用独立输出并保留此次失败证据；本记录不授权自动重试/调参/训练。

## Historical: BootsTAPIR Six-Window Adapter — Download Pending (2026-09-22)

入口`tools/probe_real10_bootstapir_tracking.py`；准备目录
`simulation_output/real10_bootstapir_tracking_v1/`。仅离线点追踪诊断，无训练命令。
已完成代码顺序SCP、4090适配检查、本机prepare与固定输入裁剪图查看。**尚无预训练模型
追踪结果**；`--stage check`的随机权重前向只证明接口形状，不能当准确率测试。
官方代码固定revision=`730cda1c730877cfedbe01bf87fb1cadb78a565d`；最小新增依赖
dm-tree0.1.10、einshape1.0已安装，Torch保持原环境版本。

原始图像不需全部上传：输入为首帧查询点确定的512方框，66张RGB无缩放裁剪合成一个
约22.5MB的`inference_inputs/frames.npz`。只同步`inference_inputs/`，未来人工点/响应/
路径参照所在`evaluation_source/`留本机供预测落盘后的评估，不能传给模型。
小输入包上传已完成，远端读取66帧并验证RGB形状与查询坐标通过，无需用户再传图片。

**用户下一步：在4090终端下载约219MB权重**（目标目录已建立，支持断点续传）：

```bash
curl -fL --retry 3 --connect-timeout 20 -C - \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt \
  -o /media/zsw/SSD1T/project_2026_weights_v1/point_tracking/bootstapir_checkpoint_v2.pt
```

期望218886140 bytes；入口检查文件大小并严格加载state_dict，不例行SHA256。
下载源来自[官方示例](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/colabs/torch_tapir_demo.ipynb)。
不要下载causal版本来替换这个权重；本轮明确使用离线版本。

下载完成后的短推理命令（预计1–3分钟，尚未执行；不训练）：

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_real10_bootstapir_tracking.py --stage infer
```

每路输出进度；完成标志为`result_v1/inference_report.json`，只有中间
`predictions.jsonl`不算完成。模型仅看固定RGB和首帧查询，输出包含未门控候选、null拒判、
未标定分数、输出恢复与离线lookahead。不加64px步长门，也不保证突进必能找回。
若中断/失败，保留目录，以新`--result simulation_output/real10_bootstapir_tracking_v1/result_v2`
完整重跑；没有自动调参/训练重试。官方权重加载与真实推理待下载后验证。

助手后续回传小结果并在本机按冻结标注评估（以下尚未执行）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r project4090:/home/zsw/project_2026/simulation_output/real10_bootstapir_tracking_v1/result_v1 simulation_output/real10_bootstapir_tracking_v1/
.\.venv\Scripts\python.exe -B tools/probe_real10_bootstapir_tracking.py --stage report
```

将生成`report.json`、`evaluation.jsonl`、`README.md`、`index.html`和6张中文对照图。
旧LK使用因果全图灰度，新入口使用离线RGB裁剪，因此不是同输入/同时间信息的纯架构消融。
不能把中间帧的离线精度写成实时可用，也不能用原6个开发难例支持泛化/训练就绪结论。

下列prepare已完成，**不要覆盖或重复创建既有work目录**：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/probe_real10_bootstapir_tracking.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/probe_real10_bootstapir_tracking.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/probe_real10_bootstapir_tracking.py --stage check"
.\.venv\Scripts\python.exe -B tools/probe_real10_bootstapir_tracking.py --stage prepare
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r simulation_output/real10_bootstapir_tracking_v1/inference_inputs project4090:/home/zsw/project_2026/simulation_output/real10_bootstapir_tracking_v1/
```

## Historical: Seeded Tracking / Abstention Probe Completed (2026-09-22)

工具`tools/probe_real10_seeded_wire_tracking.py`，结果
`simulation_output/real10_seeded_wire_tracking_v1/`。本轮已跑完，无长任务、无训练命令。
`index.html`和6张`human_*.png`显示人工点、输出点、已拒判的原始候选及不动对照；
`predictions.jsonl`先于离线评估生成，`evaluation.jsonl`/`report.json`保留误差、覆盖率与质量值，
`protocol.json`在预测前冻结参数，`interpretation.md`/`review.json`记录解释与助手查看状态。

只用原首帧visible点初始化，不用中末帧点、路径、任务、响应标签、跨视角或机器人state跟踪。
拒判后output保持null；原始候选只用于暴露误追踪，不能输入policy或作为自动标签。
12路中10路初始化，9个可评末帧输出8个；输出子集平均误差27.591px，同子集不动基线28.388px，
几乎没有实用优势。#77 Top偏差76.285px仍通过质量门；不要按此结果扩标、训练或宣称可见性识别。

已执行顺序如下，**已有输出勿重复覆盖**。先顺序同步再远端检查，真实原图只在本机读取：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/probe_real10_seeded_wire_tracking.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/probe_real10_seeded_wire_tracking.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/probe_real10_seeded_wire_tracking.py --check"
.\.venv\Scripts\python.exe -B tools/probe_real10_seeded_wire_tracking.py
```

远端OpenCV4.12仅做已知平移、空白帧拒判持续、无初值检查，不需要原图或模型权重；
本机OpenCV4.13完成6窗/80帧槽真实图像追踪与绘图4.326s，不能将两者说成真实数据双机复现。
原图无重编码/回传，5份小输入快照未改，原8794页面无需更新，也不用用户补无法可靠定位的点。

后续若获得继续执行授权，使用新`--out`保存新方案；失败输出也保留，不能覆盖v1后只报改好的
结果。本入口不支持自动阈值搜索/自动重初始化/训练，默认拒绝已有输出目录。>100MB单文件或
压缩包仍由用户传；本次只有小代码和诊断结果可直接SCP。可复用点误差和同子集不动基线口径，
但不能把当前6个开发难例当新的独立验证集。

## Historical: Six-Window 2D Direction Diagnostic (2026-09-21)

已完成，只读诊断，不是训练入口。工具`tools/audit_real10_wire_path_motion.py`，结果：

```text
simulation_output/real10_wire_path_motion_diagnostic_v1/index.html
simulation_output/real10_wire_path_motion_diagnostic_v1/reference_overview.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_004.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_039.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_062.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_065.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_077.png
simulation_output/real10_wire_path_motion_diagnostic_v1/human_100.png
simulation_output/real10_wire_path_motion_diagnostic_v1/report.json
simulation_output/real10_wire_path_motion_diagnostic_v1/interpretation.md
simulation_output/real10_wire_path_motion_diagnostic_v1/review.json
```

保留输入快照、初始入口及最终renderer快照；原8794标注页未改、无需刷新个人草稿。
9组可用首末、3组缺失；本机与远端12行数值一致。#65 Top沿向-0.028/横向24.128px，
#100 Top沿向20.040/横向7.046px；#65 Side仍有+9.566沿向，不能用任一视角正值直接判推进。
固定四点不是中心线；仅6个开发难例，不改标签、不用于模型输入/训练、contact或毫米推进。

已执行的完整顺序（**现有输出勿重复全量运行**；约3秒，不需要用户长任务）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/audit_real10_wire_path_motion.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/audit_real10_wire_path_motion.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_real10_wire_path_motion.py --numeric-only --out simulation_output/real10_wire_path_motion_diagnostic_numeric_v1"
.\.venv\Scripts\python.exe -B tools/audit_real10_wire_path_motion.py
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090:/home/zsw/project_2026/simulation_output/real10_wire_path_motion_diagnostic_numeric_v1/report.json simulation_output/real10_wire_path_motion_diagnostic_v1/remote_numeric_report.json
```

远端无需原图/Pillow即可计算；本机`.venv`仅读原图生成7张中文图版，没有模型依赖。
若有后续经批准的计算变更，使用新`--out`；入口拒绝覆盖已有输出，失败输出也先保留。
只修显示时可重绘本工件的冻结输入，不读最新点标、不改report数值：

```powershell
.\.venv\Scripts\python.exe -B tools/audit_real10_wire_path_motion.py --render-only
```

本次已用该命令修正全缺失视角的等比显示，并明确PL/PR及任务；最终代码已再同步并help回读。
用户视觉接受状态仍pending，以`review.json`为后续助手查看记录；`report.json`保留首次生成
时not_viewed状态，不由代码自动写accepted。输入未改检查与数值比对已完成，不必追加无关smoke。
未改旧人工响应标注；不要向8794自动提交、重标缺失、将6窗作为独立验证集或自动启动训练。

## Historical: Static Four-Point Reference + Per-Window W (2026-09-21)

本批已完成用户点标及对话同点确认。2026-09-21代录追加6版本后，summary为33条保存、
9/12同视角首末点对、血管8/8；24张必看帧状态显示20张有坐标，其余按用户说明因无法
可靠定位而保留缺失，不要求补点。不要重复批量确认或为凑进度补零。
确认依据`real10_wire_correspondence_v1/chat_confirmation_20260921_v1.json`；旧27条完整保留。
本轮未改代码/页面、未计算投影、未训练。以下启动/summary命令仍适用。
JSONL与确认记录已上传4090，保留`incoming_confirmation_20260921_v1`同步快照；
不覆盖复制及字节比较通过，远端summary=33条保存/9组对应/8个血管点，不用重复上传。

8794入口已改为四点版并重启，无需重建pack。先在“血管四点”标Side/Top各P0/P1/PL/PR，
保存后切“导丝W”，逐窗标首末；不再逐帧A/B。指南：
`docs/algorithm-real10-wire-correspondence-guide-20260921.md`。当前仅人审工具，无训练命令。

本机服务仍运行时直接打开`http://127.0.0.1:8794`；仅在服务未运行时执行：

```powershell
.\.venv\Scripts\python.exe -B -u tools/review_real10_wire_correspondence.py --port 8794
```

确认页面标题“固定血管四点 · 逐窗导丝对应”。若提示服务旧版，先保存旧页草稿、确认目标
进程是本入口后重启；勿结束未知进程。默认仅绑定127.0.0.1，不连相机或机械臂。

只读进度：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_wire_correspondence.py --summary
```

正式新文件（首次人工保存才创建）：

```text
simulation_output/real10_wire_correspondence_v1/vessel_reference.jsonl
simulation_output/real10_wire_correspondence_v1/annotations_v2.jsonl
```

旧`annotations.jsonl`/manifest/快照保留只读，**不要再次prepare或回传整包覆盖本机文件**。
新版协议`interface_protocol_v2.json`，新版代码/HTML快照后缀`_v2`。完成标注后先核对
本机与远端有无并发新增，再只传这两个小JSONL；原图不用上传。>100MB仍由用户传。

本次已执行代码/HTML顺序SCP→4090临时目录最小保存/兼容检查→本机原图UI检查。
`real10_wire_correspondence_v2_ui_fixture`及其任意点坐标只用于界面测试，不能并入人审、
评估或训练；测试服务8795仅测试时开启，正式用户入口始终为8794。

## Historical: Six-Window Sparse Correspondence Annotation (2026-09-21)

入口与独立索引已完成。用户点标尚未开始；不是训练命令。操作说明：
`docs/algorithm-real10-wire-correspondence-guide-20260921.md`。固定6窗、24张首末必看图，
12张中间帧选填；80个上下文原图在本机，无需上传或重编码。

本机服务已启动，打开`http://127.0.0.1:8794`。若将来服务未运行，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -u tools/review_real10_wire_correspondence.py --port 8794
```

已有服务时复用，不重复启动、不结束不明进程。前台运行时Ctrl+C停止，已保存的追加记录
保留；同一命令可继续。唯一人审写入`real10_wire_correspondence_v1/annotations.jsonl`，
不写旧响应标注或任何训练包；按原图坐标保存，模糊/不可见=null，未标注另记。

只读进度（不会创建标签）：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_wire_correspondence.py --summary
```

以下为已完成的准备记录，**不要对现有包再次prepare**：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/review_real10_wire_correspondence.py tools/real10_wire_correspondence.html project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/review_real10_wire_correspondence.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/review_real10_wire_correspondence.py --prepare"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_wire_correspondence_v1 simulation_output/
```

远端临时目录保存/重载、坐标与缺失值检查已通过；真实包未写测试坐标。页面已做显示、
窗口/帧切换、1:1与播放检查，没有提交人工标签。初始summary为0/24必看图、0/12点对；
后续进度以本机新文件为准，远端不会自动收到标注。

用户完成后先读取本机summary再处理；同步时只传新的小JSONL，先检查远端是否有人并发
追加，不能拿初始空快照覆盖用户点标。不要回传整包覆盖本机annotation。原图不传，
>100MB单文件/压缩包仍由用户传输。没有二维投影计算、训练或现场接入的自动后续动作。

## Historical: Local Supervision Semantic Audit Completed (2026-09-21)

只读审计已完成，无需补跑、没有训练命令。入口`tools/audit_real10_local_supervision_semantics.py`，
输出`simulation_output/real10_local_supervision_semantic_audit_v1`。固定10个旧代表宏请求及
6个人审案例；178帧对，原图提取/绘图16.981s。精确路径、时间和选择规则写入selection/protocol。
结论见handoff与输出summary；没有应用待验收包络或更改标签。

下列为已执行记录，原图仍只在本机，保持既有“脚本先SCP→远端入口回读→本机原图数值审计”流程：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 tools/audit_real10_local_supervision_semantics.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/audit_real10_local_supervision_semantics.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_real10_local_supervision_semantics.py --help"
.\.venv\Scripts\python.exe -B -u tools/audit_real10_local_supervision_semantics.py
```

入口拒绝覆盖已有输出。若不完整，保留原工件，修复/SCP后使用新`--out`；不删除旧标签或权重。
只读来源为`real10_elite_local_response_targets_v1`旧监督/掩膜、
`real10_pre_action_label_retest_v1/annotation_snapshot.jsonl`新标注快照及
`real10_event_windows_v1/observations.jsonl`原图索引。没有新依赖或权重下载。

小工件同步（完成SCP后才回读；单文件最大约1.66MiB，总约26.5MiB）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r simulation_output/real10_local_supervision_semantic_audit_v1 project4090:/home/zsw/project_2026/simulation_output/
```

`index.html`直接引用同目录PNG，不依赖相机/标注服务。无需为了审查重新启动浏览器服务。
`report.json`保留计算快照，后续视觉状态见`visual_review.json`。原图不上传；未来单个文件或
压缩包>100MB仍由用户传输。尚未开始下一步稀疏对应标注、模型训练或策略接入。

## Historical: Local Response Forecast Completed (2026-09-21)

固定175请求/10折的局部响应预测已完成：本机原图提取140.689s、4090拟合/评价1.035s；
20个ridge模型共19.10MB留远端。没有神经网络/PI05训练、额外视觉编码或硬件调用。
入口`tools/run_real10_elite_local_forecast.py`，协议
`docs/algorithm-real10-local-response-forecast-protocol-20260921.md`。本轮无请求增量收益，
不采纳、不搜索配置续跑。下列为**已执行记录**，现有输出不得覆盖，无需用户补跑。

代码/协议先顺序SCP并回读，随后按已有原图本机数值提取流程执行：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/run_real10_elite_local_forecast.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-local-response-forecast-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/run_real10_elite_local_forecast.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_elite_local_forecast.py --help"
.\.venv\Scripts\python.exe -B -u tools/run_real10_elite_local_forecast.py --stage prepare
```

默认源`real10_elite_request_response_pack_v1`，原图`collected_data/<episode>/frames/`；直接
读取`real10_response_local_motion_v1/preview_arrays.npz`的旧掩膜及对应protocol，不需要
访问待验收的`real10_target_vessel_region_review_v1`，也不读取45窗人工标签。
生成`real10_elite_local_response_targets_v1`数值包和10窗双视角预览，约10.7MiB、最大单
文件约5.54MiB；原图未复制。程序拒绝覆盖已有`--targets`。

数值包完整上传后才开始远端拟合（不要在scp尚未退出时读取/执行）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -r simulation_output/real10_elite_local_response_targets_v1 project4090:/home/zsw/project_2026/simulation_output/
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_real10_elite_local_forecast.py --stage evaluate"
```

远端默认复用`real10_elite_request_forecast_v1/features.npz`及20个旧fold文件中的X归一化
参数，不使用旧输出系数。72维新监督按有效训练行拟合；缺目标不填零。输出
`real10_elite_local_forecast_v1`，包含4组OOF、20模型、覆盖/分层/逐事件/配对指标。
程序拒绝覆盖已有`--out`；两端都无需安装新库或下载新权重。

小结果回读和本机重绘（render不重新提取监督、拟合或调用模型）：

```powershell
$out = 'simulation_output/real10_elite_local_forecast_v1'
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($name in @('report.json','protocol.json','prepared.json','fold_log.json','folds.json','sample_groups.jsonl','oof_event_metrics.jsonl')) {
  scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/${out}/$name" "$out/"
  if ($LASTEXITCODE -ne 0) { throw "Readback failed: $name" }
}
.\.venv\Scripts\python.exe -B tools/run_real10_elite_local_forecast.py --stage render
```

中文图为`figures/local_forecast.png/.svg`；监督包内`figures/local_target_preview_left.png`
及`...right.png`是10个固定代表窗，蓝标是网格中心上的平均向量，不是尖端检测。预览说明
文字与data manifest已完善，未重新提取或改变数值目标；原prepare入口快照保留计算版本，
evaluate入口快照为最终版本。视觉状态均viewed_not_accepted。

失败保留原目录/错误，修复并同步后使用明确新`--targets`/`--out`；不删除旧监督、标签或
权重。当前没有后续训练命令；超过5min的计算仍交用户，单文件/压缩包>100MB仍由用户传输。

## Historical: Elite Request Forecast Completed (2026-09-21)

4090已完成固定175事件、10折、三组未来预测对照，主体6.190s，无需补跑。冻结ResNet18
编码966张图，20次闭式ridge拟合；没有神经网络/PI05训练或硬件调用。入口：
`tools/run_real10_elite_request_forecast.py`；预先固定协议：
`docs/algorithm-real10-elite-request-forecast-protocol-20260921.md`。结果以handoff与
`simulation_output/real10_elite_request_forecast_v1/report.json`为准。

以下为**已执行记录**。该输出已存在，禁止用run覆盖；无alpha/clip/seed搜索入口：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/run_real10_elite_request_forecast.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-elite-request-forecast-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 6 tools/run_real10_elite_request_forecast.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_elite_request_forecast.py --help"
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_real10_elite_request_forecast.py --out simulation_output/real10_elite_request_forecast_v1"
```

默认只读`real10_elite_request_response_pack_v1`，复用其中完整样本/folds。仅使用既有
`resnet18-f37072fd.pth`缓存，无自动下载；输入历史和请求从白名单读取，未来末帧另作监督。
生成固定protocol、源manifest/folds/输入/末帧快照、features与冻结embedding NPZ、20个
fold模型、全维OOF预测NPZ、逐事件指标JSONL和report。模型总75.84MB留远端，不需传图。

小工件回读已完成。等待scp退出后再打开目标文件，避免读取正在传输的JSON：

```powershell
$out = 'simulation_output/real10_elite_request_forecast_v1'
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($name in @('report.json','protocol.json','folds.json','fold_log.json','oof_event_metrics.jsonl')) {
  scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/${out}/$name" "$out/"
  if ($LASTEXITCODE -ne 0) { throw "Readback failed: $name" }
}
```

中文图在本机生成，**render不会拟合或编码图像**，可从已回读report重绘：

```powershell
.\.venv\Scripts\python.exe -B tools/run_real10_elite_request_forecast.py --stage render --out simulation_output/real10_elite_request_forecast_v1
```

输出`figures/forecast_comparison.png`（450dpi）、`.svg`、`data-manifest.md`、视觉review。
默认字体`C:/Windows/Fonts/msyh.ttc`，只展示真实固定对照指标；不是未来图像生成器。
agent已查看，用户验收尚未进行。没有新现场或长期训练命令；不采用本评分器。

失败时保留原输出和终端错误，修复、同步后指定新`--out`，不得清理旧175包/45窗标签/模型。
本短诊断无需resume；不得为获得更好指标反复改配置重跑。未来>5min计算交用户运行，单文件
或压缩包>100MB仍由用户传输。

## Historical: Elite Request Response Pack Built (2026-09-21)

4090已完成175请求的监督包，主体0.431s，无需补跑；175个源配对/输入回读、10个episode
隔离检查、缺失填充和整episode覆盖检查通过。3424个图像引用存在，只查路径，不重编码/
训练/加载模型/调用硬件。说明：`docs/algorithm-real10-elite-request-pack-20260921.md`。

以下是**已执行记录**；当前输出已存在，不能覆盖或直接重跑：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/prepare_real10_elite_request_pack.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/prepare_real10_elite_request_pack.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/prepare_real10_elite_request_pack.py --out simulation_output/real10_elite_request_response_pack_v1"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_elite_request_response_pack_v1 simulation_output/
```

默认只读源：`real10_elite_command_timing_audit_v1`、`real10_event_source_metadata_v1`、
图像包`real10_pi05_compat_v1`及`real10_pre_action_label_retest_v1/folds.json`；均不改写。
复用既有pre_action_input/state encoder，未新增依赖或大文件传输。各JSON/JSONL均小于1MB；
未来如果有单文件或压缩包超过100MB，仍由用户传输。

后续算法脚本可显式使用下列读取函数（接口示例，不是训练命令）：

```python
from prepare_real10_elite_request_pack import load_model_inputs
inputs = load_model_inputs("simulation_output/real10_elite_request_response_pack_v1")
observation = inputs[0]["observation"]
macro_request = inputs[0]["request"]
```

模块位于`tools/`，从该目录下的算法脚本导入，并以项目根目录为工作目录解析图像相对路径。
函数只读取manifest、observations、requests，不打开未来targets、时序audit或folds。
其请求为绝对目标宏动作，**禁止直接传给旧canonical action9/world API或现场控制器**。
未来目标另在response_targets中读取，只用于监督；人工响应标签均null。没有训练入口或自动下载。

失败保留工件，修复/SCP后选新`--out`；当前短构建无需resume，不删除旧审计/标注/实验。
当前没有新的完整训练命令；下一步是预先固定未来响应预测对照，不在175窗上进行超参搜索。

## Historical: Elite Command Timing Audit Completed (2026-09-21)

已在4090标准库CPU完成，主体0.210s，无需补跑。原10条轨迹2729行，188次键盘事件中
175次新绝对目标唯一配对、8次busy skip、5次终点空请求。175次宏动作不是PI05逐步delta；
本轮没有训练、图片解码或硬件调用。结论：
`docs/algorithm-real10-elite-command-timing-audit-20260921.md`。

最终脚本及10份小CSV已顺序上传；CSV只复制到独立审计输入目录，不改原数据（已执行）：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && mkdir simulation_output/real10_elite_command_timing_inputs_v1"
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/audit_real10_elite_command_timing.py project4090:/home/zsw/project_2026/tools/
$episodes=(Get-Content -Raw simulation_output/real10_event_windows_v1/manifest.json | ConvertFrom-Json).source_episodes
foreach ($ep in $episodes) {
  scp -O -o BatchMode=yes -o ConnectTimeout=12 "collected_data/$ep/logs/controller_commands.csv" "project4090:/home/zsw/project_2026/simulation_output/real10_elite_command_timing_inputs_v1/${ep}_controller_commands.csv"
  if ($LASTEXITCODE -ne 0) { throw "CSV upload failed: $ep" }
}
```

入口首5行回读后执行以下短审计；这里只是**已完成复现记录**，输出已存在不得覆盖：

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  tools/audit_real10_elite_command_timing.py \
  --csv-root simulation_output/real10_elite_command_timing_inputs_v1 \
  --sdk-move-source /home/zsw/miniconda3/envs/sam3/lib/python3.12/site-packages/elite/_move.py \
  --out simulation_output/real10_elite_command_timing_audit_v1
```

SDK路径仅作为文本/AST读取，**不是启动sam3、加载机器人SDK或连接机械臂**。默认源为
`real10_event_source_metadata_v1`，allowlist为`real10_event_windows_v1/manifest.json`，
旧窗范围取新标签retest的snapshot，只比对frame IDs，不复制标签。当前`path/`只作佐证，
不是采集时冻结文件。脚本不提供训练、动作执行、改标签或自动delta转换开关。

小工件回读已完成：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_elite_command_timing_audit_v1 simulation_output/
```

使用目录名回读；本轮`scp -O`对源目录`/.`后缀报`unexpected filename: .`，改为上述普通
目录路径后成功，不需要关闭SSH文件名检查。report/188 events/175 pairs/source_evidence
均保留；不需要复制图像。失败保留现有工件，修复/SCP后指定新审计输出目录，无长训练resume。
当前没有新world训练命令；下一步是单独Elite请求中心监督接口设计/适配，不直接接9维world API。

## Historical: Human-Label Retest Completed (2026-09-21)

4090已完成固定45窗、10折的三阶段核对，主体0.212 s，新20次闭式拟合及保存0.0866 s，
**无需用户补跑**。新标签重拟合后BA=50.50%/53.50%，2窗改对/1窗改错；不采用评分器，
不调参续跑。0张图重编码、无神经模型加载/PI05/world训练/硬件。协议：
`docs/algorithm-real10-label-retest-protocol-20260921.md`。

最终小文件顺序上传及必要检查（已执行）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/retest_real10_pre_action_response.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-label-retest-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/retest_real10_pre_action_response.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/retest_real10_pre_action_response.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/retest_real10_pre_action_response.py --help"
```

以下仅为**已完成复现记录**；输出已存在，不能覆盖或再次裸跑：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/retest_real10_pre_action_response.py \
  --out simulation_output/real10_pre_action_label_retest_v1
```

默认reference=`real10_pre_action_response_v1`，revisions为
`real10_hardcase_human_review_20260921_v2/applied`，pack为`real10_event_windows_v1`。
现有缓存约757KiB，20个旧折模型必须齐全；新snapshot须与当前人审源逐项一致，仅批准
#100/#62响应改动。原矩阵顺序/旧标签/折模型score回放、两组请求槽及所有旧指标都核对。
仅使用NumPy CPU，无自动下载或新特征。不是通过关闭旧入口snapshot检查来重跑。

新输出保存重拟合折模型及特征，旧结果只读。失败保留目录，修复/SCP后用明确新目录，
无长训练resume；不要删已完成工件或移除当前人审修订。小文件回读与本机绘图（已执行）：

```powershell
New-Item -ItemType Directory -Path simulation_output/real10_pre_action_label_retest_v1 -Force | Out-Null
scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/simulation_output/real10_pre_action_label_retest_v1/*.json" simulation_output/real10_pre_action_label_retest_v1/
scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/simulation_output/real10_pre_action_label_retest_v1/*.jsonl" simulation_output/real10_pre_action_label_retest_v1/
.\.venv\Scripts\python.exe -B tools/retest_real10_pre_action_response.py --stage render --out simulation_output/real10_pre_action_label_retest_v1
```

中文`figures/label_retest_zh.{png,svg}`、data-manifest、summary及review_status已保存/同步。
图已agent查看，`viewed_not_accepted`；render只读report，不拟合。源代码未改旧模型逻辑，
新入口没有参数/阈值搜索开关。当前没有新world训练或硬件命令。

## Historical: #62 / #77 Human Revisions Appended — No Training (2026-09-21)

已完成第二批：#62改stationary，#77保留advance并追加窗口级可见性备注。原54条保持，
现56条对应45窗（25 advance /20 stationary）；原始记录、动作标签和旧实验不变。
复用上轮入口，未改代码、未重训或重算指标。以下为**已执行记录，勿重复应用**：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && wc -l simulation_output/real10_event_windows_v1/annotations_joint_v2.jsonl && head -n 3 tools/apply_real10_human_review.py && mkdir -p simulation_output/real10_hardcase_human_review_20260921_v2"
scp -O -o BatchMode=yes -o ConnectTimeout=12 simulation_output/real10_hardcase_human_review_20260921_v2/user_feedback.json project4090:/home/zsw/project_2026/simulation_output/real10_hardcase_human_review_20260921_v2/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/apply_real10_human_review.py --batch simulation_output/real10_hardcase_human_review_20260921_v2/user_feedback.json --out simulation_output/real10_hardcase_human_review_20260921_v2/applied --apply"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_hardcase_human_review_20260921_v2/applied simulation_output/real10_hardcase_human_review_20260921_v2/
```

已核对本机源记录与`applied/annotations_before.jsonl`全部54条一致，才回读远端源文件；
再核对56条为旧前缀＋`appended_revisions.jsonl`的2条。并发冲突或部分失败的恢复原则
与下述首批相同：保留工件，检查实际新增记录，不盲目重跑/覆盖。
优先10窗现已复核完成，无需再请求#62/#77标注。旧v1页/报告仍是原快照，不重生成或
静默修补；新标签下若获准重评，需新监督快照和新输出目录。当前无新增训练命令。

## Historical: Eight Explicit Human Revisions Appended — No Training (2026-09-21)

本轮已完成：#100改advance/证据top，其余7窗保持原响应；#65补充下移但无沿管推进，
#4补充小幅推进/尖端模糊。“不变”按保持原标签解释。全部锚点可见性不改。源追加记录
46→54，45个最新窗口为26 advance /19 stationary；原前46条逐行核对保持，未重训/重评。

以下仅为**已执行记录，勿重复应用**：入口会拒绝与原难例快照不一致的当前标注及已有
输出目录，不应通过删目录/回滚标签绕过。只用标准库和既有 `ReviewStore.save`，不读图、
不启动服务器或硬件。批次精确保留用户原文及编号→window_id映射。

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/apply_real10_human_review.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "mkdir -p /home/zsw/project_2026/simulation_output/real10_hardcase_human_review_20260921_v1"
scp -O -o BatchMode=yes -o ConnectTimeout=12 simulation_output/real10_hardcase_human_review_20260921_v1/user_feedback.json project4090:/home/zsw/project_2026/simulation_output/real10_hardcase_human_review_20260921_v1/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 3 tools/apply_real10_human_review.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/apply_real10_human_review.py --batch simulation_output/real10_hardcase_human_review_20260921_v1/user_feedback.json --out simulation_output/real10_hardcase_human_review_20260921_v1/applied --apply"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_hardcase_human_review_20260921_v1/applied simulation_output/real10_hardcase_human_review_20260921_v1/
```

`applied/annotations_before.jsonl`保留原46条，`appended_revisions.jsonl`为8条新增记录，
`annotation_snapshot.jsonl`为最新45窗，`report.json`记录字段变化/计数。已先核对本机
源标注与before一致，再回读远端源标注，并核对54条为旧前缀＋新增修订；有并发差异时
应停止覆盖并人工合并，不直接将任一旧副本覆盖另一端。若中途失败，保留输出/源尾部，
仅检查尚未应用的决定，不盲目重跑。

未来显式新批次可以省略`--apply`只读检查，但仍须提供其对应的最新审阅快照；该入口
不从模型分数推导标签。现有难例页、动作接口和训练v1仍是旧标签快照，**没有重新生成**。
修订后直接复用旧训练入口会被snapshot一致性检查拦截，不要关闭检查；获准重评后另建
新版监督/输出。当前无需用户运行命令，#62/#77尚待本轮人工复核。

## Historical: Pre-Action Piper Request Comparison Completed (2026-09-21)

4090已完成全部45窗、原10折的两组ridge诊断，运行函数1.87 s，**无需用户补跑**。
仅动作前信息 vs 加Piper请求：balanced accuracy=54.50% /50.50%，0窗改对、2窗改错。
不采用为动作评分器、不调参重跑；未加载或训练PI05/world model，现场入口/权重不变。
协议：`docs/algorithm-real10-pre-action-response-protocol-20260921.md`。

最终文件先顺序SCP并回读检查，本轮已执行：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/run_real10_pre_action_response.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-pre-action-response-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 6 tools/run_real10_pre_action_response.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/run_real10_pre_action_response.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_pre_action_response.py --help"
```

以下为**已完成的复现记录**；v1目录已存在，禁止覆盖或再次裸跑：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_pre_action_response.py --stage run \
  --source simulation_output/real10_action_effect_interface_v1 \
  --out simulation_output/real10_pre_action_response_v1
```

只读已有接口与当前人工v2修订，ResNet18必须已有缓存，无自动下载。固定alpha=1、
clip5、阈值0.5、seed20260920和两组[12,2048,2]块，不提供调参开关。只编码历史
首帧/锚点178张图，不编码response_targets里的未来图片。所有10折模型保存，非单一
全量部署权重；既有world API不调用，未知Elite candidate仍null。

结果已回读并本机绘图；特征NPZ与20个fold模型保留远端，不需搬运或重复拟合：

```powershell
New-Item -ItemType Directory -Path simulation_output/real10_pre_action_response_v1 -Force | Out-Null
scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/simulation_output/real10_pre_action_response_v1/*.json" simulation_output/real10_pre_action_response_v1/
scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/simulation_output/real10_pre_action_response_v1/*.jsonl" simulation_output/real10_pre_action_response_v1/
.\.venv\Scripts\python.exe -B tools/run_real10_pre_action_response.py --stage render --out simulation_output/real10_pre_action_response_v1
```

`--stage render`只读已完成report，不加载神经模型/重新训练；本机中文字体微软雅黑，
输出`figures/pre_action_response_zh.{png,svg}`（450dpi）和data-manifest。图已查看，
`review_status.json`记录`viewed_not_accepted`，不是用户接受或模型有效性证明。

未来修改展示只重跑render；若计算首次失败则保留故障目录、修复并SCP后使用明确新目录，
短闭式拟合无resume需求。不用已完成v1做无授权超参数/阈值搜索；当前没有world联合
训练、现场部署或硬件命令。本轮只同步小文件，无>100MB传输或例行SHA检查。

### 本次难例人工复核（只读，未改标）

用户要求先检查标注。以下构建已完成；只生成清单/离线HTML，不重新拟合模型，拒绝覆盖
已存在review目录：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/build_real10_pre_action_hardcases.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 5 tools/build_real10_pre_action_hardcases.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/build_real10_pre_action_hardcases.py"
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_pre_action_response_v1/hardcase_review simulation_output/real10_pre_action_response_v1/
```

本机文件：`simulation_output/real10_pre_action_response_v1/hardcase_review/index.html`，
文字清单`summary.md`。离线页内嵌完整cases，无fetch、保存标注接口或服务器依赖；相对
引用本机`collected_data/`原图，914个引用全部存在，勿单独移动HTML或误以为只需远端
224px图片。默认优先8窗，可切换全部22难例、2新增错例、全部45窗。原标签/模型结果
默认隐藏，先看原始双视角再展开；页内显示的是本次实验标注快照，不是动态重读标注。

自动预览服务启动和内置浏览器file URL分别被环境策略拒绝，**没有启动/打开成功**，
也未绕过；GUI播放/交互尚未验证，见`hardcase_review/validation.json`。请用户从本机
打开离线页或先阅读文字清单。此处不提供自动改标签/重训命令；用户确认图像证据后再
单独处理修订并保留旧实验。统计图`review_status.json`的viewed状态不适用于此新页面。

## Historical: Pre-Action Response Interface Completed — No Training (2026-09-21)

4090接口构建已完成，主体0.32 s，**无需用户补跑**。45窗、原10个整episode folds及
914个唯一图像引用保持；输入/监督/审计已拆分，无训练或模型/硬件调用。
协议：`docs/algorithm-real10-action-effect-interface-20260921.md`。
Elite候选保持null/无效，不能用源BC下一帧位姿差填充；原9维world入口不可直接消费这些
不完整候选。原现场模型和state_32不变，暂不提供world联合训练命令。

小文件顺序上传，回读及短检查（本轮已完成）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/prepare_real10_action_effect_interface.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-action-effect-interface-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && head -n 8 tools/prepare_real10_action_effect_interface.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/prepare_real10_action_effect_interface.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/prepare_real10_action_effect_interface.py --help"
```

以下为**已完成的复现记录**；v1已存在，入口拒绝覆盖，不要重复裸跑：

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_real10_action_effect_interface.py \
  --pack simulation_output/real10_event_windows_v1 \
  --reference simulation_output/real10_response_baseline_v1 \
  --source-root simulation_output/real10_event_source_metadata_v1 \
  --image-pack simulation_output/real10_pi05_compat_v1 \
  --out simulation_output/real10_action_effect_interface_v1
```

`--source-root`为远端已有元数据副本，不能误用本机不存在的同名目录；本机原始记录在
`collected_data/`。本机仅回读结果，没有执行适配构建或依赖本机LeRobot。
复用已有`prepare_real10_event_windows`读写/观测适配和`prepare_real_pi05_pack`状态编码。
不重新采集、转换/搬运原图或读取diagnostic_targets。全部新输出合计约650 KiB。

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_action_effect_interface_v1 simulation_output/
```

源45份人工snapshot若被修订则中止，不静默使用旧标签；构建失败时尚无输出目录可在
修正并SCP后重跑，若已存在则保留故障目录、用明确新目录，不能删除/覆盖完成的v1。
首次`user_event=null`读取问题已修复并成功运行，未生成故障目录。无训练checkpoint或
resume开关；没有例行SHA或新smoke套件，必要检查集成于这次短构建。

工件包括queries/response_targets/timing_audit/annotation_snapshot四个JSONL及
folds/protocol/report三个JSON。后续仅`queries.jsonl/model_input`为预测输入：IDs、
实际响应时长、可见性人工标签、后续请求及异步执行结果不作特征，未来图像只能用于监督。
完整动作的`complete_action9`拒绝45/45缺失Elite候选是预期结果，不是丢弃45条数据。
下一步的请求条件响应预测尚未训练，也没有counterfactual/contact/策略收益证据。

## Historical: Parameter-Matched Motion Encoding Pair Completed (2026-09-21)

4090已完成两组各100步，完整92.50 s，**无需用户补跑**。同初始化/参数量/顺序/noise/
优化器/旧post-cast接口，唯一变化为绝对历史→相邻变化＋当前内容＋时间间隔。两组均
30/30零门控恒等，旧组100步过程及最终权重/预测逐值复现。新组MAE=0.230682 mm，
仍未优于原PI05；不部署、不继续调参或加世界模型loss。
协议：`docs/algorithm-real10-motion-prefix-protocol-20260921.md`。

最终小文件顺序上传并远端回读/检查：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/pi05_motion_temporal_world.py tools/run_real10_pi05_motion_comparison.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-motion-prefix-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/pi05_motion_temporal_world.py tools/run_real10_pi05_motion_comparison.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_pi05_motion_comparison.py --help"
```

以下仅为**已完成的复现记录**；v1目录已存在，禁止覆盖或重复裸跑：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_pi05_motion_comparison.py \
  --out simulation_output/real10_pi05_motion_comparison_v1
```

默认reference=`real10_pi05_gated_action_v1`：读取其协议和固定训练日志/预测/step100权重
以复现控制组；初始化来自协议所指`real10_pi05_joint_comparison_v1/paired_checkpoint.pt`
的**untrained initial_state**，不是该文件的已训练分支。不要清理这些源工件。
默认仍固定各100步；不提供步数、学习率、gate或representation搜索开关。

恢复仅用于尚无completed report的中断：确认旧进程结束后，在同一命令后加`--resume`，
读取同目录`paired_checkpoint.pt`，协议必须一致。恢复点0/25/50/75/100，先重新核实
原模型/零门控再加载两组model/optimizer/log；没有人为中断验证完整恢复过程。若首次
checkpoint前失败，保留故障目录并在修复/SCP后用明确新目录，不覆盖已完成结果。

小JSON已回读，约41MiB恢复文件和两份约11MiB适配器留远端。本机绘图不加载模型：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/render_real10_pi05_motion_comparison.py project4090:/home/zsw/project_2026/tools/
New-Item -ItemType Directory -Path simulation_output/real10_pi05_motion_comparison_v1 -Force | Out-Null
scp -O -o BatchMode=yes -o ConnectTimeout=12 "project4090:/home/zsw/project_2026/simulation_output/real10_pi05_motion_comparison_v1/*.json" simulation_output/real10_pi05_motion_comparison_v1/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_motion_comparison.py simulation_output/real10_pi05_motion_comparison_v1/report.json
```

中文`figures/motion_encoding_comparison_zh.png`/SVG与data-manifest已查看，
`viewed_not_accepted`。本轮没有大文件传输、原权重迁移或现场改接。新metadata version
`pi05_gated_adjacent_change_v1`必须对应`MotionGatedTemporalWorld`，不可因参数键名
相同而用旧类加载；当前只由本配对实验入口显式使用，没有部署命令。

## Historical: Frozen Pre-Cast / Post-Cast Comparison Completed (2026-09-21)

4090已完成固定前向对照，70.08 s，**无需用户补跑**。0步训练，原权重未更改；新入口是
显式可选hook，未接入默认训练/现场代码。零门控30/30逐值恒等；新旧MAE为0.230531/
0.230697 mm，原PI05为0.229796 mm；交换历史仍30/30动作不变，不部署、不续训练。
协议：`docs/algorithm-real10-gated-precast-protocol-20260921.md`。

最终小文件顺序上传并在远端回读/检查后执行：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/pi05_gated_temporal_world.py tools/audit_real10_pi05_gated_precast.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 docs/algorithm-real10-gated-precast-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/pi05_gated_temporal_world.py tools/audit_real10_pi05_gated_precast.py && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_real10_pi05_gated_precast.py --help"
```

以下是**已完成的复现记录**；v1目录已存在，入口拒绝覆盖，不需再执行：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_real10_pi05_gated_precast.py \
  --out simulation_output/real10_pi05_gated_precast_audit_v1
```

默认source为`real10_pi05_gated_action_v1`，reference为已完成的
`real10_pi05_gated_history_audit_v1`。后者report、measurements、captured_vectors用于
逐值复现检查；两者及所指原PI05/Piper/pack均是受保护恢复输入。参数不提供gate/精度搜索，
不调用optimizer/backward或保存模型。新路径要求真实state投影为FP32，否则中止。
每10窗保存partial_measurements。失败时保留故障目录，确认旧进程已结束再指定明确新目录
重跑这次短前向工作；无训练resume，不覆盖已完成结果或旧权重。

本轮约5.9MiB NPZ及小JSON已回传，本机仅做数值回读与绘图；没有大文件转移：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=12 tools/render_real10_pi05_gated_precast.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=12 -r project4090:/home/zsw/project_2026/simulation_output/real10_pi05_gated_precast_audit_v1 simulation_output/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_gated_precast.py simulation_output/real10_pi05_gated_precast_audit_v1/report.json
```

中文图`figures/gated_precast_comparison_zh.png`/SVG与data-manifest已查看，
`viewed_not_accepted`。图中数值变化不等同信息保留或动作收益。展示修改只重跑renderer；
当前没有新训练/世界模型/硬件执行命令，显式运动变化表征仅为下一步候选。

## Historical: Frozen Gated Precision / History Audit Completed (2026-09-21)

4090已完成只读审计，48.28 s，**无需用户补跑**。0步训练，没有修改精度、连接或源权重。
协议：`docs/algorithm-real10-gated-history-audit-protocol-20260921.md`。具体结果见handoff：
加法后仅7.26074%坐标变化；交换两个过去内容后30/30个xyz动作不变。

最终小文件先传后执行，生产门控/训练代码不动：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/audit_real10_pi05_gated_history.py tools/render_real10_pi05_gated_history_audit.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=10 docs/algorithm-real10-gated-history-audit-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/audit_real10_pi05_gated_history.py tools/render_real10_pi05_gated_history_audit.py"
```

**已完成的复现记录**，当前v1目录存在，入口拒绝覆盖：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_real10_pi05_gated_history.py \
  --out simulation_output/real10_pi05_gated_history_audit_v1
```

source默认`real10_pi05_gated_action_v1`：只读report、panel_predictions及固定step100
`gated_action_adapter.pt`，从源协议定位原pack与PI05/Piper。当前noise/输入保持不变，
不调用optimizer或backward，不更换现有BF16计算；真实预测必须逐值复现历史记录。
每10窗保存partial_measurements，中断时保留故障输出，确认旧进程结束后以明确的新目录
重新做这次短只读检查；没有训练resume，不删除/覆盖已完成审计或源权重。

整个审计输出仅小JSON和约2.76MB数值NPZ，已全部回读；本机只读绘图：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 -r project4090:/home/zsw/project_2026/simulation_output/real10_pi05_gated_history_audit_v1 simulation_output/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_gated_history_audit.py simulation_output/real10_pi05_gated_history_audit_v1/report.json
```

`captured_vectors.npz`的15数组可直接用于后续离线数值分析；它不是新增policy observation
或训练集。中文图`figures/gated_precision_history_zh.png`（450dpi）/SVG及data-manifest已
检查，`viewed_not_accepted`。渲染不依赖LeRobot，不需重做模型推理。当前没有新的训练、
精度修正或硬件执行命令；“转型前相加”仅为下一步提议。

## Historical: Gated State Residual — Identity + Action-Only 100 Steps Completed (2026-09-21)

本轮已在4090完成，总55.68 s，**不需要用户补跑**。没有世界模型辅助loss或实机执行。
30/30初始输出/动作loss逐值复现原模型；100步后MAE略高0.3919%，因此不部署、不直接续联合训练。
协议：`docs/algorithm-real10-gated-prefix-protocol-20260921.md`。

最终小文件先传后运行，旧v1配对代码和工件不变：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/pi05_gated_temporal_world.py tools/run_real10_pi05_gated_action.py tools/render_real10_pi05_gated_action.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=10 docs/algorithm-real10-gated-prefix-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/pi05_gated_temporal_world.py tools/run_real10_pi05_gated_action.py tools/render_real10_pi05_gated_action.py"
```

以下为**已完成的复现记录**，当前目录存在，勿裸跑重训：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_pi05_gated_action.py \
  --out simulation_output/real10_pi05_gated_action_v1
```

参数不开放步数/学习率搜索。默认reference为`real10_pi05_joint_comparison_v1`，读取其
report/protocol、panel_predictions及paired_checkpoint中的**initial_state**，不是其最终参数。
沿用100步顺序、30窗面板和seed123。若初始一致性失败，立即停在更新前，保留failure记录。

每25步及0步写`checkpoint.pt`（约15MiB）。恢复只用于没有completed report的中断：
确认旧进程结束后，在同一命令后加`--resume`；reference和协议必须不变。启动会重新做
原模型/零门控一致性检查，再加载已保存的模型/optimizer/log继续。最终save/reload已验证，
但未人为中断验证完整resume。没有checkpoint时保留故障、修复并SCP后用新目录；不得覆盖
已完成v1，也不得清理其依赖的前轮初始化工件。没有已授权的后续训练或硬件命令。

小JSON已回读，checkpoint保留远端；本机只读渲染不依赖LeRobot：

```powershell
New-Item -ItemType Directory -Path simulation_output/real10_pi05_gated_action_v1 -Force | Out-Null
scp -O -o BatchMode=yes -o ConnectTimeout=10 "project4090:/home/zsw/project_2026/simulation_output/real10_pi05_gated_action_v1/*.json" simulation_output/real10_pi05_gated_action_v1/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_gated_action.py simulation_output/real10_pi05_gated_action_v1/report.json
```

`figures/gated_action_diagnostic_zh.png`（450dpi）、SVG和data-manifest展示动作MAE、门控
轨迹和每轨迹配对差；灰色旧追加式结果是已冻结历史参照。图已查看，`viewed_not_accepted`。
展示问题只重跑render，不重新训练；现场入口和原权重继续保留。

## Historical: Calibrated Joint Prefix — Matched 100-Step Comparison Completed (2026-09-21)

本节已由agent在4090完成（总69.82 s），**无需用户重新执行**；不启动完整训练或实机。
两组各100步、seed123；当前结果不支持采用联合适配器。完整协议和数值见
`docs/algorithm-real10-joint-prefix-protocol-20260921.md` 与算法handoff当前节。

最终代码先上传后执行；同名旧连通检查只新增可选flow_time参数，默认0.5不变：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/check_real10_pi05_joint_temporal.py tools/run_real10_pi05_joint_comparison.py tools/render_real10_pi05_joint_comparison.py project4090:/home/zsw/project_2026/tools/
scp -O -o BatchMode=yes -o ConnectTimeout=10 docs/algorithm-real10-joint-prefix-protocol-20260921.md project4090:/home/zsw/project_2026/docs/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/check_real10_pi05_joint_temporal.py tools/run_real10_pi05_joint_comparison.py tools/render_real10_pi05_joint_comparison.py"
```

**已完成的单次复现命令**（v1目录现已存在，入口会拒绝重复执行）：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_pi05_joint_comparison.py \
  --out simulation_output/real10_pi05_joint_comparison_v1 --steps 100 --seed 123
```

入口先记录全部2719索引及30上下文/100步顺序，再按固定规则校准；两组共用一次冻结模型
加载和冻结图像特征。每25步保存同一完整配对步的`paired_checkpoint.pt`，同时保存完整
日志于checkpoint；最终输出两个独立`deployable=false`适配器、report与逐上下文预测。
实机入口仍指向原real10权重，绝不指向这些新小checkpoint。

恢复规则（实现已具备；本次没有人为中断以测试完整恢复流程）：

已完成只读checkpoint回载：最终两组张量与paired_checkpoint逐值一致；optimizer分别
16/22个状态均为step100。该检查不等同于已执行中途恢复训练。

- 先确认旧进程已经结束，检查`report.json`和失败记录；有completed report就停止，不重跑。
- 仅当有`paired_checkpoint.pt`而没有完成report时，在**同一原命令**后加`--resume`。
  pack/training_root/seed/steps必须不变；两组共同回到最新完整配对步（包括0步校准点）。
  逐步随机数由seed与step确定，校准参数从checkpoint加载，不重新调权。
- 未形成配对checkpoint的首次故障不能`--resume`。保留故障目录，修复代码并SCP后，
  使用明确新目录；不要删除或覆盖v1。`--calibrate-only`可只写0步校准点，当前无需再跑。
- 更大步数/多seed是新实验，不能通过改变`--steps`把本次结果续成另一项协议。

只回读小JSON，不下载57MiB配对checkpoint或两份10.35MiB适配器（它们安全保留在4090）：

```powershell
New-Item -ItemType Directory -Path simulation_output/real10_pi05_joint_comparison_v1 -Force | Out-Null
scp -O -o BatchMode=yes -o ConnectTimeout=10 "project4090:/home/zsw/project_2026/simulation_output/real10_pi05_joint_comparison_v1/*.json" simulation_output/real10_pi05_joint_comparison_v1/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_joint_comparison.py simulation_output/real10_pi05_joint_comparison_v1/report.json
```

只读绘图不依赖LeRobot；沿用Windows微软雅黑。PNG450dpi / SVG / data-manifest在
工件的`figures/`，图已查看，`viewed_not_accepted`；它不是用户验收或独立性能评估。
图显示初始梯度尺度、四条件训练内MAE、10条轨迹的配对差，以及persistence参照。
展示问题只重跑render，不重训。没有新的用户运行任务，也没有采用任何新权重。

## Historical: Joint PI05 Temporal Prefix / World Prediction Check (2026-09-21)

真实权重连通检查已在4090完成，v1r2总31.05秒，非完整训练或效果评估。原 PI05、Piper
权重全部冻结；新增适配器只做一次临时 future-only 更新（随后还原）和一次联合诊断更新。
共享梯度连通，但当前单上下文辅助梯度尺度过弱；**暂不使用固定0.1直接启动完整训练**。
结果及下一步尺度校准范围见算法 handoff 的当前节。

最终代码按顺序同步；不是将大权重下载到本机，也不需要重新传输图片：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/pi05_joint_temporal_world.py tools/check_real10_pi05_joint_temporal.py tools/render_real10_pi05_joint_check.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m py_compile tools/pi05_joint_temporal_world.py tools/check_real10_pi05_joint_temporal.py tools/render_real10_pi05_joint_check.py"
```

以下为 **已完成的复现命令**，无需用户再跑：

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_real10_pi05_joint_temporal.py \
  --out simulation_output/real10_pi05_joint_temporal_check_v1r2
```

入口拒绝已有目录，不自动续训、覆盖或失败后重启。若未来确需复测，先检查旧进程/日志，
给定新的 `--out`，保留已完成 v1r2 与首次失败 v1。当前默认 v1 已含失败记录，不要裸跑。
v1 的 action batch 形状故障已修复；最终小改仅为诊断 `.detach()` 取标量去警告，数值路径
不变，已 SCP/py_compile，没有为这一日志修订重复加载模型。

完成回读：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 -r project4090:/home/zsw/project_2026/simulation_output/real10_pi05_joint_temporal_check_v1r2 simulation_output/
scp -O -o BatchMode=yes -o ConnectTimeout=10 -r project4090:/home/zsw/project_2026/simulation_output/real10_pi05_joint_temporal_check_v1 simulation_output/
.\.venv\Scripts\python.exe -B tools/render_real10_pi05_joint_check.py simulation_output/real10_pi05_joint_temporal_check_v1r2/report.json
```

报告、约10.85MB适配器两端均有。`diagnostic_adapter.pt`明确 `deployable=false`，没有
替换实机加载路径。四上下文覆盖起点、左右feed、末次转移；索引仍是全部2719转移/10条轨迹，
不使用max_records，不新建validation。新图为本机只读报告渲染，不加载LeRobot；PNG450dpi、
SVG和data-manifest位于工件下`figures/`。已查看，`viewed_not_accepted`，不是用户验收。
该节时点尚未实现的尺度校准和公平配对训练已在Current节完成；不重复启动本节旧检查。

## Deferred: Target-Vessel Region Review — Geometry Only, No Fit (2026-09-21)

只读草案已完成，等待用户确认/修正几何区域。以下均为已执行复现记录，不是重新训练命令。
先上传最终小文件，再在4090建立32个原始时刻/17对图像的元数据索引；不读取图片或拟合：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/review_real10_target_vessel_region.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/review_real10_target_vessel_region.py --stage prepare"
scp -O -o BatchMode=yes -o ConnectTimeout=10 -r project4090:/home/zsw/project_2026/simulation_output/real10_target_vessel_region_review_v1 simulation_output/
```

待传回完成，本机从原始RGB和v1的`preview_arrays.npz`渲染（微软雅黑；最终2.66秒）：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_target_vessel_region.py --stage render
```

页面：`http://127.0.0.1:8793/real10_target_vessel_region_review_v1/index.html`。
校准对照图`calibration_regions.jpg`、左右各5轨迹覆盖图、10张逐轨迹图、7张问题窗图，
共20张JPG；原图链接只GET现有8792服务，无标注提交/模型预测。服务启动命令见下文历史节。
绿色包络/保留色块与红色排除色块都只是固定首帧掩膜草案；橙色39/65提示框不参与掩膜。
输出`review_only_masks.npz`不能当作已验收掩膜或直接替换v1特征输入。

prepare拒绝覆盖非空目录。展示恢复只运行render；若确需几何修订，保留本版，使用明确
的新版本目录，不覆盖旧实验/人工标签。图像及页面只需本机查看，远端保存小协议/索引/
report/review_status/候选掩膜；未上传原图，未拟合，也没有新增smoke套件或例行逐文件hash。

## Historical: Fixed Motion Clip5 Ablation — Completed, No Classification Gain (2026-09-21)

仅运动分支的固定±5σ裁剪已执行：45窗预测全部不变，BA仍69.50%；39窗score从
-0.188041到0.375216但仍判stationary。不要根据本节命令重复试阈值、覆盖v1或重训PI05。
入口固定clip=5、alpha=1、阈值0.5，复用已有288维数值特征，无需原图或新的大文件。

最终3个小文件先SCP完成，再做远端检查和评估（以下为已执行记录）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 tools/run_real10_response_baseline.py tools/test_real10_event_windows.py tools/run_real10_response_motion_clip.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026/tools && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest test_real10_event_windows.ResponseBaselineTests.test_fixed_clip_train_statistics_missing_bits_and_refit_center test_real10_event_windows.ResponseBaselineTests.test_train_only_fit_missing_values_and_controller_allowlist test_real10_event_windows.ResponseBaselineTests.test_response_roi_original_pixels_and_shared_block_scale"
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_response_motion_clip.py --stage evaluate"
```

3项检查通过；整项闭式拟合/评估约0.023秒，包含10次无裁剪复现及10次裁剪拟合；
只保存10个候选折模型。无裁剪45分数最大差0；候选保存/回载一致。输出为
`simulation_output/real10_response_motion_clip5_v1/`；源`real10_response_local_motion_v1/`
只读。已有非空输出时拒绝重跑；若中断，先检查工件，不删除旧目录，仅在确需恢复且明确
授权时用新的`--out`保持同一固定协议。展示失败只重跑render，不重新fit。

结果传回完成后，本机渲染中文对照图和全部45窗页面（已完成）：

```powershell
scp -O -o BatchMode=yes -o ConnectTimeout=10 -r project4090:/home/zsw/project_2026/simulation_output/real10_response_motion_clip5_v1 simulation_output/
.\.venv\Scripts\python.exe -B tools/run_real10_response_motion_clip.py --stage render
```

页面：`http://127.0.0.1:8793/real10_response_motion_clip5_v1/index.html`，HTTP200。
`comparison.png`已查看，视觉状态`viewed_not_accepted`；本轮没有重新标注或UI交互验收。
复用下节8793静态服务即可。完整逐轨迹指标在report，45窗变化在paired_windows，
裁剪维度/训练统计在clipping_audit；分数不是概率，全部只是开发OOF证据。

## Historical: Read-Only Local-Motion Error Localization (2026-09-21)

已完成全部25个左任务窗口及65窗的冻结分数/网格/特征分解，78个保存分数复现至1e-10，
4090约0.100秒。入口只读模型、特征、标注及observable元数据，不调用fit或新提特征。
代码先SCP、再运行（以下是已执行复现记录）：

```powershell
scp -O tools/audit_real10_local_motion.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_real10_local_motion.py --stage summarize"
scp -O -r project4090:/home/zsw/project_2026/simulation_output/real10_local_motion_error_audit_v1 simulation_output/
```

本机从原图渲染12张中文拼图和只读序列页（已执行；不修改模型ROI或人工标注）：

```powershell
.\.venv\Scripts\python.exe -B tools/audit_real10_local_motion.py --stage render
```

页面：`http://127.0.0.1:8793/real10_local_motion_error_audit_v1/index.html`。
页面包含全部6个新增左侧误判、4个同轨迹正确对照、33融合误判和65；数字audit覆盖26窗。
原始完整序列仅GET8792 `/image`，不提交标注。两项原有服务本轮已恢复；如果日后进程
退出，在两个本机终端分别运行既有入口即可，不需要SSH、传原图或重训：

```powershell
.\.venv\Scripts\python.exe -B -m http.server 8793 --bind 127.0.0.1 --directory simulation_output
```

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_event_windows.py
```

`render`依赖本机原图和微软雅黑字体；无需在4090执行。audit可以重算同一冻结版本的
数字记账，不产生新候选模型。clip[-5,5]的单因素消融目前仅是下一步建议，尚无已完成
训练结果或已实现的新训练命令；不要直接修改v1结果、mask或分类阈值。

## Historical: Local-Motion Response Prototype — Completed Diagnostic (2026-09-20)

入口`tools/run_real10_response_local_motion.py`已实现并完成一次固定对照。仅动作后响应
线性识别，不是PI05训练或实机控制。不要因本节保留命令而重复拟合/搜索超参数。
窗口/标签/折来自`real10_response_matched_roi_v1`冻结快照，不读最新标注重新选样本。

代码先传到4090并做一次补偿符号/局部变化数值检查（已通过）：

```powershell
scp -O tools/run_real10_response_local_motion.py project4090:/home/zsw/project_2026/tools/
ssh -o BatchMode=yes -o ConnectTimeout=10 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_real10_response_local_motion.py --stage check"
```

原图仅在本机，因此本地提取图像特征（39.0秒，已完成，无模型拟合）；只上传约221KB
数值包，不重传原图或新建大图像包。协议在图像提取前锁定：

```powershell
Set-Location 'D:\PycharmProjects\project_2026'
.\.venv\Scripts\python.exe -B -u tools/run_real10_response_local_motion.py --stage prepare
tar -czf simulation_output/real10_response_local_motion_v1_numeric.tar.gz -C simulation_output/real10_response_local_motion_v1 protocol.json prepared.json annotation_snapshot.jsonl folds.json features.npz motion_quality.jsonl
scp -O -o BatchMode=yes -o ConnectTimeout=10 simulation_output/real10_response_local_motion_v1_numeric.tar.gz project4090:/home/zsw/project_2026/simulation_output/
```

4090首次解压和评估（已执行，30个折模型、135条预测，约0.035秒）：

```bash
cd /home/zsw/project_2026 &&
test ! -e simulation_output/real10_response_local_motion_v1 &&
mkdir simulation_output/real10_response_local_motion_v1 &&
tar -xzf simulation_output/real10_response_local_motion_v1_numeric.tar.gz -C simulation_output/real10_response_local_motion_v1 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_response_local_motion.py --stage evaluate
```

结果回读只需report、OOF和折模型；本轮180,150字节结果归档已成功传回。本机可再生
中文校准区域/运动拼图及只读页面，不重提特征或拟合：

```powershell
.\.venv\Scripts\python.exe -B tools/run_real10_response_local_motion.py --stage render
```

页面：`http://127.0.0.1:8793/real10_response_local_motion_v1/index.html`。
它只依赖已有8793静态服务及本目录5张JPG，不需要8792标注服务、原图播放器或保存接口。
`render`需要本地`preview_arrays.npz`和Windows微软雅黑字体；4090无需运行它。

恢复原则：先查看`prepared.json`/`report.json`，不要覆盖已有结果。若数值包已经解压且
没有report，可单独执行evaluate；若拟合曾部分完成，先确认失败原因，不能把重试当作
新的参数候选。prepare遇到非空输出会停止；不完整特征应保留，并在确认原因后使用新的
`--out`。SCP中断时只重传相同的小归档，确认完整解压后再执行，不拆大文件规避传输规则。

本次控制器/仅运动/融合BA为62.50%/69.50%/62.00%。仅运动右任务改善、左任务下降，
65仍误判；冻结这次结果，尚未授权后续消融或部署。详细局限见algorithm handoff当前节。

## Historical: Read-Only Right-Task Stationary Error Audit (2026-09-20)

用户批准的只读复核已完成；不需要重新训练或重标注。工具
`tools/audit_real10_response_stationary.py`只读取冻结45窗、features、fold_models、OOF
预测和observable event-window元数据。默认选择全部7个右任务stationary窗，包括一个
正确对照；21个已有预测分数和分解加和校验通过，4090运行0.062秒。

代码先SCP，再在4090执行数字审查（已执行，以下仅为复现记录）：

```powershell
scp -O tools/audit_real10_response_stationary.py project4090:/home/zsw/project_2026/tools/
```

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  tools/audit_real10_response_stationary.py --stage summarize
```

结果`simulation_output/real10_response_error_audit_v1/audit.json`已回读本地。可在本机从
原图生成审查拼图/页面（已完成，不涉及模型推理、fit、新ROI或标签写入）：

```powershell
Set-Location 'D:\PycharmProjects\project_2026'
.\.venv\Scripts\python.exe -B tools/audit_real10_response_stationary.py --stage render
```

浏览器：`http://127.0.0.1:8793/real10_response_error_audit_v1/index.html`。
8793为已有simulation_output只读静态服务；页面首/中/末拼图可独立查看，展开的原图
播放器只GET已有8792 `/image`，不访问保存标注接口。若原图不出现，检查原来的8792
标注服务是否仍在运行；不要为此重传原图或重训。`render`依赖本地collected_data及
Windows微软雅黑字体，4090无需运行该阶段。页面已检查65窗滑条首末帧切换；用户可优先
复核85窗的整个时间段，但该建议不等于要求修改标签。

下一步建议是局部导丝运动表示/静态捷径诊断；尚未实现或授权新的训练实验。不要根据
这次已查看的45窗改变阈值或挑checkpoint。审查范围和证据限制见algorithm handoff当前节。

## Historical: Real10 Task-Matched Full-Frame / Fixed-ROI Response Comparison (2026-09-20)

固定三组同状态/同任务条件对照，入口`tools/run_real10_response_roi_comparison.py`。
它只训练离线动作后响应线性识别器，不训练PI05或操作硬件。窗口/标签/折从已有
`real10_response_baseline_v1`冻结快照读取，不从最新人工标注重新选窗。

本轮已在本机执行原图裁剪（22.263秒），生成602张224px PNG；约51.5MB上传包由agent
传输，不需要重传11GB原图。以下是分机流程记录；已完成的阶段不要重复跑：

```powershell
Set-Location 'D:\PycharmProjects\project_2026'
.\.venv\Scripts\python.exe -B -u tools/run_real10_response_roi_comparison.py --stage crop
tar -czf simulation_output/real10_response_roi_images_v1_upload.tar.gz -C simulation_output real10_response_roi_images_v1
scp -O -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 simulation_output/real10_response_roi_images_v1_upload.tar.gz project4090:/home/zsw/project_2026/simulation_output/
```

最终代码必须先顺序SCP完成，再在4090执行（本轮代码已同步并通过7项针对性检查）：

```powershell
scp tools/run_real10_response_baseline.py tools/run_real10_response_roi_comparison.py tools/test_real10_event_windows.py project4090:/home/zsw/project_2026/tools/
```

4090首次解压与评估：

```bash
cd /home/zsw/project_2026 &&
test ! -e simulation_output/real10_response_roi_images_v1 &&
tar -xzf simulation_output/real10_response_roi_images_v1_upload.tar.gz -C simulation_output
```

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_response_roi_comparison.py --stage all
```

三组固定为控制器+Elite+task、再加全图时序、再加ROI时序。后两组共享相同的40维状态，
只比较图像表示；采用固定分模态缩放，防止控制器块因新增视觉维度被稀释。控制器组
逐窗分数自动对照v1，绝对容差1e-10。不搜索融合权重、ROI、alpha或阈值。

阶段/恢复：`crop`仅本地原图派生；`prepare`仅远端特征提取；`evaluate`复用完整准备包
拟合/评估；`all`=prepare+evaluate；`examples`仅重建中文HTML、不重新拟合。已完成crop/
prepare复用对应冻结包；未完成准备则保留目录并显式使用新`--roi-pack`/`--out`。已有
report时拒绝重复评估。拟合中断但report未写入可恢复evaluate；report已写入但页面未完成
则只运行examples。不要以恢复为名更改ROI或在同一45窗上搜索更优模型。

输出：`simulation_output/real10_response_matched_roi_v1/`内protocol、标注快照、folds、
features、prepared、30个折模型、135条OOF预测、report、examples和completed；ROI图在
独立`real10_response_roi_images_v1/`。HTML只引用本地已有全图/ROI，不重复内嵌图片。
本轮已在4090完整执行：ROI特征准备2.415秒、拟合/评估0.095秒、exit0，无须重跑。
控制器45个逐窗分数复现v1至绝对容差1e-10。平衡准确率为62.50%/56.00%/59.50%，
ROI没有超过控制器；完整左右任务、混淆矩阵和逐轨迹结果见report及algorithm handoff。

本轮51,478,559字节小图归档的SCP在50,774,016字节断开；对**同一份未改变的归档**
用`SFTP reput`续传尾部后完整解压，未重新打包、拆分或替换图片。这是传输恢复，不是
重新裁剪/训练。回读工件包：`simulation_output/real10_response_matched_roi_v1_readback.tar.gz`，
远端完整长度2,959,496字节；传输成功后再解压，不能把断传副本当成完整结果。
本轮已完整传回并解压，30个折模型、135条预测和completed均已核对，不需要再运行all。

本地原有8793只读服务可直接查看：
`http://127.0.0.1:8793/real10_response_matched_roi_v1/examples.html`。
页面包括三组指标、评估前固定的原图/ROI对照，以及每个真实/ROI预测组合的首个窗口。
不按页面缩略图改人工标签，不根据这45窗的OOF分数再调ROI/融合权重。

## Historical: Real10 Lightweight Post-Action Response Baseline (2026-09-20)

用户已补第91窗并批准轻量基线。45窗/10条轨迹，25推进/20无明显推进；最新v2共46次
保存，已同步4090。入口`tools/run_real10_response_baseline.py`已实现并完成一轮固定
整轨迹留一评估，不修改PI05既有训练划分/权重、不控制硬件。结果不是接触真值或世界模型收益。

已执行（4090，特征准备约3.16秒、线性拟合/评估约0.067秒；不需要重复运行）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_response_baseline.py --stage all
```

默认输入为`simulation_output/real10_event_windows_v1`的最新v2人工修订，以及
`simulation_output/real10_pi05_compat_v1`内已有224px图像。默认输出
`simulation_output/real10_response_baseline_v1/`。采用缓存的ImageNet ResNet18，若缺失
权重直接报错，不悄悄下载或替换成real10 PI05特征；没有新的环境依赖安装。

默认不带stage时只准备冻结特征，不拟合。阶段与恢复规则：

- `--stage prepare`：固定标注快照和整episode折，提取特征；有完整prepared.json时复用
  该快照，新的人工作业不会悄悄进入旧实验。若准备中断且未写prepared.json，保留不完整
  目录，使用明确的新`--out`，不要删除或覆盖原标注包。
- `--stage evaluate`：仅在prepared.json和features.npz齐全且没有report.json时拟合并
  评估。拟合中断而report尚未写入，可用该阶段恢复；它不会再次提取图像特征。
- `--stage examples`：report及OOF预测已完成、可视报告未完成时，仅恢复examples.html/
  completed.json，不重新拟合或改变结果。示例为每种真实/预测组合的首个窗口，不是精选成功片段。
- `--stage all`：准备后评估；已有完整report会拒绝重复评估，不以反复运行挑结果。

只读结果：

```bash
cd /home/zsw/project_2026 &&
cat simulation_output/real10_response_baseline_v1/report.json
```

协议先于特征/模型输出写入protocol.json。快照保存45窗最新人工修订；folds.json保存
整episode留一索引；features.npz只存明确白名单特征和标签数组；fold_models保存30个
线性识别折模型及训练折归一化参数；oof_predictions.jsonl为5基线×45窗=225条预测。
score_not_probability是固定0.5阈值的未校准分数，不能解释为碰壁概率或安全置信度。
本轮没有全数据部署识别器、策略权重更新、测试集调参或实机验证。

平衡准确率：总预测推进50.0%、训练折多数类36.5%、控制器+Elite62.5%、静态53.0%、
时序58.5%。时序左/右任务分别71.79%/40.66%，未形成稳定视觉收益。完整混淆矩阵、
逐episode指标、来源、时序限制见report及algorithm handoff当前节。
原4项和新增2项针对性检查共6项通过（0.592秒），无须每次标注后重新跑测试。

本地结果已回读。HTML引用同级已有real10_pi05_compat_v1/images，不内嵌/重复上传图像。
可直接打开`simulation_output/real10_response_baseline_v1/examples.html`；也可只读查看：

```powershell
Set-Location 'D:\PycharmProjects\project_2026'
.\.venv\Scripts\python.exe -B -m http.server 8793 --bind 127.0.0.1 --directory simulation_output
```

浏览器地址`http://127.0.0.1:8793/real10_response_baseline_v1/examples.html`，本轮服务已
启动，勿重复占用端口。页面不写标注、不训练、不连硬件。第5/12/2/1窗已由agent查看，
`review_status.json`记为viewed_not_accepted，不代表用户验收或接触真值确认。
完整回读包为`real10_response_baseline_v1_readback.tar.gz`；首次中断的
`real10_response_baseline_v1_artifacts.tar.gz`本地副本不完整，不要拿它覆盖已解压结果。

## Historical: Real10 Joint Response Review v2; 44 Windows / 10 Episodes; No Training (2026-09-20)

以下保留标注阶段的44窗快照。最新45窗、人工标签已上传及轻量拟合状态以上节为准；
原标注UI仍可使用，新修订不会改变已有基线的冻结快照。

离线事件适配和标注入口已完成；默认包`simulation_output/real10_event_windows_v1/`
在本地和4090均已生成，复用原10条real10的train-only归属。本地旧人工标注30次保存、
26个不同窗口；用户已完成44窗v2标注（45次保存），覆盖全部10条轨迹，其余56窗尚未标注。
用户明确以新版为准：每窗口最新v2修订覆盖旧版参考，旧文件及migration字段仅供追溯。
远端没有上传旧30条或新45条人工修订，因此远端此前summary为100窗未标注；不要将其
误认为本地标注丢失。本轮只同步进度文档，没有启动训练或改动人工标签。
原图在本地`collected_data/`，4090新增的只是20份原manifest/records元数据，无原图。
不要为标注重新上传大图、训练模型或重建已有输出目录。

### 在本地Windows标注

项目根目录运行（不需要LeRobot或GPU）：

```powershell
Set-Location 'D:\PycharmProjects\project_2026'
.\.venv\Scripts\python.exe -B tools/review_real10_event_windows.py
```

浏览器访问`http://127.0.0.1:8792`（v2新默认端口，本轮已启动）。若端口被占用，
先使用已有v2服务；不要同时启动多个写入相同标注文件的实例。
旧8791服务停止时系统拒绝访问，未提权；不要继续在旧页面保存。可由用户在原启动
终端Ctrl+C关闭旧服务。服务仍只监听本机，不连接相机/机器人、不推理模型。

1. 在锚点分别标Side/Top尖端可见性：可可靠辨认、位置不确定、不可见；画面发糊但能
   可靠辨认尖端仍可选第一项。不根据未来帧倒推锚点可见性。
2. 综合双视角锚点及后续画面，只填一份联合响应：推进、无明显推进、回退或无法判断。
   Top不可见但Side能判断推进，不必改成无法判断；不根据feed日志猜测响应。
3. 选实际判断依据：Side、Top、两路共同提供依据，或两路证据不足。最后一种只能搭配
   无法判断/未标注。不从可见性标签自动推断依据。
4. 填标注人后保存。现有44窗全部完整，无需重复补依据；可从下拉框选择未标注窗口
   或复核已有窗口。允许部分保存、Ctrl+C关闭、同一命令恢复；有待补依据时可用对应按钮。

新保存路径：`simulation_output/real10_event_windows_v1/annotations_joint_v2.jsonl`，
首次人工保存才创建，只追加修订；原`annotations.jsonl`保持原样、只读兼容。
schema为`real10_joint_response_annotation_v2`；字段为`anchor_visibility.side/top`、
`joint_motion_response`及`motion_evidence`。旧两路响应相同且非空才保留为联合响应，
否则留空待人工确认；判断依据始终留空待用户选择，修订来源随v2记录保存。
最新v2修订优先于旧记录，包括用户主动改变了旧运动结论的情况；不得从
`legacy_migration.per_view_motion`回填或覆盖当前`joint_motion_response`。
旧版payload会拒绝写入并提示刷新，不静默改写其含义。
原始records/动作标签不改。备份旧`annotations.jsonl`和新`annotations_joint_v2.jsonl`
（若已创建）即可保留完整人工历史，
不要删除/覆盖整个pack重建；另一个pack的标注不可直接混用。
“未标注”与“已看过但不可见/无法判断”不同；上述标签不是接触真值或力测量。
完成标注不等于新模型有效或数据通过正式验收；本命令不授权后续训练。

只读查看兼容迁移与完成数（不写文件、不启动服务）：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_event_windows.py --summary
```

最新本地结果：legacy_revisions=30、legacy_windows=26，v2文件45次保存/44窗，
reviewed=44、needs_evidence=0、unreviewed=56。联合推进25、无明显推进19；依据Side11、
Top4、两路共同29；左任务25窗、右任务19窗，覆盖全部10条轨迹。
第2/4/55窗由旧stationary改为新advance的结论继续保留。第24窗有两次保存，以第二次
补齐后的完整修订为准；不把45次保存当作45个样本。新增18窗为建议清单中的17窗和
清单外第24窗；第91窗尚未保存，可后续补齐，但不要求先补完100窗再推进轻量原型。
下一步仅建议准备按episode隔离的动作后响应识别基线；尚未实现或执行，不改变已有
PI05训练划分、不把stationary作为碰壁真值、不将未来帧放入动作前预测输入。
这次仅只读统计和字段/来源核对，不需要重新运行测试、生成pack或上传原图。

### 已完成的适配及针对性检查（4090）

以下为执行记录，不需要用户重复跑；默认输出已存在时适配器拒绝覆盖。

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  -m unittest discover -s tools -p test_real10_event_windows.py -v
```

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  tools/prepare_real10_event_windows.py \
  --source-root simulation_output/real10_event_source_metadata_v1 \
  --pack-manifest simulation_output/real10_pi05_compat_v1/manifest.json \
  --out simulation_output/real10_event_windows_v1 \
  --history-s 1.0 --future-s 1.5 --hold-windows 3
```

初版适配保留2729观测/2719原动作转移，输出70请求窗口+30时段覆盖窗口，未重新生成。
v2改动顺序SCP/回读后，4090执行上述测试4项通过（0.516秒）；新增合成旧修订的无损
迁移/冲突处理/证据来源检查，复用原HTTP保存恢复测试。原图不用重新检查或上传。
前后窗口按原始时间选取，无重采样；历史observation与当前命令日志、未来目标分文件。
未知状态用null/missing_fields，已记录null另列recorded_null_fields；无错误不当作缺失。
70个后台完成记录不代表物理递丝成功，5窗口有后续请求，不是单动作反事实样本。
Schema和输出文件边界详见algorithm handoff同日章节。

本地已做5458/5458图像引用存在性检查；按需复核命令如下，不需每次标注都运行：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_event_windows.py --check-images
```

未发生完整图像二次解码、训练、硬件调用或data/SOFA文档改动。UI代表画面已查看并
验证回放/切窗；v2实际查看保留标签、空依据及14→15→55待补跳转，状态
`viewed_not_accepted`。用户后续完成了44个v2窗口并指定新版优先，已只读核对，无标签
回写；这不是正式数据或实机效果验收。当前不再等待已有44窗补依据。

## Historical: Full-Cohort Full-Batch Comparison Completed (2026-09-20)

用户已批准并完成完整训练集上的固定优化对照和统一validation。入口
`tools/train_diffusion_pusht_fullbatch_comparison.py`为15317字节；复用helper的schema
改为由contract提供（12998字节），训练数值逻辑未改。两文件已顺序SCP/回读后执行。
正常exit0，12模型纯优化总44.816秒；默认结果目录已存在，勿重复训练或评估。

已执行命令（4090）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_diffusion_pusht_fullbatch_comparison.py --train
```

固定train51上下文/30源，原seed20260918/19/20，两优化配置×两loss，共12模型。
每模型最多2000次全批次梯度；AdamW及L-BFGS配置与上次tiny-fit完全相同，
line-search closure计入预算。每seed沿用原initial_state；不热启动、不改网络/输入/
loss/归一化。九模型2000、三个L-BFGS普通1999，共23997；剩余1次不足新线搜索时
按既定规则结束，不添加计算凑整。三模型9条间隔日志，其余10条；final和training_report
保存了真实最后预算。step calls与gradient evaluations分开报告，不宣称等FLOPs/墙钟。

原数据快照采用NPZ按key延迟读取：训练前仅核对train数组。全部12个final完成并写入
training_completed.json后，才打开原validation15上下文/8源和success sidecar。
成功标签只用于评估；没有重划分、调阈值、选择best seed/checkpoint、集成或test读取。
每个final加载后的train预测与训练时保存值逐位相同，validation一次性报告全部12模型。

输出根目录：
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_fullbatch_comparison_v1/`。
每个`<seed>/<optimizer>__<loss>/`含last.pt、final.pt、训练预测/报告/日志。
根目录有run.json（训练前冻结规则）、training_snapshot.npz、training_completed.json、
evaluation_targets.npz、predictions_<seed>.npz、report.json和status.json。
使用独立schema/fixed_budget_final，不替换原模型或部署入口。

只读状态（4090）：

```bash
cat /media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_fullbatch_comparison_v1/status.json
```

仅真实中断且report尚未完整时，对同一命令加`--resume`。核对run contract及原数据快照，
已完成模型读取原结果，其余从各自最近完整step的原子last.pt恢复，预算不增加。
完整report已存在时直接退出，不重复训练/评估。本次没有中断、恢复或失败重试。

无权重回读位于本地`simulation_output/diffusion_pusht_fullbatch_comparison_v1/`，
同名readback tar.gz为222542字节。逐seed/source重算主要指标及预算一致。
四组验证分支success均值75%/75%/79.17%/77.08%，低于DP87.5%；L-BFGS另有严重
无界分数外推，不能以训练MSE下降宣传算法收益。具体逐seed与范围见handoff。
当前保留DP、停止这轮优化器扩展；后续目标/表征对照尚未设计执行，test仍关闭。

## Historical: Full-Batch Optimization Probe Completed (2026-09-20)

用户已批准并完成同一四上下文的全批次AdamW / L-BFGS对照。新入口12884字节，
本地语法检查、顺序SCP、远端入口/字节数回读后执行，exit0。默认输出已存在，勿重跑。

已运行命令（4090）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_scorer_fullbatch.py --train
```

复用前次TinyData的训练侧读取和快照核对，只训练510000–510003的anchor80。
每个优化配置各训练普通/效应对比loss，保留原网络、输入/target归一化、固定goal和
seed20260918初始权重。不是从前次final续训；不读取validation/test或success标签。

每模型上限2000次完整4×5候选组的loss/gradient计算；L-BFGS线搜索closure也计入。
AdamW沿用lr3e-4、wd1e-4、clip1；L-BFGS固定lr1、history20、max_iter1、strong_wolfe，
无衰减/裁剪。线搜索预算实现绑定已检查的torch2.10：max_eval=min(24,remaining-1)，
不足2次剩余预算时停止发起新step。本次四模型都实际2000次，共8000；L-BFGS step
调用为613/705。不要把梯度调用数混同于参数更新数、等FLOPs或等墙钟时间。

运行前run.json记录完整固定配置；结果写入独立目录：
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_scorer_fullbatch_optimization_v1/`。
每个`<optimizer>__<loss>`子目录有last.pt、final.pt、predictions.npz、metrics.jsonl、
training_report.json。final采用独立schema和fixed_budget_final类型，不用于部署。

只读状态（4090）：

```bash
cat /media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_scorer_fullbatch_optimization_v1/status.json
```

恢复仅用于真实中断：在原命令末尾加`--resume`。核对run contract及原tiny-fit快照；
每个模型按约200次梯度计算后的完整step保存原子last.pt，SIGINT/SIGTERM在完整step后
保存；硬中断回到最近完整checkpoint，日志可能有未随checkpoint完成的重复区间。
已完成子模型读取原结果，整个report存在则直接退出。异常非有限值/超预算视为失败，
不要选较早checkpoint冒充final或自动换配置重跑。本次无中断、无resume、无失败。

四模型纯优化共15.092秒；无DP/环境/硬件调用。报告、预测和四组各10条日志已回读至
本地`simulation_output/diffusion_pusht_scorer_fullbatch_optimization_v1/`，无权重的readback
tar.gz为12637字节。重算MSE/MAE/pair/argmax及预算与报告一致。L-BFGS普通/对比训练
排序85%/75%、Top1均3/4；不能当作泛化、成功率或世界模型创新收益。
完整51训练上下文扩展及后续固定validation尚未实施，本节不授权自动执行。

## Historical: Fixed Training-Only Tiny Fit Completed (2026-09-20)

用户已批准并完成固定四训练上下文的可拟合性实验。新入口10481字节，本地语法检查、
顺序SCP和远端回读后运行，exit0；两臂纯优化共7.129秒。默认输出已存在，不需重跑。

已运行命令（4090）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_scorer_tiny_fit.py --train
```

固定选择510000–510003各自最早anchor80，使用原train包行0/2/4/6；无标签筛选。
保留原网络、归一化、固定goal、loss和AdamW；seed20260918初始权重与原实验逐位一致，
batch16有放回共享抽样、每上下文含全部5候选，每臂固定2000步，不提供step/seed扫参。
只读取train PairPack、training-only normalization、协议和原initial_state；不实例化
DPData，不读取validation/test或success sidecar，不运行DP、环境或硬件。

远端输出：
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_scorer_tiny_fit_v1/`。
其中run.json在训练前记录选择规则/具体行/预算；两个final与原模型分目录、分schema，
不能作为原部署loader的权重。report.json只报告同一训练子集的initial/final。

只读完成状态（4090）：

```bash
cat /media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_scorer_tiny_fit_v1/status.json
```

只有实际中断且没有完整report时，才对上述训练命令加`--resume`：沿用原核心每100步
及graceful stop的联合checkpoint，核对原contract/训练快照/批次表，双方同一步恢复；
完整report存在时resume直接退出，不重复训练/评估。硬中断从最近完整联合checkpoint
继续，日志可能包含中断前的行，需按paired_step检查，不能相加当成新增独立实验。
本次没有中断/恢复，不增加预算、不覆盖原实验，也不换子集重新挑好结果。

报告、训练快照、预测、20条日志和批次表已回读到本地
`simulation_output/diffusion_pusht_scorer_tiny_fit_v1/`；不含权重的同名readback tar.gz
为26163字节。预测/报告/argmax一致，四上下文均informative，40候选对均非平局。
普通/对比final排序准确率62.5%/60%，Top1均1/4；这只是训练拟合不足诊断，不是
成功率或世界模型收益。下一步建议的优化器/全批次对照尚未实现或执行，test仍关闭。

## Historical: DP Frozen Input Audit Completed (2026-09-20)

用户批准的输入尺度/归一化/分布核查已完成。新增脚本final版11884字节，顺序SCP和
远端入口回读后执行，exit0，内部0.084秒；无模型前向、训练、环境或test。
首次审查因直接比较原始float64状态与包内float32而退出，未产生输出；仅将审查逻辑
对齐已有pack转换后通过。最大状态舍入1.5100814e-5，未修改任何数据或模型。

已运行命令（4090，默认结果目录已存在，无需重复）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/diagnose_diffusion_pusht_scorer_input.py
```

读取固定train/validation包、原训练快照、六个final的归一化buffer、原始保存RGB/state/
action及前次保存的预测。重新提取全部66帧观察网格，只写独立报告和7张原始RGB PNG；
不重新运行策略或仿真，不读取未来图像。距离比较排除训练中完整同源轨迹，候选动作
匹配任意训练chunk而非候选索引；百分位仅用于事后描述，不是运行时gate。

输出及本地回读目录：
`simulation_output/diffusion_pusht_scorer_input_audit_v1/`，同名readback tar.gz为14807字节。
六份归一化一致，66上下文读取一致；重点源大误差出现在图像更接近训练数据的160步，
不支持明显量纲/打包故障。完整逐帧数字、图像状态与限制见当前handoff。

只读结果（4090）：

```bash
cat /home/zsw/project_2026/simulation_output/diffusion_pusht_scorer_input_audit_v1/report.json
```

本次接口排查结束；不改归一化/筛样/裁剪/调阈值。建议的少量固定训练样本可拟合性
实验尚未实现或执行，不提供虚构命令，也不由本节授权训练、结构修改或test。

## Historical: Frozen DP Train/Validation Fit Diagnosis Completed (2026-09-20)

用户批准后已执行一次冻结模型诊断。脚本final版8842字节，顺序SCP、远端入口/大小
回读后运行，exit0，内部0.747秒；不训练，不运行DP/环境/实机，不读取test。
所有六组验证预测与原报告保存的分数逐位相同，原权重和报告保持不变。

已运行命令（在4090，默认输出已存在，不必重复）：

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/diagnose_diffusion_pusht_scorer_fit.py
```

仅加载原六个fixed-final2000，比较train51上下文/30源与validation15上下文/8源。
只新增12次评分前向，optimizer_steps=environment_steps=test_sources_executed=0。
归一化、固定goal、候选排序/平局规则不变。成功/coverage标签只在前向后参与指标。
不增加优化器或参数调节入口，不覆盖旧结果；确需重跑须明确指定新的 `--out`。

输出及本地回读同路径：
`simulation_output/diffusion_pusht_scorer_fit_diagnostic_v1/`，包含report.json和三个
`predictions_<seed>.npz`；81878字节tar.gz已回读。只读查看：

```bash
cat /home/zsw/project_2026/simulation_output/diffusion_pusht_scorer_fit_diagnostic_v1/report.json
```

普通/对比训练pair accuracy=57.78%/57.39%，验证=49.28%/52.40%；训练Top1=
25.00%/33.33%，验证=12.50%/18.75%。当前结论是训练辨别有限且验证退化，不是
单纯训练已完美后的泛化问题。逐源分数偏差与动作选择损害需分开看，完整口径见handoff。
下一步建议的输入尺度核查尚未执行；本节不授权新训练、结构修改或打开测试集。

## Historical: DP Paired Training Completed; Preserve Fixed Results (2026-09-20)

固定40源数据生成、组包和只读加载检查已完成：train51上下文/30有效源，
validation15上下文/8源。已获用户同意并执行下列训练命令，正常exit0；不要重跑
生成、prepare或新建同名训练。现有网络/loss/协议/数据未修改。

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_diffusion_pusht_action_effect.py --train
```

三个配对seed20260918/19/20，每臂2000更新，全部六个fixed-final之后才评估。
正常完成状态为 `completed_fixed_DP_paired_training_and_validation`，共12000更新。
按同一runner旧耗时估计完整任务1–3分钟，故agent执行；此次实际纯优化累计20.559秒，
不含启动/保存/验证。训练没有运行DP推理、新环境步、测试源或硬件。

原报告、日志、快照、预测和六个权重位于：
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1/`。
本地不含权重的回读目录：
`simulation_output/diffusion_pusht_action_effect_training_v1_readback/`。
同名tar.gz为198389字节。保存数组的argmax、source宏平均success/coverage已重算一致；
无需重复运行smoke或逐文件hash。

只读状态命令（在4090执行）：

```bash
cat /media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1/status.json
```

只有实际中断且尚无完整report时才用原训练命令加 `--resume`，恢复双方同一步的联合
checkpoint；本轮已完成，不需要resume。不要删除目录、增加step/seed、改loss或换验证样本。

固定开发验证的source宏平均成功指标：原DP87.50%、普通77.08%、效应对比70.83%；
对比减普通三seed为-12.5/-6.25/0个百分点。离线oracle100%，两组均未救回DP失败。
指标只覆盖8源15个已到达决策上下文，不是新鲜测试或反复rerank的闭环策略成功率。
决策与全部固定候选控制见当前handoff。冻结本轮，不启用test530000–530019。
下一项建议是固定模型的训练/验证排序诊断，尚未实现或执行，不新增训练命令。

## Historical: DP Scorer Pair Prepared; Generate Independent Data (2026-09-19)

Remote `collect_diffusion_pusht_action_effect.py --stage prepare` PASSED. Do not
repeat prepare or overwrite its output. This check used explicitly synthetic
unit labels, zero optimizer/environment/hardware steps and no pilot future labels.
Current data_ready_for_fixed_scorer_experiment=false until all40 sources finish.

**Next command ON THE4090: generate the fixed32 train +8 validation sources.**
Expected2–3h; estimate scales the completed DP pilot, not a measured new full run.
This command generates/assembles data only; it does not train any model.

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_diffusion_pusht_action_effect.py --stage collect
```

Source seeds: train510000–510031; validation520000–520007. Future test530000–
530019 is reserved but unsupported by this collector. Two fixed DP anchors80/160,
five native candidates, candidate8 then same DP to first done/global300. Maximum
80 contexts,400 branches,132000 environment steps. Keep early success and actual
terminal targets; no replacement seed/anchor or fabricated missing future.

Output: `simulation_output/diffusion_pusht_action_effect_pairs_v1/`. Completed
sources have independent done.json; interrupted attempts remain. Resume:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_diffusion_pusht_action_effect.py --stage collect --resume
```

An optional `--stop-after-sources N` pauses after N newly completed sources,
without changing the cohort or allowing partial-cohort training. Packing is
automatic when all40 sources complete. If only packing fails, use the same
entrypoint with `--stage pack`; no model inference/environment steps are rerun.
Do not use the old headroom pilot's collect/run commands as the next step.

### After Data Completion And Readback, Not Part Of The Command Above

The DP training adapter is implemented but has not run on a complete new pack.
It blocks synthetic, incomplete, wrong-family or cross-split data. A read-only
data check, once generation/packing has completed:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_diffusion_pusht_action_effect.py --check-data
```

Future explicitly invoked training (do not launch before the data readback):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_diffusion_pusht_action_effect.py --train
```

Same command plus `--resume` restores atomic joint checkpoints; both arms finish
the same update before graceful stop. Three paired seeds,2000updates per arm,
all six finals before validation, no best-validation selection/ensemble. Output:
`/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1`.
This is a new namespace and never overwrites ACT/real10 or old scorer weights.
Coverage is the regression target; separate actual-success sidecar is evaluated
offline only. Validation reports both success and coverage with source-macro
aggregation, all fixed/uniform/oracle controls and all training seeds. It is not
fresh-test or closed-loop performance. Training runtime should be refreshed from
the prior matched runner after data completion; none was executed this turn.

Protocol: `docs/diffusion-pusht-action-effect-protocol-v1.json`; design/limitations
in scorer-plan section29. Preparation readback archive15102 bytes. Old eight-source
pilot stays frozen and excluded. No new data collection outside this public Push-T
algorithm experiment; no data-track/shared/weekly/hardware modifications.

## Historical: DP Candidate Headroom Complete (2026-09-19)

The user completed `--stage run --resume`. All8 fixed sources/15 reached contexts/
75 candidate branches are complete;500004 terminated at125, so its160 anchor was
not reached. Source500001 uses attempt_002; the incomplete attempt_001 is preserved
but excluded from the comparison. Do NOT rerun prepare/run/resume for more samples.

Output: `simulation_output/diffusion_pusht_candidate_headroom_v1/report.json`.
Source-macro reference/uniform/hindsight success is62.50/56.25/87.50%; available
headroom25 percentage points, with four rescue contexts in three sources. This
is not trained-scorer success. Fixed candidates0..4:62.50/56.25/43.75/56.25/62.50%.
Whole nominal episodes5/8; all-fail context source-macro12.5%; short/final coverage
headroom0.008097/0.020134. Full evidence and decision: scorer-plan section28 and
current handoff. No training or new test command is authorized by these results.

Completed readback archive646626 bytes on both machines:
`simulation_output/diffusion_pusht_candidate_headroom_v1_completed_readback.tar.gz`.
Saved trace audit verified75 branches,1997 reference suffix steps, recorded
proposals/metrics/early stops and20634 completed-attempt environment steps.
All attempts22336 includes failed1702; prepare/repair diagnostics are separate.
Maximum reference coverage delta4.441e-16. Resume invocation1546.429s.
First rescue/harmful terminal pairs inspected, viewed_not_accepted.

Only if the summary needs to be reconstructed from these SAME saved results:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_candidate_headroom.py --stage summarize
```

Summarize does not load a model or execute an environment. It was not necessary
to rerun it during readback. Next proposal is DP-matched independent training/
validation and a fair selector comparison, not reusing this pilot's oracle labels.
No implementation, generation or training for that next proposal has started.

## Historical: DP Pilot Resume After Coverage Roundoff Fix (2026-09-19)

User direction: defer the geometry/scorer redesign and first ask whether a
stronger native DP candidate set contains better actions. Implemented
`tools/diffusion_pusht_candidates.py` and
`tools/probe_diffusion_pusht_candidate_headroom.py`; plan section27 owns scope.
No training, manual XY proposals, old ACT test reuse or hardware execution.

The first user run completed source500000, then stopped in source500001/attempt_001
at the anchor160 reference comparison. Identical saved actions/observations/
discrete states can differ in geometry-derived coverage by2.776e-16. The runner
now uses coverage-only atol1e-12/rtol0; all other checks remain exact, raw metrics
and the frozen experiment are unchanged. Preserve both existing attempts.
The short `--stage verify-reference` checks only the failed source's reference
branches; it passed remotely in74.447s with648 environment steps. The saved216
nominal actions match; anchor80/160 reference suffixes run136/56 steps with exact
actions/observations/discrete outcomes. Maximum coverage delta2.776e-16 including
the prefixes; six true-mismatch cases remain rejected. No alternative execution,
training or hardware action. Evidence: `reference_verification/attempt_001/report.json`
under the output root. Do not repeat the check or prepare. The full cohort has
not been resumed by the agent.

**Historical resume command (already completed; do not rerun):**

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_candidate_headroom.py --stage run --resume
```

This skips completed500000 and restarts incomplete500001 from native reset in
attempt_002, then continues500002–500007. Do not concatenate partial branches.
This is evaluation only, not training. Remaining time is workload-dependent;
the original35–55min estimate was for the entire eight-source job.

SSH and existing official migrated checkpoint load passed. Completed command:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_candidate_headroom.py --stage prepare
```

Do NOT repeat prepare: its output already exists and is protected. It verified
five legal distinct [5,8,2] native chunks and exact ordinary-DP/reference parity
for16 steps, including one continuation chunk.48 total environment steps in
69.153s,0 optimization/hardware steps; at preparation time no alternative or full
pilot source had run. Subsequent user-run failure and repair are above; full-cohort
resume remains user-run. Grid/geometry four-arm design remains unimplemented.

Original initial-run command (already used; the full run is now complete):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_candidate_headroom.py --stage run
```

Fixed8 sources500000–500007; only the unchanged DP's80/160 anchors,5 genuinely
sampled chunks,8 candidate steps then the same DP until done/global300. Common
post-reference sampling RNG; all native prefixes and reference suffixes must
match. No outcome-selected anchors, replacement seeds or extra training. Up to
16 contexts/80 branches/26400 environment steps including replay. Progress
prints after each branch and completed source. Preparation seed499999 is separate.
An optional `--stop-after-sources 1` requests a source-boundary stop without
changing the cohort; it is not needed for the normal full command.

After Ctrl+C/failure, preserve the folder and inspect status/traceback. Resume:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/probe_diffusion_pusht_candidate_headroom.py --stage run --resume
```

Completed sources are skipped; incomplete sources restart from reset in a new
attempt, never concatenate partial branches. Complete-source manifests and
all attempts remain. Repeated attempts can exceed the one-pass budget; recorded
attempt steps are reported separately. Abrupt process death can lose tail counts.
If all sources finished and only reporting failed, use `--stage summarize`;
it reads saved results only, no model load or environment run. Completed
`--stage run --resume` also summarizes without rerunning sources.

Output: `simulation_output/diffusion_pusht_candidate_headroom_v1/` on4090.
It contains protocol/seed_inventory/preparation_report/status, each source's
attempt logs, observable contexts/proposals, isolated diagnostic_outcomes,
terminal PNGs, first-source proposal sheets and finalreport. Primary is
source-macro hindsight rescue on the complete-five-valid population. Report
candidate validity and all-fail fraction, all five fixed indices/uniform, short
and terminal coverage headroom, per-source details and ordinary DP episode
success separately. This is NOT a learned selector's achieved success rate.
No reliable upper-space evidence means do not train a selector by default.

Preparation artifacts were read back in the55526-byte
`simulation_output/diffusion_pusht_candidate_headroom_v1_preparation_readback.tar.gz`.
`preparation_candidates_zh.png` is a split-panel command visualization, not an
executed-alternative or predicted trajectory; reviewed, viewed_not_accepted.
Its initial SSH inline redraw hit a quote error; the final saved-array-only
`--stage render-preparation` succeeded, with zero newforward/environment steps.
Only rendering/CLI changed after the passed numerical prepare; do not repeat it.
No data-track, shared contract, hardware entrypoint or weekly-log edit.

## Geometry / Relative-Gain Design Only (Deferred, 2026-09-19)

The approved design is in section26 of
[the scorer plan](algorithm-action-selection-plan-20260916.md). Four proposed
arms cross observable44-dimensional geometry off/on with absolute terminal
coverage / ACT-reference gain loss. All arms use the same planned123265-param
network, original train/validation data, shared target scale, saved batches,
three seeds and fixed2000updates. Old weights/protocols remain untouched.

There is NO implemented entrypoint or authorized training/test command for
this design yet. This turn only inspected code and the installed4090 renderer
through SSH; zero model forwards, optimizer/environment/hardware actions and
no opening of the used20-source test artifacts. Only the three algorithm docs
are updated. Do not run the historical commands below as the next step.

Next, if approved: implement the common feature/scorer, separate new schema
and prepare/train/summarize entrypoint. Preparation should minimally check
coordinate mapping on one fixed training example, shared parameter shapes/
initialization/sample order, candidate/reference permutation and equal-action
consistency, without optimization or environment execution. Complete exact
paths, run/resume commands and timing estimate before handing off training.
Do not overwrite old experiments or silently add advantage normalization,
pair weights, new seeds, data or model-selection rules. The reused validation
set remains development only; fresh confirmation needs its own later protocol.

## Input-Dependence Audit Completed (Historical Stage, 2026-09-19)

Implemented `tools/audit_pusht_scorer_input_dependence.py`, AST-checked locally,
SCP-synced (16506 bytes), and executed once on the4090:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_scorer_input_dependence.py
```

This is a completed-command record, not a request to repeat it. The script
requires the six frozen final2000 checkpoints and exact original train/validation
array snapshot. It opens no test artifact, creates no environment/ACT runtime,
and performs no optimization or hardware action. It does not modify the model,
loss, normalizers, labels or original reports. Existing output is refused.

Output on both machines: `simulation_output/pusht_scorer_input_dependence_v1/`.
Files: `protocol.json` (fixed before forward), `report.json`,
`scores_and_donor_mappings.npz`, `status.json`. The complete small readback
archive is `simulation_output/pusht_scorer_input_dependence_v1_readback.tar.gz`
(1488863 bytes); all weights stay on SSD. A failed partial audit must be
inspected/preserved, not overwritten; if a justified rerun is approved, use a
fresh `--out` and record why. There is no iterative tuning/resume search here.

Actual scope:92 training/21 validation contexts,2740/126 other-source same-anchor
donor pairs, all four input swaps, both candidate-wiring controls, all three
paired training repeats.324 forward calls/70818 context-batch rows/354090
candidate scores; these are repeated diagnostic combinations, not independent
episodes. The reported0.497-second segment excludes process startup, initial
preparation and final writes.0 optimizer/environment/hardware steps, no test
access. All candidate permutation/identical-action control errors are0; original
validation scores reproduce within the fixed2e-6 tolerance.

Mismatched-input outputs have no paired real outcome; their MAE/gain/harm are
deliberately not evaluated. Original-input fit and score sensitivity remain
separate. Per-source aggregation first averages donors, then anchors; all
training repeats are retained. See plan25 for results and the explicitly
post-hoc saved-array target/error variance decomposition.

Conclusion: inputs are active, but candidate differences are severely
under-fitted even in training; no evidence justifies increasing pair weight
or promoting a selector. Keep ACT/real10, freeze the used test. An explicit
relative-geometry/action-interaction design is only a proposed next step;
there is no new training/rollout command to execute. No weekly/data-track edit.

## Twenty-Source Test Completed; Preserve Results, No Rerun (Historical Stage, 2026-09-19)

The user ran `tools/run_pusht_action_effect_test.py --stage run`. Final status:
`completed_frozen_twenty_source_candidate_test`. All20 sources completed with
one attempt each,28021 native steps and45.972 seconds before summarization;
0 optimizer steps and0 hardware actions. Full execution passed; interruption/
resume was not exercised in this run. **Do not repeat training or test.**

The full raw artifact is on both machines:
`simulation_output/pusht_action_effect_fresh_test_v1/`. Its758532-byte complete
readback archive is `simulation_output/pusht_action_effect_fresh_test_v1_readback.tar.gz`.
This includes frozen protocol/identities, report/status, per-source journals
and RGB, saved predictions/targets, manifest and first-source case sheet;
no weights were copied. The six original final2000 weights remain on SSD.

Readback verified57 eligible contexts/285 candidates over20 sources, from58
executed anchors/290 candidate continuations.430000/160 and430011/160 were
unreached;430003/160 was excluded for invalid current extraction. All planned
exclusions remain; no replacement. Saved arrays match all342 eligible model/
context prediction journals and actual outcomes. The prescribed main95% and
five fixed-control99% intervals reproduce from the saved arrays. This inspection
ran no model, environment or optimizer; no new inference/evaluation is needed.

Main coverage difference is-0.000830,95% source-cluster interval
[-0.002717,+0.000978]. Harm rises12.50% ->19.72%; both learned mean coverages
are below ACT0.181083. All fixed-control intervals include zero. Keep ACT/
real10; do not pick a favorable training seed, constant direction or loss
weight using this now-inspected test. Detailed results/limits are in plan24.

Viewed artifact: `first_test_source_zh.png`, fixed source430000/all planned
anchors/first training seed. It is viewed_not_accepted, not user acceptance;
the raw report's original visual flag is unchanged. No data-track or weekly
document was modified. A train/validation-only input-dependence audit is only
a next-step proposal; there is no new training/test command to run now.

The following preparation/launch commands are retained as historical provenance,
not as instructions to execute the completed experiment again.

## Frozen Twenty-Source Test Prepared; User Run Next (Historical Stage, 2026-09-19)

Entrypoint `tools/run_pusht_action_effect_test.py` and the separate test
supplement are on the4090. The six completed final2000 weights are frozen;
**do not retrain**. Remote `--stage prepare` has already completed: all six
models exactly reproduced the saved21-context validation predictions, old
metric slots are unchanged, and the zero-effect grouped-statistics check
passed. This was six existing-validation forward calls,0 fresh-test sources,
0 native steps,0 optimizer steps and0 hardware actions. It does not validate
the full new test or interrupt recovery, neither of which has run yet.

Run once in the4090 terminal:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_action_effect_test.py --stage run
```

No download/image upload is needed. Weights remain on SSD at
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_action_effect_contrast_v1/`.
Default small test output:
`/home/zsw/project_2026/simulation_output/pusht_action_effect_fresh_test_v1/`.
Preserve its existing preparation files; **do not rerun `--stage prepare`**.

Fixed workload:20 source seeds430000–430019, anchors0/80/160, unchanged ACT
and four XY ramps,5 candidates/horizon8, six frozen scorers sharing all actual
outcomes. Maximum60 contexts/300 continuations/29760 environment steps in a
single pass; no substitutions for early termination or invalid observations.
Estimate approximately1–3 minutes from the previous40-source92-second run
plus model startup/scoring; this new full-run timing is not yet measured.
No selector-driven closed loop, optimizer or robot connection is involved.

The script logs each completed source. Ctrl+C preserves the partial attempt;
resume with:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_action_effect_test.py --stage run --resume
```

Completed sources are skipped. An incomplete source restarts from its native
reset in a new `attempt_NNN/`; previous attempts are not overwritten or spliced.
For an intentional source-boundary pause, add `--stop-after-sources 1` to the
run command (and `--resume` if any attempt already exists). This changes only
where that invocation pauses, not the frozen cohort or reporting rule. Do not
view a partial cohort as the final test or choose new seeds. Retry work can
exceed the single-pass step bound; reports separate completed-source steps
from all recorded attempts. Abrupt power loss may omit unflushed step counts.

When all20 sources are complete, the script automatically summarizes saved
predictions/outcomes. If only this stage failed, use the following offline
recovery; it performs no inference, reset or physics step:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_action_effect_test.py --stage summarize
```

An already-completed `--stage run --resume` is a no-op. Do not delete results
or change weights/protocol to bypass a frozen-binding failure; inspect that
error first. The new entrypoint reuses the native reset/prefix replay already
verified for paired data, but its complete run/recovery path is still pending.

Expected artifacts (only preparation files exist at this handoff):

```text
protocol.json / seed_inventory.json / preparation_report.json / status.json
sources/<source_seed>/attempt_NNN/
  predictions.jsonl           # six scores/choices persisted before each future
  contexts.jsonl / nominal_prefix.jsonl / outcomes.jsonl / candidate_steps.jsonl
  reset.json / report.json / observations/*.png
sources/<source_seed>/done.json
invocations.jsonl
predictions.npz / targets.npz / contexts.json / manifest.json
report.json / first_test_source_zh.png
```

The report keeps the unchanged main comparison and ALL fixed-index0–4
controls, with source-level coverage/gain/regret/harm, ranking/MAE, per-source
and per-training-seed details, selection histograms and compute counts. Three
training repeats are not extra independent test sources.10000 paired source-
cluster bootstrap draws produce a95% main interval and five99% control
intervals (Bonferroni nominal family95%, percentile approximation, conditional
on the fixed three training pairs). All anchors/repeats from a source stay
together. Fewer than two eligible sources gives no interval; no replacement.

The fixed first source430000/all anchors/first training seed determines the
case sheet before outcomes exist. Keep ACT/real10 and the six frozen weights;
no test-based tuning, best seed/constant promotion or cross-domain conclusion.
See plan section23. This preparation did not change data-track or weekly docs.

## Fair Scorer Training Completed; Readback Only (Historical Stage, 2026-09-19)

The user completed `tools/train_pusht_action_effect_contrast.py --train`.
Remote status is `completed`, root report is
`completed_fixed_paired_training_and_validation`. **Do not rerun training.**
Each of the six final checkpoints reached2000 updates;12000 retained updates
total, no resume in these logs. All six final files exist on the SSD and were
loaded by the trainer for final-only validation. No fresh-test source ran.

Completed training root:
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_action_effect_contrast_v1/`.
It occupies13,440,081 bytes. Pointwise finals are487559 bytes each; contrast
finals487623 bytes each. These are lightweight size identities, not hashes.
Keep all three seed pairs, not just a best validation seed.

Metadata, logs, batch schedules and saved validation predictions were packed
without copying weights, then SCP-transferred as the148655-byte archive:

```text
simulation_output/pusht_action_effect_contrast_training_readback_v1.tar.gz
simulation_output/pusht_action_effect_contrast_training_readback_v1/
```

The archive/readback contains root report/run/status plus each training seed's
metrics.jsonl, sampled_rows.npz and validation_predictions.npz. Original weights
remain on SSD. The separately copied root report is retained at
`simulation_output/pusht_action_effect_contrast_training_report_v1.json`.

Completed local saved-output analysis (no model, LeRobot, optimizer or environment):

```powershell
.\.venv\Scripts\python.exe -B tools/inspect_pusht_action_effect_contrast.py `
  --out simulation_output/pusht_action_effect_contrast_inspection_v2
```

Do not rerun into the existing output directory. For a genuinely necessary
readback repeat use a fresh `--out`; this is not a training/evaluation rollout.
v1 is the initial readback, v2 adds all-five fixed-candidate controls and their
warning. `report.json` and `validation_selection_readback_zh.png` are the final
analysis artifacts. Both image versions were inspected; v2 is the current
viewed_not_accepted artifact, not user acceptance. Raw run reports are unchanged.

Mean validation coverage is0.207096 ->0.218072, harm11.11% ->4.76%, but
pooled pair accuracy65.20% ->64.71%. All three paired coverage differences are
positive.99.19% of positive source differences come from420003/420006;
fixed candidate1 reaches0.217884, only0.000188 below the contrast mean.
Controls are post-hoc diagnosis of all five indices, not a promoted
validation-picked selector. See plan section22 for complete numbers and limits.

Training computation averages3.546/3.451 seconds per2000-update arm (excludes
initialization, transfers, checkpoint writes and final analysis). Warm forward
averages0.295/0.276ms per context/five candidates; no end-to-end latency or
speedup claim. Runtime is4090/PyTorch2.10.0+cu128/FP32/no AMP or TF32.

Next: implement/predeclare the reserved20-source fresh-test evaluator, retaining
the main comparison and adding all fixed-index controls as a diagnostic
supplement. There is **no fresh-test launch command yet**. Do not substitute
`--resume`, old scorer comparisons, validation reruns or new training for it.
No fresh test, hardware, data-track edit or weekly-log update occurred in this readback.

## Fair Pointwise / Effect-Difference Training Entrypoint (Historical Stage, 2026-09-18)

Implemented `tools/train_pusht_action_effect_contrast.py`. **Not executed or
runtime-tested this turn**, following the user's request to leave verification
and return to implementation. No training is implicitly authorized by this
document. Do not use the old terminal-score trainer for this experiment.

Run in a terminal on the4090, when starting this fixed training is intended:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_pusht_action_effect_contrast.py --train
```

No new download, image transfer, dataset generation or ACT rollout is required.
The entrypoint requires CUDA and defaults to the already completed
`simulation_output/pusht_action_effect_pairs_v1/` and original goal cache
`/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/`.
It reads only the fixed training-goal row from that cache, not a new training
dataset. Source IDs/anchors remain metadata; future coverage is a target only.

Fixed budget (no CLI hyperparameter sweep): seeds20260918/19/20, two arms per
seed,2000 optimizer updates per arm,16 contexts x5 candidates per batch,
AdamW lr3e-4/wd1e-4/clip1, same initialization and ordered batches. Total12000
optimizer updates; no additional environment/test/hardware steps. Runtime has
not been measured for this new entrypoint; do not infer completion from the
short duration of the previous data-generation job.

Output lives on the SSD, not the home filesystem:

```text
/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_action_effect_contrast_v1/
  run.json                       # frozen contract, runtime/device and source paths
  data_snapshot.npz              # exact small arrays/key order for resume
  status.json
  <training_seed>/
    initial_state.pt             # single shared initial state_dict
    sampled_rows.npz             # one shared ordered context schedule
    last.pt                     # atomic BOTH-arm models/optimizers at equal steps
    metrics.jsonl               # losses, paired step, resume origin, compute seconds
    pointwise/final.pt
    pointwise_plus_effect_difference/final.pt
    training_report.json
    validation_predictions.npz  # both raw score arrays; no future targets
    validation_report.json
  report.json                    # all three pairs; no best-seed or ensemble selection
```

All six final weights are saved before any validation inference. The final
reader reports gain/regret/harm/headroom, informative top1/pair accuracy,
true/predicted ties, raw coverage MAE, source macros, per-source/per-training-seed
results and warm forward time (1 context/5 candidates,10 warmups/50 timed calls;
excludes image feature extraction and input transfer). Uniform is an analytical
expectation; oracle is an offline upper bound. No clipping/abstention gate is
added. Pair accuracy excludes target ties but does not count prediction ties
as correct. The validation report is descriptive, not the fresh-test primary
result or closed-loop success.

Graceful stop: Ctrl+C once; the active paired update finishes, then both arms
are saved together. Resume the same run with:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/train_pusht_action_effect_contrast.py --train --resume
```

Automatic joint checkpoints occur every100 paired updates. Abrupt failure
discards only work after the last joint checkpoint; it cannot resume a saved
half-pair. Completed seed pairs are skipped. If only final evaluation was
interrupted, resume repeats evaluation without further optimization. A fully
completed root is a no-op on resume. Loss log rows carry `resume_from_step`;
checkpoints, not duplicate log rows after a crash, determine retained steps.
Reported optimizer totals describe retained final state, not all crash-lost
compute. Do not delete old results to restart or change split/seeds/weights.
If interruption happens before `run.json` exists, inspect the partial root
instead of deleting it; it has no initialized resumable training contract.

Default output creation refuses an existing directory without `--resume`.
Resume checks the frozen contract, exact observation/target/key snapshot,
shared initialization, batch schedule and both optimizer steps. Checkpoint
reload for later inference uses this new module's `load_final()`, not the old
DS0/terminal-score loader. `predict_effect()` is a legacy method name: for
these checkpoints its result is **raw terminal coverage score**, not an effect
delta, probability, calibrated safety score or robot action.

Do not run fresh-test sources430000–430019 from this entrypoint. Their evaluator
must be fixed separately after the six weights exist. Keep ordinary ACT and
onsite real10 unchanged. This implementation-only stage adds no evidence to
the weekly log and does not modify data-track documents.

## Paired Train/Validation Data Completed (Historical Stage, 2026-09-18)

`tools/collect_pusht_action_effect_pairs.py` is synchronized and the entire
fixed32/8-source public Push-T data job is complete. **Do not rerun collection.**
No training or hardware command belongs to this completed step. The reserved20
test sources have not been run and are not supported by this collection CLI.

Completed commands, retained for provenance rather than another run:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_pusht_action_effect_pairs.py --stage prepare

/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_pusht_action_effect_pairs.py --stage collect --stop-after-sources 1

/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_pusht_action_effect_pairs.py --stage collect --resume
```

The first source completed in5.538s, then resume skipped it and completed39
remaining sources in92.443s. Total55992 environment steps,0 optimizer steps,
0 hardware actions. Original runtime warning about pygame/pkg_resources and
dynamic gym_pusht registration did not prevent native execution. All actual
resets/prefixes/candidate0 outcomes were checked against the unchanged reference.

A variable-count typo in final packing raised ValueError after all40 source
records had completed. It was fixed locally, SCP-synced, and recovered using
only the following **already completed** command (no ACT or environment load):

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/collect_pusht_action_effect_pairs.py --stage pack
```

Results at `simulation_output/pusht_action_effect_pairs_v1/` on both machines:

- `report.json`:completed_train_validation_candidate_pairs;
- `train/pack/`:92 contexts/32 sources/460 candidate outcomes;
- `validation/pack/`:21 contexts/7 eligible sources of8/105 candidate outcomes;
- `normalization.json`:original ACT input stats + new training-only target stats;
- `execution_contract.json`, `runtime_binding.json`, `environment_parity.json`;
- `{train,validation}/<seed>/attempt_001/`:raw current/future RGB, actions,
  native execution journals, reset identity and source report;`done.json` marks completion;
- `first_sources_pairs_zh.png`:fixed first train/validation source, all anchors,
  inspected as viewed_not_accepted; raw report not modified to imply acceptance.

The full archive `simulation_output/pusht_action_effect_pairs_v1.tar.gz` is
1,305,880 bytes and has been transferred/extracted locally. The raw tree is
about4MB. No large-file transfer or user upload is needed for this batch.

Recovery behavior for a genuinely incomplete attempt: `--resume` skips every
`done.json`, restarts only the incomplete source from native reset in the next
`attempt_NNN` directory and preserves earlier logs. Stop normally with Ctrl+C
or use `--stop-after-sources N` for a source-boundary pause; the latter counts
newly completed sources in that invocation and never changes the fixed cohort.
SIGINT/SIGTERM save the current counter where possible. Abrupt process/machine
loss may leave its final step count incomplete; no crash-proof accounting claim.
When all40 sources are done, `collect --resume` only calls offline packing.
`--stage pack` can rebuild derived packs after interrupted packaging without
rerunning physics. Never delete the source attempts or change seeds to recover.

Maximum single-pass budget59520 steps; the completed run used55992. Scope is
public diagnostic scorer data, not formal guidewire data. Four train anchors
were unreached, and validation420007's three current grids were invalid; no
replacement samples were added. Training target mean/std only uses460 train
outcomes. Validation/test data must not enter fitting or checkpoint selection.

Next implement the new matched fixed2000 trainer and final-only validation
reader. **There is still no new training launch command**, and old terminal-score
training commands below do not implement this experiment. Hold the test sources
until final weights and their evaluation protocol are fixed.

## Action-Effect Pair Interface Prepared (Historical Stage, 2026-09-18)

The user is away from the site. All hardware commands below are deferred, not
the active next step. Do not connect the robot, cameras or feeder for this work.
Only offline public Push-T algorithm preparation is in scope.

Completed once after small-file sequential SCP and remote byte-size readback:

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_pusht_action_effect_contrast.py
```

The default `simulation_output/pusht_action_effect_contrast_prepare_v1/` already
exists. **Do not rerun by default.** A necessary explicit repeat must use a new
`--out simulation_output/pusht_action_effect_contrast_prepare_v2` and preserve
v1. Failure leaves partial output for inspection; no training or environment
execution occurs, and there is no optimizer/checkpoint to resume.

Report: `prepared_interface_only_missing_independent_training_pairs`, internal
runtime 0.138 s; 57 valid contexts / 285 actually executed candidates / 155
informative coverage pairs. Both arms have 117633 parameters with matching
initial outputs and different finite loss gradients. The old source remains
development-only. `observations.npz` contains only current/agent_xy/actions;
`targets.npz` contains only terminal_coverage; `contexts.jsonl` is metadata.
`random_forward.npz` contains untrained mechanical-check scores, not a trained
checkpoint, selected policy, evaluation gain or deployment recommendation.
All artifacts are on both machines; optimizer/environment/hardware steps=0.

Protocol: `docs/pusht-action-effect-contrast-protocol-v1.json` and action-selection
plan section 19. There is **no training/collection launch command yet**:
`training_ready=false`. Next implement the bounded public-benchmark same-start
candidate execution entrypoint and inspect independent train/validation data;
then complete the matched trainer and fresh-test evaluator. Do not adapt an
old terminal-score training command or train on the old inspected dev20 cache.
The projected 32/8/20 source-seed split and 3 paired training seeds x 2000 steps
are a fixed experimental plan, not authorization or evidence of completed work.

## Deferred Onsite: Home Elite Before One Policy Action (2026-09-17)

Other people currently use the arm. **Wait for device handover before running**;
the agent implemented/synchronized this change and tested mocked control flow
only, without connecting Elite/cameras/feeder or loading real model weights.

The same entrypoint now has two onsite confirmations when `--execute` is used:

1. HOME: confirm equipment handover, other controllers stopped, return path and
   guidewire setup clear, correct TCP/tool, emergency stop accessible. Only then
   does the script connect Elite and return to the recorded capture start.
2. After measured arrival and model/camera startup, EXECUTE: capture fresh inputs
   and execute one bounded policy action. Homing itself never feeds.

Do not use an unattended pipeline to supply these confirmations. Future onsite
command, once the equipment is free:

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_once.py \
  --task left --elite-ip 192.168.5.66 \
  --feeder-host 192.168.5.6 --feeder-local-host 192.168.5.11 \
  --home-speed 5 --max-step-mm 1 --speed 5 --execute \
  --out "simulation_output/real10_once_left_homed_$(date +%Y%m%d_%H%M%S)"
```

Default start (same for left/right) is XYZ
[-336.181546, 251.652310, 321.987936] mm and RPY
[3.0661665148309702, 0.03647610536354759, 0.06842115274422467] rad, from the first
point of both recorded paths plus their fixed orientation. It is not mechanical
zero. `--home-pose X Y Z RX RY RZ` explicitly overrides it if the calibrated
setup changes; no default was inferred from the other user's current pose.

The return move uses IK/reference joints and joint speed <=5 percent, not TCP
mm/s. It can be much larger than the policy's 1 mm action. Initial distance is
limited to 500 mm (lower with --home-max-distance-mm), maximum joint change is
60 degrees, and arrival timeout is 60 s. Arrival tolerance is 0.5 mm/0.01 rad.
These caps do not guarantee a collision-free joint-space path. If outside the
bounds, reposition manually along a known clear route; do not disable checks
or repeatedly retry. No automatic intermediate/obstacle-avoidance path is added.

HOME cancellation connects no devices. Failure/timeout attempts stop if a move
was pending and prevents inference/feed. Once home, a subsequent pose change
or pre-home camera frame cancels before inference; there is no automatic
re-homing loop. Cancelling EXECUTE may leave the completed homing move: inspect
`report.json`'s nested `homing` as well as top-level policy action fields.
Without `--execute`, no homing or action is performed (live input/IK reading
still connects devices). New reports use schema `real10_once_v2_homing`.

Nine mocked lifecycle cases and wrapped-RPY/CLI checks passed remotely:
`simulation_output/real10_homing_logic_check_20260917_v1.json` on both machines.
Real homing and its visual validation remain pending the later operator run.
Keep the previous feeder no-retry and partial-report recovery instructions;
inspect `before.png`/`after.png` for the policy step after an actual attempt.

## One Elite Action Completed, Piper Hold (Historical Stage, 2026-09-17)

The user-run attempt at
`simulation_output/real10_once_left_20260917_123949/` is complete; evidence is
on both machines. Status is `completed_one_action_attempt`, Elite move reply
true, target reached true, cleanup_errors empty, and worker.log empty.
Raw translation norm was 0.018802880 mm (not clipped); SDK TCP readback changed
0.018172600 mm, with XYZ target error 0.002462381 mm. This is controller feedback,
not an independently calibrated physical displacement measurement.

Piper predicted ID=1 (hold probability 0.809075), mapped to command 0. Thus
`piper_send_attempted=false`, `piper_packets_sent=0`, and `hold_no_packet` are
expected. Do not repeat or force feed to make this result look like the earlier
all-feed display run. The nested policy diagnostics' `hardware_executed=false`
describes inference-only behavior; the top-level move/arrival fields record
the separate controller execution. Model-selected physical feed remains untested.

Both before/after PNGs were inspected (`viewed_not_accepted`, raw report unchanged).
The input TCP differs numerically by 409.015071 mm from the earlier live pose
near the recorded nominal start, and orientation differs too. Confirm intended
start pose and unchanged TCP/tool definition before another trial; do not infer
task quality or automatically return home. No new motion, inference or packet
was sent during this readback. The command below is retained for an explicitly
chosen later attempt, not an instruction to repeat the completed run.

## One Live Inference And One Bounded Action Entrypoint (Historical Before Homing, 2026-09-17)

This paragraph records the initial version. The entrypoint's current behavior
and HOME/EXECUTE confirmations are defined in the top section, including for
an old invocation that still supplies `--execute`.

The user requested one real attempt and deferred observed feeder-history work.
The separate `tools/run_real10_pi05_once.py` is synced to the 4090. It reuses the
tested persistent model process/camera reader; the display-only bridge and
trained policy remain unchanged. Ten focused mocked logic/import checks passed
remotely without motion or UDP packets. A read-only Elite check also found
STOP/REMOTE, servo and sync enabled, estop=0, and consistent current-pose IK.
The installed SDK uses `RobotState.PLAY` for the running state; the final
compatibility check caught and corrected an initial `RUN` spelling before
delivery. Dispatch checks were repeated with actual SDK enums. This does not
establish that the new physical attempt has already run.

Run in the **onsite 4090 terminal**, with all other robot/feeder controllers and
camera readers stopped. Confirm the feeder is idle and keep the emergency stop
accessible. After the model loads, type `EXECUTE` once; any other input cancels.
The observation is acquired after confirmation, not replayed from a prior run.

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/run_real10_pi05_once.py \
  --task left --elite-ip 192.168.5.66 \
  --feeder-host 192.168.5.6 --feeder-local-host 192.168.5.11 \
  --max-step-mm 1 --speed 5 --execute \
  --out "simulation_output/real10_once_left_$(date +%Y%m%d_%H%M%S)"
```

This executes **at most one** Elite move (translation norm <=1 mm, unchanged
orientation, joint speed 5 percent), waits for measured arrival, then sends one
feed event only for Piper ID=2 -> +1. Hold sends no packet; this first adapter
rejects retract. It neither enables servos nor homes the robot. A move rejection,
abnormal state or arrival timeout blocks feed and attempts a stop. The first
test's 1 mm cap is not a guarantee of a collision-free path.

Feeder UDP destination is 192.168.5.6:8888 and source bind 192.168.5.11:37011.
The verified adapter is imported by path from the real_collection root; do not
replace hardware sources. There is no automatic resend on absent ACK. One
event is not a calibrated feed distance, and packet send/ACK is not measured
wire motion. History stays missing/validity=0 for this attempt, not synthesized.

Inspect `report.json`, `worker.log`, `before.png` and `after.png`. The report
logs raw/guarded outputs, current/target/measured TCP, IK, attempted/acknowledged
motion and UDP outcomes separately. Failure may happen after a command was
sent: inspect the actual device and partial report before any new invocation.
Use a fresh output directory; never overwrite or blindly retry a run. Ctrl-C
during a pending Elite move attempts stop, but the onsite emergency stop is
still the response to loss of control. No feeder stop primitive is invented.
Omit `--execute` for inference plus IK only, without any motion/feed dispatch.

## Live Camera And Model Inference Completed (Historical Stage, 2026-09-17)

The user reported successful live inference. Read-only SSH inspection and local
artifact readback confirmed the latest run at
`simulation_output/real10_live_display_20260917_120953/`: source=live, task=left,
20 predictions, status=completed, no error or cleanup failure, empty worker log.
No camera/model run or hardware command was repeated during this readback.

Inputs were both expected color-camera serials and Elite current TCP at
192.168.5.66. All Elite deltas were finite, rotation stayed zero, and all Piper
ID=2 outputs mapped to legacy +1 correctly. Typical request roundtrip was about
140 ms (not a validated control rate). No action was sent. The latest preview
was inspected, `viewed_not_accepted`, with the earlier laptop obstruction absent;
raw report visual status remains unchanged. Metadata does not record whether
the run used the desktop window or `--headless`.

Feeder history was missing in all 20 samples, all Piper predictions were feed,
and the maximum raw translation norm was 6.450672 mm. This verifies the live
interface only. The command below remains display-only; there is no execution
flag or completed physical policy test. Proposed next: separately approved,
manual-confirmation Elite single steps with translation limit and low speed;
keep feeder operation manual. No executable single-step entrypoint exists yet.

## Camera-Only SSH Diagnosis Completed (Historical Stage, 2026-09-17)

The user authorized SSH diagnosis after reporting no camera window. Both
expected D435 serials were present on USB3.2, with no device owner blocking
them. A camera-only read through the existing `LatestColorCamera` class passed:
side 250122079856 and top 317222072584 each supplied 25 fresh distinct frames
at the requested 1920x1080@15 color profile. Total check time was 4.707 s.
No camera reset, option tuning, dependency installation, model load, Elite
connection or feeder command occurred. Depth was not tested.

Evidence on both machines:
`simulation_output/real10_camera_diagnostic_20260917_v1/` contains report.json,
preview.png and display_report.json. The saved Chinese dual view was inspected
(`viewed_not_accepted`); it is actual onsite camera evidence, not a model output.
X11 :1 access and a bounded static OpenCV window call exited successfully, with
Qt thread warnings, but the user did not observe the window and explicitly
said reading images is sufficient. Do not claim the GUI problem was fixed or
that its original cause was established. No production code was changed.

The existing live bridge supports `--headless --samples 20`; it writes
predictions and a preview without calling imshow. Camera+model inference was
still pending at this camera-only stage; its subsequent completion and the
cleared top-view obstruction are recorded above.

## Onsite Feeder Address Correction (2026-09-17)

The user confirmed the one-forward feeder test passed after changing only the
destination IP to **192.168.5.6**. Port 8888 and source bind 192.168.5.11:37011
remain the parameters of that command. This supersedes the old 192.168.5.22
destination for current onsite use, not historical capture metadata. Specify
the new address explicitly; hardware-module defaults were not changed.

Operator-run command below sends **one real forward event**; do not repeat it
just to acknowledge the reported pass. Run locally on the onsite computer with
other feeder controllers stopped. No response does not establish that no motion
occurred; inspect the device before considering any retry. One event is not a
calibrated insertion distance. The agent has not rerun this command.

```bash
cd /home/zsw/PycharmProjects/real_collection &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u - <<'PY'
from hardware.feeder_device.udp_controller import UdpFeederConfig, UdpFeederDevice

config = UdpFeederConfig(
    host="192.168.5.6", port=8888,
    local_host="192.168.5.11", local_port=37011,
    timeout_s=2.0,
)
with UdpFeederDevice(config) as device:
    reply = device.feed_once(wait_response=True)
    print("Sent one forward event; response:", reply)
PY
```

Use the collection root for this test: its adapter has source-bind support,
whereas the algorithm-root copy inspected on 2026-09-17 did not. This does not
affect the display-only bridge, which imports neither feeder adapter. No code
was synchronized between those roots. Elite return-to-start and live model
display are separate from this user-reported feeder pass.

## Current: Real10 Display-Only Bridge (2026-09-16)

The user-approved `tools/real10_pi05_bridge.py` is synced to the 4090 project.
It uses the existing hardware Python 3.12 and automatically starts one persistent
model Python 3.10 worker. No conda activation, installation, weight copy, HTTP
service or second terminal is needed. This version has no motion/IK/feeder-send
code or enable-execution flag. Policy output remains canonical; only the legacy
preview computes `piper_step_command = piper_intent_id - 1`.

### Onsite: display live inputs and predictions, without movement

Run in the **4090 desktop terminal** after checking the actual Elite IP and
camera connections. `192.168.5.66` below was read successfully in the 2026-09-17
live run; change it before running if the controller differs.
Do not run the collector concurrently against the same cameras. The serials
below are the ten training episodes' side/top configuration, not the older
July side-camera serial. This opens the two cameras and reads Elite current
pose; it does not enable servos, go to a start pose, run IK, move or feed.

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/real10_pi05_bridge.py \
  --source live --task left --elite-ip 192.168.5.66 \
  --side-serial 250122079856 --top-serial 317222072584 \
  --out simulation_output/real10_live_left_display_v1
```

Wait about 30 seconds for the model to load. Side/top, task, raw Elite delta,
canonical Piper ID and mapped -1/0/+1 command appear in the window. Q/Esc or
Ctrl-C exits and closes only this program's cameras, read connection and model
worker. For the right task, restart with `--task right` and a new output such
as `simulation_output/real10_live_right_display_v1`; do not reuse a completed
directory. The default 0.5 s refresh interval is display timing, not a trained
control rate. It does not make the raw, unclipped delta executable.

If running over SSH without a desktop display, append `--headless --samples 20`:
the bounded run still writes `preview.png`, `predictions.jsonl`, `metadata.json`,
`status.json` and `worker.log` in the output directory. Inspect status/error and
worker.log on failure; preserve partial evidence and use a new output suffix
after resolving the cause. Do not fall back to the old BC motion script.

The bridge does not own or invent current feeder state. Without
`--controller-state-json`, history stays missing (state indices 6/16/27 are zero),
which is schema-compatible but not equivalent to a known idle/count snapshot.
An optional observed-state file must be atomically updated by the actual state
producer with fields like this (timestamp is current Unix time, not a constant):

```json
{"timestamp": 1789545600.0, "piper_step_after_command": 3, "piper_busy": false}
```

The previous iteration's valid snapshot is supplied to the model. Missing or
stale history is never replaced with the current prediction. The count denotes
observed controller events, not millimetres or proof of physical feed success.
First live prediction uses no history. No state-file producer was added here.

### Completed cross-environment replay

The following command already completed; do not overwrite its evidence:

```bash
cd /home/zsw/project_2026 &&
/media/zsw/SSD1T/conda_piper/envs/sam3/bin/python -B -u \
  tools/real10_pi05_bridge.py --source replay \
  --input-json simulation_output/real10_pi05_onsite_preflight_20260916_v1/example_request.json \
  --headless --samples 3 \
  --out simulation_output/real10_pi05_bridge_replay_20260916_v1
```

One model load, three repeats of one recorded input: load 29.272 s; roundtrips
0.316/0.114/0.115 s. First raw delta exactly matches direct inference; class
mapping, invalid-ID rejection, lossless 224x224 BGR transport, missing-history
encoding and worker shutdown pass. Repeated flow-policy outputs need not be
identical: sampling state advances without changing the checkpoint. No robot
or camera SDK was instantiated. All evidence is copied locally; Chinese visual
review is tracked in the handoff. Live SDK/device/GUI interaction remains the
onsite check, not something established by this recorded-input run.

## Real10 Onsite Preflight Completed (Historical Stage, 2026-09-16)

The user requested a readiness audit, not hardware execution. The existing
checker ran on the onsite 4090 and passed; no source or dependency changed.
Completed offline command, **do not overwrite its evidence**:

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_real10_pi05_inference.py \
  --out simulation_output/real10_pi05_onsite_preflight_20260916_v1
```

`report.json` and `example_request.json` were copied locally (38,076 and 561
bytes). All 29 selected observations passed; load 29.177 s, warm median 0.106737 s.
No hardware connection, real IK, UDP send, robot movement or training occurred.
The current CLI is offline inference, **not a live robot startup command**.
See [onsite audit](algorithm-real10-onsite-readiness-20260916.md) for the missing
bridge, two Python environments, camera provenance and critical Piper ID mapping.
Bridge implementation and onsite execution remain separate next steps.

## Saved Terminal-Score Ranking Complete (Historical Stage, 2026-09-16)

The user-approved saved-candidate comparison completed in0.795 s, excluding
startup/SSH. New entrypoint `tools/compare_pusht_terminal_score_ranking.py`
(16,967 bytes) was locally compiled, sequentially SCP-transferred and size-read
back before execution. Its only model calls were one per fixed-final arm,
285 candidate scores each. No training, ACT inference, environment/reset,
candidate continuation, video decode or hardware action occurred.

Completed command, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/compare_pusht_terminal_score_ranking.py
```

Inputs: both `final.pt` files under the paired training root on SSD, plus
`simulation_output/pusht_fresh_scorer_pair20_v1/` current RGBs, observable XY,
saved actions and, only after prediction persistence, its saved outcomes.
Both arms retain the same57 eligible contexts /20 inspected seeds. Original
evaluator analysis reproduces exactly; metric slots are renamed for the new
terminal scores without changing metric math. No original evaluator is edited
or environment entrypoint executed. The source's old fresh-seed claim is not
applied to this reused-development comparison.

Output on both machines:
`simulation_output/pusht_terminal_score_ranking_dev20_v1/` (552,950 bytes).
It contains `started.json`, `status.json`, `report.json`, `predictions.jsonl`,
`predictions.npz`, `per_context.jsonl`, `terminal_score_candidates_zh.png`.
The complete small output directory was SCP-copied locally; no checkpoint
or >100MB transfer was needed. The Chinese figure was inspected and is
`viewed_not_accepted`; raw report retains `not_viewed`.

Reward/visual seed-macro coverage gains:0.003422261/0.000138652; harms9/57 vs10/57;
both informative top1=6/17 and coverage pair agreement81/155. Paired seed mean
+0.003283609 is concentrated: seed300009 contributes83.18%. This is not stable
policy improvement or a new best; keep ACT/real10 and do not promote either.
See algorithm handoff and plan §18 for uncertainty and historical DS0 comparison.

The entrypoint refuses an existing output directory. If interrupted, preserve
partial evidence and inspect status/report; do not delete or automatically
append a new experiment. A completed figure can be redrawn without inference
using the same command with `--render-only`; that recovery path was not needed
or exercised. No new training, fresh-seed or rollout command is provided here.

## Terminal-Score Pair Trained And Verified (Historical Stage, 2026-09-16)

The subsequent user-approved paired run is complete. Entrypoint
`tools/run_pusht_terminal_score_pair.py` (19,960 bytes) was locally compiled,
SCP-transferred and size-read back before execution. No existing scorer,
dataset/cache, ACT/real10 weight, candidate rule or training loss was modified.

Completed command, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_terminal_score_pair.py --train
```

Input: the unchanged prepared contract/targets in
`simulation_output/pusht_terminal_score_pair_pretrain_v1/` and original object
cache. Output root:
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_terminal_score_pair_v1/`.
The `visual_terminal/` and `reward_terminal/` subdirectories each contain
`last.pt`, `final.pt`, `metrics.jsonl`, `status.json`, `report.json` and
`validation_predictions.npz`. Root `initial_trainable.pt` / `sampled_rows.npy`
record the identical starting parameters and all ordered sampled rows;
`run.json` records this authorization separately from the historical preparation.
Root `report.json` and `validation_score_readback_zh.png` summarize the run.
Total output is 9,029,907 bytes, no large-file transfer or cleanup needed.

Both arms finished10,000 updates; training took16.436/15.926 s, excluding
startup and final validation/reload. Each arm's100 log rows and final optimizer
state/reload were verified. Each preserves21,958/2,002 windows and117,633
parameters. Current-observation/action inputs exclude future/reward/raw pose.
Final-only own-target MAEs are0.052068737/0.049758308; different units, no
cross-target winner. Environment steps, candidate continuations and hardware
actions are0. Small reports/logs/predictions/image are copied locally under
`simulation_output/pusht_terminal_score_pair_v1/`; weights remain on SSD.

Recovery for an **incomplete instance of this same authorized run only**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_terminal_score_pair.py --train --resume
```

The runner saves every1,000 updates and after a graceful SIGINT/SIGTERM/SIGHUP;
it completes the current update, saves and does not start the next arm. Resume
restores model/optimizer and starts at the next fixed schedule row, skipping
completed arms. A completed root is refused even with `--resume`. Recovery is
implemented but was not exercised in this uninterrupted run. Do not delete
artifacts to bypass the refusal; a new run requires an explicit new scope.
The small new checkpoint loader is `run_pusht_terminal_score_pair.load_final`;
the reused model's historical `predict_effect` method returns a terminal
score for these checkpoints, not the old DS0 effect delta.

Next proposed task is same-context/same-candidate development ranking of the
two fixed checkpoints, separately scoped; no new training or environment
command is provided here. Keep ACT/real10 and defer DS1. See plan §17.

## Native Reward Audit And Paired Labels Complete (Historical Stage, 2026-09-16)

Both approved numerical preparations completed; **no training or environment
run**. SSH passed before work. New entrypoints were SCP-transferred and size-
read back; current sizes are 12,550 and 7,658 bytes. The geometry script's
post-run change only removed an unused path and clarified a decoder docstring;
there was no numerical rerun. Existing sources/checkpoints were not modified.

One small decoding dependency was added remotely, without dependency upgrades:

```bash
/home/zsw/miniconda3/envs/project2026-pi/bin/python -m pip install \
  --disable-pip-version-check --no-deps numcodecs==0.13.1
```

Completed commands, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_reward_geometry.py

cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_pusht_terminal_score_pair.py
```

The first downloads only the pinned official state/action/episode-boundary
members (327 files, 430,143 bytes), not images, then statically computes native
and historical-pose geometry. It took 44.188 s including download, 6.746 s for
geometry. The second took 0.063 s, using existing RGB features and reward labels.
The harmless installed Pygame `pkg_resources` deprecation warning was observed;
it did not affect either result and no unrelated dependency repair was made.

Inputs: existing snapshot/cache/reward-v1 directories plus the raw numerical
cache `/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_raw_numeric_f93fc579/`.
`numeric_source.json` pins `lerobot-raw/pusht_raw@f93fc57921866e96d8ae00efbe5bb146a04b088f`
and lists every downloaded member/size. No hash sweep or full video download.

Outputs:

- `simulation_output/pusht_reward_geometry_audit_v1/`: `report.json`,
  `geometry_diagnostic_targets.npz` (privileged, offline only).
- `simulation_output/pusht_terminal_score_pair_pretrain_v1/`:
  `comparison_contract.json`, `paired_terminal_targets.npz`, `report.json`.

Both refuse an existing output directory. After an interrupted future
authorized download, the audit reuses matching pinned-size numerical members.
After a completed download, `--local-only --output <new-path>` avoids retrieval;
point the paired preparation's `--audit` to that new completed audit if needed.
Never remove successful old output, silently change split, or rerun training.

All 25,650 native next-frame reward values match stored float32 exactly;
both paired targets retain the original 21,958/2,002 windows. The future paired
run is fixed to two terminal targets, seed20260915, 10,000 steps each and batch64
with identical model initialization/sampling and unchanged architecture/loss.
See the prepared JSON contract and plan §16. No training runner or training
execution is provided/authorized by these commands. Keep ordinary ACT/real10;
the next scoped work is implementing the paired runner, not expanding DS1.

## Recorded-Reward Sidecar Prepared (Historical Stage, 2026-09-16)

The authorized label-interface preparation completed once in 0.277 s excluding
startup/SSH, CPU only. Two new small files were SCP-transferred and size-read
back before execution (5,906 and 9,554 bytes). Neither LeRobot nor video decode
is needed by this entrypoint; it uses remote NumPy/PyArrow/Torch and the existing
object-grid cache. No checkpoint/model forward, optimizer or environment run.

Completed, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_pusht_recorded_reward_targets.py
```

Inputs (read-only):

- `/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/`
- `/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_7628202a/data/chunk-000/file-000.parquet`

Outputs, copied back locally:
`simulation_output/pusht_recorded_reward_targets_v1/{manifest.json,targets.npz,report.json}`.
The default refuses an existing output directory; if preparation fails, inspect
the error and partial output, then use an explicitly named `--output` directory
for any authorized retry. Do not delete/overwrite successful artifacts or
regenerate the original RGB cache. This successful run needed no retry.

Adapter API: `RecordedRewardData(existing_object_data).batch(anchors)` returns
`((current_grid, agent_pos, actions), terminal_reward)`; labels are not inside
the input tuple. The documented index is `next.reward[t+7]` for outcome t+8.
The original 21,958/2,002 eligible windows and inputs are unchanged. Train-only
target mean/std: 0.2984111475702125 / 0.2741044058242909. All 16 real batch
input values and target indices checked equal. Missing initial reward remains
NaN with a validity mask (180 train / 20 val windows); no delta-training mode.

This is **recorded reward**, not verified exact current-environment coverage.
Historical converter semantics and all 206 episode tails are consistent, but
the original block pose / exact conversion runtime is absent. Future training
also needs a matched terminal-vs-terminal or recovered-delta comparison, since
old DS0 used visual improvement. Read plan §15 before any next action. No
training command is authorized or implemented by this section; keep ACT/real10
unchanged and do not use inspected candidate outcomes as fresh test data.

## Saved Goal Alignment Audit Complete (Historical Stage, 2026-09-16)

The approved read-only audit completed once in 0.209 s excluding startup/SSH.
The 12,715-byte script was sequentially SCP-synced and remotely size-read back
before execution. No new model/environment run, training or checkpoint load.

Completed, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_goal_alignment.py
```

Inputs: `simulation_output/pusht_fresh_scorer_pair20_v1/` report/predictions/
outcomes/RGBs and original training goal under
`/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/`.
Only NumPy/Pillow/SciPy and the unchanged RGB-object parser are used; no torch,
policy or environment entrypoint. It preserves the original 57-context group,
reproduces original scores/gain/regret exactly, and decomposes coverage regret
into the proxy-goal term and signed model-ranking term. The 96x96 comparison
is post-hoc sensitivity with identical masks/goal, not a changed policy target.

Results on both machines: `simulation_output/pusht_goal_alignment_audit_v1/`
(`protocol.json`, `report.json`, `per_context.jsonl`,
`goal_alignment_cases_zh.png`). All four outputs are small. Chinese figure is
`viewed_not_accepted`; generated report remains `not_viewed`.

DS0 harms: 4 proxy-ranking mistakes + 1 proxy-optimal coverage harm; residual:
3 + 2. Three proxy-oracle harms survive both the tie-set check and full96 RGB
mask scoring. Keep ACT/real10; defer DS1 and propose a task-aligned target/data
contract before another architecture experiment. See plan section 14.

The script refuses an existing output directory. If a future authorized audit
fails, inspect its traceback/partial output without changing old evidence or
silently rerunning model/environment jobs. This completed audit needed no retry.
No training/label rewrite command is introduced by this section.

## Fresh-Seed Scorer Comparison (Historical Stage, 2026-09-16)

The saved-evidence follow-up proposed by this completed comparison is recorded
above; the original frozen comparison and metrics were not rerun or modified.

The separately approved fixed20-seed comparison completed once in48.541 s
excluding process startup/SSH. The small entrypoint/protocol was sequentially
SCP-synced and size-read back before execution. Prepare inspected70 metadata
files and froze seeds300000-300019 with no recorded project-seed overlap;
public demonstration-generation seeds remain unknown. No new training.

Completed once, **do not repeat, extend or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_fresh_scorer_pair.py --stage prepare

cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_fresh_scorer_pair.py --stage run
```

`prepare` has zero model/environment execution and writes the protocol,
checkpoint identity and seed inventory. `run` requires these unchanged; it
refuses an attempted output directory, rechecks seed overlap, and saves both
predictions before each anchor's future. Fixed ACT prefixes, five unchanged
8-step candidates/offset8 and anchors0/80/160; both scorers share the same actual
outcomes. Frozen ACT final100000, DS0/residual final10000 and original goals.

Actual budget: nominal3255 + replay22800 + candidate2320 =28375 environment
steps, below29760;20 seeds,58 contexts,290 complete candidate futures.300011
ended at63, leaving two anchors unavailable without replacement. Invalid current
vision at300006/0 retains ACT;57 contexts/20 seeds enter the common metric group.
No clipped/rejected candidates or future-visibility exclusions. Exact native
replay/reference checks passed. Optimizer/hardware steps and model changes0.

Results on both machines: `simulation_output/pusht_fresh_scorer_pair20_v1/`
(`protocol.json`, `seed_inventory.json`, `started.json`, `status.json`,
`report.json`, `resets.jsonl`, `predictions.jsonl`, `outcomes.jsonl`,
`nominal_prefix.jsonl`, `candidate_steps.jsonl`,348 RGBs under `observations/`,
`first_seed_shared_candidates_zh.png`). The small compressed readback archive
is also retained locally; no >100 MB user-transfer requirement was triggered.

Primary seed-macro coverage gain/regret: DS0 0.003946307/0.001316460, residual
0.003860830/0.001401937. Paired gain delta+0.000085477, sample SD0.002551210,
3 better/13 tied/4 worse seeds. Both harm5/57; DS0 worst loss-0.019613539 vs
residual-0.014045329. Only17 contexts are coverage-informative; top1 6/17 vs9/17.
No stable DS0 advantage or closed-loop success evidence. See plan§13 for the
object-goal metrics and separately labeled read-only proxy diagnostic.

The fixed-seed Chinese image was reviewed `viewed_not_accepted`; report retains
its original `not_viewed`. If a future approved variant fails, preserve its
report/traceback/partial predictions first; this entrypoint has no resume or
automatic seed replacement. Do not delete attempted output or rerun environment
execution for a figure. This completed run required no execution recovery.

Stop at this budget. Keep ACT/real10; no DS1 command or training is authorized
by this result. Do not reuse these inspected seeds as an untouched final test
after changing a model, goal, candidate generator or threshold.

## DS0 Saved-Candidate Offline Comparison (Historical Stage, 2026-09-16)

The pending fresh-test statements below describe the earlier development stage
and are superseded by the completed frozen comparison above.

The separately approved offline run completed once in0.731 s excluding startup/
SSH. The small new entrypoint was sequentially SCP-synced and size-read back.
One DS0 forward scored135 candidates on27 eligible contexts, using the unchanged
28-context/140-future saved development set. No optimizer/environment/hardware
steps, ACT inference, history recovery, full validation rerun or weight change.
Predictions were persisted before future targets were opened; the original
residual summary reproduces exactly. This is reused development evidence.

Completed once, **do not repeat or overwrite**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/compare_pusht_direct_scorer_ranking.py
```

Read-only sources: `simulation_output/pusht_object_action_ranking_dev10_v1/`
and `simulation_output/pusht_object_residual_ranking_dev10_v1/`; DS0 checkpoint
remains `/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_direct_action_scorer_v1/final.pt`.
Results on both machines: `simulation_output/pusht_direct_scorer_ranking_dev10_v1/`
(`report.json`, `started.json`, `status.json`, `predictions.npz/jsonl`,
`analysis.jsonl`, `per_context.jsonl`, `saved_candidate_direct_scorer_zh.png`).

DS0 coverage gain/regret0.003484408/0.003932389 vs residual0.003460159/0.003956638.
Both harm1/27 contexts; worst gain-0.025082150 vs-0.041008277. Informative top1
falls4/7->2/7; only7 contexts across5 seeds are coverage-informative. Mean gain
delta+0.000024250 is not stable superiority or a success-rate improvement.
Uniform is an analytical expectation; both oracles remain post-hoc only.

A figure-only layout correction was completed with the same entrypoint plus
`--render-only`, after syncing the revised file. It reads saved predictions and
metrics, changes only the PNG, and performs no model/environment execution.
The final Chinese figure was checked: `viewed_not_accepted`; original report
remains `not_viewed`. No extra smoke or checksum sweep was added.

If interrupted/failed, inspect report/status and preserve partial output. The
default command refuses an existing directory; do not delete it or rerun model
inference just to regenerate a figure. `--render-only` requires a completed run.
No recovery was needed for this completed comparison.

Keep ACT/real10 unchanged. No further training or candidate rollout is started.
Next proposed scope is an unused-seed comparison with frozen models and shared
candidate outcomes, after its own seed/protocol/budget approval; no command for
such an independent test has been prepared or executed in this turn.

## DS0 Fixed Training And Readback (Historical Stage, 2026-09-16)

The pending-selection paragraph in this older stage is superseded by the
separately approved completed offline comparison above.

The separately approved fixed10000-step run completed once, without resume.
Final checkpoint/reload,100 finite log intervals and optimizer step10000 are
verified. Training/final-evaluation timer16.704 s excludes startup/SSH. The
existing2002-window/20-episode validation gives DS0 effect MAE0.041895673,
RMSE0.066026986,bias-0.004384350; episode-macro MAE0.042326281. Same-goal MAE
is lower than zero-effect0.064914733 but higher than residual0.038677901 and
history0.037660427. No candidate-selection or policy-benefit evaluation.
Keep ACT/real10 unchanged; do not repeat the completed training or add steps.

Implemented `tools/pusht_direct_action_scorer.py` and
`tools/run_pusht_direct_action_scorer.py`. SSH/cache availability passed; final
small files were sequentially SCP-synchronized and remote sizes/`--help` read
back before the single completed preparation check. No LeRobot was installed
locally; existing models, data sources and checkpoints were not modified.

Completed once (0.379 s excluding Python startup/SSH), **do not repeat**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_direct_action_scorer.py --stage check
```

Result on both machines: `simulation_output/pusht_direct_scorer_pretrain_v1/`
(`report.json`, `started.json`, `target_example.npz`, `supervision_alignment_zh.png`).
117633 parameters; unchanged21958/2002 windows and186/20 episodes; train-target
mean/std0.043009500695/0.104149264582. Positive/near-zero/negative11247/5699/5012.
One64-row training batch passes target-index/NumPy-score, gradient and unchanged-
parameter checks; all8 action steps participate. In-memory roundtrip and minimal
candidate mask/tie/fallback checks pass. No optimizer/checkpoint file, validation
model inference, old development-candidate read, environment or hardware step.
Target-only Chinese image reviewed `viewed_not_accepted`; report `not_viewed`.

Completed once after the separate training approval; **do not repeat**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_direct_action_scorer.py --stage train
```

Fixed seed20260915,10000 updates,batch64,AdamW3e-4,wd1e-4,clip1. MSE of
train-standardized observed goal improvement; no fake negative labels, dynamics
auxiliary, intermediate validation or checkpoint selection. Final evaluation
reports raw effect MAE/RMSE/bias against zero-effect persistence and training-
mean effect, including sign/near-zero strata and episode macros. This is the
existing reused validation set, not a fresh policy test. No automatic candidate
ranking, environment replay or policy switch follows training.

```text
read-only cache:
  /media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/
completed training output (weights remain remote):
  /media/zsw/SSD1T/project_2026_weights_v1/training/pusht_direct_action_scorer_v1/
present:
  run.json / metrics.jsonl / status.json / last.pt / final.pt / report.json
local small-file readback:
  simulation_output/pusht_direct_scorer_train_v1/
  run.json / metrics.jsonl / status.json / report.json
```

The prior runtime estimate was tens of seconds to a few minutes; the completed
run recorded16.704 s. Neither that aggregate nor the preparation's0.0605 s
cold forward/backward establishes online-control latency. Training and final
evaluation are now verified; the resume path was not exercised.

For an authorized run, Ctrl+C/SIGTERM requests a save after the current update.
Resume only an interrupted run by appending `--resume` to the same command;
it loads last.pt including optimizer and sampler state, preserving the total
10000-step budget. Completed reports refuse resume. Unexpected termination can
leave log rows newer than the last periodic checkpoint; `resume_from_step`
identifies rerun log segments, not additional unique completed updates.
Do not delete a partial run or overwrite old checkpoints to obtain a fresh run.

Completed final readback once (0.660 s excluding startup/SSH):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/inspect_pusht_direct_scorer.py
```

The new small inspector was SCP-synced before execution. It reads final reports,
checks matching cache/split/goal/normalization/shared budgets/final sampler
states, and makes one forward on the already-fixed validation episode0/frame76.
No full validation rerun, old candidate scoring, RGB decoding, environment step
or hardware action. Both machines have:
`simulation_output/pusht_direct_scorer_inspection_v1/{report.json,started.json,fixed_validation_direct_scorer_zh.png}`.
The picture shows actual current/future/goal grids and scalar predictions; DS0
does not generate images. Fixed-example target+0.031422,prediction-0.100628.
Chinese sheet inspected as readable, `viewed_not_accepted`; original report
retains `not_viewed`. The training approval is not user visual acceptance.

Next proposed scope, not yet executed or authorized by this training approval:
offline DS0 scoring of the already-saved development candidates, comparing
actual selection gain/regret with ACT/uniform/residual/oracles. No new fitting,
history recovery or environment replay is required. The run is not a fresh test.

## Candidate/Scorer Design And Saved-Result Readback (2026-09-16)

Historical design-stage record; its pending implementation statements are
superseded by the completed DS0 preparation above.

See [the current plan](algorithm-action-selection-plan-20260916.md) and the top
of the algorithm handoff. This milestone is documentation plus read-only JSON /
source inspection: no new model, training, inference or environment command.
The former adjacent-history recovery/ranking proposal is deferred. Do not
automatically run old training/replay commands below.

Completed local PowerShell readback (no LeRobot dependency, files read only):

```powershell
$rows = @(Get-Content simulation_output/pusht_object_residual_ranking_dev10_v1/analysis.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { $_.variant -eq 'residual' -and $_.eligible })
$headroom = ($rows | ForEach-Object {
  ($_.actual_coverages | Measure-Object -Maximum).Maximum - $_.actual_coverages[0]
} | Measure-Object -Average).Average
$gain = ($rows | ForEach-Object { $_.strategies.model.coverage_gain_vs_ACT } |
  Measure-Object -Average).Average
[pscustomobject]@{
  eligible = $rows.Count
  informative = @($rows | Where-Object { $_.coverage_varies }).Count
  oracle_headroom = $headroom
  selected_gain = $gain
  selected_regret = $headroom - $gain
  ratio_of_means = $gain / $headroom
}
```

Observed27/7 contexts; headroom0.007416796852, gain0.003460158696,
regret0.003956638156, ratio46.65%. This is8-step development coverage, not
policy success or a fresh held-out result. Per-context details, candidate
semantics, supervision limits and future experiment order are in the plan.
The real10 single-step boundary is confirmed in `build_config` of
`tools/probe_pi05_lerobot_adapter.py` and `Real10PI05Policy.predict` of
`tools/real10_pi05_policy.py`.

Next implementation would reuse the existing object-grid cache and five
candidates. No direct-scorer entrypoint exists yet; do not invent/run a launch
command. After implementation, synchronize only final changed small files and
perform the one relevant remote batch check. A training command will be added
when runnable and separately authorized. All prior checkpoints remain protected.

Small documentation synchronization, sequentially from the local project root:

```powershell
scp docs/algorithm-action-selection-plan-20260916.md project4090:/home/zsw/project_2026/docs/
scp docs/algorithm-track-handoff.md project4090:/home/zsw/project_2026/docs/
scp docs/algorithm-track-commands.md project4090:/home/zsw/project_2026/docs/
```

This transfers documentation only, not weights or a training request. Read back
the new plan heading and current-priority headings after transfer; no SHA or
model smoke is needed for these document edits.

## BC/ACT Results Audited; World-Model Research Resumed (2026-09-14)

The user resumed offline BC/ACT/world-model research. Real10 weights and its
inference interface are preserved; live camera/robot integration waits for the
user to be onsite. The original BC/ACT benchmark already completed: do not run
its historical launch command again.

Verified clean Push-T results: BC0/100, ACT22/100; mean episode-max coverage
0.202458 /0.512645. Both have zero action-bound violations and strict protocol
pass. Fixed100 seeds100000..100099 and final step100000 weights are unchanged.
DP67/100 is an external pretrained reference, not same-data/budget training.

Completed short post-run audit/recorded-action replay (not a new benchmark):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_bc_act_benchmark.py --replay-first \
  --out simulation_output/pusht_bc_act_benchmark_audit_v1_diagnose
```

Outputs: `report.json` and `first_seed_replay_zh.png`, retrieved locally.
All57783 original BC/ACT steps/action records were audited; a600-step recorded
first-seed replay matched observations and outcomes exactly in this attempt.
No checkpoint loading, model inference, optimizer or hardware. The earlier
default `..._audit_v1/` replay stopped on ACT outcome equality; the diagnostic
retry changed only error reporting and did not reproduce it. Preserve that
failure; no general cross-process determinism claim. No automatic retry.
The completed command is provenance, not a request to rerun. A future required
log-only readback can omit `--replay-first` and use a fresh `--out`.

World-model compatibility and the first candidate/goal-score interface are now
checked below. No rerun of the completed projection-scale study is needed.
The user's subsequent approval advanced one fixed native visual-dynamics run,
now completed below. Do not repeat/extend it or download new weights/packages;
refer to the current algorithm handoff for the mixed-result decision.

## ACT + Visual-Goal Interface Check: Completed (2026-09-15)

New independent entrypoints: `tools/pusht_world_model_adapter.py` and
`tools/check_pusht_world_model_adapter.py`. Their small final versions were
transferred by sequential SCP and read back before execution. They reuse the
existing frozen ACT loader; they do not modify the pinned BC/ACT implementation,
training data/split, protocols, checkpoints or real10 interface.

Completed4090 command (18.1060 s; provenance, not a request to repeat):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_pusht_world_model_adapter.py \
  --out simulation_output/pusht_act_wm_interface_v1
```

Output directory creation is exclusive. If a later implementation change makes
a repeat necessary, use a fresh `--out`; preserve any failed partial output and
read its terminal error first. No checkpoints are written and no resume is
needed for this short check. No auto-download, full-dataset decoding or training.

Outputs: `report.json`, `candidate_interface_zh.png` (both retrieved locally).
Status `passed_interface_only`; real frozen ACT candidate0 exactly matches its
first8 queued actions; partially consumed queue unchanged. Actual feature shape
`[1,9,512]`, candidate shape `[1,5,8,2]`; all five reset candidates are valid.
The sole goal image is the current RGB identity fixture. Scoring uses explicitly
fabricated features with known costs `[4,1,9,0.25,16]`; its index3 is **not** a
world-model action recommendation. No simulator step, optimizer step, hardware
action or new policy success result. The existing gym import notice and pygame
`pkg_resources` deprecation warning did not prevent completion; no environment
upgrade was made to suppress them.

API sequence for the future predictor (not a complete runnable policy):

1. `load_final("act")` from the existing inference module, then construct
   `FrozenACTPrior(policy, binding)`.
2. `prior.prepare(observation, offset_xy=8)` returns only current spatial visual
   features, raw observable2D agent position, and explicit absolute-coordinate
   candidates. Reference+four ramp offsets, horizon8; no clipping.
3. Explicitly encode a declared task-goal RGB using `prior.encode_image(goal_rgb)`.
   The identity fixture in this check is not that task goal. Do not select a goal
   from validation/future frames or simulator truth.
4. A separately trained compatible predictor must produce `VisualFeatures`
   shaped `[B,5,8,9,512]` with the same `space_id`, predicting each post-action
   control step. It owns its train-only state/action normalization. That model
   is not implemented or loaded by this adapter.
5. `score_visual_goal(candidates, predicted, goal)` returns terminal spatial-MSE
   costs, selected index/chunk and first action. Invalid alternatives are masked;
   ties keep the reference. Nothing executes automatically. The existing ACT
   baseline keeps its own original queue; a future combined executor is separate.

This provisional encoder is ACT's frozen3x3 ResNet map, **not DINOv2**. DINO-WM
defaults differ in relative/scaled actions, velocity input and224px image
normalization; see the primary-source compatibility links in the handoff. No
old LIBERO/guidewire world-model checkpoint is native2D-compatible. Next work
required task-goal provenance and compatible learned visual dynamics before
any policy comparison; that next implementation/run is recorded below.

## Native Spatial Visual Dynamics: Completed First Run (2026-09-15)

Independent new files: `tools/pusht_visual_dynamics.py` and
`tools/run_pusht_visual_dynamics.py`. Existing BC/ACT/real10 sources, checkpoints,
source dataset and split are unchanged. Only small code/report/image transfers;
the472780800-byte cache remains on the remote weight disk. No dependencies added.

Completed preparation and one cohesive real-batch check:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_visual_dynamics.py --stage prepare-check
```

Preparation7.6948 s, check0.5721 s. All25650 frozen image features,22000/2002
full train/val windows, zero cross-episode padding. Existing186/20 split and
train-only normalization; model380288 parameters. Real-batch forward/backward,
causal prefix, action gradient, candidate masks, goal-score wiring and exact
initial state-dict reload passed; zero optimizer/environment steps in the check.
No repeated fine-grained smoke suite. Target RGB is episode1/frame117/index278,
fixed from training only, visually inspected without using success/pose labels.

Cache: `/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_act_wm_v1/`
contains `features.npy`, `arrays.npz`, `manifest.json`, `goal_rgb.png`, and
`goal_source_zh.png`. Manifest/goal images and check report are local in
`simulation_output/pusht_visual_dynamics_pretrain_v1/`. Never fetch the large
cache automatically. It is re-creatable from the pinned local source only if
explicitly needed; use a fresh `--cache`/`--out`, not overwrite or rerun now.

Measured runtime made the approved10000-update run a <5-minute logical job,
so the agent completed it after reporting the estimate. Completed command:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_visual_dynamics.py --stage train
```

Fixed seed20260915/10000 updates/batch64/AdamW0.0003; final-only validation.
Completed in40.6483 s including final evaluation/reload, optimizer10000,
environment0. Training root:
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_act_wm_v1/`.
Artifacts: `run.json`, `status.json`, `metrics.jsonl`, `last.pt`, `final.pt`,
`report.json`. Small evidence retrieved locally under
`simulation_output/pusht_visual_dynamics_train_v1/`. No RGB decoder/rollout.

Interruption recovery, only for an incomplete authorized run: SIGINT/SIGTERM
requests saving after the current update. Reuse the exact command with
`--resume`; it restores model, optimizer and sampler generator from `last.pt`
at the same fixed budget, without changing normalization. Completed report
blocks resumption. A hard-killed run replays from the preceding1000-step save;
metric logs can then contain duplicated resumed steps, so do not count log rows
as total updates. This completed run had100 unique100-step log records and no
interruptions. No command above is a request to repeat it.
Resume is implemented but an interrupted/resumed training path was not exercised
in this uninterrupted run; do not claim crash-recovery validation.

Completed original-RGB / feature-error inspection (fixed first-val-episode
middle window; no selection by error, no new optimizer/environment steps):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_visual_dynamics.py --stage inspect
```

Outputs in `simulation_output/pusht_visual_dynamics_inspection_v1/`:
`report.json`, `prediction_inspection_zh.png`; downloaded locally. The3x3
heatmaps are feature errors, **not** future RGB/pixel errors. Future validation
RGB is inspected as offline evidence, never used as the planner's goal.
The Chinese goal/inspection sheets were viewed by the agent and remain
`viewed_not_accepted`; reports preserve their creation-time `not_viewed` state.

Result relative to persistence: normalized MSE -10.90%, terminal raw MSE
-14.44%, but raw MAE +11.17% and goal-cost MAE +7.71%. Mean-action input ablation
only worsens terminal MSE1.26% relative to observed actions. Same20 episodes,
single training seed, no learned candidate-rank or policy-success evidence.
Do not adopt the checkpoint for ACT reranking or extend training automatically.
Keep the original ACT22/100. See the handoff for per-episode/macros, limitations
and the proposed small closed-loop diagnostic (not yet implemented or run).

Future API: `load_final(final_path)` from `pusht_visual_dynamics` returns a frozen
model in the same ACT feature space. Use
`model.predict_candidates(current_features, agent_xy, candidates)` to predict
only valid native chunks, then the existing `score_visual_goal`. This is a
callable inference interface; the subsequently user-approved public-task
diagnostic executor is below, not a hardware/controller integration.

## ACT vs ACT+WM Fixed20-Pair Diagnostic (2026-09-15)

The user approved the small closed-loop comparison. New independent runner
`tools/run_pusht_act_wm_paired.py`; no edits to the existing ACT/BC model,
world-model checkpoint, training code or real10 interface. Fixed20 paired seeds
100000..100019, maximum300 steps each, alternating condition order. Both execute8
from the same ACT chunk16 prior; WM scores the unchanged five8-unit XY-ramp
candidates using the fixed training-only image goal. No new optimizer steps,
oracle policy input, action clipping, goal/mask/candidate tuning or hardware.

Two failures before environment construction/stepping were inspected and kept
as `..._paired20_v1/` and `..._paired20_v1_cached_goal/`: single/batched goal
re-encoding differed at approximately2e-6. Original PNG pixels are exact and
the original source batch reproduces the goal cache exactly. Scoring now uses
that exact cache row, RGB normalization follows CPU data-adapter order, and
the independent RGB-feature readback uses a rounding-scale tolerance only.
No model or score setting was adjusted. Current fresh-attempt command:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_act_wm_paired.py \
  --out simulation_output/pusht_act_wm_paired20_v1_fixed_cache
```

Expected logical runtime1–3 minutes, executed by the agent within approved
short-job scope. The output must be new. On failure/interruption, preserve
`report.json`/partial logs, inspect the reason, and use a separately justified
fresh `--out` only if needed. There is no auto retry, mid-episode resume or
automatic continuation beyond20 pairs. Completed outputs must not be rerun.

Outputs: `started.json`, `status.json`, `report.json`, `episodes.jsonl`,
`steps.jsonl`, `decisions.jsonl`, `first_seed_pair_zh.png`. First-seed image sheet
records actual reset/100/200/end snapshots, not model-predicted RGB or a
cherry-picked success. Success/coverage are read only from actual post-step
environment info. Decision logs record costs, valid candidates and selected
versus reference chunks; execution logs record every action and queue position.
Synchronized decision/policy latency accounts for extra WM compute. Metrics
and source data never become policy observations.

The runner includes one zero-step17-action reference-queue equivalence check
and a trained-WM forward. It then completes exactly40 episodes, checking all
reset identities against the unchanged benchmark schedule. Do not claim this
known-seed diagnostic is a fresh100-seed benchmark, equal-training-compute
comparison, multi-seed training stability or real-system validation.

Completed:40 episodes /10978 environment steps in37.1794 s; optimizer0,
hardware0, reference-only17-action check exact. ACT5/20 versus ACT+WM5/20;
two ACT-only and two WM-only successes cancel. Mean maximum coverage
0.581292 ->0.497891. Decision latency5.4181 ->10.0018 ms. WM selected offsets
at all694 decisions (3470 candidate predictions), never reference0. No executed
action exceeded the native bounds or fixed8-coordinate offset budget.
Keep the original ACT checkpoint and100-seed report; no positive policy-gain
claim and no automatic expansion/retraining. The command above is now provenance
for a completed run, not a request to rerun. Detailed paired outcomes,
compute accounting and limitations are in the algorithm handoff.
All small run artifacts and both failed zero-step reports are local as well as
remote. Post-run read-only log audit passed all10978 steps and694 decisions,
including exact float32 ramp reconstruction/queue execution and aggregates.
The first-seed Chinese actual-rollout sheet was viewed and is
`viewed_not_accepted` (agent only); reports retain creation-time `not_viewed`.
Historical ACT subset success and step counts match; two coverage numbers
differ by <=5.6e-16. The fresh paired results are not a bitwise-replay guarantee.

## ACT-Reference-Preserving Gate: Training-Only Check Completed (2026-09-15)

New independent modules `tools/pusht_act_wm_abstention.py` and
`tools/calibrate_pusht_act_wm_abstention.py`. Both small files were transferred
sequentially and parsed remotely before execution. Existing greedy runner and
all model/training/checkpoint/split interfaces are unchanged. No new training,
environment stepping, benchmark extension or hardware actions.

Completed4090 command (7.8547 s; provenance, **do not rerun automatically**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/calibrate_pusht_act_wm_abstention.py \
  --out simulation_output/pusht_act_wm_abstention_v1
```

This entrypoint has no train/rollout or threshold-sweep options. It computes
all22000 training-window observed-action goal-cost residuals, freezes
`2 * q95(method=higher)` in `calibration.json`, then checks one middle full
window per186 training episodes and17 queued inference calls. Existing loaders
load shared containers but no validation index is predicted/evaluated. No20/100
benchmark log is opened. The threshold is in-sample error scale, **not** a95%
confidence or counterfactual-improvement guarantee.

Output must be new. On failure, preserve partial files/read `report.json`;
there is no automatic retry, training resume or episode continuation. Only a
separately justified code correction permits a fresh output; never overwrite
this calibration or silently change the rule in response to diagnostic gains.
The initial remote syntax-precheck command had a shell-quoting error, not a
Python-source failure; stdin-based parsing then passed before the single run.

Outputs (about1.1 MB total, retrieved locally): `started.json`,
`calibration.json`, `training_residuals.npz`, `training_decisions.jsonl`,
`report.json`, `training_gate_example_zh.png`. No large cache/checkpoint transfer.
Training residual q95=0.0063769482; threshold=0.0127538964. All186 inspected
windows preserve ACT exactly;17 actual queued calls equal official ACT across
two boundaries. Threshold/records/split membership were recomputed from saved
artifacts. The Chinese training-observation/action-target sheet was inspected,
`viewed_not_accepted`; immutable report retains `not_viewed`. No new success score.

Callable inference interface for a future explicitly scoped experiment:

```python
from pathlib import Path
from pusht_bc_act_inference import load_final
from pusht_world_model_adapter import FrozenACTPrior
from pusht_act_wm_abstention import load_selector

act, binding = load_final("act")
selector = load_selector(FrozenACTPrior(act, binding), Path(
    "simulation_output/pusht_act_wm_abstention_v1/calibration.json"))
selector.reset()  # per episode only; not per action
native_xy = selector.select_action(observation)  # existing RGB/state allowlist
```

`load_selector` binds the completed calibration to the original feature space,
source/split/goal metadata and fixed final WM path, and loads the exact cached
goal row. No environment is constructed or action dispatched. Each new chunk
still evaluates the same five candidates, so abstention does not save WM compute.
Current decision is ordinary ACT, not adoption of this wrapper. Keep real10 and
existing benchmark results unchanged; see the handoff for evidence limits and
the future paired-action-ranking research question.

## Candidate Action Ranking: Development Diagnostic Completed (2026-09-15)

New independent runner `tools/check_pusht_wm_action_ranking.py`, finalized,
SCP-synchronized and syntax-read back before execution. Fixed development seeds
200000..200009 and ACT-prefix anchors0/80/160, unchanged five candidates/8 steps,
models, goal, score and residual gate. Only this public-task algorithm diagnostic
is run; no collector/environment/model edits, optimizer updates or hardware.

Completed4090 command (26.6329 s; provenance, **not a rerun request**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_pusht_wm_action_ranking.py \
  --out simulation_output/pusht_wm_action_ranking_dev10_v1
```

The fresh output is exclusive. No auto retry, mid-seed resume, replacement
sampling or parameter sweep. If execution fails, preserve the partial report
and inspect its cause before a separately justified fresh attempt. There is no
new training/policy-success-benchmark stage in this entrypoint. Existing report
metadata are read for environment/source parity and reset disjointness only.

Each prediction is written before its continuation. Every candidate starts from
the same native reset+recorded ACT prefix; all replay RGB/state must match and
candidate0 must reproduce actual ACT queue actions and outcomes. No privileged
state injection, post-termination stepping or outcome input to the policy.
Actual future-image visual costs and native coverage are separate diagnostic
targets. Early-done candidate groups are excluded from the8-step primary
comparison; missing anchors are listed without replacement.

Completed28/30 planned contexts,140 full8-step candidate continuations. Nominal
seed200005 ends at76, so anchors80/160 are unavailable. All replay/reference
checks passed. Environment steps13508 =1588 nominal +10800 prefix replay +1120
candidate; optimizer/hardware0. WM visual top1=4/28 (14.29%; analytical uniform
reference20%), visual pair ordering126/280 (45%). Only7 contexts vary in task
coverage: WM choice gives4 better/22 tie/2 worse versus ACT, while choosing the
actual minimum-visual-cost future gives0 better/25 tie/3 worse. This is evidence
to inspect both prediction ranking and goal-task alignment, not a success score,
statistical worse-than-random claim, or authorization to train a new variant.

Remote outputs:

```text
started.json / status.json / report.json
predictions.jsonl              # frozen before candidate execution
nominal_prefix.jsonl          # original ACT actions/outcomes
candidate_steps.jsonl         # every counterfactual executed action
outcomes.jsonl                # future RGB costs/coverage, diagnostic only
analysis.jsonl                # offline combined rankings and regret
observations/                 # individual actual96px RGB PNGs
first_seed_counterfactual_zh.png
```

Reports/logs and the Chinese sheet are local as well as remote; individual
`observations/` PNGs remain remote (whole remote artifact about1.9 MB). Saved-log
readback passed counts, all1120 actions/bounds, ranking and aggregate metrics.
The fixed first-seed all-anchor/all-candidate sheet was viewed,
`viewed_not_accepted`; immutable report remains `not_viewed`. It shows actual
futures, not reconstructed/generated model images. Gym import/pygame deprecation
notices were non-blocking; no environment upgrade was performed.

Keep ordinary ACT active. The next proposed work is an image-derived task-goal
representation/scoring interface, not a threshold sweep or more benchmark runs.
No such new representation, training or real10/hardware change follows
automatically; see the algorithm handoff for the full evidence boundary.

## RGB Object/Goal Interface: Saved-Image Check Completed (2026-09-15)

New files `tools/pusht_object_goal.py` and `tools/check_pusht_object_goal.py`.
Task-specific RGB96 -> gray T mask ->24x24 occupancy, retaining image position
and orientation; same fixed training goal RGB. Terminal quadratic soft-Dice
loss, no alignment or simulator truth. Fixed RGB thresholds and border rejection
are recorded in code, `goal.json` and the report. This does not change/load ACT,
WM, their losses, action interface or checkpoints.

Completed once after sequential SCP and remote syntax readback. Command on4090
(provenance only, **do not rerun**):

```bash
cd /home/zsw/project_2026 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_pusht_object_goal.py
```

Fixed sources:

```text
simulation_output/pusht_wm_action_ranking_dev10_v1/
  predictions.jsonl / outcomes.jsonl / analysis.jsonl / observations/
/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_act_wm_v1/
  manifest.json / goal_rgb.png
```

Only existing small saved files are read. NumPy/SciPy/Pillow are already in
`project2026-pi`; the reused pair-ordering helper also imports the existing
Torch/baseline modules, but no model is loaded and no environment is created.
The goal stays train episode1/frame117/global278. All RGB-only outputs are
saved before the coverage sidecar join. Actual future masks are offline
targets, never observations or predictions for a deployable policy.

Default output `simulation_output/pusht_object_goal_dev_v1/` is exclusive:

```text
started.json / goal.json / report.json
object_grids.npz              # current/future grids and fixed goal grid/mask
image_observations.jsonl      # image-only validity, centroid and source roles
rgb_only_scores.jsonl         # persisted before coverage join
comparison.jsonl             # offline join, actual-future ranking/coverage
goal_object_template_zh.png
first_seed_object_goal_zh.png
```

No automatic retry/resume, resampling or threshold sweep. On failure preserve
the partial output and inspect its report before declaring a fresh attempt.
There is no training/model-evaluation stage hidden in this command. Recorded
analysis-body runtime0.167 s excludes process/import/SSH startup; all checkpoint,
optimizer, environment and hardware counters are0.

Completed28 retrospective comparisons,167/168 RGBs valid:27/28 current plus
140/140 terminal. Goal has219 visible pixels. Identity-score/tie and incompatible
old-feature rejection checks pass. The failed current frame200004/0 touches
the lower image border (252-pixel component at x25..50/y71..95), and is reported
invalid without fallback/retuning. It does not exclude its valid *terminal*
futures from offline scoring; no current-input planning readiness is implied.

Actual-future score/coverage pair ordering53/58 (91.38%), versus old27/58
(46.55%); only7 contexts from5 seeds have differing coverage. Best coverage
choice5/7 versus0/7. Coverage gain vs ACT6 better/22 tie/0 worse, mean+0.007015518;
old0/25/3, mean-0.001767961. These are retrospective choices, **not a new policy
success rate**. All28 prior-analysis choices match; reports/logs/NPZ/PNGs were
retrieved locally. Both Chinese sheets were inspected, `viewed_not_accepted`;
immutable report remains `not_viewed`. The failed current RGB is also copied
locally as `invalid_current_200004_0.png` for evidence review.

Next proposed scope: action-conditioned object-grid predictor interface and
training-only objective declaration. Old `[B,5,8,9,512]` dynamics outputs/gate
must not be padded, reshaped or reused as `[B,5,8,24,24]` object forecasts. This
check broadcasts only the measured terminal grid across8 steps as an explicit
API fixture, not an observed/predicted trajectory. Keep ordinary ACT; no new
training, gate tuning or benchmark follows automatically. See handoff for
border/occlusion, disjoint-mask flat-score and development-evidence limits.

## Object-Grid Dynamics: Preparation And Training Commands (2026-09-15)

Preparation was completed first; the subsequent approved training is now also
complete. See the final-training readback section below; do not rerun either.

New `tools/pusht_object_dynamics.py` and `tools/run_pusht_object_dynamics.py`:
248256-parameter GRU128 predicts native8-step24x24 visible-object grids from
current RGB-derived grid, observable agentXY and native XY action prefixes.
No old model checkpoint, environment, hardware or coverage is an input. Goal
and frozen RGB parser are unchanged. Training loss is all8-step quadratic
soft-Dice to subsequent RGB-derived grids, not to the fixed task goal.

Completed once on4090 after finalized small-source SCP and remote syntax check
(provenance, **do not rerun**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_dynamics.py --stage prepare-check
```

This decoded/extracted all25650 pinned RGBs in8.633 s and checked one real
training batch in0.425 s, without creating an optimizer, training checkpoint,
validation-model forward, environment or hardware action. Reused existing
loader provenance checks; no new hash suite or independent smoke files.

Paths:

```text
source: /media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_7628202a
cache:  /media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1
  started.json / manifest.json
  grid_counts.npy              # uint8[25650,24,24], divide by16, not RGB/features
  arrays.npz                   # state/action/episode/frame/valid/pixel counts
  invalid_frames.jsonl         # source bookkeeping, not model input
  goal_rgb.png                 # unchanged training goal1/117/global278
  training_window_targets_zh.png
check: /home/zsw/project_2026/simulation_output/pusht_object_dynamics_pretrain_v1
  started.json / report.json
train: /media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_dynamics_v1
```

All episode ownership186/20 stays fixed.25626 valid frames;24 invalid frames
are in training only and invalidate42 full windows. Train21958/22000 eligible,
all186 episodes; validation2002/2002/all20 unchanged. No zero-filled missing
object target, padding, resplit, parser tuning, or statistics refit. Cache all
frame positions; only training sampling filters invalid9-frame windows. Full
cache is small (grid payload14.77 MB) and remains remote; manifest, invalid
ledger, target PNG and check report are also in the local check directory.

Check passed real `[64,1,8,24,24]` forward/backward, action and grid gradients,
exact causal prefix, unchanged parameters and in-memory initial state reload.
Invalid-current advice retains candidate0 with no forecast; invalid candidates
are masked; Torch/NumPy goal scores match. There was no trained predictor at
this pretraining check; the later trained final is recorded below.
The viewed Chinese sheet shows fixed training episode1/frame55 and subsequent
56/59/63 RGB/grid **targets**, not predictions. `viewed_not_accepted`; immutable
report/manifest keep `not_viewed`.

The training command below was **subsequently authorized and completed**; it
is provenance, not a rerun request. Fixed seed20260915,10000 updates/batch64/AdamW3e-4,
float32/no TF32, clip1; only final10000 is evaluated. Timing estimate1–2 minutes
from0.004566 s/forward-backward was a pretraining estimate. Actual training
including final evaluation/reload took55.950 s.

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_dynamics.py --stage train
```

Final evaluation compares persistence and observed/mean-action model inputs on
the same2002 validation windows. Logs include Dice/MSE, terminal goal-cost error,
centroid and visible-area errors, episode macros and all per-episode metrics.
Mean-action is an inference ablation, not a separately trained baseline. No
development ranking/coverage lookup or rollout is performed by `train`.

Output is exclusive: `run.json`, `metrics.jsonl`, `last.pt`, `status.json`, then
`final.pt` and final-only `report.json`. Ctrl-C/SIGTERM requests a save after
the current update; an authorized interrupted run resumes using the same
command plus `--resume`. Hard termination can lose updates since `last.pt`
(saved each1000); those may be replayed and logs beyond that saved step must
be distinguished from the resumed trajectory when auditing compute. Do not
overwrite completed outputs, change seed/budget on resume, or compare last.pt
variants. Training/optimizer/final-checkpoint paths have now run successfully;
the interruption/resume path remains unexercised (the run was uninterrupted).

If only preparation/check fails, preserve outputs and diagnose first. A
completed cache can be reused by `--stage check --out <fresh_check_directory>`
without decoding/rebuilding it. Do not reuse an incomplete cache or rerun a
successful check solely to accumulate evidence. Continue with ordinary ACT;
no new gate, policy switch or real10/hardware change follows automatically.

## Object-Grid Final Training And Inspection Completed (2026-09-15)

Fixed training above completed once:10000 updates,21958 eligible train windows,
2002 unchanged validation windows/20 episodes,55.950 s including final-only
evaluation/reload. Original ACT, old WM, real10 and environment sources unchanged.
No seed/budget/loss/goal tuning, extra training or policy rollout. The matching
pretraining report/plan and absent output directory were checked before launch.

Final/last weights remain at
`/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_dynamics_v1/`.
Only final.pt is the assessed checkpoint; last.pt is recovery. Small
`run.json`, `metrics.jsonl`, `status.json`, `report.json` are also local under
`simulation_output/pusht_object_dynamics_train_v1/`.

New `tools/inspect_pusht_object_dynamics.py` was synchronized and parsed before
the completed short inspection (2.727 s; provenance, **do not rerun**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/inspect_pusht_object_dynamics.py
```

It reads100 finite training intervals100..10000, all AdamW step states10000,
checkpoint/run/report/cache identity, and9 source RGB/grid correspondences.
The lowest-ID validation episode's middle eligible window was fixed before
final metrics were read: episode0/frame76, displayed futures77/80/84. One
observed-action and one mean-action sequence are inferred; no optimizer,
environment/hardware steps or full validation rerun. Predictions are saved
before future RGB is decoded for display. No policy input contains future RGB.

Outputs on both machines:

```text
simulation_output/pusht_object_dynamics_inspection_v1/
  started.json / report.json
  fixed_window_predictions.npz
  fixed_validation_prediction_zh.png
```

Final validation window means (lower is better): persistence -> observed model:

```text
all8 Dice loss          0.109447 ->0.116308  (+6.27%, worse)
all8 grid MSE           0.004659 ->0.004431  (-4.89%)
terminal Dice loss      0.196137 ->0.154807  (-21.07%)
terminal goal-cost MAE  0.064915 ->0.048559  (-25.20%)
terminal centroid MAE   0.019308 ->0.015964  (-17.32%, normalized image XY)
terminal area rel.err   0.024968 ->0.125413  (+402.30%, worse)
```

Terminal Dice/goal-cost/centroid improve in17/20 episodes; all8 Dice only10/20,
area0/20. Per-episode and macro values are in the report. Mean-action input
gives terminal Dice0.415386 and goal-cost MAE0.179149, showing input dependence,
not a separately trained baseline or causal-counterfactual proof. One training
seed, no multi-seed stability or policy benefit claim.

The Chinese prediction sheet was inspected (`viewed_not_accepted`; immutable
report stays `not_viewed`). It retains an unfavorable case: nearly stationary
observed grids versus blurred/changing predictions, terminal Dice0.104786 vs
persistence0.005990. Upper RGBs are actual images; only the grid row is a learned
forecast. No representative case, thresholds or model were changed afterward.

Keep ordinary ACT. Next proposed check concerns candidate ordering, not more
training. Existing development logs lack current observable `agent_pos`, so a
faithful comparison first requires a separately declared short native replay
of saved ACT prefixes and exact current-RGB matching, before reusing saved
candidate outcomes. Do not substitute action targets/mean XY or hidden object
state and call it a pure offline check. This replay/ranking has no executed
command from the current turn; no new candidate continuation, gate or policy
switch was performed. See the handoff for the full mixed-result boundary.

## Object-Grid Candidate Ranking And XY Recovery Completed (2026-09-15)

Added `tools/check_pusht_object_action_ranking.py` and ran it once after local
syntax check, sequential SCP and remote entrypoint/plan readback. Successful
runtime4.109 s; use the saved artifacts, **do not rerun a completed diagnostic**.
Executed command (provenance, not another requested run):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_pusht_object_action_ranking.py
```

Inputs kept fixed:

- `simulation_output/pusht_wm_action_ranking_dev10_v1/`: recorded ACT actions,
  current RGB,140 actual candidate futures and diagnostic coverage.
- `simulation_output/pusht_object_goal_dev_v1/`: unchanged actual-future object
  scores/grid goal, checked only after new model predictions are persisted.
- `/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/`:
  existing goal/cache manifest; no source dataset decode or re-preparation.
- `/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_dynamics_v1/final.pt`:
  fixed final10000; no ACT/old-WM weights loaded or changed.

Ten native resets plus1440 recorded-prefix steps recover observable agentXY at
the28 existing anchors (seed200005 reset-only). All28 saved current RGBs and
replayed nominal coverage/done flags match. XY is read from native observation
before anchor action through the original preprocessor; historical XY was not
serialized and cannot be directly compared. No action-as-state substitution,
hidden object pose, state injection, new candidate continuation, training,
controller/hardware actions or policy benchmark. Twenty-seven valid current
images yield135 candidate-sequence forecasts; the unchanged invalid200004/0
case retains ACT with NaN forecasts and is excluded from ranking metrics.

Predictions and their NPZ are written before reading saved future targets;
all140 valid terminal objects reproduce previous actual-object scores. This
is reused development evidence, not a fresh held-out policy test. Same27
contexts for model/ACT/uniform/oracle comparisons, with1e-7/1e-6 cost/coverage
reporting tolerances and explicit target ties. Main results:

```text
predicted vs actual object cost     41/66 (62.12%);204 target ties
predicted vs actual coverage        32/58 (55.17%);212 target ties
actual-future object vs coverage    53/58 (91.38%;not model accuracy)
coverage-best selection             3/7 model,0/7 ACT,1.4/7 uniform expectation
mean coverage change vs ACT         +0.003460;better/tie/worse=5/21/1
mean coverage change incl fallback  +0.003336 across all28
optimizer/new candidate/hardware    0/0/0
native prefix replay steps          1440
```

Only7 contexts in5 seeds have varying coverage. Seed200000/80 accounts for
89.41% of net context-sum coverage gain;200006/160 reverses all10 coverage pairs
and loses0.041008 coverage. Do not claim stable improvement or success-rate gain.
No new gate is calibrated and old latent thresholds do not apply. Keep ACT.

Outputs on both machines:

```text
simulation_output/pusht_object_action_ranking_dev10_v1/
  started.json / report.json / status.json
  recovered_observations.jsonl
  predictions.jsonl / predictions.npz
  analysis.jsonl / offline_targets.npz
  first_seed_object_prediction_ranking_zh.png
```

The fixed first-seed/all3-anchor Chinese sheet is inspected
(`viewed_not_accepted`, immutable report `not_viewed`). Predictions are grids;
the RGB panels are previously observed images. One SSH retrieval reset affected
only the small predictions.npz copy; it was retransferred, not recomputed.
Successful output directories are exclusive and protected. If a future
authorized run fails, inspect its report/traceback and retain partial outputs;
fix the specific failure and use an explicitly named fresh `--out`, never
overwrite successful evidence or automatically extend seeds/budgets.

Next proposal: analyze the saved stationary-object and direction-ordering
failures, with no additional environment steps or fitting, before choosing a
minimal predictor change. No next training/rollout command is authorized here.

## Saved Object-Error Analysis Completed (2026-09-15)

Added `tools/audit_pusht_object_prediction_errors.py`; final small file was
SCP-synchronized and remotely parsed before the completed0.100 s run:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_object_prediction_errors.py
```

Command is provenance only, **do not rerun the completed analysis**. It imports
NumPy/Pillow, not LeRobot/Torch/environment tools, and reads only saved ranking
NPZ/JSONL plus the previous RGB files. No checkpoint load, model forward,
optimizer, replay, candidate execution, hardware or new training-data decode.

Same27 eligible contexts/135 candidate sequences. Exact terminal/current grid
equality defines an offline endpoint group, not full-horizon stationarity or
contact. No thresholds, candidates, scores, action signs or model are changed.
Source model goal scores are reproduced from its saved predictions; the three
examples are previously identified cases, not a newly selected validation set.

```text
terminal=current: 66 candidates; model/persistence Dice =0.223944/0
terminal changed: 69 candidates; model/persistence Dice =0.432782/0.141851
all135: model/persistence Dice =0.330683/0.072501; model better/worse=14/121
unchanged endpoint predicted centroid error:4.275 image pixels
reversed-ranking case200006/160: reference displacement0.433 predicted/6.786 actual
same case ±X/±Y response direction cosine:+0.958/+0.905
same case ±X/±Y response magnitude ratio:6.27%/10.00%
checkpoint loads / new forwards / optimizer / environment / hardware:0/0/0/0/0
```

Candidate/context/seed means and individual rows are saved separately; do not
treat135 candidates as independent scenes. Goal-ranking reversal here is not
the same as a command-axis reversal: severe under-response and inaccurate
position/shape are visible. See the handoff for interpretation and limits.

Outputs on both machines:

```text
simulation_output/pusht_object_error_analysis_v1/
  started.json / report.json / cases.json
  per_candidate.jsonl
  saved_object_error_cases_zh.png
```

The Chinese sheet is inspected (`viewed_not_accepted`, report `not_viewed`).
Current and terminal RGB are actual observations; +1/+4/+8 grids are previously
saved forecasts. No actual intermediate frames or hidden contact state were
recovered. Keep original ACT/model weights and successful output directories.
On an analysis failure, preserve partial files/traceback and inspect the exact
issue; do not recreate source predictions or start an environment as a fallback.

Next proposal: implement a minimal current-grid identity/residual path up to
training preparation, not a larger model. This is **not implemented**, and no
new training or policy-evaluation command is authorized by this analysis.

## Residual Object Predictor: Preparation Before Training (2026-09-15)

New model and entrypoint:

```text
tools/pusht_object_residual_dynamics.py
tools/run_pusht_object_residual_dynamics.py
```

`tools/run_pusht_object_dynamics.py::train` now takes an explicit `model_api`
keyword, defaulting to the original model. Residual training uses the same
optimizer/sampler/resume/final-evaluation implementation with the new versioned
model. Original model definition and trained weights are unchanged.

Completed check after sequential SCP and remote syntax/entrypoint readback
(0.437 s; provenance only, **do not rerun**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_residual_dynamics.py --stage check
```

Same original cache/schema,21958/2002 windows and248256 parameters. Output is
`clamp(current_grid + residual,0,1)`, with a zero-initialized decoder and the
same-seed GRU/encoder/state/action trunk. Initial64-window output is exactly
persistence; decoder gradients are nonzero. First-backward trunk gradients
are zero by design. A separate discarded nonzero-head fixture demonstrates
action/grid gradient flow and prefix causality; it is not optimized or saved
as weights. Initial parameter/state roundtrip, initial ACT tie and invalid-
observation paths pass. No new broad smoke suite or per-file hashes.

Outputs on both machines:

```text
simulation_output/pusht_object_residual_pretrain_v1/
  started.json / report.json
  initial_identity_example.npz
  initial_identity_zh.png
```

Fixed training example episode1/frame55/global216; Chinese sheet inspected
(`viewed_not_accepted`, immutable report `not_viewed`). It shows initialization,
not learned prediction. No optimizer, checkpoint, trained-weight load,
validation/development inference, environment or hardware execution. The new
training output directory remains absent.

Command prepared at this preparation milestone; subsequently approved and
completed in the next section. **Do not repeat the successful run**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_residual_dynamics.py --stage train
```

Fixed seed20260915,10000 updates,batch64,AdamW3e-4,weight decay1e-4,clip1, no
TF32, final-only original6 prediction metrics. No warm start or expanded data.
The source cache is
`/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1/`.
Only the new output directory is used:

```text
/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_residual_dynamics_v1/
  run.json / metrics.jsonl / status.json / report.json
  last.pt   # interrupted-run recovery only
  final.pt  # fixed step10000, not selected from intermediate checkpoints
```

The new checkpoint schema is `pusht_rgb_object_residual_dynamics_v1`; data cache
schema stays `pusht_rgb_object_dynamics_v1`. Decoder meanings differ despite
matching weight shapes: use the matching residual loader and do not copy raw
base-model state_dicts. The train stage requires this preparation report's plan
and cache identity to match. No ACT, PI05, mixed-head or real10 interface changes.

Future interruption: Ctrl-C/SIGTERM finishes the current update and saves
last.pt plus optimizer/sampler state. Inspect status before resuming the same
authorized unfinished run with `--stage train --resume` and the same other
arguments. Never resume a completed report, overwrite the old model output,
extend the budget, or silently create a new trial. If final evaluation fails
after final.pt is saved, inspect that failure before any new fitting. Historical
base training took55.950 s; the subsequent residual timing is recorded below.

Stop here for this approval: training preparation is complete, but neither
new trained predictions nor policy gains are available.

## Residual Object Predictor: Fixed Training And Readback Completed (2026-09-15)

The next explicit user approval advanced the prepared train stage once, without
resume or changed parameters. The command above completed10000 optimizer steps
in53.584 s including final evaluation/reload; do not rerun or extend it.
New weights remain onSSD1T in the residual training directory above. It contains
run/metrics/status/report plus last.pt (recovery only) and final.pt (selection).
Completed status is step10000, reload exact,100 finite log intervals. All
optimizer states reached10000; the original base output is preserved.

Added read-only `tools/compare_pusht_object_residual.py`; after sequential SCP
and remote syntax/size/entrypoint readback, this completed in0.743 s:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/compare_pusht_object_residual.py
```

This is completed provenance, not a command to rerun. It reads both saved final
evaluations, verifies shared training plan/cache/split/normalization and final
sampler RNG equality, and compares window/episode means and per-episode error
directions. It loads both typed final models only for the same fixed validation
episode0/frame76 (two single-window forwards). The input/target/base prediction
exactly match the prior saved inspection. Zero full-validation reruns, optimizer,
candidate ranking, source-RGB decode, environment or hardware steps.

Same2002 windows/20 episodes; lower is better:

```text
metric                       base       residual    persistence   residual vs base
Dice all8                    0.116308   0.091994    0.109447      -20.90%
grid MSE all8                0.004431   0.003770    0.004659      -14.92%
terminal Dice                0.154807   0.130259    0.196137      -15.86%
terminal goal-cost MAE       0.048559   0.038678    0.064915      -20.35%
terminal centroid XY MAE     0.015964   0.012841    0.019308      -19.56%
terminal area relative error 0.125413   0.097230    0.024968      -22.47%
```

All six episode-macro means improve vs base; per-episode improvements in that
order are20/20,20/20,20/20,16/20,18/20,16/20. Area error still worsens289.42%
against persistence and loses in20/20 episodes. Mean-action inference ablation
all8 Dice0.327305 is worse than observed0.091994; it is not separate training.
This single-seed prediction comparison does not establish policy improvement.

Artifacts:

```text
remote training: /media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_residual_dynamics_v1/
local small logs: simulation_output/pusht_object_residual_train_v1/
both machines: simulation_output/pusht_object_residual_comparison_v1/
  report.json / started.json
  fixed_window_predictions.npz
  fixed_validation_comparison_zh.png
```

Fixed example terminal Dice: base0.104786, residual0.026563, persistence0.005990.
Actual/future grids come from the unchanged RGB-derived cache, not generated
RGB or simulator hidden state. Chinese comparison sheet inspected:
`viewed_not_accepted` (immutable report `not_viewed`); faint detached occupancy
artifacts remain despite the sharper silhouette. Output creation is exclusive; preserve the
successful comparison. If a later necessary readback fails, inspect the actual
error/partial output and use a fresh `--out`; do not retrain to repair reporting.

Stop after this completed final readback. Keep ACT and real10 unchanged. The
next proposed task is a separately approved inference-only comparison using
already saved development observations/actions/future targets; no environment
replay or new candidate continuations are needed. It is not started here.

## Residual Saved-Candidate Comparison Completed (2026-09-16)

After the next scoped approval, the new small
`tools/compare_pusht_object_residual_ranking.py` was SCP-synchronized and parsed/
read back on4090. One inference-only run completed in0.791 s (provenance only;
**do not rerun the successful comparison**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/compare_pusht_object_residual_ranking.py
```

Input: `simulation_output/pusht_object_action_ranking_dev10_v1/` saved current
RGB grids/recovered observable XY/native actions/goal/base forecasts. Only the
typed residual final10000 checkpoint is loaded. New forecasts are persisted
before joining that source's `offline_targets.npz`; actual future/coverage and
endpoint groups never enter prediction. Base ranking/per-seed and endpoint
summaries reproduce the prior reports exactly. Reuses old metrics/tolerances.

```text
contexts / eligible / candidate forecasts        28 / 27 / 135
checkpoint loads / batched forwards               1 / 1
optimizer / environment / ACT inference / hardware 0 / 0 / 0 / 0
object ordering, base -> residual                 41/66 ->44/66
coverage ordering, base -> residual               32/58 ->41/58
coverage-best selection /7, base -> residual       3 ->4
goal-cost MAE, base -> residual                    0.049536 ->0.030482
relative action-gain MAE, base -> residual         0.004629 ->0.005595 (worse20.89%)
mean coverage gain vs ACT, base -> residual        0.003459878 ->0.003460159
residual minus base coverage gain                 +0.000000281
terminal Dice, base / residual / persistence       0.330683 /0.181148 /0.072501
```

Only7 contexts/5 seeds have varying coverage; persistence has58 predicted pair
ties and retains ACT0, not58 reversed pairs. The extra top1 success is a tiny
coverage change0 ->0.000007588 at200008/160. All other informative selected
actions are unchanged, including the harmful index3 at200006/160 (coverage
-0.041008 vs ACT). Do not infer policy success from pair ordering or4/7 top1.
Residual endpoint Dice improves45.22% vs base but loses to persistence117/135.
Fixed under-response case: actual displacement6.786 px, base0.433/residual0.753.
No tuning, full validation rerun, new candidate continuation or policy rollout.

Outputs on both machines:

```text
simulation_output/pusht_object_residual_ranking_dev10_v1/
  started.json / report.json / status.json
  predictions.jsonl / predictions.npz
  analysis.jsonl / per_candidate.jsonl / cases.json
  saved_candidate_residual_comparison_zh.png
```

The prediction NPZ stores new forecasts/keys/validity and references the original
saved input/target directory; preserve that source. Three unchanged selected
cases200000/0,200006/160,200000/80 use one grid scale and all five candidate
scores. Chinese image inspected: `viewed_not_accepted`; report `not_viewed`.
No raw source dataset decode or new RGB generation. Files are small and were
retrieved directly; no weights transfer. For a future necessary reporting fix,
inspect the failure and preserve partial outputs, then use a fresh `--out`;
never rerun environments or retrain as an automatic recovery.

Stop here: keep ACT/real10 unchanged; residual remains a prediction reference.
The next proposal is implementation/alignment preparation for a minimal
two-frame observable-history variant. No history model, new training, gate
calibration or rollout is started/authorized by this completed comparison.

## Two-Frame History Preparation Complete; Training Pending (2026-09-16)

This section records the preparation milestone. Its pending-training wording
is superseded by **Two-Frame History Fixed Training And Readback Completed**
below, following the user's subsequent explicit approval. Do not rerun the check.

New `tools/pusht_object_history_dynamics.py` and
`tools/run_pusht_object_history_dynamics.py` were sequentially SCP-synchronized
and parsed/read back on4090. Completed one0.427 s check (provenance only;
**do not repeat it**):

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_history_dynamics.py --stage check
```

Uses the unchanged `pusht_object_dynamics_v1` cache/schema, not a new dataset.
Keeps21958/2002 anchors and all original labels/splits. History is strictly
same-episode adjacent t-1; reset, gap or invalid predecessor copies current
grid/XY with valid=false. Available train/val history counts21770/1982; train
fills180 resets+8 invalid predecessors, val fills20 resets. No new exclusions.

Shared encoder delta128 + normalized agent-XY delta2 + availability1 are mapped
by a bias-free131->128 projection to the original residual hidden context.
Total265024 parameters (+16768/+6.75%); GRU/loss/output/normalization unchanged.
Forward receives explicit `ObservableHistory`, not the old single-frame XY
tensor. Missing-history input yields an exact zero history branch; no implicit
episode cache, velocity/physical-unit reinterpretation, goal or future input.
Use this version's checkpoint loader; do not warm-start/interchange base or
residual weights. A history-off same-weight untrained fixture is compatibility
evidence, not a separately trained capacity-control result.

Initial64-window forecasts exactly equal current-grid persistence. A discarded
nonzero-decoder fixture checks nonzero history/input/action gradients, zero
masked-history gradients, prefix causality and exact same-weight single-frame
fallback, including unavailable NaNs. Current/actions/targets match base;
five-candidate output shape and in-memory roundtrip pass. Initial parameters
unchanged; optimizer/checkpoint/trained-weight loads/environment/hardware all0.
Validation metadata is audited, but no validation/development model inference.

Outputs on both machines:

```text
simulation_output/pusht_object_history_pretrain_v1/
  started.json / report.json
  history_index.npz / history_examples.npz
  history_alignment_zh.png
```

The Chinese sheet uses fixed training episode1/frame0 and frame55, with reset
fill and true frame54->55 history. Inspected `viewed_not_accepted`; report
`not_viewed`. It shows input alignment and untrained identity, not learned
prediction improvement. Small files retrieved directly; source cache unchanged.

Prepared next command, **not executed; requires subsequent training approval**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_history_dynamics.py --stage train
```

Same seed20260915,10000 updates,batch64,AdamW3e-4,weight decay1e-4,clip1, noTF32,
final-only original6 metrics. Requires matching successful preparation report
and unchanged source cache identity. Generic training/evaluation/resume loop
is reused without edits; future output is isolated and currently absent:

```text
/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_history_dynamics_v1/
  run.json / metrics.jsonl / status.json / report.json
  last.pt   # unfinished-run recovery only
  final.pt  # fixed step10000; schema pusht_rgb_object_history_dynamics_v1
```

Runtime not measured; prior residual10000 updates took53.584 s, so estimate
roughly1-2 minutes for this modest history addition on4090. Never overwrite an
existing successful run or extend its budget. On interruption inspect status;
the same unfinished authorized run can use `--stage train --resume`, preserving
last.pt/optimizer/sampler. If final evaluation fails, inspect it before fitting
again. The old candidate NPZ lacks adjacent previous grid/XY: future ranking
requires a separately scoped history recovery, not an automatic environment
replay or an80-step-old anchor substituted for t-1. Stop before training here.

## Two-Frame History Fixed Training And Readback Completed (2026-09-16)

User subsequently approved the prepared fixed run and comparison. The following
command completed once on4090; **provenance only, do not repeat or resume it**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_object_history_dynamics.py --stage train
```

Completed10000 updates in57.362 s, without interruption/resume, budget extension
or intermediate checkpoint selection. Same21958/2002 windows,186/20 episodes,
seed20260915, batch64, AdamW3e-4/weight decay1e-4/clip1, quadratic Dice, noTF32.
Final checkpoint reload is exact. Status is completed/step10000; all parameter
optimizer states reached10000 and all100 logged loss/gradient intervals are
finite. History265024 parameters vs residual248256 (+6.75%); final sampler RNG
states/cache/split/normalization match exactly. Source cache is unchanged.

Added `tools/compare_pusht_object_history.py`, reusing existing readback and grid
rendering. It reads already completed final validation reports; no full
validation rerun. Two fixed-window forwards compare residual/history at the
previously selected validation episode0/frame76, using actual adjacent frame75.
Original current/XY/actions/targets/goal/residual forecast reproduce exactly.

Initial comparison `..._comparison_v1/` stopped before model forwards: the check
had mistakenly treated intentional `base_model_schema` and `comparison`
metadata differences as a shared-training-plan violation. Preserve its
started.json. Only the comparison check was corrected, syntax-read, sequentially
SCP-synchronized and remotely read back. No training/model/checkpoint changed.
Fresh successful comparison (0.766 s), **already run; do not repeat**:

```bash
cd /home/zsw/project_2026 &&
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/compare_pusht_object_history.py \
  --out simulation_output/pusht_object_history_comparison_v2
```

| Validation error, lower better | Residual | History | Relative change |
|---|---:|---:|---:|
| Dice, all8 | 0.091994 | 0.080585 | -12.40% |
| Grid MSE, all8 | 0.003770 | 0.003319 | -11.95% |
| Terminal Dice | 0.130259 | 0.122244 | -6.15% |
| Terminal goal-cost MAE | 0.038678 | 0.037660 | -2.63% |
| Terminal centroid XY MAE | 0.012841 | 0.011865 | -7.60% |
| Terminal area relative error | 0.097230 | 0.085550 | -12.01% |

All six episode-macro means improve too, but goal-cost is better/worse in10/10
episodes. Area still loses to persistence in20/20 episodes. Fixed-example
terminal Dice regresses0.026563 ->0.038157; persistence0.005990. Do not replace
that example or call aggregate improvement uniformly better motion prediction.
The mean-action ablation uses existing final evaluation only; no history-off
training or inference sweep was added. No policy/candidate ranking evidence.

```text
remote protected training root:
  /media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_history_dynamics_v1/
    run.json / metrics.jsonl / status.json / report.json
    last.pt / final.pt
local small-log mirror (no copied weights):
  simulation_output/pusht_object_history_train_v1/
failed comparison retained on both machines:
  simulation_output/pusht_object_history_comparison_v1/started.json
successful comparison on both machines:
  simulation_output/pusht_object_history_comparison_v2/
    started.json / report.json / fixed_window_predictions.npz
    fixed_validation_history_comparison_zh.png
```

Chinese sheet inspected `viewed_not_accepted`; generated report stays
`not_viewed`. Current approval was for execution, not visual acceptance. Keep
all original/residual/history checkpoints and use their own typed loaders.
No ACT/PI05/real10 changes, environment steps, hardware actions or source decode.
Readback/helper preparation is complete, not a request for more smoke tests.

Historical proposal, now deferred by the current candidate/scorer plan:
determine/recover actual adjacent
observable history for the old saved development contexts and reuse their
candidate actions/actual futures for ranking. Do not supply an80-step-old
anchor as t-1, fake valid history, add fitting or start a new policy benchmark.
A history-specific attribution would also require a capacity-matched trained
history-off control; it was not run or authorized by this fixed training.
Retain ordinary ACT until there is separate policy evidence. Weekly log unchanged.

## Ready For Onsite Later: Real10 Observation-Only Inference (2026-09-14)

Both final weights load on4090. The short check completed in31.6713 s with29
observation-only PI05 inferences and a2719-row state-only Piper fit readout.
The learned state token is consumed exactly once per PI05 sample and matches
the normalized input. No optimizer or hardware action occurred. The sampling
context fix is in `tools/pi05_state_conditioning.py`; no retraining is needed
for that correction.

Current report/example:

```text
simulation_output/real10_pi05_inference_v1/report.json
simulation_output/real10_pi05_inference_v1/example_request.json
```

Warm single-observation median101.8 ms /p95110.6 ms; load27.8 s. Selected29-frame
Elite MAE1.346058 mm; all-row Piper accuracy96.36%, balanced accuracy57.10%,
feed recall11/70. These are training-data diagnostics, not held-out performance
or real-system success. Do not use them to initiate automatic hardware motion.

Single recorded observation -> action JSON (no camera/robot connection):

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/real10_pi05_policy.py \
  --input-json simulation_output/real10_pi05_inference_v1/example_request.json \
  --out simulation_output/real10_pi05_inference_v1/single_prediction.json
```

The example points to uploaded pack images using absolute4090 paths. Output
creation is exclusive; after a successful call use a new `--out` to repeat.
This optional CLI example is not a prerequisite to rerun the completed check.
Keep the pair at the default training paths: the Piper file references Elite.

For a persistent onsite client, put `tools/` on its Python import path, load
once, and supply in-memory OpenCV BGR uint8 images:

```python
from pathlib import Path
from real10_pi05_policy import Real10PI05Policy

model = Real10PI05Policy(
    Path("simulation_output/real10_pi05_train_v1/elite/final_policy.pt"),
    Path("simulation_output/real10_pi05_train_v1/piper/mixed_head_policy.pt"),
)
result = model.predict(
    side_bgr=side_frame, top_bgr=top_frame,
    elite_tcp_pose_6d=measured_pose_xyz_mm_rpy_rad,
    task="left",  # or "right"
    previous_controller_state=previous_snapshot,
)
```

`previous_snapshot=None` on episode reset/missing history; otherwise supply
only the previous observation/decision's `piper_busy` and
`piper_step_after_command` (event count, not feed distance). Do not substitute
the current target command. The adapter derives the validity bit and task
one-hot. Result keys remain `elite_tcp_delta_6d` and `piper_intent_id`, with a
separate diagnostics dictionary. Rotation is zero; xyz is displacement in mm,
not velocity. Canonical intents0/1/2 are returned unchanged; unsupported
training intents are flagged, not masked into hold. No clipping or dispatch
is performed. Live capture/controller plumbing and bounded physical execution
are the next stage, not implemented or verified by this observation interface.

Completed short-check command, retained for reproduction only:

```bash
cd /home/zsw/project_2026 &&
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/check_real10_pi05_inference.py
```

It refuses an existing report directory. Inspect a failed attempt before
retrying and use a fresh `--out`, preserving prior artifacts. The local Chinese
preview was generated with
`tools/check_real10_pi05_inference.py --render-report simulation_output/real10_pi05_inference_v1/report.json`
using the local Windows interpreter; no model is loaded in render mode.

## Completed: User-Run Real10 Training Pair (2026-09-14)

Remote conversion and data preflight passed:10 complete source episodes,
2719 train transitions, no validation, labels retract0 / hold2649 / feed70.
Every converted state/action/explicit target equals the uploaded source pack.
All-row normalization is fresh; a side/top pair from each episode was read back.
The existing pinned PI05 base weight/config files are locally available.
Both user-run stages completed: Elite3000 / Piper1000 steps, with finite
losses/gradients and final checkpoints saved. No hardware action has run.

The following command is retained for provenance; do not rerun the completed
pair. Logged training-loop times were549.1116 s and121.3577 s (excluding some
startup/save work). The original20–40 minute allowance was a planning estimate.
One command runs Elite then Piper; the launch log is protected
against overwrite and the trainers refuse existing stage output directories:

```bash
cd /home/zsw/project_2026 &&
(set -C; nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_real10_pi05_training.py --execute \
  > simulation_output/real10_pi05_train_v1.launch.log 2>&1 < /dev/null &)
```

Follow progress; Ctrl-C stops this viewer only:

```bash
tail -n 30 -f /home/zsw/project_2026/simulation_output/real10_pi05_train_v1.launch.log
```

The runner's default without `--execute` is data/weight-path preflight only.
That preparation has already passed; no need to run it separately again. Its
report is `simulation_output/real10_pi05_train_v1/preflight.json` and includes
the exact two child commands. The training command also checks data before
dispatching either trainer, without constructing a model in that preparation.

Fixed first-run settings:

- Elite:3000 steps, batch1, lr2.5e-5, seed123; existing PI05 base and
  state-conditioning token, translation-only loss, frozen vision and existing
  train-expert-only setting. All2719 rows enter each epoch; no max-record cap.
- Piper:1000 steps, batch32, seed123, balanced CE, state_only head128/256,
  head lr1e-3; load and freeze the new Elite checkpoint. Batch32 enables the
  existing class weights to act across samples; no loss formula change.
- Both use `--train-only --save-checkpoint` and fresh all2719-row normalization.
  No validation or best-checkpoint selection; final validation fields are null.
  No simulation overlay, old real captures, or world-model component.

Verified outputs on the home filesystem (not SSD):

```text
simulation_output/real10_pi05_train_v1/
  preflight.json
  elite/final_policy.pt
  elite/train_log.json
  elite/summary.json
  piper/mixed_head_policy.pt
  piper/train_log.json
  piper/summary.json
```

Both final weights and full logs exist. Elite is7,473,655,457 bytes and Piper
is163,481 bytes. Keep the pair on4090, also the onsite inference GPU. Final checkpoints record normalization and
the real input/timing contract. The frozen-Piper checkpoint references the
Elite checkpoint rather than duplicating its large weights. These are final
inference artifacts, not optimizer-resume checkpoints.

If SSH disconnects, inspect the launch log and process before relaunching;
`nohup` may still be running. Errors stop the pair, never auto-retry. Preserve
a completed Elite checkpoint if Piper fails. If Elite completed but Piper was
never started, `--component piper --execute` continues that stage only. If a
stage created a partial output, inspect the error and use a new output path
for the corrected stage command printed in `preflight.json`; do not delete
checkpoints or blindly restart the pair. These recovery instructions are for
future failed attempts; this pair is complete and needs no resume/retrain.

## Completed: Real10 Data Adaptation And Transfer (2026-09-14)

The active objective is the user-authorized ten-episode real prototype.
Historical benchmark, failure-analysis and low-visibility commands below are
paused. `project4090` is also the onsite inference machine. No training or
hardware command is part of this adaptation step.

Local preparation completed in211.1 s; this command is retained for reproduction,
not a request to rerun it over the existing output:

```powershell
cd D:\PycharmProjects\project_2026
.\.venv\Scripts\python.exe -u tools/prepare_real_pi05_pack.py --archive
```

The explicit allowlist is left002..006 plus right001..005 of
`real_pilot_20260817_*_s_bend_replacement_tip_fixedudp_*`. It produces a portable
224x224 image/state/action pack. All2,729 observations remain represented;
2,719 real next-observation transitions supply action targets, including all70
feed requests. Terminal manufactured-zero targets are not supervised. No
quality filtering or mixing with older captures. All ten episodes are train-only.

Output preparation refuses an existing directory/archive. If preparation is
interrupted, keep the partial output and use a new explicit name, for example
`--out simulation_output/real10_pi05_compat_retry1 --archive`; do not delete raw
captures or overwrite a completed package. Update the paths below consistently
for a different attempt.

Large-file upload is assigned to the user (not agent SCP). The ready ZIP is
537,478,661 bytes (537.5 MB), with a passing archive CRC test. Only this ZIP is
needed, not the raw captures/depth folders:

```powershell
scp "D:\PycharmProjects\project_2026\simulation_output\real10_pi05_compat_v1.zip" project4090:/home/zsw/project_2026/simulation_output/
```

The following remote extraction/conversion commands completed successfully;
do not rerun them over the existing outputs:

```bash
cd /home/zsw/project_2026 &&
test ! -e simulation_output/real10_pi05_compat_v1 &&
/home/zsw/miniconda3/envs/project2026-pi/bin/python -m zipfile -e \
  simulation_output/real10_pi05_compat_v1.zip simulation_output
```

```bash
cd /home/zsw/project_2026 &&
HF_HUB_OFFLINE=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/export_openpi_compat_to_lerobot.py simulation_output/real10_pi05_compat_v1 \
  --out simulation_output/real10_pi05_lerobot_v1 \
  --repo-id project2026/real10-prototype-v1 --fps 5 --image-size 224 --no-videos
```

These outputs use the home filesystem (97 GB free at readback), not the SSD
(25 GB free). No deletion is needed for this small package. Checkpoint storage
is a separate next-step budget. Leave archives/data in place until conversion
is verified; no automatic cleanup.

`--fps 5` is only the LeRobot indexing clock; original irregular image/pose
times and action durations remain in the source index, copied to the export
as `project2026_source_index.jsonl`. Actions remain displacement in mm/rad,
not velocity. No `--max-episodes`, `--max-frames-per-episode`, `--force`, or
default15 FPS. A later trainer must keep one-observation/one-action operation
and use freshly computed real-data normalization, not old simulation stats.

The exporter refuses an existing output directory. After an interrupted
conversion inspect the process first; for a confirmed failed attempt use a
new `--out` name, leaving the prior output intact. Do not relaunch a live job.

Focused verification already passed (not a prerequisite to rerun each time):

```powershell
.\.venv\Scripts\python.exe tools/test_prepare_real_pi05_pack.py
```

```bash
cd /home/zsw/project_2026 &&
HF_HUB_OFFLINE=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B \
  tools/test_prepare_real_pi05_pack.py
```

Two tests passed locally plus one expected LeRobot skip; all three passed
remotely. The remote test uses a temporary synthetic four-frame dataset,
not the deleted real captures. Full-real-data remote conversion and the training
entrypoint's data preflight subsequently passed, as recorded above.

Use the new training-pair command above: explicit real10 train-only mode now
supports fixed final-checkpoint saving without fake held-out scores. Legacy
trainer calls without that flag still require validation. Architecture,
mixed-head semantics, and loss formulas are unchanged.

## Completed: User-Run BC/ACT Clean Push-T Benchmark (2026-09-14)

Completion was discovered and audited when research resumed. Final scores are
BC0/100 and ACT22/100 (strict pass); see the current section at the top. The
following preparation description/launch is historical and must not be rerun.

Final-checkpoint inference and the trained-policy smoke passed.44 selected
tests passed locally/remotely (25 new + 19 frozen shared protocol tests).
Both full2,162-frame validation replays match training-time results within
1e-6 (BC exact; ACT max difference1.806613e-7). Official ACT chunk16/execute8,
partial-queue reset, full20-step environment reset replay and unchanged final
weight hashes are verified. Only80 diagnostic environment steps were executed
(20 primary +20 replay per model); neither model completed the task within
that short window. At that preparation stage no full benchmark score existed;
the subsequent completed results are recorded above.

Run the complete paired benchmark on the remote host. It is user-run as
assigned; reserve5–10min. The short-smoke extrapolation is about3.1min combined,
not a measured benchmark duration. This command refuses existing log/output
paths and protects the launch log from overwrite. BC then ACT; no training:

```bash
cd /home/zsw/project_2026 &&
if [ -e simulation_output/pusht_bc_act_benchmark_v1 ] || [ -e simulation_output/pusht_bc_act_benchmark_v1.launch.log ]; then
  echo 'Existing benchmark/log: inspect status and processes; do not relaunch.'
else
  (set -C; nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
    tools/run_pusht_bc_act_baseline.py --stage benchmark --model both --execute \
    > simulation_output/pusht_bc_act_benchmark_v1.launch.log 2>&1 < /dev/null &)
fi
```

Follow progress (Ctrl-C stops only the viewer):

```bash
tail -n 30 -f /home/zsw/project_2026/simulation_output/pusht_bc_act_benchmark_v1.launch.log
```

Outputs:

```text
simulation_output/pusht_bc_act_benchmark_v1/
  bc/ or act/
    started.json        protocol/source/checkpoint/environment bindings
    status.json         actual PID and completed episodes
    per_step.jsonl      reset, raw action, queue and actual environment outcomes
    per_episode.jsonl   success, max coverage, bounds counts and trace hashes
    report.json         complete or failed result; inspect strict_protocol_pass
```

Frozen schedule:100 episodes per model, seeds100000..100099, maximum300 steps,
float32 CUDA, deterministic, no AMP/TF32/noise/clipping. Each episode resets
policy/controller queues and RNG, checks its initial observation hash against
the corresponding completed DP seed, steps the single underlying native
environment directly and stops on first done. Success/coverage come only from
actual environment info. Native bounds are0..512. Bounds violations are logged
without clipping and make `strict_protocol_pass=false`; completion alone is
not a valid benchmark score. The official DP is still an external pretrained
reference with different/unknown training-data revision and budget.

Both model invocations require the existing strict-pass smoke under exactly
the same new runner/inference/dependency hashes and final checkpoint bindings.
The benchmark reads the pinned normalization/source metadata but does not
build a demonstration image cache, compute losses or execute an optimizer.
All training checkpoints remain untouched on the SSD. Do not rerun training,
preflight or the completed smoke to start a benchmark.

If SSH disconnects, the nohup job may still be live. Inspect the log, each
model's status and its PID first. Do not relaunch a live job. No automatic
resume/retry or deletion is supported. Preserve a completed BC result if ACT
fails. After inspecting a confirmed ACT failure, an explicitly authorized
fresh ACT-only attempt would use the following foreground form (do not run
preemptively; it repeats the full frozen100 seeds, not just favorable cases):

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_bc_act_baseline.py --stage benchmark --model act --execute --attempt retry1
```

That explicit attempt writes `pusht_bc_act_benchmark_v1_retry1/act/`, leaving
all previous outputs intact. If source code must change after a failure, the
old smoke binding is invalid: investigate, then run a new named smoke for the
changed implementation and explicitly pass its parent via `--smoke-root` to
the new benchmark attempt. Never silently substitute checkpoints or parameters.

Completed verification commands, retained as provenance only:

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest \
  tools.test_pusht_bc_act_inference tools.test_pusht_bc_act_baseline tools.test_diffusion_pusht_baseline -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_bc_act_baseline.py --stage preflight
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/run_pusht_bc_act_baseline.py --stage smoke --model both --execute
```

The strict loader restores the trusted local model/optimizer/RNG format solely
to verify final state, then discards the optimizer and freezes all parameters.
Saved AdamW states are step100000; no optimizer update occurs during loading.
This is not verification of interrupted-training continuation. The validation
adapter's `predict()` resets the queue intentionally; the environment loop uses
the separate `select_action()` that preserves ACT's8-action queue.

Source pins (synced and verified before execution):

```text
tools/pusht_bc_act_inference.py
  c535ec45fba5a8d900bb0fe5675c875568536ba38c66f979fde0c7f4484e6782
tools/run_pusht_bc_act_baseline.py
  e9bc06cefb5529624bdcdfaa932f9534868c77519c549f94c0ee586f49893862
tools/test_pusht_bc_act_inference.py
  07caf814d6ef1f4b16c95627014e5ae2569d44c0da1d74d7f6d63ba5ff3ef632
tools/test_pusht_bc_act_baseline.py
  f99f0010a3aba266cc3a82c3da7be362a8eaa788f4f3f4059a388ca85e16e6bd
```

Smoke artifacts under `simulation_output/pusht_bc_act_smoke_v1/` (small files
retrieved; report and preview hashes verified locally):

```text
bc/report.json
  623b345e18a707e67baf86bd9506cb96cb67f9842098472b1b1e88c62d30b723
act/report.json
  37da909b3234b59a60222875285ec5ec90a536265b81c2820feb31d75e178b9c
bc/smoke_sheet_zh.png
  1f827027cc63ca4e9e3fb59ab386811d723342017b5530e934c4aae67b27dd72
act/smoke_sheet_zh.png
  21c79192849af33bc684f20715753224953d276e3bb27aea9de42cdd0c1d3903
```

Both sheets were inspected with readable Chinese labels; user visual acceptance
is pending (`viewed_not_accepted` in the handoff; generated report creation-time
status is unchanged). The pygame `pkg_resources` deprecation warning and lazy
gym_pusht registration message did not prevent completion. No version changes,
data/SOFA/weekly edits, R0 reads or world-model work were performed.

## Completed: BC/ACT Training And Read-Only Audit (2026-09-14, Before Inference Smoke)

Both user-run models completed exactly 100,000 updates. The complete metrics,
sampler replay, source bindings and all 40 checkpoint payload hashes passed
the post-run audit; do not relaunch/resume training or rerun successful preflight.
Final full-split frame-mean MAE is BC 11.102428 / ACT 10.150438 in native Push-T
coordinates. These are open-loop validation errors, not closed-loop scores.
Training session wall times are 9.55 / 61.44 minutes (startup excluded).

Read the local or remote audit, without executing a model:

```bash
cd /home/zsw/project_2026
cat simulation_output/pusht_bc_act_training_audit_v1/report.json
```

Completed verification commands, retained as provenance only:

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest \
  tools.test_audit_pusht_bc_act_training -v
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/audit_pusht_bc_act_training.py
```

The new 15 tests pass locally/remotely. Audit v1 explicitly accepts only this
single-session completed pair; a resumed/partial run fails instead of being
silently merged. It independently replays every CPU sampler batch, checks all
200,000 logged steps and hashes all payloads without unpickling a checkpoint,
constructing a policy, decoding images or stepping an environment. It refuses
an existing report; no duplicate successful audit is needed. Preserve failed
reports and use an explicitly named `--out` only after investigating a failure.

Inputs and outputs stay in place:

```text
/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1/
  bc/checkpoints/step_100000/       selected final BC, checkpoint.pt 16,478,157 bytes
  act/checkpoints/step_100000/      selected final ACT, checkpoint.pt 618,912,401 bytes
/home/zsw/project_2026/simulation_output/
  pusht_bc_act_training_audit_v1/report.json
  pusht_bc_act_pair_v1.launch.log
```

All 40 payloads total 12,707,811,160 bytes and are protected experiment evidence.
The SSD had approximately 25 GiB free after training. This audit fetched only
the report/log (202,837 bytes total), not checkpoints; files above 100 MB remain
user-transfer only. Do not clean up earlier checkpoints or runs to free space
without the applicable recovery/authorization procedure.

Post-run pins, small report/log copies verified locally:

```text
tools/audit_pusht_bc_act_training.py
  d053ba02543c84c14ab30991cc3b4aa11e1d54b9302eebfaedc5a7ef89379a43
tools/test_audit_pusht_bc_act_training.py
  6c97e013f26d23af3f64f9963df030225f4fc048b233145d07d4e479e6bac7bb
simulation_output/pusht_bc_act_training_audit_v1/report.json
  e5f005742df350758ffca19f053f8a5df6c47718e8a96ecd674189196393276a
simulation_output/pusht_bc_act_pair_v1.launch.log
  097a35fc38bc6554f16f0f60ceaefec19cde1e48d3332297160b0dc52a7abd2d
bc/sessions/1789316667946083501_1064553/metrics.jsonl
  da9af4a67b3b327d56adeda2aa9bcfa0e5a6eced2d5efbd9775c6e3d33a8e923
act/sessions/1789317245351472461_1082837/metrics.jsonl
  0a6099b7bdc363929d1adc71ac174bc4ca5b47ecc2df72653fe66d2aa9dbce4a
bc/checkpoints/step_100000/checkpoint.pt
  f65bdef47d9db651e20d1c5bda0442d123fa2270a404f79ea47af008899fee43
bc/checkpoints/step_100000/manifest.json
  fdbdd0924f32b3f21e139164a13d401d9263d411e6f0e5487a413ee187aa77f8
act/checkpoints/step_100000/checkpoint.pt
  8d7c1e1b5a99b82e99be9e8c1393cc74c03edf51ea8560a1e9acc24ced712393
act/checkpoints/step_100000/manifest.json
  15e8ee42dc733b4bdefe83710bb01d9af518f3bdb056d146a3a9c3abd37c330b
```

At this audit gate the next step was a separate final-checkpoint loader and
short clean Push-T smoke. That implementation/check is now completed above,
where the current user-run benchmark command is provided. Preserve ACT
chunk16/execute8 queues across steps; the
trainer's validation `predict()` resets every call and must not be copied as
the rollout policy. After that gate, hand the user the frozen seeds100000..100099,
max300-per-episode clean benchmark command. No new training, rollout, corruption
test or world-model work was executed during the audit. Final trained checkpoint
loading was then unverified (now passed above); actual interrupted-training
continuation remains unverified.

## Historical: User-Run BC/ACT Paired Clean Training Launch (2026-09-14)

Implementation and zero-update real-data preflight passed 63 local/remote tests
before launch. The user has now completed the long job audited above; these
launch/recovery examples are historical, not a request to run again. The job
was user-run: each model100,000 updates, batch64, same186/20 episode split
and frame anchors, no augmentation/pretraining. Final step only, no best-val
selection. See `docs/pusht-bc-act-training-protocol-v1.json` for all parameters.

The pre-run reservation was about1–2hours for the sequential pair. Measured
gradient-only lower components extrapolate to5.6min BC and43.1min ACT, excluding
optimizer, batch preparation, validation and checkpoint I/O. Actual duration
must come from training logs. Full run plus checkpoints exceeds five minutes.
Around13GB of checkpoint payloads are expected, with an additional runtime
storage guard; checkpoints stay on the999GB SSD, not the system partition.

Historical fresh-pair launch command; do not execute for the completed run.
This guarded command refuses an existing
launch log. The script also refuses either existing model run. It executes BC
then ACT, exits if BC fails, and never automatically retries/resumes:

```bash
cd /home/zsw/project_2026 &&
if [ -e simulation_output/pusht_bc_act_pair_v1.launch.log ]; then
  echo 'Existing launch log: inspect status and processes before resuming.'
else
  (set -C; nohup bash tools/run_pusht_bc_act_pair.sh \
    > simulation_output/pusht_bc_act_pair_v1.launch.log 2>&1 < /dev/null &)
fi
```

Follow progress(Ctrl-C on tail only stops the viewer):

```bash
tail -n 30 -f /home/zsw/project_2026/simulation_output/pusht_bc_act_pair_v1.launch.log
```

Outputs on the SSD:

```text
/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1/
  bc/ or act/
    run.json                 immutable protocol/data/source/preflight binding
    status.json              actual Python PID, completed step, latest checkpoint
    run.lock                 process lock; not a completion marker
    sessions/<time>_<pid>/    started.json, per-update metrics.jsonl, report.json
    checkpoints/step_005000/  checkpoint.pt + manifest.json
    ...
    checkpoints/step_100000/  only this completed step is the selected final model
```

If disconnected, inspect the launch log, each `status.json` and its live PID
before doing anything. Never relaunch a live job. Training-stage invocation
requires `--execute` and an exact successful preflight binding. A code, source,
data, protocol or runtime drift fails closed. Whole-dataset caching/normalization
checks repeat on startup without downloads; cache is about676MiB of RAM.

Recovery example ONLY when inspection confirms BC stopped and its latest
committed checkpoint really is step005000; substitute the verified model and
step if different. This is foreground training; a new outer nohup/log can be
used deliberately after inspection. It must not be run just because SSH closed:

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/train_pusht_bc_act.py \
  --stage train --model bc --execute \
  --resume /media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1/bc/checkpoints/step_005000
```

`--stop-after N` optionally saves a checkpoint and stops at an explicit total
completed-update index without changing the100,000-step final budget. It must
be greater than the resumed step. A deliberately stopped or incomplete model
is not the selected final model. Only explicit, same-run committed checkpoints
are accepted; later committed checkpoints block resuming an older one. Existing
`.partial` or failed checkpoint directories are never overwritten. Inspect and
preserve them before planning recovery; do not delete output directories or
silently restart from random weights. If BC was resumed separately, start ACT
separately only after BC completes and ACT has no existing run:

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/train_pusht_bc_act.py \
  --stage train --model act --execute
```

Preflight completed at `simulation_output/pusht_bc_act_training_preflight_v1_retry2/`:
25,650 frames fully decoded with exact PTS and6 matching earlier image hashes;
each model repeats the same real64-frame training batch for3 forward/backward
checks, performs zero optimizer updates, saves/reloads step0 and reproduces
predictions exactly. A64-frame validation probe also passed. This proves the
training interfaces/serialization path, not an optimizer-update trajectory,
full validation, closed-loop score or interrupted-training continuation.
Preserved failed v1/retry1 outputs document the fixed version-string and
postprocessor-device integration issues. Do not rerun successful preflight.
Its step0 checkpoints cannot be used as trained/resume checkpoints.

Final source/protocol pins(remote verified before execution):

```text
docs/pusht-bc-act-training-protocol-v1.json
  22e26bb73f40cd408b2df4a90f6bc113aaa811abbeb80b90711844f624ba86ea
tools/train_pusht_bc_act.py
  d9403ccd63d85c33adbdd71200ec132f487cbbaaac70328f7acbce30cfa89da2
tools/test_train_pusht_bc_act.py
  e38c2a2382ff679da160b67cea5c2be5316cea59e26f45d02e28d2251ff9bf7f
tools/run_pusht_bc_act_pair.sh
  656fa2bfe7e428834178f1b611e5b281fbb08a5048d37067e33d06e227d4ef64
tools/pusht_bc_act_training_data.py
  f92046e9f813e37fcaa692f44d9c972f0d6e34e44f18eff081e450502c9339b0
tools/test_pusht_bc_act_training_data.py
  e7c8ae8f75f5199dfda668fe373f39d3efceb7f6d546949f60163224afcb7db8
tools/pusht_bc_act_checkpoint.py
  669e179333a010371024ce006b897b26681296802ef0f62e75571fa95fc91f89
tools/test_pusht_bc_act_checkpoint.py
  fb0b04d7031305413995b3ebb692bc28c2a52369f697959a672da2f63951c40c
```

Successful report pins:

```text
simulation_output/pusht_bc_act_training_preflight_v1_retry2/bc/report.json
  28c2b98e064ba06eacf2fa8d13231abbb26446b3fe165a1e5135cb578d8b3bb4
simulation_output/pusht_bc_act_training_preflight_v1_retry2/act/report.json
  c45bc3551681a4cb5861203bee051465f12b21a04e3162968295e2e9ff75da8c
```

Only small reports/manifests/logs are copied locally. After training, inspect
all logs/checkpoints and run a separate trained-policy rollout smoke before the
frozen100-episode test; this trainer never starts that benchmark automatically.
No BC/ACT score exists yet. Official pretrained DP remains a separate reference;
R0 stays unopened, world-model work paused, and no project formal/real claim is
authorized by public Push-T results.

## Completed: BC/ACT Public Push-T Preparation (2026-09-13)

The user-approved acquisition, data audit and synthetic CUDA forward checks
are complete.22 local/remote tests pass. No optimizer, backward call, rollout
or training was executed. Do not rerun successful output paths. At this earlier
gate the trainer, recovery and frozen training budget/seed/selection protocol
were still pending; the completed implementation is recorded above.

Fixed source `lerobot/pusht` revision
`7628202a2180972f291ba1bc6723834921e72c19` has8 files totaling7,686,801 bytes.
All payload SHA256/size pins passed. Installed LeRobot0.4.4 reads its v3.0
metadata and PyAV video without conversion. Source is on the999GB volume:

```text
/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_7628202a
```

Completed commands, retained as provenance only:

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest \
  tools.test_prepare_pusht_bc_act_data tools.test_pusht_bc_act_models -v
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_pusht_bc_act_data.py --stage acquire
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/prepare_pusht_bc_act_data.py --stage audit --decode-smoke
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u \
  tools/pusht_bc_act_models.py --smoke --models both --device cuda \
  --out simulation_output/pusht_bc_act_models_smoke_v1.json
```

Acquisition skips byte-matching files but refuses mismatches or existing
`.partial` files; do not delete/overwrite those to hide a failure. If a transfer
is interrupted, inspect files and processes before deciding how to resume.
Audit and model smoke refuse existing output paths and preserve failed reports.
Only an inspected failure may use an explicitly named fresh `--output`(audit)
or `--out`(model smoke); do not create duplicate successful checks.

Remote and local small artifacts are identical:

```text
simulation_output/pusht_bc_act_data_gate_v1/report.json
  607f8b81bef37bbbb1b74ec00646a6062357cf06e501e32f0163834a7646eafb
simulation_output/pusht_bc_act_data_gate_v1/split.json
  24131e9146576d1aaa050b7726e8ae7c178a20ca92a63490ca9f1c74c5b22d59
simulation_output/pusht_bc_act_data_gate_v1/normalization.json
  01bb762f7174920e778326ac5585bc90eca64bd7d5374ef6e327a43baf781c89
simulation_output/pusht_bc_act_models_smoke_v1.json
  c82a931ff36ce3a24655fc9591136984f66caa3bce951d79bb94447965e1718a
```

The SHA values above are file-byte hashes. The audit also records canonical
JSON hashes(not interchangeable with the file hashes): split
`dbf8a2ccb3e58c41b72d9ab84c46b99f803e3e5dab4b5a90a51a8a94003ef4eb`,
normalization `82dbb1f14165ac8c41a75ae732b01e3789baf3d331ef2d5cf36eda52bd26adcc`.
Freeze186 train episodes/23,488 frames and20 val episodes/2,162 frames from
this split. Use the saved train-only state/action population mean/std plus
fixed ImageNet image statistics; never use global source stats or the model
smoke's synthetic stats for training/evaluation. The decoded probes cover only
six first/last frames(episodes1,0,205), not every image. Torchvision's video-API
deprecation warning did not prevent decoding.

Source pins verified before remote execution:

```text
docs/pusht-dataset-source-v1.json
  937cf36380eaf6c7623c4afc29a1120e00a461a61af78fea0be42146d116f016
tools/prepare_pusht_bc_act_data.py
  363db452dd729f5642ea93dedc8848ffc847cbf563d3f31de9c58ca254d12e00
tools/test_prepare_pusht_bc_act_data.py
  0b920f09e37382f82d221ca438163a35e9aaf81ca4dea56f043954b632970020
tools/pusht_bc_act_models.py
  6e35404670e9e13c8f425f0c9d5be77091360401507eef8a08a6eb9b36888332
tools/test_pusht_bc_act_models.py
  e87b63283bd52370291ec5912e56474a9e5b119b314a6d70465374b5d620bd96
```

The compact BC is a new lightweight reference, not an exact senior-model
reproduction. ACT uses full official512-dimensional architecture with random
ResNet18/VAE weights, chunk16/execute8 and unchanged L1+KL loss. Both accept
only image and observable state; target/padding remain separate. Synthetic
batch2 finite losses/output[2,2] and exact reset replay passed, but neither model
has a trained score. Do not compare their synthetic loss magnitudes. Preserve
the existing DP benchmark/source files, world-model pause and unopened R0.

## Completed: Trained Diffusion Policy / Clean Push-T (2026-09-13)

Acquisition, strict migration,32 local/remote tests, import-only preflight and
the20-step CUDA smoke have passed. The user-run100-episode benchmark is now
complete and audited:67/100 success, mean maximum coverage0.9090111,
Wilson95%57.3053%..75.4369%, zero action-bound violations,21862 transitions,
3003.842840s, strict protocol passed. **Do not rerun prepare, smoke or this
completed benchmark.** No training/world-model work is authorized by this log.

Historical user-run launch command, retained for provenance only. The guarded
background form tolerates an SSH disconnect and refuses existing run/log paths.
The original estimate was1-1.5hours; actual runtime was50.064minutes because
successful episodes can terminate before the300-step cap:

```bash
cd /home/zsw/project_2026 &&
if [ -e simulation_output/diffusion_pusht_benchmark_v1 ] || [ -e simulation_output/diffusion_pusht_benchmark_v1.launch.log ]; then
  echo 'Existing run/log: inspect it first; do not start a duplicate.'
else
  nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_diffusion_pusht_baseline.py \
    --stage benchmark --execute \
    --migration-report-sha256 fd7357f9030e07c50c76e6e2a5a47f7b3a3937b884e05bed30ae2ac15c2cfb05 \
    > simulation_output/diffusion_pusht_benchmark_v1.launch.log 2>&1 < /dev/null &
fi
```

Read progress (Ctrl-C on `tail` stops the viewer, not the background run):

```bash
tail -n 30 -f /home/zsw/project_2026/simulation_output/diffusion_pusht_benchmark_v1.launch.log
```

Outputs under `simulation_output/diffusion_pusht_benchmark_v1/`:
`started.json`, `status.json`, `per_step.jsonl`, `per_episode.jsonl`,
`report.json`. The status records the actual Python PID and completed episodes;
each finished episode prints progress. A completed report is not necessarily
a strict pass: inspect `strict_protocol_pass` and bounds counts as well.
If the connection drops, inspect status/log and that PID before any restart.
Failure/partial outputs are preserved. No automatic resume/retry; an inspected
failure needs a deliberately named fresh `--attempt` and a distinct launch log,
with the original protocol unchanged. Never delete the old output to rerun.

Protocol:100 clean episodes with seeds100000..100099, max300steps, native2D
actions, float32 CUDA, deterministic, no AMP/TF32 or action clipping, source
obs2/horizon16/action8/DDPM100. Every episode resets RNG, policy/processor queues
and the native environment, stops at first done, and reads actual success and
coverage from info. Different benchmark tasks are not combined into a ranking.
The official checkpoint lacks a pinned training-dataset revision, so this is
pretrained-policy reproduction, not yet a controlled BC/ACT training comparison.
Benchmark enforces the audited migration SHA plus successful same-binding smoke,
source hashes, environment parameters and versions. World-model research stays
paused. The clean DP result is now available; controlled same-data/budget
BC/ACT comparison and corruption benchmarks are still pending.

Post-run result pins:

```text
simulation_output/diffusion_pusht_benchmark_v1/report.json
  c1d4c4e582af1116ec41d554c8ae8be76480cee73488acff4471097bafe314b9
simulation_output/diffusion_pusht_benchmark_v1/per_episode.jsonl
  d27f2869b7c482d9fc4358314ad7f488d862dc721a4ac7aedc3a1a3fcd159c07
simulation_output/diffusion_pusht_benchmark_v1/per_step.jsonl
  45b2c591d72eab18b88c4027514329e94f940c6a4ff132b9a601072013d122a7
simulation_output/diffusion_pusht_benchmark_v1.launch.log
  07cecc6952a1f0d8329ac172c9b22fed79d47e11d893d34bcedec76c38781f2c
```

The independent read-only audit checks all43824 rows (100 resets,21862 actions,
21862 returned steps),100 unique seeds/reset hashes, exact action pairing,
first-done stop,67 successful terminations and33 failures truncated at300.
Root also rechecked original/migrated weight files and all source/runtime/smoke
bindings. No model was loaded for that audit. Do not rerun the benchmark to
obtain a better score.

Separate recorded-action visual verification, completed commands below are
history only (no policy inference or new score episodes):

```bash
cd /home/zsw/project_2026
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/replay_diffusion_pusht_benchmark.py --execute --attempt numeric_diagnostic
```

Replay tool SHA256:
`d3292341e97a9bdd06c9126a44178c7e019a03b43573e34a035ea2cc90260fae`.
Fixed seeds100000/100049/100099 replay706 recorded steps in3.552284s. All reset
image/state hashes and success/done flags agree. The initial exact-float attempt
is preserved as failed; the separate numeric diagnostic reports
`completed_with_numeric_differences`, max absolute coverage error4.440892e-16
and reward error5.551115e-16. No cross-process bitwise reproducibility claim;
the original benchmark scores are not changed. See handoff for the source-level
candidate explanation and visual-marker caveat.

Numeric replay outputs:
`simulation_output/diffusion_pusht_benchmark_replay_v1_numeric_diagnostic/report.json`
and `recorded_action_replay_zh.png` in that same directory. The original failed
exact attempt is under `simulation_output/diffusion_pusht_benchmark_replay_v1`;
do not delete or relabel it.

Numeric replay report SHA256:
`654a2c149c87e49322d83cc9c27eab3e23f7b074f91cfb8bd21b3927f0d5678c`;
image SHA256:
`ace2c41f7220d44ae65bc2aeac3cf8fa0febb3592d95ec84c9cf8c75a85e2c52`.
Root viewed the Chinese sheet: `viewed_not_accepted`, not user acceptance.
Small main reports/per-episode/log and replay report/image are locally
hash-verified. Full10.14MB `per_step.jsonl` stays remote (audited there); the
local zero-byte `per_step.jsonl.partial` from interrupted SCP is not a trace.
For later small transfers use `-o ServerAliveInterval=5 -o ServerAliveCountMax=2`
in addition to BatchMode/ConnectTimeout; do not restart the completed benchmark
because a file transfer stalls.

Verified source pins:

```text
tools/diffusion_pusht_checkpoint.py
  dda8a8571fe52ba014e595a3c245054e45870a2116b9c8974680bed3c5c91485
tools/test_diffusion_pusht_checkpoint.py
  93b18e6f54a93571db988447fd175cdac6128fb7cdb2f78761bac21a0f892580
tools/run_diffusion_pusht_baseline.py
  631876fed646e2d5d1de577ac1c556a76d04ea863697806e29aa107abc7b925a
tools/test_diffusion_pusht_baseline.py
  cc08dbcab2ec614355ce421e62b168c770ef97d9235f5ccfe958884516d2aab9
protocol SHA256
  e24a8d2e287fa8ebf038260cfb72193d51ea73964e243cd721c90d0cd88fcb89
```

Safe short checks, no policy inference:

```bash
cd /home/zsw/project_2026
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest discover -s tools -p 'test_diffusion_pusht*.py' -q
/home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_diffusion_pusht_baseline.py --stage preflight
```

Completed commands below are **history only**, not future reruns:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/diffusion_pusht_checkpoint.py --stage prepare
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_diffusion_pusht_baseline.py --stage smoke --execute --migration-report-sha256 fd7357f9030e07c50c76e6e2a5a47f7b3a3937b884e05bed30ae2ac15c2cfb05
```

Migration:8.176880s,213 model tensors byte-identical, eight original stats ->
six unique stats, strict reload, all CPU/CUDA transform probe errors0. Raw and
migrated >1GB weights remain on SSD; no large transfer. Remote artifact:
`/home/zsw/models/project_2026/diffusion_pusht_84a7c231_lerobot044_v1`.
Local small migration report:
`simulation_output/diffusion_pusht_checkpoint_gate_v1/migration_report.json`.

Smoke:10.879861s total;20 transitions3.482529s, no nonfinite/OOB actions,
optimizer0. It reached the short cap, not episode termination: no score claim.
All six files copied locally, SHA-verified, and41 trace rows checked (one reset,
20 pre-execution actions,20 returned steps). Representative Chinese image at
`simulation_output/diffusion_pusht_smoke_v1/smoke_sheet_zh.png` viewed by root,
`viewed_not_accepted`; immutable report keeps generation-time `not_viewed`.

```text
migration_report SHA256
  fd7357f9030e07c50c76e6e2a5a47f7b3a3937b884e05bed30ae2ac15c2cfb05
simulation_output/diffusion_pusht_smoke_v1/report.json
  b536a39ecba111e07c2cfec3c64af995f746fd963b8e61af5db82ffe99f2232b
simulation_output/diffusion_pusht_smoke_v1/smoke_sheet_zh.png
  ce4c1427ce1bdbba4968d0e5a481a90ca133780d4d6d750e346504dcfefc5b5e
```

## Baseline First: Acquire Official Diffusion Policy / Push-T (2026-09-13)

Acquisition is complete; the following is a recovery/reference command,
**download only**. New world-model experiments are paused. No training or
evaluation is launched by this command. The pinned
source is [the official checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653).

Run in the remote Linux shell (`ssh project4090`). The installed CLI syntax,
destination parent and available space were verified. About1.05GB is required;
the project skill assigns >100MB transfers to the user. This directory resolves
onto the999GB SSD, not the home filesystem. Do not replace existing PI05 files.

```bash
/home/zsw/miniconda3/envs/project2026-pi/bin/hf download lerobot/diffusion_pusht config.json model.safetensors train_config.json README.md \
  --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir /home/zsw/models/project_2026/diffusion_pusht_84a7c231 \
  --max-workers 1
```

Physical destination:
`/media/zsw/SSD1T/project_2026_weights_v1/models/diffusion_pusht_84a7c231`.
If interrupted, rerun the same pinned command and directory, without
`--force-download`; preserve download metadata/partial files. Do not delete or
overwrite another checkpoint. Do not run an old random-weight smoke as a score.

Required file pins (four files total1050872795 bytes):

| File | Bytes | SHA256 |
|---|---:|---|
| model.safetensors | 1050862408 | 995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8 |
| config.json | 1509 | d391a7bf488accd1c26b2043482f0060b0855b1ec236f3d7358486918472c0a5 |
| train_config.json | 5939 | 500ea79ba1bef13810219697ce60eca580a0f259f5c9bf4f847f8f843b06b14a |
| README.md | 2939 | 122b7d5067abfaf7e2ac44e6ab184bee8fb5c58d777e82d5c92bd3849dd308c3 |

After download, the agent checks every size/hash and prepares a separate
normalization-migrated artifact using the
[official legacy migration contract](https://huggingface.co/docs/lerobot/backwardcomp).
The raw model has eight embedded normalization tensors and213 policy tensors;
no saved pre/postprocessor files. Source stats must be preserved exactly,
with all213 model tensors unchanged and strict loading checked. Do not use
environment-bound statistics from earlier smokes. The pinned configuration
passes remote meta-device construction with `pos_grid=[9,2]`, crop84, no resize;
this is not a full-weight load, forward or rollout. No compatibility workaround
or policy architecture change is currently justified.

The migration/short-rollout gate and full evaluation are now complete and
audited; see Completed above, and do not rerun the historical launch. Official
model-card scores remain upstream references, not project results. BC/ACT
controlled training and compatible LIBERO comparators remain separate
preparation tasks, not implicitly launched by this acquisition.

## Remote Environment

```powershell
ssh project4090
```

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
```

The local Windows environment does not contain LeRobot. Before remote
execution, synchronize and verify the relevant final changed version under the
project skill's transfer rules (including the 100 MB user-transfer boundary):

```powershell
scp tools\changed_algorithm_file.py project4090:/home/zsw/project_2026/tools/
```

For a local syntax check that does not import LeRobot:

```powershell
.\.venv\Scripts\python.exe -c "import ast,pathlib; ast.parse(pathlib.Path(r'tools\changed_algorithm_file.py').read_text(encoding='utf-8')); print('ast_ok')"
```

## Remote Project Weight Storage

The 2026-09-12 migration moved every existing project weight file and the
pinned PI0.5 Hub cache from `/home/zsw` to the 999 GB SSD volume. Original
logical paths are preserved by symlinks:

```text
physical root:
  /media/zsw/SSD1T/project_2026_weights_v1

PI0.5 cache logical path:
  /home/zsw/.cache/huggingface/hub/models--lerobot--pi05_base

future model logical path:
  /home/zsw/models/project_2026
```

The future-model logical path resolves to
`/media/zsw/SSD1T/project_2026_weights_v1/models`; therefore commands later in
this file that download to `/home/zsw/models/project_2026` already write to the
SSD. Do not replace these symlinks with new regular directories. Before a large
download or new checkpoint, check both filesystems. After the task-finetuned
LIBERO checkpoint download, `/home` had `107,133,259,776` bytes free and the
SSD had `40,721,780,736` bytes free at `96%`.

Migration script and read-only preflight:

```powershell
scp tools\migrate_project_weights_to_ssd1t.sh project4090:/home/zsw/project_2026/tools/

ssh project4090 "cd /home/zsw/project_2026 && bash -n tools/migrate_project_weights_to_ssd1t.sh && sha256sum tools/migrate_project_weights_to_ssd1t.sh && bash tools/migrate_project_weights_to_ssd1t.sh --plan"
```

The completed execution used `--execute`. Do not rerun it as a cleanup command;
it is resume-aware only for the frozen migration manifest and future newly
created weights need a new inventory.

Verified result:

```text
status=passed
project_weight_count=41
project_weight_bytes=52662528608
pi05_cache_bytes=14467168162
total_source_bytes=67129696770
result_count=42
project paths preserved by symlink=true
HF cache path preserved by symlink=true
future model path redirected=true
project weight links=41
broken project weight links=0
partial/backup/temp residue=0

/home free bytes after=107136139264
/media/zsw/SSD1T free bytes after=48194990080

tools/migrate_project_weights_to_ssd1t.sh
  sha256=0f7983a45d3a087799edebf4c218c8edd8b2e6855a53ca3879df4395e40437c0
simulation_output/weight_migration_to_ssd1t_v1/source_weight_paths.txt
  sha256=dc1bcccfe3feb20555c14e23eff4495aeb4e2b78c7718477e83546c3d0bd3bfc
simulation_output/weight_migration_to_ssd1t_v1/migration_results.tsv
  sha256=0e1beeb1d3c89b734726d30eb57f35a01afca363a2b661baeb8d3c0e64a0d4ab
simulation_output/weight_migration_to_ssd1t_v1/summary.txt
  sha256=db0d8de0b8cfaf27e31dfb9edee783fadd7d4a655e4a074c2b41f2fa209d4a54
```

Offline cache-path verification:

```bash
cd /home/zsw/project_2026
HF_HUB_OFFLINE=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

p = Path(snapshot_download(
    "lerobot/pi05_base",
    revision="7de663972b7817d2c4cf2d84c821153dfea772e9",
    cache_dir="/home/zsw/.cache/huggingface/hub",
    local_files_only=True,
))
model = p / "model.safetensors"
assert str(p.resolve()).startswith(
    "/media/zsw/SSD1T/project_2026_weights_v1/huggingface/hub/"
)
assert model.stat().st_size == 14467165872
print("status=passed network_used=false model=" + str(model.resolve()))
PY
```

The older `probe_pi05_pretrained_loading.py --metadata-only` helper still calls
the Hub API before resolving local files, so an explicit offline invocation of
that helper fails with `OfflineModeIsEnabled`. The direct local snapshot check
above passed and is the accepted post-migration path check; the failed helper
run is retained as diagnostic evidence, not a cache failure.

## Installed PI05 Interface Probe

```powershell
scp tools\probe_pi05_feature_interface.py project4090:/home/zsw/project_2026/tools/

ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/probe_pi05_feature_interface.py --out simulation_output/pi05_feature_interface_probe_v1.json --source-out simulation_output/pi05_modeling_source_v1.py"
```

Verified result:

```text
LeRobot version: 0.4.4
PI05 forward/sampling inputs: images + language
observation.state consumed by Elite diffusion: no
```

## Current Stop Condition

Do not rerun the existing mixed-head v1 long training or full evaluation.

```text
final piper_acc=0.53125
majority_baseline=0.53125
balanced_accuracy=0.5
final predictions=all class 2
```

The current base checkpoint was created with `PI05Policy(config)`, not
`PI05Policy.from_pretrained(...)`. It is architecture/loss feasibility only.

## Verified Pretrained Metadata Probe

The loader is pinned to:

```text
repository: lerobot/pi05_base
revision: 7de663972b7817d2c4cf2d84c821153dfea772e9
model.safetensors: 14,467,165,872 bytes
blob id: 822bdb619cecbe41045c54788df264e57ec03572
```

Metadata-only verification, which does not download model weights:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/probe_pi05_pretrained_loading.py \
  --out simulation_output/pi05_pretrained_metadata_probe_v1.json \
  --metadata-only
```

## Download Pinned PI05 Base

This is a long 14.5 GB download and should be run by the user on the 4090:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/probe_pi05_pretrained_loading.py \
  --out simulation_output/pi05_pretrained_download_v1.json \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --cache-dir /home/zsw/.cache/huggingface/hub \
  --download-only
```

## Verified Full Load And Backward

This uses the local cache and fails instead of silently returning a random
model:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/probe_pi05_pretrained_loading.py \
  --out simulation_output/pi05_pretrained_load_forward_backward_v2.json \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --cache-dir /home/zsw/.cache/huggingface/hub \
  --local-files-only \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --max-stats-records 32 \
  --batch-size 1 \
  --backward
```

Verified result:

```text
resolved revision: 7de663972b7817d2c4cf2d84c821153dfea772e9
checkpoint/model/direct-match keys: 812 / 813 / 812
raw/effective missing keys: 1 / 0
unexpected/shape mismatch: 0 / 0
verified tied aliases: 1
raw duplicated-state coverage: 0.872895
unique-parameter coverage: 1.0
loss: 1.043866
gradient tensors / L2: 208 / 39.2795
```

The raw missing key is the omitted PaliGemma embedding in tied-weight
serialization. `tools/probe_pi05_tied_weights.py` verified that it shares shape,
data pointer, and storage with the stored LM-head parameter. The loader accepts
only that exact alias and rechecks both fingerprints and shared storage after
loading; do not lower the coverage threshold to handle other missing keys.

## Controlled Piper Head Comparison

Run these as two separate user-owned jobs on the 4090. Keep every argument
except `--out` and `--head-mode` identical so the comparison uses the same
records, deterministic shuffle, validation subset, weighting, and pretrained
revision.

State-only diagnostic:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_piper_state_only_pretrained_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 20260717 \
  --class-weighting balanced \
  --head-mode state_only \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Dimension-balanced prefix plus state:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_piper_prefix_state_balanced_pretrained_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 20260717 \
  --class-weighting balanced \
  --head-mode prefix_state \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

After both finish, enforce the controlled-run contract and compare results:

```bash
python tools/compare_pi05_piper_head_runs.py \
  --state-only-log simulation_output/pi05_piper_state_only_pretrained_frontup_step024_v1/train_log.json \
  --prefix-state-log simulation_output/pi05_piper_prefix_state_balanced_pretrained_frontup_step024_v1/train_log.json \
  --out simulation_output/pi05_piper_head_comparison_pretrained_frontup_step024_v1.json
```

Do not select a head from cross-entropy alone. Inspect accuracy versus majority,
balanced accuracy, predicted-class support, and the confusion matrix. Require
performance above the majority baseline and non-degenerate predictions before
any full open-loop or rollout-like evaluation. The comparison report records
both the best step-level validation point and the final point so late-training
degradation cannot be hidden by a final-only comparison.

Verified controlled result:

```text
state_only best: step 100, accuracy 0.6250, balanced accuracy 0.6167,
                 predicted support [0,46,82]
prefix_state best: step 800, accuracy 0.5234, balanced accuracy 0.5515,
                   predicted support [0,121,7]
majority baseline: 0.53125
prefix_state minus state_only best balanced accuracy: -0.0652
```

The current `prefix_state` fusion is rejected. Reproduce the selected
`state_only` point in a short run using the best-checkpoint-aware trainer:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_piper_state_only_pretrained_frontup_step024_best_v1 \
  --max-steps 100 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 20260717 \
  --class-weighting balanced \
  --head-mode state_only \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

The output must contain both `mixed_head_policy.pt` and
`best_mixed_head_policy.pt`; at 100 steps they should represent the same
selected validation point.

Verified reproduction: step 100 exactly matched the original long-run
accuracy `0.625`, balanced accuracy `0.6167`, CE `0.669158`, and confusion
matrix. The two checkpoint files contain identical head weights and distinct,
correct selection metadata.

## Full Held-Out Open-Loop Evaluation

This is a user-owned full evaluation over all 616 held-out records. It loads
the pinned pretrained PI05 base, requires the selected best head, and rejects
training/evaluation provenance mismatch:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --mixed-head-checkpoint simulation_output/pi05_piper_state_only_pretrained_frontup_step024_best_v1/best_mixed_head_policy.pt \
  --out simulation_output/pi05_piper_state_only_pretrained_open_loop_fullval_v1.json \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --max-val-batches 616 \
  --seed 123 \
  --eval-seed 20260717 \
  --save-samples 32 \
  --require-best-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

The report must show 616 records, best-checkpoint step 100, matched pretrained
provenance, Piper metrics/confusion, and Elite TCP-delta absolute-error metrics.
This remains sim-data algorithm feasibility, not rollout or real-system
validation.

Verified full result:

```text
Piper accuracy / balanced accuracy / majority: 0.5552 / 0.5643 / 0.5292
Piper target support: [0,290,326]
Piper predicted support: [0,402,214]
Piper class-1 / class-2 recall: 0.7207 / 0.4080
Elite 6D MAE mean / median / P95 / max: 0.7360 / 0.5146 / 2.6789 / 8.3575
Elite dim-1 MAE: 1.8585
```

The explicit Piper head is weak but above majority. The unfine-tuned
pretrained Elite policy is not ready for tactile injection or rollout-like
evaluation.

## Pretrained Elite-Only Fine-Tune Scan

This user-owned scan optimizes only Elite TCP-delta dims `0:6`; Piper one-hot
compatibility dims are excluded. It intentionally does not save the large
policy checkpoint. Inspect the validation trajectory first, then rerun only to
the selected best step with `--save-checkpoint`.

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_pretrained_elite_only_train_scan_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-only-loss \
  --lr 2.5e-5 \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

The log must report `active_action_dims=6`, loss scope
`elite_tcp_delta_0_6`, pinned pretrained coverage `1.0`, and validation points
at each 100-step interval.

Verified scan result:

```text
initial validation loss: 2.425742
best/final step: 1000
best/final validation loss: 0.575906
final minus initial: -1.849836
elapsed training/evaluation time: 245.2 seconds
```

Reproduce the selected step and save the approximately 7 GB policy checkpoint:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_pretrained_elite_only_train_ckpt_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-only-loss \
  --lr 2.5e-5 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

The output must contain `final_policy.pt` with checkpoint step `1000`, active
action dims `6`, Elite-only loss scope, and the pinned pretrained provenance.

Verified saving rerun:

```text
every validation point versus scan: exact match
maximum validation-loss difference: 0
checkpoint size: 7,473,380,005 bytes
SHA256: 43c084e5b49d2a3019506305212b075db825d0e8ff2a9f31602c665ff31e98f9
remaining 4090 disk space after save: about 99 GB
```

## Combined Elite Policy And Piper Head Evaluation

The evaluator accepts the Elite-only fine-tuned policy with the retained
`state_only` Piper head only when both derive from the same pinned pretrained
source and revision. A one-record remote smoke passed with Elite checkpoint
step `1000`, Piper best-checkpoint step `100`, active action dims `6`,
Elite-only loss scope, and pretrained coverage `1.0`. Its one-record task
metrics are not performance evidence.

Run the user-owned full evaluation over all 616 held-out records:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --base-policy-checkpoint simulation_output/pi05_pretrained_elite_only_train_ckpt_frontup_step024_v1/final_policy.pt \
  --mixed-head-checkpoint simulation_output/pi05_piper_state_only_pretrained_frontup_step024_best_v1/best_mixed_head_policy.pt \
  --out simulation_output/pi05_pretrained_elite_only_state_head_open_loop_fullval_v1.json \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --max-val-batches 616 \
  --seed 123 \
  --eval-seed 20260717 \
  --save-samples 32 \
  --require-best-checkpoint \
  --require-elite-only-finetune
```

The report must show 616 records, matched pinned pretrained provenance,
Elite checkpoint step `1000`, Piper checkpoint step `100`, and the full Elite
MAE and Piper classification metrics. This is held-out simulation-data
algorithm feasibility, not rollout or real-system validation.

Verified full result:

```text
Piper accuracy / balanced accuracy / majority: 0.5552 / 0.5643 / 0.5292
Piper result versus unfine-tuned-base evaluation: exactly unchanged
Elite mean / median / P95 / max MAE: 0.4664 / 0.0722 / 2.5340 / 7.4591
Unfine-tuned Elite mean / median / P95 / max: 0.7360 / 0.5146 / 2.6789 / 8.3575
Aggregate mean / median improvement: 36.6% / 86.0%
Per-dim mean-MAE change dims 0:6: +12.8%, -4.6%, -1.3%, -90.1%, -94.8%, -92.7%
```

The aggregate gain is dominated by rotation dims 3:6. Their target means are
all zero and their zero standard deviations are replaced by `1` during
normalization. Translation dims 0:3 remain weak, including a regression in
dim 0, so do not start tactile injection from this result.

## Translation-Only Elite Fine-Tune Scan

`--elite-translation-only-loss` optimizes only Elite TCP translation dims
`0:3`. It is mutually exclusive with `--elite-only-loss` and records active
action dims `3` plus an explicit loss scope. A two-step remote smoke passed at
pinned pretrained coverage `1.0`; its one-batch loss is not a performance
result.

Run the controlled user-owned 1000-step scan without saving a large policy:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_pretrained_elite_translation_only_train_scan_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --max-stats-records 2048 \
  --val-fraction 0.15 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --lr 2.5e-5 \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Require `active_action_dims=3`, loss scope
`elite_tcp_translation_delta_0_3`, pinned coverage `1.0`, and three
per-dimension validation losses at every 100-step point.

Verified scan result:

```text
initial translation validation loss: 3.426407
best/final step: 1000
best/final translation validation loss: 1.121533
final per-dim losses dims 0:3: 0.860086 / 1.162485 / 1.342028
existing six-dim run losses dims 0:3: 0.794087 / 1.156091 / 1.338871
translation-only minus six-dim translation mean: +2.3%
```

Decision: do not rerun with `--save-checkpoint`. Translation-only loss did not
beat the six-dimensional Elite-only run on any translation dimension. The next
algorithm step is the data-readiness gate in
`docs/vla-training-data-readiness-audit.md`, before a full state-conditioned
PI05 ablation or tactile context injection.

## Read-Only PI05 Training-Data Audit

After editing the audit script locally, sync it to the 4090 and run it in the
LeRobot environment:

```powershell
scp tools\audit_pi05_training_dataset.py project4090:/home/zsw/project_2026/tools/
```

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/audit_pi05_training_dataset.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_training_data_audit_frontup_step024_v1.json \
  --contact-sheet simulation_output/pi05_training_data_audit_frontup_step024_v1.png \
  --max-records 4096 \
  --val-fraction 0.15 \
  --seed 123 \
  --ridge-alpha 10 \
  --image-audit
```

The audit is read-only. It reports schema checks, action degeneracy, temporal
redundancy, simple held-out baselines, image similarity, and representative
train/validation frames. It does not relabel data or modify collectors.

## Contact-Risk Existing-Adapter Smoke

This probe validates the policy-safe diagnostic pack and feeds matched
risk-off/risk-on observations through the existing `PI05ProbeDataset` and
collate path. It does not instantiate or modify PI05/mixed-head models and does
not train on risk.

Run locally because the source images are currently in the Windows workspace:

```powershell
.\.venv\Scripts\python.exe tools\probe_contact_risk_adapter_views.py `
  simulation_output\contact_risk_diagnostic_pack_v1 `
  --project-root . `
  --out simulation_output\contact_risk_adapter_probe_v1.json `
  --max-adapter-records 24 `
  --image-size 224 `
  --check-all-images `
  --allow-policy-disabled-risk-diagnostic
```

The last flag is deliberately required because the manifest declares
`policy_input_allowed=false`. It authorizes only this in-memory comparison; do
not add it to training commands.

Verified result:

```text
schema: pass
samples / episodes / checked images: 7096 / 52 / 14192
risk flag off / on: 6172 / 924
Piper hold / feed: 3320 / 3776
forbidden exact-contact/wall keys in policy samples: 0
risk-off/on changed state dims: 8:14 only
non-state adapter fields changed: none
model structure changed: false
risk used as training label: false
```

The report ID-checks `diagnostic_targets.jsonl` only to verify one-to-one
separation. It never merges diagnostic contact/wall fields into an observation.

## Stride-5 Fixed-Split PI05 Baseline

Authoritative dataset:

```text
root: simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1
repo-id: project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1
split manifest: project2026_lerobot_export_manifest.json
frames / episodes: 1005 / 40
train / validation: 851 / 154 records, 34 / 6 episodes
```

All new training and evaluation commands must pass `--split-manifest` and must
not pass `--max-records`. The code rejects that combination so an episode
cannot be truncated accidentally.

Verified full fixed-split audit:

```text
previous-label Piper accuracy: 0.5135
majority Piper accuracy: 0.5325
state-ridge Piper accuracy: 0.8182
image-ridge Piper accuracy: 0.4545
previous-action Elite translation MAE: 0.9662
train-mean Elite translation MAE: 0.7057
state-ridge Elite translation MAE: 0.5520
image-ridge Elite translation MAE: 0.5080
exact validation image hashes present in train: 0
```

Pinned-pretrained one-step smokes passed for Elite translation and the
`state_only` Piper head. The mixed open-loop evaluator also emitted fixed-split
translation, transition/steady, and temporal-baseline fields. Smoke metrics are
not performance evidence.

The user-owned 1000-step Elite translation scan completed without saving a
large checkpoint:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_stride5_translation_scan_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --lr 2.5e-5 \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Require `split=export_manifest_episode_split`, `max_records=null`, pretrained
coverage `1.0`, active action dims `3`, and three validation loss dimensions at
every evaluation point. Do not add `--save-checkpoint` to this scan.

Verified scan result:

```text
initial validation diffusion loss: 2.945760
best step / loss: 900 / 1.031572
step-1000 loss: 1.040866
best per-dim losses dims 0:3: 0.927054 / 1.103755 / 1.063905
pretrained checkpoint keys: 812/812
effective missing / unexpected / shape mismatch: 0 / 0 / 0
unique-parameter coverage: 1.0
```

The two LeRobot warnings naming vision `patch_embedding.weight` and `.bias`
are unconditional informational messages in LeRobot 0.4.4's PI05 state-dict
fixer. Both keys are retained unchanged, and the fail-closed project report
confirms complete loading for the pinned revision.

Reproduce the selected step and save the approximately 7 GB policy checkpoint:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_stride5_translation_ckpt_step900_v1 \
  --max-steps 900 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --lr 2.5e-5 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Do not start the 154-record open-loop evaluation until the saving rerun has
reproduced the scan through step 900 and `final_policy.pt` metadata has been
checked.

Verified saving rerun:

```text
final validation loss: 1.0315715289 (exactly reproduced)
checkpoint: simulation_output/pi05_stride5_translation_ckpt_step900_v1/final_policy.pt
checkpoint SHA256: fc6b863f9f7ddb8a5a50d6337080e766fcae1618d4fe0c5816931bc5ec61f116
```

Run the user-owned full fixed-split Elite open-loop evaluation:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/eval_pi05_lerobot_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --checkpoint simulation_output/pi05_stride5_translation_ckpt_step900_v1/final_policy.pt \
  --out simulation_output/pi05_stride5_translation_open_loop_fullval_step900_v1.json \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --max-val-batches 154 \
  --seed 123 \
  --eval-seed 10007 \
  --save-samples 32 \
  --sample-mode random
```

The report must contain `records=154`, `split=export_manifest_episode_split`,
Elite translation MAE, Piper transition/steady metrics, and the temporal
baselines. This remains a held-out simulation-data feasibility evaluation.

Verified open-loop result:

```text
records / split: 154 / export_manifest_episode_split
Elite translation MAE mean / median / P95 / max: 0.7660 / 0.4378 / 2.5704 / 6.1637
Elite translation transition / steady MAE: 0.4866 / 0.9260
previous-action Elite translation MAE: 0.9662 (148 eligible records)
Piper accuracy / majority: 0.3247 / 0.5325
Piper transition / steady accuracy: 0.2917 / 0.3026
```

The Elite result beats the previous-action shortcut but remains worse than the
readiness-audit train-mean, state-Ridge, and image-Ridge baselines. The Piper
numbers are not a model result because this checkpoint intentionally excludes
Piper loss. The next experiment is a state-conditioned Elite translation
ablation with the same fixed split and no tactile/contact fields.

## State-Conditioned Elite Translation Ablation

The `--state-conditioning` route appends a trainable projection of the 32D
`observation.state` as one prefix token after image/language embeddings. It is
attached only after pinned pretrained loading, and the checkpoint records
`state_prefix_token_v1`; no exact contact, wall, route, or simulator-truth
fields are used.

Verified remote interface smokes:

```text
pretrained coverage: 1.0
state adapter: enabled, state_dim=32, projected_dim=2048
parameter increase: 67,648
checkpoint round-trip: pass
fixed-split open-loop inference smoke: 1 record, pass
```

Run the user-owned 1000-step scan before saving a large checkpoint:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_stride5_state_conditioned_scan_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --state-conditioning \
  --lr 2.5e-5 \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Do not add `--save-checkpoint` to this scan. Select the best validation step
first, then rerun that step with `--save-checkpoint` and use the existing
open-loop evaluator; it automatically enables the adapter from checkpoint
metadata.

Verified state-conditioned scan:

```text
best step / validation loss: 900 / 1.035247
image/language-only step-900 loss: 1.031572
relative loss change: +0.36%
state-conditioned per-dim loss dims 0:3: 0.875492 / 1.121874 / 1.108376
image/language-only per-dim loss dims 0:3: 0.927054 / 1.103755 / 1.063905
```

Reproduce step 900 and save the checkpoint for the final raw-MAE comparison:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_stride5_state_conditioned_ckpt_step900_v1 \
  --max-steps 900 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --state-conditioning \
  --lr 2.5e-5 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

After saving, use `eval_pi05_lerobot_open_loop.py` with the same fixed split
and `--max-val-batches 154`, writing
`simulation_output/pi05_stride5_state_conditioned_open_loop_fullval_step900_v1.json`.

Verified saving rerun:

```text
final validation loss: 1.0352473862 (exactly reproduced)
checkpoint: simulation_output/pi05_stride5_state_conditioned_ckpt_step900_v1/final_policy.pt
checkpoint SHA256: 6d37fae648c2cf9a0d894c3f5a0b8da2e748e853932f406417537dabef45aae4
```

Run the final user-owned state-conditioned full-validation comparison:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/eval_pi05_lerobot_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --checkpoint simulation_output/pi05_stride5_state_conditioned_ckpt_step900_v1/final_policy.pt \
  --out simulation_output/pi05_stride5_state_conditioned_open_loop_fullval_step900_v1.json \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --max-val-batches 154 \
  --seed 123 \
  --eval-seed 10007 \
  --save-samples 32 \
  --sample-mode random
```

The evaluator reads `state_conditioning` from the checkpoint, reattaches the
adapter before loading weights, and then uses state during action sampling.

Verified state-conditioned open-loop result:

```text
records / split: 154 / export_manifest_episode_split
Elite translation MAE mean / median / P95 / max: 0.7478 / 0.4169 / 2.4515 / 6.3311
Elite translation transition / steady MAE: 0.4433 / 0.9192
image/language-only Elite translation mean: 0.7660
mean improvement: 2.37%
Piper accuracy / majority: 0.3571 / 0.5325 (untrained compatibility output)
```

The state prefix is retained as the current Elite candidate, but the gain is
modest and remains below train-mean/state-Ridge/image-Ridge readiness baselines.

## Stride-5 Piper State-Only Head

Train Piper separately on all stride-5 records and the same episode manifest;
do not pass `--max-records` and do not combine its metrics with the untrained
Piper compatibility output from Elite translation checkpoints:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_stride5_piper_state_only_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --class-weighting balanced \
  --head-mode state_only \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Select the best Piper step using balanced accuracy, majority baseline,
predicted support, and transition/steady metrics before any combined report.

Verified stride-5 Piper head result:

```text
best step: 1000
accuracy / balanced accuracy / majority: 0.5974 / 0.6194 / 0.5325
transition / steady accuracy: 0.6111 / 0.5526
previous-label accuracy: 0.5135
```

Run the final combined Elite/Piper open-loop evaluation:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --base-policy-checkpoint simulation_output/pi05_stride5_state_conditioned_ckpt_step900_v1/final_policy.pt \
  --mixed-head-checkpoint simulation_output/pi05_stride5_piper_state_only_v1/best_mixed_head_policy.pt \
  --out simulation_output/pi05_stride5_state_conditioned_piper_combined_fullval_v1.json \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --max-val-batches 154 \
  --seed 123 \
  --eval-seed 10007 \
  --save-samples 32 \
  --require-best-checkpoint \
  --require-elite-translation-finetune
```

The evaluator restores the state adapter from the Elite checkpoint metadata;
do not add a separate `--pretrained-*` source when using `--base-policy-checkpoint`.

Verified combined full-validation result:

```text
records / split: 154 / export_manifest_episode_split
Elite translation MAE mean / median / P95: 0.7478 / 0.4169 / 2.4515
Elite transition / steady MAE: 0.4433 / 0.9192
Piper accuracy / balanced accuracy / majority: 0.5974 / 0.6194 / 0.5325
Piper transition / steady accuracy: 0.6111 / 0.5526
Elite checkpoint step / Piper checkpoint step: 900 / 1000
```

The mixed evaluator report now explicitly records that the base consumes the
state prefix adapter while exact contact/wall/route truth remains excluded.
This is the current simulation-data algorithm baseline, not real validation.

## Inspect Remote Results

```powershell
ssh project4090 "cd /home/zsw/project_2026 && cat path/to/summary.json"
```

Copy a small result back to Windows:

```powershell
scp project4090:/home/zsw/project_2026/path/to/result.json simulation_output\result.json
```

Long training and full evaluation remain user-run operations. The agent should
inspect the resulting summaries, logs, checkpoints, and open-loop metrics after
completion.

## E2 Spatial Weighted-Overlay Paired Diagnostic

This is a diagnostic overlay comparison, not formal training or real-system
validation. Run all commands on the Ubuntu 4090. Do not add `--max-records`,
change the split manifest, use E2 for validation, or pass offline role/weight
fields as observations.

The adapter smoke is already verified, but this command reproduces it:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/test_pi05_e2_overlay_smoke.py \
  simulation_output/e2_spatial_candidate_pack_v1 \
  --project-root . \
  --out simulation_output/pi05_e2_overlay_adapter_smoke_v1.json
```

Expected contract: 836 records, 1672 valid images, hold/feed mapped to action
dims 7/8, explicit diagnostic opt-in, and no offline-only field in the model
batch.

### 1. Fresh Paired Elite Baseline

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_e2_paired_elite_baseline_step900_v1 \
  --max-steps 900 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --state-conditioning \
  --weighted-training-sampler \
  --lr 2.5e-5 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

### 2. E2 Weighted-Overlay Elite

```bash
python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_e2_paired_elite_weighted_overlay_step900_v1 \
  --max-steps 900 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --elite-translation-only-loss \
  --state-conditioning \
  --weighted-training-sampler \
  --training-overlay-pack simulation_output/e2_spatial_candidate_pack_v1 \
  --training-overlay-project-root . \
  --allow-diagnostic-overlay \
  --lr 2.5e-5 \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Both Elite summaries must report 900 optimizer steps, seed 123, 851 base
records, and the same base-only normalization statistics. Only the overlay run
may report 836 overlay records and expected overlay draw fraction `0.495554`.

Verified paired Elite baseline result:

```text
final validation diffusion loss: 1.0335452076
best step: 900
sampler optimizer steps / draws: 900 / 900
overlay records: 0
checkpoint SHA256: 29e4c0c8d37277769f94a466694f3de9270446377ba18134a462c1b057fc22bb
checkpoint round-trip: pass
```

Verified paired Elite weighted-overlay result:

```text
same initial validation loss as baseline: 2.7678067472
final validation diffusion loss: 1.2942433481
baseline final loss: 1.0335452076
overlay minus baseline: +0.2606981405 (+25.22%)
lowest overlay logged loss / step: 1.2577329827 / 700
sampler optimizer steps / draws: 900 / 900
overlay records / expected draw fraction: 836 / 0.495554
checkpoint SHA256: dfe6a279ade28f9965ec2c1e16d71869947d18e87aec8c38ca04fea2860b9943
checkpoint round-trip: pass
```

The controlled training-side loss result is unfavorable in all three Elite
translation dimensions. Keep the fixed step-900 checkpoint for raw-MAE
evaluation; do not replace it with the overlay-only step-700 point.

### 3. Fresh Paired Piper Baseline

```bash
python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_e2_paired_piper_baseline_step1000_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --class-weighting balanced \
  --head-mode state_only \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --weighted-training-sampler \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

### 4. E2 Weighted-Overlay Piper

```bash
python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_e2_paired_piper_weighted_overlay_step1000_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --val-batches 154 \
  --eval-every 100 \
  --log-every 10 \
  --seed 123 \
  --eval-seed 10007 \
  --class-weighting balanced \
  --head-mode state_only \
  --head-feature-dim 128 \
  --head-hidden-dim 256 \
  --head-lr 1e-3 \
  --freeze-pi05 \
  --weighted-training-sampler \
  --training-overlay-pack simulation_output/e2_spatial_candidate_pack_v1 \
  --training-overlay-project-root . \
  --allow-diagnostic-overlay \
  --save-checkpoint \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Both Piper runs must report class counts `[0,400,451]`, class weights derived
from `base_training_split_only`, and 1000 optimizer steps. Use
`mixed_head_policy.pt` for the comparison so both endpoints are step 1000;
do not compare different `best_mixed_head_policy.pt` steps.

Verified paired Piper baseline result:

```text
step-1000 accuracy / balanced accuracy: 0.5649350882 / 0.5567411924
step-1000 transition / steady accuracy: 0.5555555820 / 0.5394737124
step-1000 hold/feed confusion: [[31, 41], [26, 56]]
best logged balanced accuracy / step: 0.5640243902 / 900
class counts / source: [0,400,451] / base_training_split_only
sampler optimizer steps / draws: 1000 / 1000
overlay records: 0
final checkpoint SHA256: 235a2b3b881a5f3174cd948f29ab9a4f2f25f4c7dca164b48f52c7a2941ae815
combined checkpoint round-trip: pass
```

Verified paired Piper weighted-overlay result:

```text
step-1000 accuracy / balanced accuracy: 0.5714285970 / 0.5687669377
absolute changes versus baseline: +0.006493509 / +0.012025745
step-1000 transition / steady accuracy: 0.5555555820 / 0.5526315570
transition / steady changes: 0.0 / +0.013157845
step-1000 hold/feed confusion: [[38, 34], [32, 50]]
best logged balanced accuracy / step: 0.5821476965 / 700
class counts / source: [0,400,451] / base_training_split_only
sampler optimizer steps / draws: 1000 / 1000
overlay records / expected draw fraction: 836 / 0.495554
final checkpoint SHA256: 71d3bb680521e8fbe6a9f9da74331f6797f250deff62914e3667fbe1483a8729
combined checkpoint round-trip: pass
```

The strict comparison retains the final step-1000 checkpoint. The Piper gain
is small and does not establish overall overlay benefit before Elite raw-MAE
evaluation.

### 5. Fixed-Validation Evaluation

Baseline:

```bash
python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --base-policy-checkpoint simulation_output/pi05_e2_paired_elite_baseline_step900_v1/final_policy.pt \
  --mixed-head-checkpoint simulation_output/pi05_e2_paired_piper_baseline_step1000_v1/mixed_head_policy.pt \
  --out simulation_output/pi05_e2_paired_baseline_fullval_v1.json \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --max-val-batches 154 \
  --seed 123 \
  --eval-seed 10007 \
  --save-samples 32 \
  --require-elite-translation-finetune
```

Overlay:

```bash
python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --base-policy-checkpoint simulation_output/pi05_e2_paired_elite_weighted_overlay_step900_v1/final_policy.pt \
  --mixed-head-checkpoint simulation_output/pi05_e2_paired_piper_weighted_overlay_step1000_v1/mixed_head_policy.pt \
  --out simulation_output/pi05_e2_weighted_overlay_fullval_v1.json \
  --batch-size 1 \
  --max-stats-records 851 \
  --split-manifest simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1/project2026_lerobot_export_manifest.json \
  --max-val-batches 154 \
  --seed 123 \
  --eval-seed 10007 \
  --save-samples 32 \
  --require-elite-translation-finetune
```

Verified paired baseline full-validation report:

```text
records / validation episodes: 154 / 6
Elite translation MAE mean / median / P95: 0.7732550502 / 0.4046515226 / 2.7439994812
Elite transition / steady MAE: 0.4907198548 / 0.9324470758
Piper accuracy / balanced accuracy: 0.5649350882 / 0.5567411924
Piper hold/feed confusion: [[31, 41], [26, 56]]
Piper transition / steady accuracy: 0.5555555820 / 0.5394737124
report SHA256: cdcd781a9279be2f67541f9068d19beafff816c612698101084c19141a6858b5
```

Use this fresh weighted-sampler report, not the older epoch-shuffled baseline,
for the E2 comparison.

### 6. Paired Report

```bash
python tools/compare_pi05_e2_overlay_runs.py \
  --baseline-report simulation_output/pi05_e2_paired_baseline_fullval_v1.json \
  --overlay-report simulation_output/pi05_e2_weighted_overlay_fullval_v1.json \
  --out simulation_output/pi05_e2_weighted_overlay_comparison_v1.json
```

The comparison fails closed if the split, validation support, seeds,
checkpoint steps, sampler steps, base-only Piper class weights, or overlay
provenance differ. It reports Elite translation MAE, Piper
accuracy/balanced accuracy, hold/feed confusion, transition/steady metrics,
and absolute/relative changes.

Verified overlay full-validation and paired comparison:

```text
records / validation episodes: 154 / 6
Elite translation MAE: 0.7732550502 -> 0.9085206985 (+17.49%)
Elite transition MAE: 0.4907198548 -> 0.6626856923 (+35.04%)
Elite steady MAE: 0.9324470758 -> 1.0214800835 (+9.55%)
Piper accuracy: 0.5649350882 -> 0.5714285970 (+1.15%)
Piper balanced accuracy: 0.5567411924 -> 0.5687669377 (+2.16%)
Piper transition accuracy: 0.5555555820 -> 0.5555555820 (0.00%)
Piper steady accuracy: 0.5394737124 -> 0.5526315570 (+2.44%)
baseline / overlay confusion: [[31,41],[26,56]] / [[38,34],[32,50]]
overlay report SHA256: af6f65cc72003d25855e8bf5faf9839a1d4c2eec986180b62b84af5c4c166780
comparison SHA256: 2c1a2af642690a4f59ddcc1b958c5111809da424445ca6468c2c9706317aa0b4
```

Decision: stop expanding or tuning the current CABD/DABC families. Retain the
fresh paired baseline. Do not run the optional in-sample role diagnostic for
this no-go result; it cannot be used for validation or model selection.

### 7. Optional Offline Role Diagnostic

Run the evaluator twice with the same checkpoints as section 5, adding:

```bash
  --offline-overlay-pack simulation_output/e2_spatial_candidate_pack_v1 \
  --offline-overlay-project-root . \
  --allow-diagnostic-overlay \
  --max-val-batches 836
```

Write the two outputs as
`pi05_e2_paired_baseline_role_diagnostic_v1.json` and
`pi05_e2_weighted_overlay_role_diagnostic_v1.json`, then rerun the comparison
with:

```bash
  --baseline-role-report simulation_output/pi05_e2_paired_baseline_role_diagnostic_v1.json \
  --overlay-role-report simulation_output/pi05_e2_weighted_overlay_role_diagnostic_v1.json
```

This reports `feed_request`, `recovery_response`, `clear_response`,
`observe_effect`, and `execution_hold` after prediction. It evaluates the E2
training overlay itself, so it is descriptive in-sample evidence only and must
not be used for validation or model selection.

## E2 Observable Event-State Adapter Smoke

This is an adapter/schema check only. It does not load PI05 weights or run
training. Run after syncing the two small tool files to the 4090:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/test_pi05_e2_overlay_smoke.py \
  simulation_output/e2_spatial_candidate_pack_v1 \
  --project-root . \
  --base-openpi-arrays simulation_output/openpi_compat_pack_frontup_step024_temporal_stride5_v1/openpi_arrays.npz \
  --out simulation_output/pi05_e2_event_state_adapter_smoke_v1.json
```

Required checks: 836 E2 records and 1672 images; 1005 base records with zeros
in event-state slice `16:27`; nonconstant busy, executed-feed, age, age-valid,
and four status dimensions; cooldown/request-accepted synthetic toggle wiring;
raw event id ignored; diagnostic targets unopened. Current E2 cooldown and
request-accepted values are both constant false and must be reported as such.

## Action-Effect World Model Pre-Training Checks

These commands validate implementation only. They do not read the new real or
simulation data and do not train a model.

### Local Static And Synthetic Smoke

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\pi05_action_effect_world_model.py `
  tools\pi05_action_effect_dataset.py `
  tools\train_pi05_action_effect_world_model.py `
  tools\test_pi05_action_effect_world_model_smoke.py `
  tools\probe_pi05_action_effect_feature_interface.py

.\.venv\Scripts\python.exe tools\test_pi05_action_effect_world_model_smoke.py `
  --work-dir simulation_output\pi05_action_effect_world_model_smoke_v1_work `
  --out simulation_output\pi05_action_effect_world_model_smoke_v1.json
```

Expected:

```text
status=passed windows=18 selected_candidate=1 trainer_dry_run=passed
```

The report must record `training_started=false`, exact sim/current-real sampler
mass `0.75/0.25`, five one-left/one-right validation folds, a finite positive
paired-degradation consistency loss, training-episode-only normalization,
feature/split fingerprints, and rejection of raw event id and privileged policy
input.

### Sync Small Algorithm Files

Run sequentially because SSH is unstable:

```powershell
scp tools\pi05_action_effect_world_model.py project4090:/home/zsw/project_2026/tools/pi05_action_effect_world_model.py
scp tools\pi05_action_effect_dataset.py project4090:/home/zsw/project_2026/tools/pi05_action_effect_dataset.py
scp tools\train_pi05_action_effect_world_model.py project4090:/home/zsw/project_2026/tools/train_pi05_action_effect_world_model.py
scp tools\test_pi05_action_effect_world_model_smoke.py project4090:/home/zsw/project_2026/tools/test_pi05_action_effect_world_model_smoke.py
scp tools\probe_pi05_action_effect_feature_interface.py project4090:/home/zsw/project_2026/tools/probe_pi05_action_effect_feature_interface.py
```

### Remote Synthetic Smoke

```bash
cd /home/zsw/project_2026
/home/zsw/miniconda3/bin/conda run -n project2026-pi \
  python tools/test_pi05_action_effect_world_model_smoke.py \
  --work-dir simulation_output/pi05_action_effect_world_model_smoke_v1_work \
  --out simulation_output/pi05_action_effect_world_model_smoke_v1.json
```

This invokes the trainer only with `--dry-run`; no optimizer step is allowed.

### Existing PI0.5 Feature-Interface Probe

This uses one record from the existing accepted stride-5 simulation baseline
to verify the real pinned PI0.5 vision path. It is not Gate A2 for the future
sim/current-real pack and performs zero optimizer steps.

```bash
cd /home/zsw/project_2026
export HF_HUB_OFFLINE=1

/home/zsw/miniconda3/bin/conda run -n project2026-pi \
  python tools/probe_pi05_action_effect_feature_interface.py \
  --root simulation_output/lerobot_project2026_frontup_step024_temporal_stride5_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-temporal-stride5-v1 \
  --out simulation_output/pi05_action_effect_feature_interface_probe_v1.json \
  --sample-index 0 \
  --max-stats-records 851 \
  --device cuda \
  --freeze-vision-encoder \
  --pretrained-name-or-path lerobot/pi05_base \
  --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 \
  --pretrained-cache-dir /home/zsw/.cache/huggingface/hub \
  --pretrained-local-files-only
```

Required report fields: two-view finite latent, explicit nine-dimensional
active action, pinned pretrained coverage, `optimizer_steps=0`,
`training_started=false`, and no exact truth or historical senior data.

### Future Accepted Feature-Pack Dry Run

Do not run until data Gate A1 passes. Replace the placeholders with paths from
the accepted handoff:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/train_pi05_action_effect_world_model.py \
  --feature-pack <accepted_action_effect_feature_pack> \
  --split-manifest <accepted_action_effect_feature_pack>/episode_split.json \
  --out simulation_output/pi05_action_effect_world_model_dry_run_v1 \
  --context-len 4 \
  --horizon 3 \
  --hidden-dim 256 \
  --batch-size 8 \
  --max-steps 1000 \
  --real-draw-fraction 0.25 \
  --seed 123 \
  --dry-run
```

Starting an actual training run requires removing `--dry-run`; that is
forbidden until both source acceptance and the real PI0.5 feature-pack smoke
pass and the user approves the controlled training command.

The dry-run `summary.json` is also the inference-normalization contract. It
must retain the training-only state/Elite statistics and both feature-pack
fingerprints; do not recompute them from validation or deployment records.

## Baseline And Public-Benchmark Architecture Gate

These checks do not open the current real records, load public benchmark data,
or take optimizer steps.

### Local Contract Smoke

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\vla_benchmark_contract.py `
  tools\test_vla_benchmark_contract_smoke.py `
  tools\probe_lerobot_baseline_architectures.py

.\.venv\Scripts\python.exe tools\test_vla_benchmark_contract_smoke.py `
  --out simulation_output\vla_benchmark_contract_smoke_v1.json
```

Expected:

```text
status=passed profiles=4 corruptions=4 project_action_round_trip=passed privileged_fields=blocked optimizer_steps=0
```

### Sync And Verify Small Files

Run the copies sequentially:

```powershell
scp tools\vla_benchmark_contract.py project4090:/home/zsw/project_2026/tools/
scp tools\test_vla_benchmark_contract_smoke.py project4090:/home/zsw/project_2026/tools/
scp tools\probe_lerobot_baseline_architectures.py project4090:/home/zsw/project_2026/tools/

ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/vla_benchmark_contract.py tools/test_vla_benchmark_contract_smoke.py tools/probe_lerobot_baseline_architectures.py"
```

### Remote Contract And Model Probe

The SmolVLA random-construction smoke requires the small `num2words` processor
dependency. It was installed on 2026-09-11 with:

```powershell
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip install num2words"
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip check"
```

Verified: `num2words==0.5.14` and `docopt==0.6.2` installed; `pip check`
returned `No broken requirements found.`

Run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/test_vla_benchmark_contract_smoke.py --out simulation_output/vla_benchmark_contract_smoke_remote_v1.json"

ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/probe_lerobot_baseline_architectures.py --out simulation_output/lerobot_baseline_architecture_probe_v5.json --lightweight-forward --device cuda"
```

Verified final output:

```text
status=passed policies=4/4 lightweight_forwards=5 failures=0 pusht_env=False libero_env=False optimizer_steps=0
```

The five lightweight executions are ACT on project/Push-T, Diffusion on
project/Push-T, and reduced random SmolVLA on the project profile. PI0.5 uses
the existing full pinned feature-interface probe instead of reconstructing a
second 3B random model. The SmolVLA processor currently emits a non-blocking
upstream deprecation warning about `preprocessor.json`.

Evidence SHA256:

```text
vla_benchmark_contract_smoke_v1.json        b541852873fd565c7d886ac7157601bfb737cdefc747820d7e642ae3d0677290
vla_benchmark_contract_smoke_remote_v1.json 2f18417bcc273e9315a1a19fda18b809bdf05613471ef65388f2fd20bd7d0a63
lerobot_baseline_architecture_probe_v5.json  07b07b941e27e8da163e2d42cdf8044a55135bbe3846810ebcfddf35d681915f
```

### Public Environment Dependency Gate

Installed and import-verified on the 4090 on 2026-09-11. A direct extras
installation first failed while building `egl-probe` / `hf-egl-probe`: CMake
4.1 removes compatibility with the packages' old policy declaration, and pip
build isolation hid the environment's Python CMake module. The reproducible
compatible sequence is:

```powershell
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip install 'cmake==3.31.6'"
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip install --no-build-isolation 'egl-probe==1.0.2' 'hf-egl-probe==1.0.2'"
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip install 'lerobot[pusht,libero]==0.4.4'"

# Preserve the OpenPI-compatible package versions and avoid an unnecessary
# OpenCV 5 import takeover from the public-environment extras.
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi python -m pip install --force-reinstall --no-deps 'transformers==4.53.2' 'tokenizers==0.21.4'"
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi python -m pip install 'opencv-python==4.12.0.88'"

# PI0.5 in this installed LeRobot path requires OpenPI's official Transformers
# replacement files. Any Transformers reinstall overwrites them, so reapply
# them after the version pin. Fail if the local OpenPI source is not the audited
# commit.
ssh project4090 'cd /home/zsw/openpi && test "$(git rev-parse HEAD)" = "15a9616a00943ada6c20a0f158e3adb39df2ccac" && test -z "$(git status --short)"'
ssh project4090 "cp -r /home/zsw/openpi/src/openpi/models_pytorch/transformers_replace/* /home/zsw/miniconda3/envs/project2026-pi/lib/python3.10/site-packages/transformers/"
ssh project4090 "/home/zsw/miniconda3/bin/conda run -n project2026-pi pip check"

ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/probe_lerobot_baseline_architectures.py --out simulation_output/lerobot_baseline_architecture_probe_v7.json --lightweight-forward --device cuda"
```

Verified package versions:

```text
lerobot==0.4.4
gym-pusht==0.1.6
pymunk==6.11.1
hf-libero==0.1.4
egl-probe==1.0.2
hf-egl-probe==1.0.2
cmake==3.31.6
transformers==4.53.2
tokenizers==0.21.4
opencv-python==4.12.0.88
```

The version pair alone is insufficient for this PI0.5 port. The installed
Transformers tree must also match the five OpenPI replacement files from commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`:

```text
models/gemma/configuration_gemma.py    26fd0d0f52730ab32b6f337eb73ae9a26cb71ae0915bc49eced1d0e2ed8d0ec6
models/gemma/modeling_gemma.py         17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde
models/paligemma/modeling_paligemma.py c945d330b829264f927f3ee5339f977238631772de07a28e855e265bb85fd53c
models/siglip/check.py                 466e8eb7887ac5ccc98118a3169298d6ab65a284d834bae74751e3854dbc4245
models/siglip/modeling_siglip.py       ef2e99500f263fdd78db9d4d9a255e29d844eb92f53c25ad640f8056700ef84b
```

Lightweight fail-closed patch audit after the integration tool is synced:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && PYTHONPATH=/home/zsw/project_2026/tools /home/zsw/miniconda3/bin/conda run -n project2026-pi python -c \"from smoke_pi05_pusht_policy_integration import _transformers_patch_evidence; x=_transformers_patch_evidence(); print(x); assert x['status']=='passed'\""
```

An intermediate reinstall without this copy produced 37 missing and 74
unexpected AdaRMS keys with unique-parameter coverage `0.999989`; that run was
correctly rejected before inference. Reapplying the audited files restored
`forward(x, cond=None)`, `dense.weight/dense.bias`, and final coverage `1.0`.

Verified results:

```text
pip check: No broken requirements found.
CUDA available: True (NVIDIA GeForce RTX 4090)
status=passed policies=4/4 lightweight_forwards=5 failures=0 pusht_env=True libero_env=True optimizer_steps=0
lerobot_baseline_architecture_probe_v7.json sha256=07452578be20d1586f1e2f1ec63c98700631fea24b6dc59a27e303be7fa37689
```

This completes dependency and import verification only. The bounded
reset/one-step runtime evidence is recorded below; it is still not a benchmark
score or authorization to start training.

### Bounded Public Environment Runtime Smoke

The smoke does not load demonstrations, policies, or checkpoints and takes no
optimizer steps. Push-T is verified with:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_lerobot_public_envs.py --envs pusht --out simulation_output/lerobot_public_env_runtime_smoke_pusht_v1.json"
```

Verified result:

```text
status=passed pusht=passed optimizer_steps=0
raw keys: pixels, agent_pos
processed keys: observation.image, observation.state, task
native action: shape=(2,), low=0, high=512
same-seed independent reset replay: state exact, image exact
report sha256=4bb1dd410557a600fe3212c22e0d4fe3ed1ec12e9d5265d40b49a2ad0ba6aeea
```

LIBERO package import passes, but the wheel excludes its runtime assets. The
official asset repository is `https://huggingface.co/datasets/lerobot/libero-assets`
and is approximately 422 MB. Under the project skill's transfer boundary, the
user transfers it as one complete directory:

```text
suggested local source:
D:\PycharmProjects\project_2026\simulation_output\libero_assets_v1\

required remote destination:
/home/zsw/project_2026/simulation_output/libero_assets_v1/
```

The remote destination must directly contain all six directories:

```text
articulated_objects/
scenes/
stable_hope_objects/
stable_scanned_objects/
textures/
turbosquid_objects/
```

After transfer, the accepted LIBERO-only run is:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_lerobot_public_envs.py --envs libero --libero-task-id 0 --out simulation_output/lerobot_public_env_runtime_smoke_libero_v3.json"
```

The accepted final paired run is:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_lerobot_public_envs.py --envs all --libero-task-id 0 --out simulation_output/lerobot_public_env_runtime_smoke_all_v1.json"
```

Verified LIBERO and combined result:

```text
libero status=passed; all 13 checks=true
raw keys: pixels.image, pixels.image2, robot_state.*
processed keys: observation.images.image, observation.images.image2, observation.state, task
processed state: shape=(1,8)
native relative action: shape=(7,), low=-1, high=1
same-seed independent reset replay: all raw leaves exact
effective assets: /home/zsw/project_2026/simulation_output/libero_assets_v1
optimizer_steps=0; dataset_loaded=false; policy_or_checkpoint_loaded=false
lerobot_public_env_runtime_smoke_libero_v3.json sha256=826a38a84ffac3798444b53acd1606d7f4e5f38024accbcf5eb1132e7a0ccde0
lerobot_public_env_runtime_smoke_all_v1.json   sha256=3beab3a6fa41875c3b80ca23db4d9e272de314cfd997153a36ab73c054f9f5aa
```

The tool creates `simulation_output/libero_runtime_config_v1/config.yaml`
non-interactively and points it only to packaged BDDL/init files and the
explicit transferred asset directory. It also pins LIBERO's process-local
asset cache to that directory and checks the effective path, because a rejected
v2 run showed that upstream `hf-libero` otherwise ignores the config asset path
and auto-populates `/home/zsw/.cache/libero/assets`. The rejected run left a
407 MB duplicate cache; do not remove it without explicit cleanup authority.
The accepted v3 and combined runs do not alter `~/.libero` or auto-download
assets/demonstrations. The pygame `pkg_resources` deprecation and missing
private robosuite macro warnings are non-blocking for the verified offscreen
path. Current tool SHA256:

```text
d3158421135046c0d7bc544b3fc0655af4733eae753f5f6c363a0148e7ccb8b4
```

### Paired Clean/Noisy Environment Episode Smoke

The runtime wrapper changes only declared raw image leaves after environment
observation and before LeRobot preprocessing. The smoke uses fixed native
actions independent of observation; it is not a policy score and loads no
dataset, policy, or checkpoint.

Local dependency-light verification:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\vla_visual_noise_wrapper.py `
  tools\test_vla_visual_noise_wrapper_smoke.py `
  tools\smoke_lerobot_public_noise_pairs.py

.\.venv\Scripts\python.exe tools\test_vla_visual_noise_wrapper_smoke.py `
  --out simulation_output\vla_visual_noise_wrapper_smoke_v2.json
```

Sync the three small files sequentially and verify their SHA256:

```powershell
scp tools\vla_visual_noise_wrapper.py project4090:/home/zsw/project_2026/tools/
scp tools\test_vla_visual_noise_wrapper_smoke.py project4090:/home/zsw/project_2026/tools/
scp tools\smoke_lerobot_public_noise_pairs.py project4090:/home/zsw/project_2026/tools/

ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/vla_visual_noise_wrapper.py tools/test_vla_visual_noise_wrapper_smoke.py tools/smoke_lerobot_public_noise_pairs.py"
```

Verified code hashes:

```text
vla_visual_noise_wrapper.py            642e4ec1f84a1eb658d433436f53ae5305eeabb581bde06cdbfa35394759f986
test_vla_visual_noise_wrapper_smoke.py  9278ad35e97227a75813b25a4026bda2f67c6c05d0d89d404c68dabb55a978a9
smoke_lerobot_public_noise_pairs.py     68291693118c23f479563db20ba75829c91fcfb5353e98b985e2d28dbc016dd2
```

Remote wrapper contract smoke:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/test_vla_visual_noise_wrapper_smoke.py --out simulation_output/vla_visual_noise_wrapper_smoke_remote_v2.json"
```

Expected:

```text
status=passed corruptions=4 wrapper_position=preprocess_input optimizer_steps=0
```

Final paired 20-transition run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_lerobot_public_noise_pairs.py --envs all --episode-length 20 --libero-task-id 0 --corruption gaussian_sensor_noise --severity 2 --artifact-dir simulation_output/lerobot_public_noise_pair_artifacts_v3 --out simulation_output/lerobot_public_noise_pair_smoke_all_v3.json"
```

Verified final result:

```text
status=passed pusht=passed libero=passed optimizer_steps=0
transitions: Push-T=20, LIBERO=20
source observations before corruption: exact
raw non-image leaves: exact
processed state/task: exact
rewards and termination flags: exact
processed images: changed on every frame
deterministic duplicate corruption: exact
native action contract: preserved
wrapper report sha256=bc5a047beff4fcc27e07ebe5b84d32a9700fd5a6e48a305e00d55ef8dfb6347c
paired report sha256=25fc8035ead08b0b3b8ef81de27bef107baaa223420472b67836037823d95399
```

Visual artifacts:

```text
simulation_output/lerobot_public_noise_pair_artifacts_v3/pusht_gaussian_sensor_noise_s2_reset_pair.png
sha256=1c8838e36ab73a99027176d115614e2e9e379bfd60712f360e2479c393c6dcf5

simulation_output/lerobot_public_noise_pair_artifacts_v3/libero_gaussian_sensor_noise_s2_reset_pair.png
sha256=3821b68e72cc80561e666de50af4509f1878b6b4a6c49e30d6c165bc18fc8095
```

Both Chinese sheets were agent-viewed without missing glyphs, clipping, or
overlap. Record `viewed_not_accepted` until the user explicitly accepts them.
The first run (`...all_v1.json`) failed closed because processed LeRobot
observations use literal dotted keys; the wrapper now supports both dotted and
nested mappings. The v2 run passed numerically but its comparison titles were
clipped, so v3 is the accepted implementation evidence.

### ACT + Push-T Policy Integration Smoke

This Gate B3 slice constructs a reduced random ACT and runs one real Push-T
observation batch plus one 20-transition episode through the official LeRobot
environment and policy processors. It loads no dataset, checkpoint, or
pretrained weights and takes no backward or optimizer step.

Local syntax and smoke-stat contract check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_act_pusht_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_act_pusht_policy_integration import _smoke_normalization_stats; s=_smoke_normalization_stats(); assert s['action']['mean'].tolist()==[256.0,256.0]; assert s['action']['std'].tolist()==[128.0,128.0]; print('status=passed stats_source=bounds smoke_only=true')"
Remove-Item Env:PYTHONPATH
```

Sync and verify:

```powershell
scp tools\smoke_act_pusht_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/smoke_act_pusht_policy_integration.py"
```

Verified code SHA256:

```text
78a446dfe92af1a9085b061ac8cf7667612a9d98fdfa608ed06475fc87278831
```

Remote run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_act_pusht_policy_integration.py --episode-length 20 --device cuda --artifact simulation_output/act_pusht_policy_integration_v1/rollout_sheet.png --out simulation_output/act_pusht_policy_integration_v1/report.json"
```

Verified result:

```text
status=passed policy=act env=pusht transitions=20 random_weights=true optimizer_steps=0
parameter_count=11286402
first image batch=[1,3,96,96]
first state batch=[1,2]
normalized/native action=[1,2]
all actions finite and inside [0,512]^2
observed action range=165.30738830566406..345.0386047363281
normalization stats source=push_t_native_bounds_smoke_only_not_dataset_statistics
report sha256=03b1dcd4a6594fa3620035657d1a8ec3295c4dc11e96c19aa5e2cd4bca90a71e
rollout sheet sha256=93f470c84a111eb763f7c1ba8bb5c50e757673e2a543f18ac62e45a7a3671a93
```

The rollout sheet was agent-viewed without missing glyphs, clipping, or
overlap. Its status remains `viewed_not_accepted` until explicit user
acceptance. The policy is random and the stats are bounds-derived, so do not
interpret reward, success, behavior, or action distribution as a baseline
result or reuse these statistics for training.

### Diffusion Policy + Push-T Integration Smoke

This Gate B3 slice constructs a reduced random Diffusion Policy and checks its
two-observation history queue and four-action execution queue in addition to the
same real first-batch and 20-transition Push-T loop. It loads no dataset,
checkpoint, or pretrained weights and takes no backward or optimizer step.

Local syntax and MIN_MAX boundary-stat check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_diffusion_pusht_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_diffusion_pusht_policy_integration import _smoke_normalization_stats; s=_smoke_normalization_stats(); assert s['action']['min'].tolist()==[0.0,0.0]; assert s['action']['max'].tolist()==[512.0,512.0]; print('status=passed normalization=min_max stats_source=bounds')"
Remove-Item Env:PYTHONPATH
```

Sync and verify:

```powershell
scp tools\smoke_diffusion_pusht_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/smoke_diffusion_pusht_policy_integration.py"
```

Verified code SHA256:

```text
6b21d9d0712f094f538c4a2e4ec8cb8571ce69d8b4fc6219842b047aa69fe7e8
```

Remote run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_diffusion_pusht_policy_integration.py --episode-length 20 --device cuda --artifact simulation_output/diffusion_pusht_policy_integration_v1/rollout_sheet.png --out simulation_output/diffusion_pusht_policy_integration_v1/report.json"
```

Verified result:

```text
status=passed policy=diffusion env=pusht transitions=20 random_weights=true optimizer_steps=0
parameter_count=12357890
first image batch=[1,3,96,96]
first state batch=[1,2]
observation history after first action: state=2, image=2
remaining action queue after first pop=3
all actions finite and inside [0,512]^2
observed action range=0.0..512.0
normalization stats source=push_t_native_bounds_smoke_only_not_dataset_statistics
report sha256=75d75954dc287d3ad29c440649ed8c2cfc705067943ea6bf43981f21c209fb14
rollout sheet sha256=b4a51e8e4475c90fa96b4e3964662c505f39cf03439a11690d293c084d449ad2
```

The random model reaches both native bounds through Diffusion's configured
sample clipping and MIN_MAX unnormalization. This is safe interface evidence,
not a useful action distribution or policy-quality result. The sheet was
agent-viewed without text or layout defects and remains
`viewed_not_accepted`; target crosses cut by the environment frame boundary
are evidence of the saturated action, not report-layout clipping.

### Reduced SmolVLA + Push-T Integration Smoke

This Gate B3 slice constructs a reduced random SmolVLA, injects the Push-T
profile's allowed constant instruction, and runs one real batch plus a
20-transition episode. It is forced offline and fails closed unless the small
cached SmolVLM config/tokenizer/processor files are present and no cached model
weight file is found. It does not load a dataset, checkpoint, or pretrained
weights and takes no backward or optimizer step.

Local syntax and MEAN_STD smoke-stat check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_smolvla_pusht_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_smolvla_pusht_policy_integration import _smoke_normalization_stats; s=_smoke_normalization_stats(); assert s['action']['mean'].tolist()==[256.0,256.0]; assert s['action']['std'].tolist()==[64.0,64.0]; print('status=passed normalization=mean_std stats_source=bounds')"
Remove-Item Env:PYTHONPATH
```

Sync and verify:

```powershell
scp tools\smoke_smolvla_pusht_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/smoke_smolvla_pusht_policy_integration.py"
```

Verified code SHA256:

```text
517ae343ff52d75547be80fad0d5049ddc280dd00bf1baa4a21f03e9f30897c1
```

Remote offline run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_smolvla_pusht_policy_integration.py --episode-length 20 --device cuda --artifact simulation_output/smolvla_pusht_policy_integration_v1/rollout_sheet.png --out simulation_output/smolvla_pusht_policy_integration_v1/report.json"
```

Verified result:

```text
status=passed policy=smolvla env=pusht transitions=20 random_weights=true offline=true optimizer_steps=0
runtime_seconds=8.704
parameter_count=204069056
first image batch=[1,3,96,96]
first state batch=[1,2]
constant task="Push the T-shaped block to the target."
language tokens=[1,11], valid tokens=11
remaining action queue after first pop=1
all actions finite and inside [0,512]^2
observed action range=98.57557678222656..334.5440368652344
cached model-weight files before/after=0/0
normalization stats source=push_t_native_bounds_smoke_only_not_dataset_statistics
report sha256=edaa3910c2d961ee49134c0086d7f0b795ad50ebefc7b5446f9f930188d18deb
rollout sheet sha256=8e0830c77df10606518daab627ec65dcd6207d2f054dd795a068c7d6dd9485dd
```

The policy uses one VLM layer, one expert layer, a width multiplier of 0.25,
two flow steps, and `load_vlm_weights=false`. The only Hugging Face resources
used are the existing approximately 4.7 MB config/tokenizer/processor cache.
The upstream video-preprocessor filename deprecation warning is non-blocking.
The sheet was agent-viewed without missing glyphs, clipping, or overlap and
remains `viewed_not_accepted`. These random-weight actions and bounds-derived
statistics are interface evidence only, not a baseline score or training
configuration.

### Pinned PI0.5 + Push-T Integration Smoke

This Gate B3 slice loads the full pinned PI0.5 base checkpoint from the local
4090 cache, while reducing only the action chunk and flow-inference step counts
to two for a bounded interface run. It uses the Push-T profile's constant task,
checks that the official PI0.5 processor encodes all 32 padded state bins into
the prompt, and executes 20 environment transitions. It remains fully offline
and takes no backward or optimizer step.

Local syntax, QUANTILES smoke values, and prompt parser check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_pi05_pusht_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_pi05_pusht_policy_integration import _smoke_normalization_stats,_prompt_state_bin_count; s=_smoke_normalization_stats(); assert s['observation.state']['q01'].tolist()==[0.0,0.0]; assert s['observation.state']['q99'].tolist()==[512.0,512.0]; assert s['action']['q01'].tolist()==[192.0,192.0]; assert s['action']['q99'].tolist()==[320.0,320.0]; assert _prompt_state_bin_count('Task: x, State: '+' '.join(['127']*32)+';\nAction: ')==32; print('status=passed quantile_smoke_contract=true prompt_bins=32')"
Remove-Item Env:PYTHONPATH
```

Sync and verify:

```powershell
scp tools\smoke_pi05_pusht_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/smoke_pi05_pusht_policy_integration.py"
```

Verified code SHA256:

```text
899e4e9e6ce73bff16b4dc90b0407d09f205631a29cf31f9b64bff12d9c242be
```

Remote offline run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_pi05_pusht_policy_integration.py --episode-length 20 --device cuda --pretrained-name-or-path lerobot/pi05_base --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 --pretrained-cache-dir /home/zsw/.cache/huggingface/hub --artifact simulation_output/pi05_pusht_policy_integration_v1/rollout_sheet.png --out simulation_output/pi05_pusht_policy_integration_v1/report.json"
```

Verified result:

```text
status=passed policy=pi05 env=pusht transitions=20 loaded_fraction=1.0 offline=true optimizer_steps=0
runtime_seconds=32.533
parameter_count=3616757520
resolved revision=7de663972b7817d2c4cf2d84c821153dfea772e9
checkpoint size=14467165872 bytes
missing / unexpected / shape-mismatch keys=0 / 0 / 0
verified tied-weight aliases=1
first image batch=[1,3,96,96]
first state batch=[1,2]
constant task="Push the T-shaped block to the target."
language tokens=[1,200], valid tokens=147
state bins encoded into prompt=32
remaining action queue after first pop=1
all actions finite and inside [0,512]^2
observed action range=234.06015014648438..270.4274597167969
normalization stats source=push_t_native_bounds_smoke_only_not_dataset_quantile_estimates
OpenPI Transformers patch audit=passed
report sha256=a9744fcd0504daf3d1d3087d542798d81fa6243da786d68f5479f3155a7b9d13
rollout sheet sha256=d84c662952e8e3b3f3e549ae601b03a599556aaa6d43867536e9a0a083f43bfb
```

The state q01/q99 values are the native Push-T bounds. The action q01/q99
values `[192,192]/[320,320]` are deliberately conservative, bounds-derived
smoke values that leave headroom for normalized pretrained outputs; they are
not empirical quantiles and must not be reused for training or scoring. The
two vision-embedding warnings occur during known key normalization, but the
final strict audit has full unique-parameter coverage and zero effective
missing, unexpected, or shape-mismatched keys. The Chinese four-frame sheet
was agent-viewed without missing glyphs, clipping, or overlap and remains
`viewed_not_accepted`. Do not interpret the action trace, reward, or resulting
motion as PI0.5 benchmark performance.

### Reduced SmolVLA + LIBERO-Spatial Task-0 Integration Smoke

This language-conditioned Gate B3 slice preserves LIBERO's native task text,
two image views, eight-dimensional state, and native relative `[-1,1]^7`
action contract. The policy is reduced and randomly initialized. It uses no
demonstrations, checkpoint, backward pass, optimizer, or training.

Local syntax and smoke-stat check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_smolvla_libero_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_smolvla_libero_policy_integration import _smoke_normalization_stats; s=_smoke_normalization_stats(); assert s['observation.state']['mean'].tolist()==[0.0]*8; assert s['action']['mean'].tolist()==[0.0]*7; assert all(abs(x-0.2)<1e-6 for x in s['action']['std'].tolist()); print('status=passed normalization=mean_std source=semantic_and_bounds_smoke_only')"
Remove-Item Env:PYTHONPATH
```

Sync and verify the small algorithm file:

```powershell
scp tools\smoke_smolvla_libero_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "sha256sum /home/zsw/project_2026/tools/smoke_smolvla_libero_policy_integration.py"
```

Verified code SHA256:

```text
08d474af8acd0029a8c892e4bd42ca5a0f88271355b10cf364e67fbcf925745b
```

Remote offline run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_smolvla_libero_policy_integration.py --out simulation_output/smolvla_libero_policy_integration_v1/report.json --artifact simulation_output/smolvla_libero_policy_integration_v1/rollout_sheet.png --episode-length 20 --libero-task-id 0 --libero-config-dir simulation_output/libero_runtime_config_v1 --libero-assets-dir simulation_output/libero_assets_v1"
```

Verified result:

```text
status=passed policy=smolvla env=libero_spatial task=0 transitions=20 random_weights=true offline=true optimizer_steps=0
runtime_seconds=11.314
parameter_count=204069056
native task="pick up the black bowl between the plate and the ramekin and place it on the plate"
environment camera batches=[1,3,256,256] and [1,3,256,256]
state batch=[1,8]
language tokens=[1,20], valid tokens=20
normalized/native action=[1,7]
remaining action queue after first pop=1
all actions finite and inside [-1,1]^7
observed action range=-0.5699618458747864..0.5662946701049805
cached model-weight files before/after=0/0
normalization stats source=libero_state_semantic_units_and_native_action_bounds_smoke_only_not_dataset_statistics
report sha256=c7ae77e7619eeb6bef773184f47cc0d3d4265dbb31d2632b0877f2ec6f4d38fa
rollout sheet sha256=d2cf565f7a61bde4c64064a61fe5992ae2d3da0f9fce1ab5df4764a41c378bd1
```

The environment processor emits native 256x256 tensors; the benchmark feature
declaration is 224x224 and SmolVLA performs its native padded 512x512 visual
resize internally. The report records these distinct layers. The upstream
video-preprocessor filename warning is non-blocking. The Chinese dual-camera
sheet was agent-viewed without missing glyphs, clipping, or overlap and remains
`viewed_not_accepted`. Do not interpret reward, success, motion, or the random
action distribution as a baseline result.

### Pinned PI0.5 + LIBERO-Spatial Task-0 Integration Smoke

This matching language-conditioned Gate B3 slice loads the full pinned PI0.5
base checkpoint fail-closed from the existing cache. Only chunk length,
executed action count, and flow steps are reduced to bound runtime. It uses no
demonstrations, backward pass, optimizer, or training.

Local syntax and smoke-stat check:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\smoke_pi05_libero_policy_integration.py

$env:PYTHONPATH=(Resolve-Path tools).Path
.\.venv\Scripts\python.exe -c "from smoke_pi05_libero_policy_integration import _smoke_normalization_stats; s=_smoke_normalization_stats(); assert all(abs(x+0.2)<1e-6 for x in s['action']['q01'].tolist()); assert all(abs(x-0.2)<1e-6 for x in s['action']['q99'].tolist()); print('status=passed normalization=quantiles source=semantic_and_bounds_smoke_only')"
Remove-Item Env:PYTHONPATH
```

Sync and verify the small algorithm file:

```powershell
scp tools\smoke_pi05_libero_policy_integration.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "sha256sum /home/zsw/project_2026/tools/smoke_pi05_libero_policy_integration.py"
```

Verified code SHA256:

```text
9c8a6e54415c522194b321b0785c44faf741feb464e81c6c95e9a8304ed8f2d5
```

Remote offline run:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/smoke_pi05_libero_policy_integration.py --out simulation_output/pi05_libero_policy_integration_v1/report.json --artifact simulation_output/pi05_libero_policy_integration_v1/rollout_sheet.png --episode-length 20 --libero-task-id 0 --libero-config-dir simulation_output/libero_runtime_config_v1 --libero-assets-dir simulation_output/libero_assets_v1 --pretrained-name-or-path lerobot/pi05_base --pretrained-revision 7de663972b7817d2c4cf2d84c821153dfea772e9 --pretrained-cache-dir /home/zsw/.cache/huggingface/hub"
```

Verified result:

```text
status=passed policy=pi05 env=libero_spatial task=0 transitions=20 loaded_fraction=1.0 offline=true optimizer_steps=0
runtime_seconds=31.152
parameter_count=3616757520
resolved revision=7de663972b7817d2c4cf2d84c821153dfea772e9
checkpoint size=14467165872 bytes
missing / unexpected / shape-mismatch keys=0 / 0 / 0
verified tied-weight aliases=1
native task="pick up the black bowl between the plate and the ramekin and place it on the plate"
environment camera batches=[1,3,256,256] and [1,3,256,256]
PI0.5 image resolution=[224,224]
state batch=[1,8]
language tokens=[1,200], valid tokens=155
state bins encoded into prompt=32
normalized/native action=[1,7]
remaining action queue after first pop=1
all actions finite and inside [-1,1]^7
observed action range=-0.05868533253669739..0.20431967079639435
normalization stats source=libero_state_semantic_units_and_native_action_bounds_smoke_only_not_dataset_quantile_estimates
OpenPI Transformers patch audit=passed
report sha256=24f10123b918e0b7526996493902e53b48745417a66d2824831bedb1e49e2dfc
rollout sheet sha256=bb0cf933a1e9c7e2223f422b466f62a974d9c3a1e99cc3684237855c2a152a48
```

The two vision-embedding mapping warnings are the same known warnings as the
Push-T run. They are non-blocking because the final strict load has full unique
parameter coverage and zero missing, unexpected, or shape-mismatched keys.
The Chinese dual-camera sheet was agent-viewed without missing glyphs,
clipping, or overlap and remains `viewed_not_accepted`. Base PI0.5 is not a
LIBERO task-finetuned checkpoint, so reward, success, behavior, robustness, and
cross-model ranking must not be reported as benchmark performance.

## Frozen LIBERO Score Protocol Audit

This stage freezes the score-bearing task/checkpoint/data route before any
large resource acquisition. It reads only small Hub JSON/config metadata and
installed runtime metadata. It does not load a model or dataset, construct an
environment, run an episode, train, or report a score.

Local syntax, JSON, and Hub-metadata audit:

```powershell
cd D:\PycharmProjects\project_2026

.\.venv\Scripts\python.exe -m py_compile `
  tools\validate_libero_score_protocol.py

.\.venv\Scripts\python.exe -c "import json; json.load(open('docs/libero-spatial-score-protocol-v1.json', encoding='utf-8')); print('status=passed json=docs/libero-spatial-score-protocol-v1.json')"

.\.venv\Scripts\python.exe tools\validate_libero_score_protocol.py `
  --manifest docs\libero-spatial-score-protocol-v1.json `
  --out simulation_output\libero_score_protocol_audit_local_v1.json `
  --check-hub
```

Sync the small files sequentially and verify them:

```powershell
scp docs\libero-spatial-score-protocol-v1.json project4090:/home/zsw/project_2026/docs/
scp tools\validate_libero_score_protocol.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && sha256sum docs/libero-spatial-score-protocol-v1.json tools/validate_libero_score_protocol.py"
```

Remote runtime audit:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/validate_libero_score_protocol.py --manifest docs/libero-spatial-score-protocol-v1.json --out simulation_output/libero_score_protocol_audit_remote_v1.json --check-hub --check-runtime"
```

Verified result:

```text
local status=passed checks=52 false_checks=0 hub=true runtime=false
remote status=passed checks=60 false_checks=0 hub=true runtime=true
remote package versions: lerobot=0.4.4, hf-libero=0.1.4,
  transformers=4.53.2, tokenizers=0.21.4
remote suite/task/horizon: 10 tasks, task-0 init states=50, max steps=280
large_blob_downloaded=false
checkpoint_loaded=false
dataset_loaded=false
environment_constructed=false
episode_executed=false
optimizer_steps=0
score_claim_allowed=false

docs/libero-spatial-score-protocol-v1.json
  sha256=b72b854d04b05930c2c1b35f384938a875e144002963f08771a2e6fc649dbdd1
tools/validate_libero_score_protocol.py
  sha256=d6b0c3ac861003b3103449b38b649d8fc3349c61754c43161c2d0b9bf8757dfb
simulation_output/libero_score_protocol_audit_local_v1.json
  sha256=cd61f61715a2127d72038a1294a08c0e4b67df50af871668f15a85f541ab1e39
simulation_output/libero_score_protocol_audit_remote_v1.json
  sha256=ce0d5696dc0a8ca4d4795754193ae3e1d40a3f95946e50581158a83c131e8497
```

The first repeat of the remote audit encountered a transient Hugging Face TLS
EOF while reading SmolVLA metadata. A retry with the same immutable manifest
passed all 60 checks. This was a metadata-network failure, not a runtime or
protocol failure.

## PI0.5 LIBERO Task-Finetuned Checkpoint Verification

The user completed the greater-than-100-MB download. The primary route needs
only this task-finetuned PI0.5 checkpoint for evaluation; it does not need the
LIBERO demonstration dataset. The completed acquisition and hash command was:

```powershell
ssh project4090 "mkdir -p /home/zsw/models/project_2026/pi05_libero_finetuned_8e174154 && /home/zsw/miniconda3/bin/conda run -n project2026-pi hf download lerobot/pi05_libero_finetuned --revision 8e174154ef5f6c60a8da12ae99c303d8963138c1 --local-dir /home/zsw/models/project_2026/pi05_libero_finetuned_8e174154"

ssh project4090 "sha256sum /home/zsw/models/project_2026/pi05_libero_finetuned_8e174154/model.safetensors"
```

Expected model SHA256:

```text
877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae
```

The downloaded model is `7,473,096,344` bytes and resolves through
`/home/zsw/models/project_2026` to the SSD-backed physical directory. Run the
fail-closed schema, processor, and effective-parameter coverage audit with:

```powershell
scp tools\audit_pi05_libero_finetuned_checkpoint.py project4090:/home/zsw/project_2026/tools/

ssh project4090 "cd /home/zsw/project_2026 && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/audit_pi05_libero_finetuned_checkpoint.py --checkpoint /home/zsw/models/project_2026/pi05_libero_finetuned_8e174154 --protocol docs/libero-spatial-score-protocol-v1.json --official-smoke-info simulation_output/pi05_libero_finetuned_loader_smoke_v1/eval_info.json --out simulation_output/pi05_libero_finetuned_checkpoint_audit_v1.json --device cuda"
```

Verified result:

```text
status=passed checks=26 false_checks=0 optimizer_steps=0 gate_b4a_executed=false
checkpoint keys / model keys / direct matches: 812 / 813 / 812
raw missing / effective missing / unexpected / shape mismatch: 1 / 0 / 0 / 0
unique-parameter coverage: 1.0
verified tied aliases: 1
tied source matches checkpoint: true
tied target matches checkpoint: true
tied source and target share storage after load: true
model SHA256: 877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae
audit report SHA256: 577a756c10d665f8e1159f00a2a212c673454aed58c69f3f0836d9e509d0eaef
```

The single raw missing key is the runtime alias
`model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight`;
the checkpoint serializes its tied `lm_head.weight`. The audit proves both
runtime tensors match the checkpoint and share storage. The vision
patch-embedding mapping messages are likewise non-blocking because all
serialized keys match and effective parameter coverage is complete.

The referenced official integration smoke used LIBERO-Spatial task 0 for only
one episode and one environment step, with `n_action_steps=1`, two inference
steps, no optimizer, and no training. It produced the following evidence:

```text
simulation_output/pi05_libero_finetuned_loader_smoke_v1/eval_info.json
  sha256=22a8d06116134bc0ae3ed45aeb9f2f34445eb6c5d2e8f4df73dfa9725a52648e
simulation_output/pi05_libero_finetuned_loader_smoke_v1/videos/libero_spatial_0/eval_episode_0.mp4
  sha256=2313896583f88e13d0acb2feeb68afb94ec4fbcd8c109e20d75c1e42b2dbd823
```

Do not report its reward or success as a baseline. The audit records
`gate_b4a_executed=false`; at that checkpoint-verification stage, no
score-bearing evaluation had run. The separately authorized B4a result follows.

The user explicitly authorized Gate B4a after the checkpoint hash and loader
audit passed. The completed command was:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && LIBERO_CONFIG_PATH=/home/zsw/project_2026/simulation_output/libero_runtime_config_v1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /home/zsw/miniconda3/bin/conda run -n project2026-pi lerobot-eval --policy.path=/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154 --policy.n_action_steps=10 --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' --env.episode_length=280 --env.control_mode=relative --env.init_states=true --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=10 --seed=1000 --output_dir=simulation_output/pi05_libero_task0_b4a_v1"
```

Verified score output:

```text
task group / id: libero_spatial / 0
episodes / successes: 10 / 10
task-0 success rate: 100.0%
average sum reward / max reward: 1.0 / 1.0
evaluation seconds / seconds per episode: 325.934 / 32.593
eval_info SHA256: e6a1e2f015f763f88421d0f1626fd6299f08f13783c14101f6a471b7d088385d
```

Audit the score, schedule, checkpoint binding, videos, installed LeRobot seed
and init-state semantics, and effective asset payload with:

```powershell
scp tools\audit_pi05_libero_b4a_result.py project4090:/home/zsw/project_2026/tools/
scp simulation_output\pi05_libero_task0_b4a_v1\visual_audit\b4a_final_frames_sheet_zh.png project4090:/home/zsw/project_2026/simulation_output/pi05_libero_task0_b4a_v1/visual_audit/

ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/audit_pi05_libero_b4a_result.py --out simulation_output/pi05_libero_task0_b4a_audit_v1.json"
```

Verified audit:

```text
status=passed checks=27 false_checks=0 task0_success_rate=100.0 full_suite=false
episode seeds: 1000..1009 (source-inferred in this historical audit)
init-state indices: historical audit assumed 0..9; NOT verified, see correction below
videos: 10 nonempty, 10 unique SHA256 values
checkpoint unique-parameter coverage: 1.0
effective asset path: /home/zsw/.cache/libero/assets
configured/effective asset payloads: 586/586 files, byte-identical
visual status: viewed_not_accepted
visual sheet SHA256: a5b6c80fad4bb69ee4497ff16c3e479ccc38a0983fe6c927e9c4b93b165de73c
audit SHA256: d8d2520d3b6146e08b36cae439d2237bc807b9aed6482466483aa17eb58db158
```

The historical audit JSON is retained unchanged. On 2026-09-12 the user
accepted this exact final-frame sheet ("认可"), so its current visual status is
`accepted`, scoped to SHA256
`a5b6c80fad4bb69ee4497ff16c3e479ccc38a0983fe6c927e9c4b93b165de73c`.

The effective asset path differs because upstream `lerobot-eval` resolves its
existing LIBERO cache even when `LIBERO_CONFIG_PATH` names the project config.
The audit excludes only Hugging Face download metadata and verifies every
runtime payload path, size, and hash against the configured project asset copy.

This task-0 result is only a score-pipeline preflight and must not be presented
as a ten-task LIBERO-Spatial aggregate. Preserve B4a's ten observed successes;
its schedule-audit claim is qualified below. Do not rerun it for model
selection. Do not run the paired noise matrix until a clean
evaluation scope has been selected, and do not use noisy results for training
or checkpoint selection.

### B4b User-Run Command And Recovery

The user assigned the hour-long run to themselves on 2026-09-12. The following
entrypoint supersedes the previous direct `lerobot-eval` command. It preserves
the frozen evaluation parameters and explicitly selects each initial state:
upstream `LiberoEnv.step()` performs an extra unseeded reset on termination,
which advances its counter before the next rollout's seeded reset. Thus the
old B4a audit's constructed `0..9` list was not runtime evidence of that schedule.
Its observed `10/10` score remains, but its actual initialization indices were
not traced. B4b now records the applied state payloads and can establish them.

The small runner has been synced to the remote host. Its `--check` preflight
hashes checkpoint resources, verifies pinned package versions, creates one
real task-0 environment, and checks resets without constructing a policy.
Observed `set_init_state` indices were `0,1,1,2,1,2`: explicit starts `0,1,1`
with intervening unseeded resets `1,2,2`. Invalid seeds were rejected.
The correction passed; this is not a full policy/evaluator run.

Reproducible short check (already executed):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_b4b.py --check"
```

Final verified file hashes (local/remote runner match):

```text
tools/run_pi05_libero_b4b.py
  sha256=02473bf81232b037ac0aef88d952966e7a45bd62c19643dcf2e0a4de30e64783
simulation_output/pi05_libero_spatial_b4b_preflight_v1.json
  sha256=b0fe2a4bd7f529d408b761c7af9f278c8698038ca3f04ae7d183b34d76964695
policy_loaded=false; b4b_started=false; optimizer_steps=0
```

Start the full run from local PowerShell. It returns immediately and keeps
running in tmux on the remote host after SSH disconnects:

```powershell
ssh project4090 "tmux new-session -d -s pi05-b4b '/home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u /home/zsw/project_2026/tools/run_pi05_libero_b4b.py --run'"
```

This runs `10` episodes per task (`100` total), reusing seeds `1000..1009` and
explicit init-state indices `0..9` for each task, with relative control,
`n_action_steps=10`, a `280`-step horizon and one synchronous environment.
The initial estimate is about one hour; task difficulty and compilation can
change runtime. Models and processors retain their checkpoint settings.

Inspect status or follow the log (Ctrl+C stops log following, not evaluation):

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_spatial_b4b_v1/status.json"
ssh project4090 "tail -n 30 -f /home/zsw/project_2026/simulation_output/pi05_libero_spatial_b4b_v1/run.log"
```

Outputs under `simulation_output/pi05_libero_spatial_b4b_v1/`:

```text
run.log                        stdout/stderr, including upstream warnings
run_manifest.json              exact argv, resource and runtime source hashes
effective_env_config.json      actual parsed environment configuration
effective_policy_config.json   actual loaded policy configuration
episode_trace.jsonl            applied init-state hashes, resets, episode results
eval_info.json                 upstream per-task and aggregate result
videos/                        upstream evaluation videos
status.json                    running / failed / completed, exit code
```

Completion requires `status=completed`, `exit_code=0`, `episodes=100` and
`schedule_verified=true`. Success percentages are allowed to be below 100;
audit validity does not depend on every episode succeeding. After the run,
inspect the result/trace arithmetic, action ranges, video completeness and
representative failure videos before publishing the result. Use the wording
"LIBERO-Spatial, 10 tasks x 10 episodes, our frozen protocol"; published-score
equivalence has not been established.

Recovery: an SSH disconnect alone does not require a restart. Check the status
file and `ssh project4090 "tmux list-sessions"` first. `status=failed`, or a
stale running status with no live process, means the attempt needs inspection.
Existing output is never overwritten. Once the old process is confirmed
stopped and the failure is understood, a full retry can use a fresh directory:

```powershell
ssh project4090 "tmux new-session -d -s pi05-b4b-retry1 '/home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u /home/zsw/project_2026/tools/run_pi05_libero_b4b.py --run --attempt retry1'"
```

This writes `simulation_output/pi05_libero_spatial_b4b_retry1/` and reruns all
100 episodes. There is no episode-level resume; never merge partial attempts
or choose a score based on which retry performed best.

### Native LIBERO World-Model Adapter / CPU Preparation (2026-09-12)

Status: implementation and local/remote synthetic CPU tests complete. No
demonstrations, checkpoint load, GPU feature extraction, world-model training,
or guidewire changes. B4b's independent review is now complete with a raw-action
bound qualification; see B4b Offline Result Audit below.
Do not rerun B4b while checking this separate preparation stack.

Files (all small; already sequentially synced and SHA256 matched):

```text
tools/pi05_libero_world_model_adapter.py
  924cfe56e8b1d6653b86073d08988f09491d6c02b40439965315d492de37fab0
tools/pi05_libero_feature_extraction.py
  41fd2201f824ed2b1e41b85d00edc8e3efd90155e864ee32e1ad1257768db6e0
tools/prepare_pi05_libero_world_model.py
  6d4af25bc674cc4af665c62cdc9d0e26f987d3bdb50ea76b5f7365f306549814
tools/test_pi05_libero_world_model_adapter.py
  852d9db050c47a6363c6caab5cf6f024368fae12d1accc70fc8ca6f97b721777
tools/test_pi05_libero_feature_extraction.py
  69bc104bce4f3752dfc0fef0a4e5a22daf055ef0af371c4c2525461168ca244e
```

The extraction entrypoint reuses only the existing frozen image-pooling helper
from `tools/pi05_action_effect_world_model.py`, not its guidewire model or
loss. That unchanged helper file's local/remote SHA256 is
`8678d90331eeb6ee64522e63d1f38fce2ff3129367de448cb6cc82afa6ea6617`.

Describe the interface without loading a dataset or policy:

```powershell
.\.venv\Scripts\python.exe tools/prepare_pi05_libero_world_model.py --describe-contract
.\.venv\Scripts\python.exe tools/pi05_libero_feature_extraction.py --describe
```

Repeat the complete bounded smoke using a new output name. The already
executed runs used suffix `_v1`; these examples use `_v2` to preserve them:

```powershell
.\.venv\Scripts\python.exe tools/test_pi05_libero_world_model_adapter.py --out-root simulation_output/pi05_libero_native_adapter_smoke_local_v2
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/test_pi05_libero_world_model_adapter.py --out-root simulation_output/pi05_libero_native_adapter_smoke_remote_v2"
```

Expected: 45 tests, 180 synthetic rows, 20 episodes, 10 fixture task mappings,
30 train + 30 validation windows, `optimizer_steps=0`, `training_ready=false`.
The test bodies took about 1.25 s locally and 0.12 s remotely, excluding Python
imports/SSH. Both runs use synthetic latents and a tiny CPU fake vision model;
they do not establish real PI0.5 feature parity or a trained predictor.

Outputs inside each smoke root:

```text
report.json
synthetic_fixture/manifest.json
synthetic_fixture/split.json
synthetic_fixture/{episode_index,frame_index,task_id,state,state_valid,action,visual_latent,visual_valid}.npy
preparation/normalization.json
preparation/train_windows.jsonl
preparation/validation_windows.jsonl
preparation/report.json
```

Replay the preparation entrypoint on the existing local fixture (no optimizer).
The `_v1` recheck passed; this example uses a fresh `_v2` output:

```powershell
.\.venv\Scripts\python.exe tools/prepare_pi05_libero_world_model.py --feature-pack simulation_output/pi05_libero_native_adapter_smoke_local_v1/synthetic_fixture --split simulation_output/pi05_libero_native_adapter_smoke_local_v1/synthetic_fixture/split.json --allow-synthetic-fixture --out simulation_output/pi05_libero_native_preparation_recheck_v2
```

The source fixture is an executable schema example, not public training data.
A real pack must omit synthetic opt-in and declare the pinned public-demo
source, actual source metadata hash, checkpoint/preprocessing/extractor hashes,
native fps, exact state/action/timing layout, explicit source-task/instruction
to Spatial mapping, complete episode registry and source/leakage-group IDs.
Every `.npy` is hash-bound, with float32 state `[N,8]`, action `[N,7]`, latent
`[N,2,D]`, boolean state/view masks and integer episode/frame/task vectors.
The split is separately bound to `manifest.json` SHA256. Do not copy the
fixture's all-zero provenance hashes, invented instructions or latent size
into a real feature pack. The checker validates declarations and hashes; it
does not independently certify upstream source identity/completeness.

`adapt_native_record` takes a deliberately selected CPU record: integer
episode/frame/source-task IDs, exact native `task`, native 7D `action`, optional
8D `observation.state` and boolean `observation.state_valid`. Do not forward a
raw simulator dictionary or LeRobot reward/success/extra fields. Images have a
separate batch boundary `extract_native_visual_features(policy, image_batch)`:
two BCHW floating [0,1] views, already-loaded fully eval-mode policy, native
preprocessing before embedding, and optional verified empty camera excluded.
It returns `[B,2,D]`; no task/state tokens or future targets are pooled into
the visual latent. It is not yet a full demonstration-download/export CLI.

Model-ready window sections are separate: `inputs` has history latent/state
and validity, native task text, normalized candidate actions `[1,3,7]`;
`targets` has future two-view latent and masked 8D coordinate residual;
episode/source IDs remain `metadata`. Default history spans 0.3 s and target
horizon 0.3 s at the fixture's 10 Hz; actual source cadence must be audited.
Native axis-angle residuals are coordinate differences, not SO(3) rotations.
Training-only normalization is for the future world model, not PI0.5 policy
processors. Do not invoke `train_pi05_action_effect_world_model.py` on this pack:
that unchanged trainer/model requires the project-specific 9D interface.

Recovery: existing output directories are never overwritten. On a schema/test
failure inspect the error and fix the source/adapter, then use a fresh output
name; do not change declarations merely to bypass a gate. Do not start a large
download, real feature extraction or optimizer step with these smoke commands.
Anything over 100 MB is transferred by the user under the project skill.

### B4b Offline Result Audit (2026-09-12)

Completed locally and on remote CPU, with no policy or environment load:
39 artifact/schedule/checkpoint/video checks passed and ten negative trace
tests passed. Observed96/100 is confirmed, but `strict_frozen_protocol_passed`
is **false** because all 100 episodes include some raw action beyond [-1,1].
The audit's exit0 means auditing completed, not that strict protocol compliance
passed; inspect the report's separate check groups and `status`.

New small files were sequentially synced and hash-matched before remote use:

```text
tools/audit_pi05_libero_b4b_result.py
  eb217b435bc7f05cf12a2756de84bd126d133e118c409d47865ce4960f573f66
tools/test_pi05_libero_b4b_audit.py
  cf30ea09445418c0ceb1fb51534397e05174ffad101b51fa0cf1e246a1bfe8a2
```

Repeat only the offline audit, not `run_pi05_libero_b4b.py --run`. The executed
outputs are local/remote `_v2`; use a new `_v3` name to preserve them:

```powershell
.\.venv\Scripts\python.exe tools/test_pi05_libero_b4b_audit.py
.\.venv\Scripts\python.exe tools/audit_pi05_libero_b4b_result.py --result-root simulation_output/pi05_libero_spatial_b4b_v1 --out simulation_output/pi05_libero_spatial_b4b_audit_local_v3 --font C:/Windows/Fonts/msyh.ttc
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/test_pi05_libero_b4b_audit.py"
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/audit_pi05_libero_b4b_result.py --result-root simulation_output/pi05_libero_spatial_b4b_v1 --out simulation_output/pi05_libero_spatial_b4b_audit_remote_v3 --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
```

Expected summary: `status=audited_with_protocol_deviations`, successes96,
episodes100, checks39, `strict_protocol_passed=false`, out-of-bounds episodes100.
Per-task success counts: `[10,10,10,10,9,8,10,10,10,9]`; failures task4/seed1002,
task5/1005, task5/1008, task9/1001, each280 steps. Full recorded eval runtime is
477.087539s, not the earlier conservative one-hour estimate. Offline audits
took seconds; they generate no new simulation steps.

All 100 video files are fully decoded and matched to episode mappings. Four
eight-frame failure sheets and ten successful reference last-saved frames are
rendered in Chinese, preserving aspect ratio. Note the saved video excludes
the final post-action frame; it cannot supply a precise terminal observation.
The raw report initially marks rendered sheets `not_viewed`; local exact hashes
are subsequently recorded as `viewed_not_accepted` in
`simulation_output/pi05_libero_spatial_b4b_audit_local_v2/visual_review.json`.
That review does not accept unviewed remote-font variants or later rerenders.

Metadata caveat: `effective_env_config.json` serializes360x360, but actual
saved main-camera videos are256x256 and `configs.LiberoEnv.gym_kwargs` does not
forward sizes. Runtime defaults are256, with the raw-main-view render path
doing only flips. Do not treat serialized360 as an observed raw input tensor
shape; use audit v2 rather than superseded local v1. Both policy image tensors
were not recorded directly. Do not change camera dimensions or rerun based on
this metadata discrepancy alone.

Native action extrema are [-1.03310227394104, 1.0188164710998535]. Episode min/max
cannot identify dimensions or counts. Current installed evaluator source
hash `cb0e1c2d891b93b41cb7b97921188fe2fd20ea6fa05675b87835b06c648ecce7`
matches the run manifest and records postprocessed actions passed to env.step.
Current relative OSC code calls `scale_action`; base controller clips input
before scaling, while Panda gripper uses sign and clipped internal state.
Post-run source inspection hashes (not execution-time captured controller
provenance):

```text
robosuite/controllers/base_controller.py
  61f99a2daeb4eb6e39bfb996d94c5a5689ef3bc771d030d070193ea56d6a8332
robosuite/controllers/osc.py
  cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb
robosuite/models/grippers/panda_gripper.py
  a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb
```

Original run directory has107 files (~5.4MB). Per-file scp was slow; that one
agent-owned transfer was stopped without touching remote originals, then the
same directory was packaged as a3,429,247-byte archive and SHA256-verified
before safe extraction over the partial local mirror. The archive is
`simulation_output/pi05_libero_spatial_b4b_transfer_v1.tar.gz`, SHA256
`f9a156dd247283c3bcbe48fb54cb4f2f8dd926fe28bacc77eb3f96c39578cb57`.
It is below the100MB user-transfer threshold. Local and remote audit reports
contain identical original-file inventories; raw artifacts are not modified.

Retain observed96/100 with its qualification. A new short action-boundary probe
or a changed clean/noisy protocol is a next task, not part of these read-only
commands. No checkpoint selection, additional wrapper clip, training, public
dataset acquisition or noisy100episode evaluation was started.

## Short Native Action Boundary Probe (2026-09-12)

Completed remotely with task0, seed1000, actual init0 payload matching B4b:
20 steps, original280 horizon,67.952s including resource checks/model load.
No optimizer, full score, controller change or additional clipping. Twelve
CPU tests passed on Windows and4090; ten real boundary checks passed.

Executed version hashes:

```text
tools/probe_pi05_libero_action_boundary.py
  a8808eb284fbf4ca980943e8c4729c2674b5c4b9af2e553dadadaad45e6e1d86
tools/test_pi05_libero_action_boundary.py
  3614625ea370616c1c62a256903811587164186dadeab9e8ae3eeefba2ab8a41
tools/render_pi05_libero_action_boundary.py
  87db93c468fa4367a745cdc80074a32c1b268b7c25d5b1eecff46fe1f2928545
```

Read existing results without loading the model:

```powershell
Get-Content simulation_output/pi05_libero_action_boundary_v1/report.json
Get-Content simulation_output/pi05_libero_action_boundary_v1/run_manifest.json
.\.venv\Scripts\python.exe tools/test_pi05_libero_action_boundary.py
```

If a repeat of this bounded diagnostic is requested, choose a fresh attempt
suffix; do not rerun v1 or delete failed attempts. Do not use these commands
to launch a full score/noisy run or training:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "pwd"
scp tools/probe_pi05_libero_action_boundary.py project4090:/home/zsw/project_2026/tools/probe_pi05_libero_action_boundary.py
scp tools/test_pi05_libero_action_boundary.py project4090:/home/zsw/project_2026/tools/test_pi05_libero_action_boundary.py
scp tools/render_pi05_libero_action_boundary.py project4090:/home/zsw/project_2026/tools/render_pi05_libero_action_boundary.py
Get-FileHash tools/probe_pi05_libero_action_boundary.py,tools/test_pi05_libero_action_boundary.py,tools/render_pi05_libero_action_boundary.py -Algorithm SHA256
ssh project4090 "cd /home/zsw/project_2026 && sha256sum tools/probe_pi05_libero_action_boundary.py tools/test_pi05_libero_action_boundary.py tools/render_pi05_libero_action_boundary.py"
```

Compare all three hashes before execution; transfers are sequential. Then:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/test_pi05_libero_action_boundary.py"
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/probe_pi05_libero_action_boundary.py --attempt v2 --steps 20"
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python tools/render_pi05_libero_action_boundary.py --root simulation_output/pi05_libero_action_boundary_v2 --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
```

`--steps` accepts1..60, defaults20; it only bounds the diagnostic loop, not
the native environment horizon. Output is an atomic fresh directory with
run.log/status.json/run_manifest.json, effective configs, steps.jsonl,
native_events.jsonl, report.json and raw pre-action PNGs. Failure remains
inspectable with a traceback/status; there is no resume and no fabricated
completion or success score. If SSH disconnects, first inspect status/PID
and run.log before issuing another run; avoid concurrent policy instances.

v1 results: arm6D0/20 saturated; gripper8/20 outside[-1,1], minimum
-1.009158492, maximum-0.990336955. All20 native7 inputs were preserved;
OSC1call/step, gripper25calls/step; all500 gripper outputs match the original
sign/increment/clipped-internal-state formula. Controller arm limits
[-1,1] map to translation+/-0.05 and rotation+/-0.5; arm clipping was not
activated in this real window. Both raw cameras are256x256, both policy
image tensors are BCHW256, and the two inference calls produce two valid
224x224 images plus the masked empty camera. State8 was checked before
policy preprocessing. No 360pixel-runtime violation is supported.

The report SHA256 is
`7f77f545904340078361e9c30a9053295c40e22fc36776d9eb967c02e470505a`.
All107 original B4b artifacts stayed unchanged, with inventory in the probe
directory. Original B4b runner/protocol hashes remain02473bf8.../b72b854d....
The small transfer archive is712975bytes, SHA256
`1b527f32146bcca76df70c657705a3c39cca8396395a0ed86575ba32af2c32bb`.
It was checked after scp completion and member names were validated before
fresh local extraction. The Chinese sheet is byte-identical to the remote
render; direct local inspection and exact-hash visual status are recorded in
`visual_review.json` as `viewed_not_accepted`. The raw camera orientation is
preserved, before the native environment/policy preprocessing; saved samples
are not terminal or success evidence.

Interpretation: this probe explains slight gripper overshoot without changing
native sign semantics. It cannot recover historical100episode dimensions,
explain failures or turn the frozen raw-bounds contract into a strict pass.
Oneenv construction/global RNG history differs from B4b's tenenv construction;
only checkpoint/config, processors, init payload and mechanics are pinned.
Next propose an explicit new native-controller/no-extra-clipping contract
before paired clean/low-contrast evaluation; do not edit frozen v1 silently.

## Fixed Paired Low-Contrast Screen (2026-09-12)

The separate new protocol is
`docs/libero-spatial-low-contrast-protocol-v1.json`, SHA256
`11dbb573db0bab86e883863ec38f0721f695bb345ad037b845c2bbd37d06bf72`.
Old B4b score protocol/result remain unchanged. Native finite7 overshoots
are diagnosed by dimension; original OSC/gripper semantics are retained and
additional wrapper clipping is prohibited. No checkpoint/loss/training change.

Fixed version hashes, all sequentially uploaded and matched before execution:

```text
tools/run_pi05_libero_low_contrast.py
  6a8dd7a1e891d4ef350967a93b9feebf26bcc3b8f408e38ba9188dc2a9156586
tools/pi05_libero_low_contrast_contract.py
  5027b653fc0145ad641d10ff87f7a323f10ed94c45d3584475868a49f773e734
tools/test_pi05_libero_low_contrast_contract.py
  885fabc88fa0b76b13f08d0c3443881f4dabc887a98889d4a6350cc1920c33a9
tools/test_pi05_libero_low_contrast_episode.py
  f9c75c714069430ee2536e0c9a40340212c3d314c1aff4853cd0b7424377a641
tools/render_pi05_libero_low_contrast.py
  7614f0250f1041a371100b81a724cccd3a376580034a6c676ac8df6205d7fe33
```

23 CPU tests passed on both Windows and4090 (20 contract/helper +3 fake
episode-hook tests). Real `--stage check --attempt v1` completed in67.715s:
one task0 pair, both conditions successful at82 steps, paired delta0pp.
This is a preflight observation, not a10task robustness score or evidence of
architecture/world-model improvement. No screen was started by the agent.

```text
simulation_output/pi05_libero_low_contrast_check_v1/report.json
  sha256=34043b8a6681732d4fbe788707a9d80d5508a04e12531be8c1805e70035a1047
run_manifest.json
  sha256=2d36a576cdfd08e53b8770a50a1ab98dadc21b53ff8a73eda27336953af0e8df
episodes.json
  sha256=188b03c4379c720cd3ac0c9464dffded712393015805c0048efd79d64887d9ed
```

Historical user-run command (v1 has now FAILED; do not rerun this suffix):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_low_contrast.py --stage screen --attempt v1 && cat simulation_output/pi05_libero_low_contrast_screen_v1/status.json"
```

Conservative runtime budget5–10min for20 full episodes; may finish sooner.
It evaluates tasks0..9, each init0/seed1000, two conditions, native280 horizon.
Alpha0.45 (existing severity2) affects both rawRGB views about per-channel
spatial mean; no other noise. Fresh clean episodes are required because RNG
is reset independently per arm, unlike historical B4b's global trajectory.
Check episodes are not reused for screen or checkpoint/model selection.

The default prerequisite is
`/home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_check_v1`.
Before loading the model, screen verifies its status/report/manifest/episode
hashes and exact runner/contract/protocol/resources. If implementation changes,
first rerun an explicitly scheduled short check into a fresh directory and use
`--check-root /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_check_v2`;
do not modify check records or bypass version mismatch. Do not silently edit
the frozen new JSON to tune severity after seeing outcomes.

Logs are redirected to the output directory; another terminal can inspect:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_screen_v1/status.json"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_screen_v1/episodes.json"
```

If disconnected, inspect the recorded PID/status and remote process before
retrying. Do not launch concurrent copies. A failed/partial directory is kept;
there is no episode resume or automatic promotion of partial metrics. Use a
new attempt suffix only after the prior process has exited and the failure is
understood. Each completed episode is immediately saved with action arrays,
observation/policy-input traces and actual pairing fingerprints.

Read result after completion:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_screen_v1/report.json"
```

Expected completion is stage=screen, pair_count10, episode_count20,
paired_schedule_verified=true, optimizer_steps0. Metrics to inspect:
clean_success_count, low_contrast_success_count, delta_pp (low-clean),
relative_drop (null if cleanzero), four paired_outcomes, per_task. First
actual init payload/fullrawpostsettle observation/firstinference RNG must match
within each pair; later states/actions may legitimately diverge. Per-episode
native action extrema/counts and exact base/inner identity remain diagnostic.

For image-only evidence after a completed stage:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python tools/render_pi05_libero_low_contrast.py --root simulation_output/pi05_libero_low_contrast_screen_v1 --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
```

This renderer compares the actual FIRST supplied rawimage pair at task0,
not the still-clean env.render() view and not terminal behavior. Eight initial/
last pre-action rawinput PNGs per pair support further inspection; last frames
are on different trajectories and must not be called matched observations.
Main screen conclusions remain pending all ten pairs and result inspection.

If local files change, scp each relevant final small file sequentially and
compare SHA256 before remote checks. The old wrapper modules are pinned and
unchanged. Files/archives over100MB remain user-transfer; the check transfer
archive is771559bytes, SHA256
`445492fd5defd169a9f22c77985a5adbd0c4a3c0e975e762b6ff9d727d8dfe21`.

## Task4 Reset-Only Failure Diagnosis (2026-09-12)

Screen v1 stopped after10 completed episodes with
`ValueError: Unpaired actual start: task 4 initial_observation_sha256`.
It has no report/full-screen score. Keep the failed directory and all previous
check/B4b outputs. This section supersedes the earlier pending-screen next step.

Implemented `tools/probe_libero_task4_reset.py` (SHA256
`715a44e3dbbe068f6897307fc265cd3f7519bc429a30a43b2474d88410d157b8`)
and `tools/test_libero_task4_reset_probe.py` (SHA256
`ab3fc4c450b6f6a0014d33d96e4b069d0895ffb2ead170b1251f15e3b2fcf941`).
Five helper tests passed on both machines; sources synced and hash-verified.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_libero_task4_reset_probe.py -v
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_libero_task4_reset_probe.py -v"
```

Already executed successfully (v1 retained; do not overwrite):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 240s /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/probe_libero_task4_reset.py --attempt v1"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/libero_task4_reset_probe_v1/status.json"
```

The external240s timeout bounds an unchanged native retryloop; if it fires,
retain partial output, inspect process exit/logs/attempts.json, and do not read
a stale running status as completion. Fresh attempts require a new suffix.
This is a short reset diagnostic, not a policy rollout or model evaluation.

Evidence root: `simulation_output/libero_task4_reset_probe_v1/`.
Report SHA256 `333dff2e45beffa6775ab9be97c2fb3e3903cbde19c6fb1ceb7a1ace04277eb2`.
Completed in15.182s, five measured resets and50 measured settlingsteps
(constructor activity excluded), no PI05 loading/inference/training. All218
protected old B4b/check/failed-screen files unchanged. Each measured reset has
attempts.json, reset.json,13stage JSON/NPZ snapshots and six rawPNG views.

Result: repeated shared-env resets diverge at native reset return; only NumPy
RNG differs. Fixed init restores qpos/qvel but not model body placements.
Two fresh-env controls match the first shared reset at every captured stage.
The first/third shared resets reproduce the exact failed clean/precorruption
task4 start hashes. No RandomizationError occurred in the five measured resets.
Do not infer precise sampler-branch cause or full-screen recovery from this.

Next proposed work (not run/implemented): a separately versioned fresh-native-
environment lifecycle per condition, all-task reset verification, then a fresh
PI05 check preserving strict equality gates. Existing v1 check cannot certify
a changed runner/protocol, and v1 partial episodes cannot be reused as scores.

Local mirror is only the review subset: report/status/manifest plus six
after_set_init_state PNGs; complete stage JSON/NPZ is remote. The340808byte
`libero_task4_reset_probe_v1_review.tar.gz` was SHA-verified and safely extracted:
`a4cd9dc33cca9f3cb93c24aa303cf147dc2a89f5c41eee893f055e79b80ce3fa`.
The much slower complete archive download was stopped; its retained partial
local `libero_task4_reset_probe_v1_transfer.tar.gz` is NOT a verified backup.
Six exact images were viewed; visual_review.json records
`viewed_not_accepted`, not user acceptance or episode success.

## Fresh-Environment Low-Contrast V2 (2026-09-12)

V2 supersedes the failed v1 lifecycle, without modifying v1 files or promoting
its partial results. The explicit delta is
`docs/libero-spatial-low-contrast-protocol-v2.json` (SHA256
`9764479b78752acbcca2b5d80bfb390877170aebfa95f24988ff01e1ac0d06c3`).
All parent model/noise/controller/metric semantics are inherited unchanged.
Fresh original single-task sync environments are constructed per condition
after seed1000, then reseeded1000 before official rollout and closed in finally.
No additional state forcing, action clipping, training or dependency patch.

Final small sources, sequentially synchronized and SHA-matched:

```text
tools/run_pi05_libero_low_contrast_v2.py
  1f45f3cdb27edcb52caf52cca416e821a234d7d0a0df0256df2c6673c3ac5654
tools/pi05_libero_low_contrast_v2_contract.py
  c564e9f3818e7b053923b07dd5b5c00c97117fc779415ad57bb1a335d49da667
tools/test_pi05_libero_low_contrast_v2_contract.py
  db8d54a07b5a255ff46ad92c4726bac0356232c02a2e068319a20a8502e0e05f
tools/test_pi05_libero_low_contrast_v2_runner.py
  3a49219982ffb245a408569fa881430d4ee88766a23071255f0d6a6a371b2e5e
```

48 tests passed on both Windows and4090 (23 prior +16 v2 contract +9 runner).

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_low_contrast*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p 'test_pi05_libero_low_contrast*.py' -q"
```

Already-run short verification commands (do not overwrite their v1 attempts):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 240s /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_low_contrast_v2.py --stage reset-check --attempt v1"
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_low_contrast_v2.py --stage check --attempt v1"
```

`reset-check` completed in45.094s:10tasks/20fresh resets, all actual init/raw
observation/post-reset RNG hashes matched, no model loaded,200 measured native
settlingsteps (constructor activity excluded),393 historical files unchanged.

```text
simulation_output/pi05_libero_low_contrast_v2_reset-check_v1/report.json
  436e0ed1f9c5c59e73d86261c13ad7e0396021fc2ceacfe9a2931109a0be3cdb
run_manifest.json
  7fe910e8d809927c929d72cb1bb8da0cce34809bb18184f814e15adb04a0e5b4
episodes.json
  03db9900d56ebfdc6c37d816ee47aad93a035a6f9047e69f5a66046fe31f7a73
```

`check` is now four full episodes: task0 AND task4 clean/low_contrast, init0,
seed1000. It is an execution regression, not a robustness estimate. It requires
the completed reset gate and prospectively checks actual init payload/raw
observation/post-reset RNG before inference, plus paired first-inference RNG.
Native terminal auto-reset is not mistaken for the initial seeded reset.

Check completed in86.004s: task0 clean82/degraded79steps, task4 clean130/
degraded131steps, all four successful, both actual initial pairs matched.
All prospective guards and native action/image boundaries passed;393 old
files unchanged. This short regression/delta0pp is NOT a robustness estimate.

```text
simulation_output/pi05_libero_low_contrast_v2_check_v1/report.json
  77d263fbdc4c8ed826f320fbf00f5148f0d4c0406794deaad657a03e94908238
run_manifest.json
  df15f2372bbd9f1c9d683ac6dd19ba3b196d9e56df259f017bfc38420647d40e
episodes.json
  0f58a95273a8651fe4eb9d5fb87b1eff6683efd086b368b077c57a3c66d0130b
```

The current reset/check gates were separately re-read and reaggregated without
loading a model; all screen prerequisites matched, including exact reset status
SHA `bb990071a1a6150bfb48b58c64b69f4771d9d93f6a4ff77a2314ee2c5688e453` and
check status SHA `9c0e53547f2fea1c356b257886154f9042fe354a3a19732c58cb3707694cb773`.
No full v2 screen was started by the agent. The subsequent user-run screen
has now completed and been audited; see the result section below.

Already completed by the user after the short check and artifact inspection.
Historical reproduction command only; do not rerun/overwrite this attempt
(new v2 namespace, separate from the failed v1 screen):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_low_contrast_v2.py --stage screen --attempt v1"
```

Original conservative full-screen budget5–10min; actual completion193.602s,
20 new full episodes,10task pairs.
No timeout is added to the full screen. Its prerequisite defaults are:

```text
--reset-root /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_v2_reset-check_v1
--check-root /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_v2_check_v1
```

The screen checks exact source/protocol/resource identity, complete row
reaggregation and that the check used the SAME reset-gate status SHA. Failed
or changed versions are not accepted. If code changes, use fresh reset/check
attempts and pass BOTH corresponding roots; do not rewrite old gate evidence.
No automatic episode resume/overwrite. After SSH loss inspect status/PID and
processes before starting anything else; after failure retain the directory,
diagnose the cause, and only then choose a new attempt suffix.

Read status even if Conda exits nonzero (the run log holds the actual error):

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_v2_screen_v1/status.json"
ssh project4090 "tail -n 30 /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_v2_screen_v1/run.log"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_v2_screen_v1/report.json"
```

A completed screen must contain20episodes/10pairs and strict actual identities.
Report clean/low success counts/rates, signed delta_pp(low-clean), relative_drop,
four paired outcomes and per-task results. Per-dimension native action overshoot
remains diagnostic. Never call the short checks a robustness/full-screen result,
or infer architecture/world-model, real-system or guidewire gains here.

The existing renderer was reused unchanged after the completed short check:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python tools/render_pi05_libero_low_contrast.py --root simulation_output/pi05_libero_low_contrast_v2_check_v1 --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
```

Root viewed its Chinese initial two-view sheet, task4 first two-view inputs and
all four last pre-action main images. `visual_review.json` records exact files
and `viewed_not_accepted`, not terminal-success or user-acceptance evidence.
The733970byte local review archive contains gateJSONs and selectedPNG only:
`pi05_libero_low_contrast_v2_checks_review.tar.gz`, SHA256
`73e1149fa6a0ad3252cd0b7219e8f862af6b0b8bdeb9f5fd5c81c4a4abf690b9`.
Full action and observation traces remain remote. Files over100MB still use
the user's transfer workflow; this small review package was transferred by agent.

### Completed V2 Screen: Read-Only Audit And Review (2026-09-12)

`simulation_output/pi05_libero_low_contrast_v2_screen_v1/status.json` is
`completed`: clean10/10, low_contrast10/10, delta0pp, relative_drop0;
both_success10 and the other three paired outcomes0. Fixed alpha0.45,
task0..9/init0/seed1000, 193.602s, zero optimizer steps. This single-init screen
does not establish general robustness, no degradation or real-oxidation validity.

```text
report.json        8b59c322385b12e2c44dd68facd9ff0a8f182dca83c696bab41982ad7369b1b5
run_manifest.json  008c45a9bc2d3ff889083b53b4ff0151ef7cf86dfa34cb2934f4e358c93be009
episodes.json      e3767c91fccdd20b40378bf23b53b8a14310c7b700c6563ae7ee1b0b8cbd24f9
status.json        c04fbefe03f50e776b1bf6e15b0938bac4c42687442978ea6de776f25dd702f8
```

Root's separate no-model remote audit reaggregated all three strict gates and
their exact code/resource/reset-proof identities; checked all20 closed fresh
constructions and actual-start guards; rehashed393 historical files and full
checkpoint resources; checked2071 action/observation/policy-input rows; and
reconstructed all80 first/last policy image tensor hashes from actual PNGs.
All passed. The two noisy views changed deterministically on every logged step;
noise genuinely entered the policy. Native overshoot counts per dim were
`[0,0,0,0,0,0,757]`, diagnostic without extra clipping. The audit loaded no model,
reset no environment and performed no optimizer steps or new rollout.
`post_run_audit.json` is an operator-recorded summary of these completed checks.

The unchanged renderer was also used on the completed screen, not on new data:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python tools/render_pi05_libero_low_contrast.py --root simulation_output/pi05_libero_low_contrast_v2_screen_v1 --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
```

Already-transferred small review package (not a full trace/checkpoint backup):

```text
simulation_output/pi05_libero_low_contrast_v2_screen_v1_review.tar.gz
bytes: 927853
SHA256: 3ebe2a2eec073362b1b3894e6cbdb1e423bb34ab555faf7643c0e957535aba19
```

Its17 members are four original JSONs, the task0 Chinese initial comparison
sheet, and task4/task9 clean/degraded first_image, first_image2 and last_image.
SCP completion and SHA were verified before path-safe extraction. Full per-step
traces remain remote. Root viewed all13 PNGs; exact hashes/scope are recorded
in `visual_review.json`, status `viewed_not_accepted`. Last images are pre-action,
not terminal-state evidence. No explicit user visual approval is recorded.

Decision: retain the result; no screen rerun, noise escalation or training.
Any future additional-init study needs a prospectively fixed schedule and the
same paired controls. Never merge check episodes into this score or use old
B4b96/100 as its clean comparator. Source/feature/training gates remain separate.

## Pinned Public LIBERO Source / Seven-Row Frozen Feature Probe (2026-09-12)

Scope: public metadata/task mapping, complete one-episode source audit and a
seven-row frozen visual-feature interface probe. No world-model training,
action selection, new rollout, guidewire/collector changes or complete feature
pack. The source choice is fixed episode1400, independent of success/quality.

New small files (sequential SCP and exact SHA check before remote execution):

```text
tools/audit_libero_public_source.py
  2b944cf131a2338fd5e9c7b31d3360372ed4a8be40399a745f6c6d896e5a740c
tools/test_libero_public_source.py
  3ef5551210e6af1f232fa193b68f00f723fbede6ca94b0443dd075419beba34e
tools/probe_pi05_libero_public_features.py
  fefef839ce8ae7d7826cb1d52a2c9827a96cd7e2035c4df69339601461812f75
tools/test_pi05_libero_public_features.py
  5e235162ce568d570f6014c50c772f1f1356f63c5a06a8a03421a4e7920af814
```

45 CPU tests pass on Windows and4090:14 source guards +31 feature/config tests.
The real config regression matters: v1 incorrectly expected only two image
keys before PI05 initialization. The pinned raw config already has three
(two real256x256, one empty224x224). The corrected guard requires that exact
layout, bfloat16/state8/action7, without adding/removing images or modifying
model code. The extractor still receives only the two real views and rejects
invalid empty-camera masking. V1 failed before policy construction; retain
`simulation_output/pi05_libero_public_feature_probe_v1/status.json` and run.log.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_libero_public_source.py -q
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_pi05_libero_public_features.py -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_libero_public_source.py -q"
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_pi05_libero_public_features.py -q"
```

Already completed source command (historical, do not overwrite/re-download):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/audit_libero_public_source.py --out simulation_output/libero_public_source_audit_v1 --episode-index 1400"
```

Source outcome:16.474s,43,217,790 bytes (seven files, all <100MB), fixed revision
`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. Full repository is1,935,512,060 bytes
and was NOT downloaded. Mapping Spatial0..9→source IDs:
`[34,37,38,35,31,32,30,33,36,39]`. Selected episode1400→source39/Spatial9,
140 complete records plus280 decoded image frames. Stored10Hz timestamps and
video offsets257.3:271.3s agree. Independent full-episode Parquet/JSON float32
state/action readback is exactly equal; upstream physical cadence/hidden-reset
or conversion provenance remains unverified.

The source report is an explicit input identity for the real feature probe:

```text
simulation_output/libero_public_source_audit_v1/report.json
SHA256: 0d431859588a2f14758fbf1c27b17e1ec6114f161c14f5cc246ae2e55317c4e5
```

The source tool caps single files at100,000,000 bytes and total selection at
90,000,000 bytes before transfer; it never splits large files to evade the
user-transfer rule. Selected Git/LFS identities and bytes are checked. Use the
source status/report to inspect failures; existing output roots are rejected.
If code, revision, selection or metadata changes, use a new audit attempt and
its new report hash, not the historical literal above. No partial audit resume.

Already completed seven-row extraction command (v2 preserves the initial
failed v1; historical only, do not rerun/overwrite):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/probe_pi05_libero_public_features.py --source-root simulation_output/libero_public_source_audit_v1 --source-report-sha256 0d431859588a2f14758fbf1c27b17e1ec6114f161c14f5cc246ae2e55317c4e5 --out simulation_output/pi05_libero_public_feature_probe_v2 --device cuda"
```

This checks all source/derived hashes before loading the offline pinned
task-finetuned checkpoint. Stored demo RGB is consumed directly (no raw-env
H/W flip), permuted to BCHW float32/255 and passed through the checkpoint's
native image preprocessing exactly once per batch. Seven batch1 rows plus a
repeat of row0 require8 preprocessing and16 real-camera embed calls; the
declared masked empty view is never pooled. Fixed repeat tolerance is
atol=rtol=1e-3, with actual absolute differences/bitwise equality also reported.
No language tokens are fused into these patch-mean visual latents; native task
instruction remains separate in the downstream adapter. No action is generated.

Inspect status and actual log after disconnection; do not rerun an existing
attempt or start a concurrent copy:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_public_feature_probe_v2/status.json"
ssh project4090 "tail -n 30 /home/zsw/project_2026/simulation_output/pi05_libero_public_feature_probe_v2/run.log"
```

Seven rows can cover a4-observation history and3 recorded successor steps, but
are NOT a complete episode feature pack or training split. Full feature-pack
generation, source-level split acceptance, native7D world-model implementation
and user-approved optimization remain separate steps; the guidewire9D trainer
must not consume these native7D records.

Verified v2 outcome:33.539s, finite float32 `[7,2,2048]`, eight preprocess /
sixteen real-camera embed calls, first-row repeat bitwise equal (maxdiff0),
row variation maxabs5.845703125. Strict loaded parameter fraction1.0,
missing/unexpected/shape-mismatch0, tied alias1. All812 parameter tensors frozen;
all modules eval, no gradients/actions/optimizer/training. Actual bfloat16
image tokens were `[1,256,2048]`, averaged in float32. Two usual vision-key
warnings did not defeat the strict load checks. No full feature pack or split.

```text
simulation_output/pi05_libero_public_feature_probe_v2/report.json
  f3f8ae8a3c234bf55a07c0c92eaa3635f06f65d96b3793c48b699c8e4975db17
simulation_output/pi05_libero_public_feature_probe_v2/visual_latent.npy
  1a84262a67870f5e4a88c91c1f650ded07465193a674dbc121671f91fed63292
```

The277959-byte SHA-verified local review archive also includes v1 failure logs:
`pi05_libero_public_feature_probe_v2_review.tar.gz`, SHA256
`60dff5e0e55f822548be3cfbc4df138732a771ac2d89989ab0b5632994028d1e`.
Source review archive:1,168,637 bytes, SHA256
`457b7cb8dc0cb0940d7cf955c25478b50d64a18b745560ea4f6618556c924412`.
Both are subsets; full selected MP4s and checkpoint remain remote. Root verified
source/feature linkage, implementation and saved artifact hashes; six source
PNGs and two processed previews were viewed, not user-accepted. Exact image
hashes and limitations are in each root's `visual_review.json`. No weekly-log
entry for interface-only preparation. Future complete-episode feature extraction
and source-level split must be separately prepared before any training request.

## Complete Two-Episode LIBERO Feature Pack / Fixed Diagnostic Split (2026-09-12)

Already completed; these commands are provenance/recovery references, not an
instruction to repeat successful jobs. Scope: complete cached public episodes,
frozen image features and native window preparation only. No training, actions,
benchmark rollout, guidewire or data/SOFA changes. The hard-coded guidewire9D
world-model trainer must not consume this native7D pack.

V1 source preparation failed before feature extraction because adjacent
episode1401 is actually source34/Spatial0, not source39. Keep its original plan,
partial files and failed status (SHA256
`5f67687561195e3489f2062be6298c79034acf3b26a892f7caaa168c5b10851d`).
The initial source code SHA was
`277e6e918889caa29dd6875cb3fb59c8e89c239d0adb1618460c63b6809d8c30`.
After a cached episode/task/frame-metadata-only scan, v2 explicitly fixes the
next same-task source episode1402. This did not follow a duplicate-test failure
or use images, rewards, actions or feature quality for selection.

Final small implementation files (sequential SCP and SHA verification completed
before remote execution):

```text
docs/libero-feature-pair-plan-v1.json  [preserved invalid-task plan]
  bc5b7e49b1d36878fe3ee1892b82faff1e6967a1bc307451756414d5a5696823
docs/libero-feature-pair-plan-v2.json  [active fixed plan]
  bad36b6a93769d4210792c60a42892a325d30694d8a087316deeafab7de48a53
tools/prepare_libero_source_pair.py
  5e7169f3039677752955931851be458e1fb41f60b54cb2d48947b0fd60b09d89
tools/test_libero_source_pair.py
  de4d224fd66a37bde212e2ad381546a8d380b7ba4e6b961688519f9b2c5ab8b9
tools/build_pi05_libero_feature_pack.py
  29db2cf2715f8440c2d374c46640da88b8144af28451fad89f0cb9c7eedc96b4
tools/test_pi05_libero_feature_pack.py
  fa485155fb74521c42e77e1bb0657c595ae7cdd70ce5d14740bf356fcad405c8
```

The two new CPU suites pass35/35 locally and remotely (16+19; builder remote
6.692s). They include fake full313-row extraction without CUDA/LeRobot; these
fixtures alone are not evidence of actual PI0.5 output.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_libero_source_pair.py -q
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_pi05_libero_feature_pack.py -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_libero_source_pair.py -q"
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_pi05_libero_feature_pack.py -q"
```

Historical successful source command (1.572s; no network downloads):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/prepare_libero_source_pair.py --base-source-root simulation_output/libero_public_source_audit_v1 --plan docs/libero-feature-pair-plan-v2.json --out simulation_output/libero_source_pair_v2"
```

Source output:1400(train140 rows) +1402(validation173 rows), Spatial9/source39,
313 complete state/action records and626 decoded RGB frames. Episode1402 global
rows237832:238005; two view offsets280.2:297.5s, disjoint from episode1400.
Both use the existing Parquet330 and video030 payloads. Full source state/action
float32 reconstruction is exactly equal to Parquet. Original cached payloads
are rehashed before/after source extraction; source report and every derived
output are also rehashed before/after feature extraction.

`duplicate_suffix_audit.json`: exact/quantized state-action and exact two-view
pixel7-frame cross-split matching hash counts all0. Transition fingerprints use
7 states and6 connecting actions, never the last no-successor action. State/
action quantization grids are1e-4/1e-3, not a continuous-distance guarantee.
The test rejects matching windows; it does not automatically swap episodes,
truncate them or establish independent families. The frozen max_records=400 is
only a fail-closed total bound, not record truncation; all313 records are used.

Source report:

```text
simulation_output/libero_source_pair_v2/report.json
  ab691452b5901836a1638096955b15b294c09f32c2b2a0915a566cc36181187f
```

Historical successful real feature build (39.399s, CUDA, offline pinned
checkpoint; no optimization):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/build_pi05_libero_feature_pack.py --source-root simulation_output/libero_source_pair_v2 --source-report-sha256 ab691452b5901836a1638096955b15b294c09f32c2b2a0915a566cc36181187f --out simulation_output/pi05_libero_feature_pair_v1 --device cuda"
```

The builder does not modify the PI0.5 processor/backbone or existing native
adapter. It accepts only the pinned plan, source revision, registry and exact
two complete episodes; verifies source records/images before model load; and
traces313 actual preprocess calls plus626 real-view embeddings. No extra image
flip or masked-empty pooling. Strict unique-parameter checkpoint coverage1.0,
effective missing/unexpected/shape mismatch0; usual two vision embedding
warnings remain non-fatal. All parameters frozen, modules eval, gradients absent,
parameter versions unchanged; zero actions, optimizer or simulation steps.

Output contract:

```text
simulation_output/pi05_libero_feature_pair_v1/
  manifest.json + split.json                # source and extractor hash-bound
  state.npy                                # float32 [313,8]
  action.npy                               # float32 [313,7], native values
  visual_latent.npy                         # float32 [313,2,2048]
  episode_index.npy + frame_index.npy + task_id.npy
  state_valid.npy + visual_valid.npy
  extraction_trace.jsonl                    # all313 rows, actual boundaries
  extractor_provenance.json                 # checkpoint, installed/repo code
  preparation/normalization.json            # train140 state /139 actions
  preparation/train_windows.jsonl           # 134 complete windows
  preparation/validation_windows.jsonl      # 167 complete windows
  preparation/report.json                  # unchanged generic schema checks
  report.json + status.json + run.log
  episode*_source.png + episode*_processed.png  # 16 first/last view PNGs
```

World-model normalization uses only episode1400, not either checkpoint's PI0.5
normalizer. Final actions remain in action.npy but never enter statistics or
candidate action windows. History4, future3, stored10Hz: history span0.3s and
prediction horizon0.3s. Task text stays a separate input; future observations
stay targets. No selected windows cross episodes; validation statistics cannot
alter the training normalization.

```text
pi05_libero_feature_pair_v1/report.json
  b30e0a2bdca6c40306b69834f6d9339d434b180bec0e2c4776b11cba05d70632
pi05_libero_feature_pair_v1/manifest.json
  e88ee58f796f8d32eb014616e094fb0a35253f988e81bb97e11a90fe536da295
pi05_libero_feature_pair_v1/split.json
  035b74a91770e75fa8d0320ba422fee9cc805c2ed1a4637da1d9303179ad509b
pi05_libero_feature_pair_v1/visual_latent.npy
  f7e21dc57b138b324d998102900753a23b4d2dbda66e97510dcc5bcb36ebb8cb
```

Recovery is inspect-first; these small jobs have no partial resume. A changed
plan/source/checkpoint needs a separately justified fresh attempt, never bypass
the checks or overwrite either failed/successful directory. After disconnect,
check status, log and recorded PID before doing anything; do not launch a
concurrent duplicate. The completed job does not need a rerun:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/libero_source_pair_v2/status.json"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_feature_pair_v1/status.json"
ssh project4090 "tail -n 25 /home/zsw/project_2026/simulation_output/pi05_libero_feature_pair_v1/run.log"
```

The local SHA-verified review archive is6,489,092 bytes:
`simulation_output/libero_complete_feature_pair_review_v1.tar.gz`, SHA256
`624bbe1f7187a02620bc4bc015712f45154e4829ac0ff53bad951b7dd503afb4`.
It includes failed-v1 and successful-v2 source JSON/representative PNGs and the
complete small feature pack, but excludes all source images.npy and raw videos.
Those remain remote; this is a review subset, not a split archive workaround.
Any single future file/archive >100MB remains user-transferred.

The complete pack's first7 feature rows are bitwise equal to the earlier
seven-row probe (max absolute difference0). Each episode's other rows all differ
from its first feature row (139/139 and172/172); this checks nonconstant
extraction, not model quality. Root directly viewed all16 first/last source and
processed PNGs; `visual_review.json` records hashes and `viewed_not_accepted`.
No user acceptance, full-video review or final-state success certification.

`postrun_audit.json` additionally records the independent read-only local
verification:31 feature output hashes,18 mirrored source output hashes,
preparation report/three output hashes and eight implementation hashes;
313 records-to-array rows,626 source-pixel trace hashes,313 latent row hashes;
independent train-only mean/std from source JSON (all maxdiff0); and all301
actual native window items. It explicitly excludes a new local audit of the
remote full source images/videos, Parquet and weights. This post-run audit and
visual sidecar are later additions, not members of the original build hash map
or the frozen review archive; preserve the original reports unchanged.

Decision: complete-episode extraction/window preparation is done for this pair;
native7D/task world-model forward implementation and zero-step tests are next.
`training_ready=false`, family independence unverified and checkpoint-training
overlap unknown. This single-task split is only a pipeline diagnostic; no
policy generalization, world-model benefit, full-suite or real-system claim.
No optimization or existing-screen rerun is authorized by these checks.

## Native7D / Task World-Model Forward-Only Check (2026-09-12)

Implementation and actual CPU zero-step verification complete. This is a
separate randomly initialized LIBERO predictor, not the guidewire9D model or
PI0.5 backbone. No loss/backward/optimizer, checkpoint update, candidate ranking,
policy action generation or environment step is in this entrypoint.

Final small files, sequentially SCP-synchronized and SHA-matched before remote
execution:

```text
tools/pi05_libero_world_model.py
  c0da1ab867264b0ced774f1ed523bdb11dd3e936d168a4729dabf63c067781d0
tools/test_pi05_libero_world_model.py
  9963a10d0b4346957a08eb19aa42cc8e22a49341ee4f12b8092eb36232a4dccb
tools/smoke_pi05_libero_world_model.py
  faddc62d7329231b75503d1263e81b2b08178af9602e7fbbc568ac661321cd3a
tools/test_pi05_libero_world_model_forward_smoke.py
  7023e9a36e2184da9c785522d4c796a48353a80e45403ba72570ddc7a34d1ec8
```

Core API:

```python
config = LiberoWorldModelConfig()  # D2048, hidden128, T4, H3, state8/action7/views2, dropout0
model = LiberoWorldModel(config, task_registry=pack.manifest["task_registry"])
batch = collate_window_inputs([window["inputs"]], device="cpu")
model = model.eval().requires_grad_(False)
with torch.inference_mode():
    predictions = model(**batch)
```

Only six fields enter `forward`: history visual/state with their boolean masks,
exact registered instruction text, and normalized7D candidate action chunks.
No `targets`, metadata, reward, object state, diagnostic fields, source IDs or
guidewire labels are forwarded. State8/action7/view2 are fixed. Model history/
horizon/visual/hidden sizes can vary for fixtures, but the real runner fixes
2048/128/4/3, dropout0, seed20260912, batch16 and CPU. Normalization is inherited
unchanged from the140-row train state/139 train actions; no second normalization
or normalized-action[-1,1] clipping. Masked NaN/Inf are cleared before projection
and current-visual residual addition; observed nonfinite values are rejected.

Output keys are exactly `pred_future_visual_latent` `[B,K,3,2,2048]` and
`pred_state_delta` `[B,K,3,8]`. Visual prediction is current masked latent plus
residual; state prediction is current-relative normalized coordinate residual,
not SO(3) rotation. Task lookup is a closed registry, not a language encoder.
Unknown instructions fail. No contact/Piper/branch or ranking heads are added.

Final bounded CPU tests:48/48 locally and remotely (36 model +12 entrypoint):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_pi05_libero_world_model.py -q
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_pi05_libero_world_model_forward_smoke.py -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_pi05_libero_world_model.py -q"
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p test_pi05_libero_world_model_forward_smoke.py -q"
```

The first remote36-test attempt had one overly tight float32 batch-permutation
assertion. Old tests SHA256
`08b1ab17f9a5252b3525e186f22f6b3c338e9aba0ca77747c46f3ce976e02eab`.
Max visual error2.384185791015625e-7 exceeded atol1e-7; all35 other tests passed.
Independent repeat with unchanged batch order gave0 for visual/state outputs;
batch flip gave visual2.384e-7/state5.960e-8; no gradients/CUDA initialization.
Only the batch-permutation test uses atol1e-6, rtol0 now. Model bytes, default
test helper and exact candidate-isolation/causality assertions stayed unchanged.
Failure retained in
`simulation_output/pi05_libero_world_model_tests_remote_v1_failure.json`.
Final remote suites completed in0.040s and0.005s respectively (test bodies).

Historical completed real-pack command (do not overwrite/repeat the result):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/smoke_pi05_libero_world_model.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --feature-report-sha256 b30e0a2bdca6c40306b69834f6d9339d434b180bec0e2c4776b11cba05d70632 --out simulation_output/pi05_libero_world_model_forward_remote_v1 --device cpu"
```

The entrypoint validates the external feature report SHA, all33 feature input
artifacts, frozen source/plan/checkpoint provenance and seven feature-producing
implementation hashes BEFORE importing torch. This inherits the existing
source audit; it does not reread raw videos, source image arrays or weights.
Fixed split and train-only stats are independently reconstructed by the native
adapter; all134 train +167 validation windows execute as K1 batches9+11. Targets
are constructed by the adapter but never passed into the model or used for loss
or MAE. Raw provenance appears only in output trace metadata.

Observed CPU result:0.790s, model1,321,352 parameters, finite predictions,
all parameters frozen/eval/no gradients, same state-dict SHA before and after
`691a7c14a7cef4c91f78f9a26a519586dacabbf85b9d1ab12d155c9eb4b573d0`.
Input feature bytes, current implementation hashes and parameter versions remain
unchanged. These times are not an end-to-end latency or efficiency comparison.

Additional B2/K3 diagnostic tensor perturbations use±0.25 in normalized action
coordinate0. Duplicate candidates, untouched candidate, candidate permutation
and future action versus earlier prediction all had maxdiff0. Changing the
last action changed final output by max0.0218520164. No candidate was ranked,
selected or dispatched; response to perturbation only verifies wiring, not
learned dynamics. Synthetic two-task tests cover task switching; this real
feature pair has one task and supplies no real multitask evidence.

```text
simulation_output/pi05_libero_world_model_forward_remote_v1/
  report.json + status.json + run.log
  model_contract.json
  forward_trace.jsonl                   # all301 input/prediction hashes + metadata
  first_train_predictions.npz          # first untrained output, not a checkpoint
  first_validation_predictions.npz     # first untrained output, not a checkpoint

report.json SHA256:
  b0476c8061208ad46a99cd715a70e26f82dff14b6c024000d558dbd4613bcaf3
forward_trace.jsonl SHA256:
  5801b0b5224edcd2a5a39db60ad4dad8009e9aadaee64c5b777ce6a154d7518b
```

The complete small result and first test failure are in the170433-byte
`simulation_output/pi05_libero_world_model_forward_remote_v1_review.tar.gz`,
SHA256 `6cfa70a28c117e730d62decf1eeec9ad825e2524b50897092ee951e2e4b05b55`.
Local download/hash/path validation completed; large model/data files were not
transferred. No image behavior changed, so no unrelated render is required.

Independent post-run local readback (`postrun_review.json`) verified4 output,
5 implementation and33 input feature hashes; all301 metadata/task records and
1,505 reconstructed input tensor hashes; and all four arrays in the two first
prediction NPZs against shape/dtype/finite/trace hashes. The reviewer did not
rerun the model or reread raw sources/weights. Only first-window predictions
are saved as full arrays; remaining prediction evidence is the bound trace and
execution report. The later review sidecar is not in the immutable run report's
output hash map or original review archive. Preserve those original bytes.

Recovery: existing output directories are rejected; there is no partial resume.
After disconnect inspect status/log/PID first. Do not rerun a completed job or
start a concurrent copy; a justified code/input change uses a fresh output name.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_world_model_forward_remote_v1/status.json"
ssh project4090 "tail -n 20 /home/zsw/project_2026/simulation_output/pi05_libero_world_model_forward_remote_v1/run.log"
```

Decision: native forward implementation and bounded zero-step verification are
complete. The model is still randomly initialized; no loss, prediction-quality,
ranking or policy-improvement result exists. Native masked objectives/metrics
and a dry-run-only training entrypoint remain next, along with an adequate
prospectively frozen public source/task split before a quality study. This
milestone does not authorize optimization or promote `training_ready=false`,
family independence, guidewire Gate A2, rotation/contact or real-system claims.

## Native LIBERO Masked Objectives And Dry-Run-Only Entrypoint (2026-09-12)

This supersedes only the pending native objective/metric/entrypoint work above.
Actual optimization is NOT implemented or authorized. The `train_` filename
does not make this a working optimizer: `--dry-run` is required, `--max-steps`
must be0, and CPU is the only allowed device. Invalid execution arguments fail
before creating output/accessing feature artifacts. No PI0.5 reload, feature
extraction, source acquisition, ranking, policy execution or environment step.

Four new files, final SHA256 (all were sequentially copied and matched remotely
before execution):

```text
tools/pi05_libero_world_model_objectives.py
  0db64bdbe5ebdc3d2de06d9d649e88fdc1d9ff92f7a4644aa6ba01934c484400
tools/test_pi05_libero_world_model_objectives.py
  e9cca4183563fe2465fc493776fef00d091eccf0878590707c95f09d16925ee5
tools/train_pi05_libero_world_model.py
  1d7be62be55f4d5ceebe5a9ea57d3d13ab5c5720b1cc8d054055d8baa4ed9713
tools/test_pi05_libero_world_model_dry_run.py
  395dd195d6572a01205fb947e9453609b7fb0ad50fd5c935ed68949b5df8dba6
```

Existing native model/adapter, frozen extraction code and guidewire model/loss/
trainer are untouched. Run short CPU regression tests after later authorized
changes and synchronization; both local and remote128/128 passed for this
version (32 adapter +36 model +12 forward guards +35 objectives +13 dry-run
guards). No tests invoke backward or an optimizer.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q"
```

Fixed objective API (targets never enter `model(**inputs)`):

```python
from pi05_libero_world_model_objectives import (
    NativeWorldModelLossConfig, NativeWorldModelMetrics,
    collate_window_targets, native_world_model_loss, objective_contract,
)
# predictions: pred_future_visual_latent [B,1,H,2,D], pred_state_delta [B,1,H,8]
# targets: future_visual_latent [B,H,2,D], future_visual_valid bool [B,H,2],
#          state_delta [B,H,8], state_target_valid bool [B,H,8]
targets = collate_window_targets([window['targets']], device='cpu')
loss_config = NativeWorldModelLossConfig()  # visual1, normalized_state0.25, beta1
metrics = NativeWorldModelMetrics(verified_train_normalization['state_std'])
with torch.inference_mode():
    predictions = frozen_eval_model(**inputs)
    loss, details = native_world_model_loss(predictions, targets, loss_config)
    metrics.update(predictions, targets)
```

The visual denominator is valid horizon/view count timesD; the state denominator
is valid horizon/coordinate count. Each branch is independently reduced and
then weighted. `details` contains float32 `visual_loss/state_loss`, detached
float64 `visual_sum/state_sum`, and integer `visual_count/state_count`. Aggregate
these sums/counts per branch over all batches, not the per-batch loss means.
An unsupported branch contributes0 loss, but all positively weighted support
missing is rejected. Shape broadcasting, K>1, extra fields, numeric masks,
target gradients and observed nonfinite values are rejected. Invalid prediction
and target entries are cleared before subtraction, not multiplied by0 afterward.

MAE/RMSE use detached float64 errors and row-ordered accumulation; summaries
include count, absolute_error_sum and squared_error_sum. A missing metric is
null. Visual reports cover horizon/view; normalized state covers horizon/
coordinate; native state additionally uses per-coordinate train_std scaling
without a mean, including horizon/coordinate and same-unit group reports. No
native mixed-unit aggregate or SO(3) claim. Constant-valid std1 is a numerical
fallback; train-unsupported coordinates stay masked by the existing adapter.
Train-only provenance is verified by the runner, not inferred by the metrics
class. Float32 elementwise loss and float64 metric-error paths are intentionally
different, so bitwise equivalence between the two is not promised.

Historical completed remote CPU command (do not overwrite/rerun this result):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/train_pi05_libero_world_model.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --feature-report-sha256 b30e0a2bdca6c40306b69834f6d9339d434b180bec0e2c4776b11cba05d70632 --out simulation_output/pi05_libero_world_model_dry_run_remote_v1 --dry-run --max-steps 0 --device cpu"
```

The fixed source1400/1402 split, normalization140 state/139 action train rows,
random seed20260912, native model config and batch16 are unchanged. Source
report SHA,33 feature input artifacts and pinned extractor provenance are
checked before heavy model imports, and feature/code bytes rechecked afterward.
All134/167 windows completed in9/11 batches, last batches6/7, in0.924968s CPU.
Observed valid scalar counts: train visual1,646,592/state3,216; validation
visual2,052,096/state4,008. All model parameters frozen/eval/no gradients;
parameter versions unchanged and the before/after state-dict SHA remains
`691a7c14a7cef4c91f78f9a26a519586dacabbf85b9d1ab12d155c9eb4b573d0`,
matching the prior random forward diagnostic. No model checkpoint was saved.

```text
simulation_output/pi05_libero_world_model_dry_run_remote_v1/
  report.json + status.json + run.log
  model_contract.json + objective_contract.json
  dry_run_trace.jsonl              # 20 batches,301 windows: input/target/pred hashes
  first_train_audit.npz            # predictions + four targets for first16 windows
  first_validation_audit.npz       # same six arrays, first16 validation windows

report.json SHA256:
  942e3dbcf76c488a02475f3d916b7b2e73b402c7119293f36acbf05c5ad88600
dry_run_trace.jsonl SHA256:
  31a81bdf60702ece95b868009e94038b1e8c99c60c62a11dd47f587e469c4261
simulation_output/pi05_libero_world_model_dry_run_remote_v1_review.tar.gz
  2,971,434 bytes
  94af82ef34ca85f888d8a6ba2508a1aa65c4ce89d40ce3555eea631fd1f25166
```

The small archive was copied locally and verified before extraction; no large
data/images/checkpoints were transferred. Existing output is rejected; there
is no partial resume. On disconnect, inspect status/log/PID before any retry;
never start a duplicate completed job. A justified change requires a fresh
output directory and relevant source synchronization/hash checks.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_world_model_dry_run_remote_v1/status.json"
ssh project4090 "tail -n 24 /home/zsw/project_2026/simulation_output/pi05_libero_world_model_dry_run_remote_v1/run.log"
```

The later `postrun_review.json`, SHA256
`f2ef2b6a250641014c49bf930e7c491e60080175517d1fe9e5a68ed5c169c5bb`,
independently checked5 output/7 code/33 feature hashes; all301 metadata entries,
100 batched input and80 target tensor hashes;12 saved arrays; and166 recorded
metric-stat arithmetic/hierarchy/scale relationships. Both global objectives
were reproduced from20 batch branch sums/counts with difference0. Independent
NumPy first-batch losses match the two float32 trace losses exactly; float64
visual sums differ by4.55e-13/9.09e-13 from reduction order (rtol1e-12,
atol1e-10), without code/report changes. Only32 saved windows allow independent
complete error recomputation; remaining269 predicted arrays were not saved or
replayed. The review did not rehash unsaved model parameters or raw sources/
weights. This added sidecar is not part of the immutable original run report's
output map or original archive; preserve their original bytes.

Metrics are `diagnostic_untrained_metrics` only. The frozen random model has
not learned dynamics;301 overlapping windows from two episodes/one task are
not independent generalization evidence. No full-source/weight reread,
backward/gradient test, optimizer, checkpointing, selection or model-quality
study has happened. `training_ready=false` and source/feature/training gates
remain separate; a future quality study needs an adequate prospectively frozen
public source/task split and explicit optimization authorization. No weekly
entry, data/SOFA edit, guidewire/contact/tactile or real-system conclusion.

## Native Persistence Reference And Frozen Diagnostic Protocol (2026-09-12)

This is a no-training baseline. Final versions were sequentially SCP-copied
and remote-SHA checked before execution. No existing model/adapter/objective/
trainer or data/SOFA files changed.

```text
tools/pi05_libero_world_model_persistence.py
  19908c48609cafab13a495f44eeb13166adaa0d4a42350b64c0cc03c7ba08c8b
tools/test_pi05_libero_world_model_persistence.py
  9ea55676ff625d77dc0fea4e0c8d175ade8d69d5ae25bc7676fc79d29e6bd027
tools/eval_pi05_libero_world_model_persistence.py
  0ae7076e14ace968be537132ff202c6d306e4e8c6591fa9c4c80cef56dd02c78
tools/test_pi05_libero_world_model_persistence_eval.py
  71daf296c5d9e1f39980631bf6974d71f3f8151346f65b6952c0b42044a669ec
docs/libero-native-world-model-learning-protocol-v1.json
  5d1b67c75c03dcaeba6c53bdf7c0f8dc14d2fc5bb4d4e0d93f5c63c2ce58c2ff
```

The protocol JSON is byte-pinned in the evaluation runner and also requires an
external matching SHA. Changing the design requires an explicit new version,
not silent editing of v1. It authorizes no optimization; the future200-step
training section is a frozen design with `execution_authorized=false` and
`implementation_status=not_implemented`. The existing dry-run-only trainer is
unchanged. Do not try to remove `--dry-run` to start actual training.

Regression suite:163/163 locally and remotely (previous128 +23 persistence
and12 evaluation guards), no backward/optimizer/model training. Local2.309s and
remote0.182s are test-body times, not performance-comparison results.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q"
```

Historical completed command (do not overwrite or rerun this completed result):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/eval_pi05_libero_world_model_persistence.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --random-reference simulation_output/pi05_libero_world_model_dry_run_remote_v1 --protocol docs/libero-native-world-model-learning-protocol-v1.json --protocol-sha256 5d1b67c75c03dcaeba6c53bdf7c0f8dc14d2fc5bb4d4e0d93f5c63c2ce58c2ff --out simulation_output/pi05_libero_persistence_remote_v1"
```

Native helper APIs:

```python
predictions = persistence_predictions(inputs)  # last visual repeated; state residual0
common_targets = persistence_scoring_targets(inputs, targets)
# Use the unchanged native objective and NativeWorldModelMetrics separately.
```

Both helpers reject non-native/extra fields and K>1, validate bool masks and
finite declared-valid input/target values, own returned storage and track no
gradients. No older-valid fallback or action/task dependency is added. The
common mask intersects each original target mask with the latest observed
validity. For this fixed pair it is elementwise identical to the original;
otherwise this runner refuses to reuse the old random metrics. All20 batches'
input/target hashes and metadata must match the old942e... trace exactly.
There is no learned-model construction/forward, feature extraction or action.

The run completed in0.807216s CPU, all134 train/167 validation windows in9/11
batches. Per-branch objectives use global sums/counts; MAE/RMSE retain horizon,
view, state-coordinate and same-unit-group detail. The comparison is
`persistence - random_untrained`; relative change is null for zero reference
or unsupported values. Validation state-normalized MAE0.0875567724 versus
0.1612492578 (-45.70%); visual latent MAE0.0891759570 versus0.1513705829
(-41.09%). These are descriptive development-pair references, not a trained
model improvement or evidence that actions are unnecessary.

```text
simulation_output/pi05_libero_persistence_remote_v1/
  report.json + status.json + run.log
  persistence_contract.json
  persistence_trace.jsonl
report.json SHA256:
  6faf0c0198d2e2940c1aff0ed84f6d0f91bad6d456a28d7d6ba6c41792036db3
simulation_output/pi05_libero_persistence_remote_v1_review.tar.gz
  47,177 bytes
  80d68cf1ca782871ed9399c0bd37dcea17fc4319d40e87a846184b31d9ab5793
```

All small artifacts were mirrored locally after SHA/path checks. Existing
output directories are rejected and there is no partial resume; inspect the
status/log after disconnect rather than launch a duplicate. A justified fresh
attempt needs a fresh output name and final-code synchronization/hash check.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_persistence_remote_v1/status.json"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_persistence_remote_v1/run.log"
```

Future training parameters and decision boundaries are in the byte-pinned
protocol, not a runnable training command: only the existing native model,
same initial hash/seed, CPU float32, AdamW0.001/wd0, batch16, exactly200 updates,
fixed separate-generator replacement samples, global clip1, no scheduler/AMP/
accumulation, final-only checkpoint and train/validation evaluation at0/200.
The planned200x16 integer sequence SHA
`75754609fbf17853c47700c0d5a63e044c9c7af7a8a1eb79d685e70e61e1c839`
matched both local and remote Torch; generating that plan performs no learning.
At this milestone gradient/checkpoint/reload support and execution were pending;
the next section records their implementation and the zero-update failure.
An independent local NumPy pass checked all301 windows without the persistence
predictor or metrics class:16 aggregate/per-horizon visual and normalized-state
nodes plus32 comparison values matched exactly (rtol1e-12/atol1e-10). It also
verified report/2 output/9 code hashes. It did not rerun the random model,
reconstruct the full trace again, or independently check every native-coordinate
metric. This bounded read-only review changed no report or source file.
The current validation episode has already been inspected; no significance,
generalization or robustness claim. A larger known-task/whole-episode source
split and repeated-seed protocol must be frozen independently before quality
runs; current closed-set embedding does not support novel-task language tests.

## Native Learning Attempt, Strict Step0 Failure And Repaired Preflight (2026-09-12)

Status at this recovery gate: training support implemented; the first authorized attempt
failed before optimizer construction at step0. After the evaluation-only repair,
short tests and full step0 preflight passed. No actual200-update run has completed.
The original one-attempt authorization/output is consumed; do not delete,
overwrite, resume or automatically retry it. A fresh attempt requires explicit
approval and a new authorization/output identity, not a changed frozen protocol.

Final synchronized source SHA256:

```text
tools/pi05_libero_world_model_training.py
  6157fa69b0092e13497f47ec5cc937934d652dd4dd17a107f445b34308bf2e7b
tools/test_pi05_libero_world_model_training.py
  10b156293804f4a1c4abcfa140f67c72292cfe0c7ce68186febd2e038bb9441d
tools/run_pi05_libero_world_model_learning_diagnostic.py
  7eb180cdfbef92f036d0c132dceaccac86683ac3335839be899595892fc3da97
tools/test_pi05_libero_world_model_learning_diagnostic.py
  f981695f72a65625b9ffbd2e300f1f207f70955a3727b0712cba8cec88160732
tools/probe_pi05_libero_step0_reproducibility.py
  3da5d2e8a0b57373bd6b2e477336873918c4cbb7719e39ccb95612318c8634a6
tools/verify_pi05_libero_learning_preflight.py
  d509926bc257b55dc2d1426547c2d935f4d6b3c31bfac97499e9d6ea9ac31ce5
docs/libero-native-world-model-learning-authorization-v1.json
  8bfe78322c44f7e4959287a9f53a920b0591c3df975b85c3b8b02d4f4f209511
```

The frozen v1 protocol remains5d1b67...c2ff and the old model/adapter/objective/
dry-run code are unchanged. The utility checks all gradients, exact fixed AdamW
and step counters, global clip1, changed finite parameters and final-only safe
checkpoint/reload. Synthetic tests exercise these mechanics; their checkpoint
fixtures must not be described as a completed200-update experiment.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p 'test_pi05_libero_world_model*.py' -q"
```

Latest200/200 passed local3.021s / remote0.884s (test bodies, not speed benchmark).
These short synthetic checks are not full-data training. New tests verify
temporary frozen evaluation, exact reference predictions and per-parameter
gradient flags/module-mode restoration on both success and failure.

Historical failed command, recorded for provenance only. **Do not run again**:
its existing output is rejected; the current runner includes the later repair.
The exact original3138a655... runner and four companion files are retained in
`simulation_output/pi05_libero_learning_diagnostic_failed_v1_code/` on both hosts.

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_world_model_learning_diagnostic.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --random-reference simulation_output/pi05_libero_world_model_dry_run_remote_v1 --persistence-reference simulation_output/pi05_libero_persistence_remote_v1 --protocol docs/libero-native-world-model-learning-protocol-v1.json --protocol-sha256 5d1b67c75c03dcaeba6c53bdf7c0f8dc14d2fc5bb4d4e0d93f5c63c2ce58c2ff --authorization docs/libero-native-world-model-learning-authorization-v1.json --authorization-sha256 8bfe78322c44f7e4959287a9f53a920b0591c3df975b85c3b8b02d4f4f209511 --out simulation_output/pi05_libero_learning_diagnostic_remote_v1 --execute-authorized-diagnostic"
```

Preserved failure: `training_started=false`, `completed_updates=0`,
`last_attempted_step=0`, no uncertain update and `automatic_retry_allowed=false`.
The step0 trace is empty; there is no checkpoint, training trace or result report.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_learning_diagnostic_remote_v1/status.json"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_learning_diagnostic_remote_v1/run.log"
```

Pure-forward factor probe (completed, no training; output must not be reused):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/probe_pi05_libero_step0_reproducibility.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --random-reference simulation_output/pi05_libero_world_model_dry_run_remote_v1 --failed-attempt simulation_output/pi05_libero_learning_diagnostic_remote_v1 --protocol docs/libero-native-world-model-learning-protocol-v1.json --out simulation_output/pi05_libero_step0_reproducibility_remote_v1"
```

All8 forward-configuration combinations were reported with two repeats on both
saved first batches. Only parameter `requires_grad` determined strict hash
reproduction in this fixed runtime. With ordinary tensors/deterministic=True,
unfrozen matches0/20 batches and frozen matches20/20. First-batch visual maximum
absolute difference1.9073486328125e-6; state difference0 is only a first-batch
observation, since the last7 validation windows also have changed state hashes.
No claim about the specific underlying kernel was verified. Probe code records
the failed runner hash; retain the original snapshot even though the current
runner has been repaired. It does not loosen the production gate.

Repair preflight (completed, also no training; directly calls repaired
`evaluate()`, not `run()` and never constructs an optimizer):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/verify_pi05_libero_learning_preflight.py --feature-pack simulation_output/pi05_libero_feature_pair_v1 --random-reference simulation_output/pi05_libero_world_model_dry_run_remote_v1 --protocol docs/libero-native-world-model-learning-protocol-v1.json --out simulation_output/pi05_libero_learning_preflight_remote_v1"
```

All20 batches/301 windows reproduce exact prediction SHAs and every historical
metric/objective node within the unchanged rtol1e-12/atol1e-10. Parameter hashes,
incoming training/gradient flags and ordinary input construction are preserved.

```text
simulation_output/pi05_libero_learning_diagnostic_remote_v1/status.json
  237f9f81a6c5736b2a643a5abf66006585c9a2803689e8664317be2d151463e7
simulation_output/pi05_libero_step0_reproducibility_remote_v1/report.json
  420d2dfd697b62ee7f933ce50386bb07197d5e095d359f4c315bf4acabc21108
simulation_output/pi05_libero_learning_preflight_remote_v1/report.json
  df0639473d1bcc7bc85de7c0c5c46d929ff2d24de8bd3cd2d156d2d239d3cc6d
```

All are small artifacts mirrored locally and SHA-verified; no large transfer,
new source acquisition, feature extraction, PI0.5 load or policy action occurred.
No weekly/data/SOFA/shared-contract change is required for this numeric repair.
After a fresh-attempt approval, preserve the frozen200x16 sampler and all model/
loss/optimizer/evaluation settings; bind the repaired code and preflight evidence
to the new attempt, and do not interpret preflight success as learned-model quality.

## Explicit Fresh Native200 Attempt Completed (2026-09-12)

The user's subsequent explicit approval authorized one new output under the
same frozen numerical protocol. The v1 failure, authorization and code snapshot
remain unchanged. No further run is implied by this command section.

```text
docs/libero-native-world-model-learning-authorization-v2.json
  05bc0012cce953d2c1d0a40020ca2de9f9b973553e21497a35bc49637cfbda66
tools/run_pi05_libero_world_model_learning_retry.py
  70747154bdcf4d7d476ecc1f0ac33b6ad9bee97758dc6279dcd12fae9975c068
tools/test_pi05_libero_world_model_learning_retry.py
  2be616d04f841b0141fd6304a5cbf72b6c6fadbc35207a86cee1695f9bc9c734
```

The wrapper verifies19 bound files before importing the repaired engine. It
temporarily changes only `engine.AUTHORIZATION_SHA` to the independently
hard-pinned v2 identity, forwards fixed arguments to `engine.main` once, then
restores the original pin on success/error. It does not change engine bytes,
model/loss, steps, seeds, sampling plan, optimizer or evaluation thresholds.
Runtime authorization identity is intentionally different. The sibling
invocation JSON captures that change and uses exclusive creation to prevent
repeated startup, even if no output directory was created.

After sequential SCP and remote hash verification, the same regression command
in the previous section passed206/206 locally (3.649s) and remotely (0.925s).
Six new tests use a mocked engine, not an actual200-step learning run.

Historical **completed** execution command; do not run again or remove its
existing output/invocation file to bypass no-overwrite/no-retry protection:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/run_pi05_libero_world_model_learning_retry.py --authorization-sha256 05bc0012cce953d2c1d0a40020ca2de9f9b973553e21497a35bc49637cfbda66 --execute-authorized-diagnostic"
```

Status/log inspection is read-only and safe after a disconnect. This runner
has no partial resume. Do not dispatch another copy while checking status.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_learning_diagnostic_remote_v2/status.json"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_learning_diagnostic_remote_v2/run.log"
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_learning_diagnostic_remote_v2_invocation.json"
```

Actual outcome:200 backward/optimizer updates, fixed sampler757546...839,
all20 step0 prediction hashes reproduced exactly, final checkpoint safely
reloaded with unchanged parameter hash and first-batch predictions, all301
fixed windows evaluated at0/200. Engine total10.534791s; training8.844469s;
save/reload verification0.113754s. The final checkpoint binds the v2 authorization,
not the old v1 approval. Numeric settings remain CPU/Torch2.10.0+cu128/one thread.

```text
simulation_output/pi05_libero_learning_diagnostic_remote_v2/
  report.json, status.json, run.log
  sampling_plan.json, training_trace.jsonl                  # exactly200 records
  evaluation_step000.json, evaluation_step000_trace.jsonl  # full301,20 batches
  evaluation_step200.json, evaluation_step200_trace.jsonl  # full301,20 batches
  final_step200.pt                                         # about16MB; remote only
  final_train_audit.npz, final_validation_audit.npz          # first16 per split; remote only
simulation_output/pi05_libero_learning_diagnostic_remote_v2_invocation.json
  separate wrapper sidecar, not in original engine report output map

report.json SHA256:
  b4e13789c3e1865e9098845506c133d340de30277a4e6fc69dc20f7cd37de02e
invocation SHA256:
  51ee18796d9b3d4b6d4093a2e215e43d6246d91dc0687171abc23c1906a9626f
final_step200.pt SHA256 (also independently rehashed remotely):
  014feaa6f7ec0392d698e84fbe5cfbc4e4e1eecb2eda5aa2efa2c316ecbe45b8
```

The full remote archive is roughly17MB with SHA
`d316d7b930cd996241d3888dd7ff7da2cde9c5f854f1b0d9bff78d118760c8a7`.
Its slow SCP was stopped without touching remote results; the incomplete local
`simulation_output/pi05_libero_learning_diagnostic_remote_v2_review.tar.gz.partial`
is **not a verified backup**. To avoid waiting for weights/NPZ transfer, a
metadata-only archive was created with these files excluded:

```text
simulation_output/pi05_libero_learning_diagnostic_remote_v2_metadata_review.tar.gz
  170,306 bytes
  75d21bb3d33d4b3171cd306148a3a77339f3674a8e3fa9f7b4ebeedea94888bb
```

That small archive was copied, SHA/path-checked and extracted locally. Its
JSON/JSONL/log files and invocation were verified; checkpoint/NPZ are not local.
Do not run a local audit that assumes all output-hash-map files were downloaded.
The full archive remains available remotely if a later authorized review needs
those arrays. Transfer rules still assign any single file/archive over100MB to
the user; this narrower retrieval was chosen for poor SSH throughput.

Validation normalized-state MAE0.065965176 versus persistence0.087556772
(-24.66%), but visual MAE0.099909134 versus0.089175957 (+12.04%) and combined
objective0.013334021 versus0.013138943 (+1.48%). Both branches improve versus
the random model. This confirms the learning/checkpoint mechanics and a limited
state-prediction signal, not an overall persistence win, action causality,
policy improvement, robustness or real-system validity. One task/two episodes/
single seed and already-inspected validation remain hard evidence limits.
No continuation or parameter tuning is authorized by completion; prospective
independent episode selection and matched-budget ablations are separate work.

The independent metadata-only review verified6 local output files,11 code
hashes,19 retry-bound evidence files, all200 rows/3200 sampler draws and both
20-batch trace input/target fingerprints. It reconstructed global objectives
and4500 arithmetic checks across332 metric-stat/668 comparison nodes within
rtol1e-12/atol1e-10 (max absolute difference7.275957614183426e-12). Trace global
gradient norms were0.0034952064–0.0563131439, below clip1 throughout; each row
records23 parameters with gradients, not independently saved gradient arrays.
No PT/NPZ was available locally, no optimizer/model forward was rerun, and no
review file was written. Checkpoint byte verification was remote SHA/root
evidence and its internal counters/reload correctness remain engine guard
evidence. Preserve that distinction when reporting completion.

## Prospective Action Ablation And Cached Metadata Projection (2026-09-12)

Preparation only; **not a new training command or source-download approval**.
The fixed native200/v2 run is complete and must not be continued. New files:

```text
docs/libero-action-ablation-study-plan-v1.json
  210ff8bbb75e3316620590cda9e5067f7447125acddb53ae7fe751bea01810ac
docs/libero-action-ablation-study-plan-v1.md
tools/plan_pi05_libero_action_ablation.py
  51457b29735ee5aef46649b367ab9fc3572a283967a3507c3ecf4a229887ac2f
tools/test_pi05_libero_action_study_plan.py
  30f3c688049926b92d20556fcfa1876a488fe7347f1e93f7c5bc85f890f8aeae
tools/pi05_libero_action_ablation.py
  8614339c8e7b43e9a0f46ddfd71dd8c776e2a78e391426cf8b6e2b63269aab83
tools/test_pi05_libero_action_ablation.py
  7e99c1520cfc86cb4daea0c0b357ac5919cf8ad26c67ffa31c9bd602a179672d
```

The new helper is for a future multiepisode trainer. It is not wired into the
old fixed-pair runner and adding an unsupported flag there will not run this
ablation. Call outside inference mode, on ordinary detached native CPU inputs:

```python
from pi05_libero_action_ablation import action_ablation_inputs
observed_inputs = action_ablation_inputs(inputs, "observed_action")
zero_inputs = action_ablation_inputs(inputs, "normalized_zero_action")
# Do not pass targets or metadata; keep them unchanged outside the model.
# Use the same arm in future training and evaluation, after shared train normalization.
```

Normalized-zero means the constant train action mean, not physical zero/hold.
Original nonfinite actions are rejected before zeroing; all other inputs and
masks are owned copies. Synthetic tests do not establish predictive quality.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -m unittest discover -s tools -p 'test_pi05_libero_action*.py' -q"
```

41/41 passed locally/remotely:20 new action-input tests,9 new metadata selector
tests,12 old synthetic action-boundary tests. No new-study backward/optimizer
or dataset read in this suite. The206 old world-model tests separately passed
locally and include their existing synthetic optimization guards.

Historical completed metadata-only projection (returns a successful diagnostic
report with a blocked selection status; do not overwrite that output):

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/bin/conda run --no-capture-output -n project2026-pi python -u tools/plan_pi05_libero_action_ablation.py --source-root simulation_output/libero_public_source_audit_v1 --plan docs/libero-action-ablation-study-plan-v1.json --out simulation_output/libero_action_study_cached_metadata_v1"
```

Source report and metadata/data330 bytes are hash-pinned. The planner projects
only episode/task/frame/index/timestamp columns, not state/action/reward/image
content. Remote PyArrow is already installed; local `.venv` lacks a Parquet
reader, so do not try to run the actual cache scan locally or install LeRobot.
No new download or feature extraction is performed by this command.

```text
status: blocked_incomplete_task_metadata
declared/parsed metadata episodes: 1693/1693
complete task-known episodes: 6 (1399–1404)
missing task identities: 1687, missing projection shards: 376
new eligible task9/source39 known episodes after exclusions: 0
selection: episodes=[], split=null
report remote: simulation_output/libero_action_study_cached_metadata_v1/report.json
report local:  simulation_output/libero_action_study_cached_metadata_v1_report.json
SHA256: 56f090d20e7c4002574c217aa289b02918d0c6b131c797d338043e84ba5e1898
```

The recipe requires complete task coverage before hash-rank selection of12 new
whole episodes into8/4. It never chooses from the cached subset, uses no quality
filter, and refuses all-or-reject row-budget/duplicate failures rather than
truncating/replacing IDs. Full source/group/video-interval audits remain pending;
ID separation does not certify family independence or unseen PI0.5 pretraining.

Next step is a separately scoped metadata-index acquisition entrypoint with
pinned revision/file identities and a total byte budget, not videos/weights or
training. The376 required paths are in the report; their sizes have not been
audited. Be explicit if full Parquet bytes are transferred although only five
columns are interpreted. Above100MB per file/archive remains user transfer.
Once the complete task inventory is verified, freeze actual IDs, then prepare
and audit complete sources/features. Future3seeds×2arms×200steps remain a
design only, with no multiepisode trainer/source payload or execution permission.

## Bounded LIBERO Task-Index Acquisition (2026-09-12)

The user separately approved completing public task-index metadata and its
transfer audit, not videos/features/training. The old study JSON and selector
stay byte-identical; this new scope is recorded separately:

```text
docs/libero-task-index-acquisition-authorization-v1.json
  94fd464a73f338fe84e719ce7016b47f4865d9a11c64fdfab6d1b5e85d3bd016
tools/acquire_libero_task_index.py
```

The pinned Hub inventory b8e421be46b1dc29de7e7d01a37eb934f187d8a0a1f203952739be4572221108
lists376 required new Parquet shards:20,208,168 bytes total,65,531 bytes maximum.
Their immutable revision and LFS SHA256 are required before requests start.
Four workers, two transport attempts per file/invocation,10s connect/30s read
timeouts,50,000,000-byte cumulative conservative request reservations (including
failed/interrupted attempts). Existing valid files are rehashed and reused;
corrupt cache is rejected, not overwritten. `.part` files are retained on error.
Actual received payload bytes are also reported; HTTP/TLS overhead is not included.
Whole Parquet files cross the network, but only five episode/task/frame/global
index/timestamp columns are interpreted. No action/state analysis, video/image
decode, feature extraction, weight load, optimizer or rollout is in this tool.

After final sequential SCP and remote SHA verification, the read-only budget
command is:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/acquire_libero_task_index.py --stage budget --source-root simulation_output/libero_public_source_audit_v1 --cached-report simulation_output/libero_action_study_cached_metadata_v1/report.json --plan docs/libero-action-ablation-study-plan-v1.json --authorization docs/libero-task-index-acquisition-authorization-v1.json --authorization-sha256 94fd464a73f338fe84e719ce7016b47f4865d9a11c64fdfab6d1b5e85d3bd016 --out /home/zsw/project_2026/simulation_output/libero_action_study_task_index_v1"
```

The corresponding separately authorized metadata acquisition command (do not
repeat if the output already exists/completed) is:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/acquire_libero_task_index.py --stage acquire --execute-index-acquisition --source-root simulation_output/libero_public_source_audit_v1 --cached-report simulation_output/libero_action_study_cached_metadata_v1/report.json --plan docs/libero-action-ablation-study-plan-v1.json --authorization docs/libero-task-index-acquisition-authorization-v1.json --authorization-sha256 94fd464a73f338fe84e719ce7016b47f4865d9a11c64fdfab6d1b5e85d3bd016 --out /home/zsw/project_2026/simulation_output/libero_action_study_task_index_v1"
```

After a disconnect first inspect status and its PID; never dispatch a second
copy while one may be running:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/libero_action_study_task_index_v1/status.json"
```

Explicit `--resume` appended to the same acquisition command is permitted only
for `status=failed`, identical code/authorization/plan/source hashes, and no
published task_inventory/selection/selected_payload_inventory/report. It rehashes
complete files and retains the conservative budget ledger; it is not permission
to add budget, change IDs, fix corrupt cache silently, or start training.
`status=running` after a hard interruption requires process/state review, not a
manual status edit. Partial finalization is deliberately non-resumable; preserve
all files for review instead of deleting them or changing the output name.

Only full1693-episode/273465-row index coverage may invoke the original frozen
12→8/4 selector. Checks include global interval continuity, whole-episode frame
and timestamp consistency, fixed task identity and metadata-declared shard
ownership. Selected video intervals and deduplicated file size/SHA inventory are
reported, not downloaded. A future file above100MB is marked user-transfer;
that inventory warning does not alter the selected IDs. Source duplicate/family
audit, video integrity, feature preparation and training remain separate gates.

Execution completed; the acquisition command above is now historical. First
invocation used relative `--out` and failed in attempt-log path formatting after
four verified files were published. Failure status, four reservations, empty
attempt log and executed source were preserved as `first_attempt_*` and
`executed_acquire_libero_task_index.py` inside the output. The same-code explicit
resume with absolute `--out` completed in264.512066s, downloading372 and reusing4.
No selected IDs or source bytes were replaced. Two0-byte SSL errors at099/193
were successfully retried within the two-attempt limit.

```text
original report SHA256:
  ff4be0d04a5a79282b59307a646932f7756ceb0412e9b4e7c4392cabaa3aa781
selection SHA256:
  755e6c4684fa127eb6adbd02d27faa5700069acf4de5c7ab164ef97e90c65692
executed code SHA256 (archived inside output):
  b2a5b055341c0c219302c2158f227e13139e18290122121a203b062049f38505
current tool SHA256 (one-line absolute-path normalization repair, not rerun):
  dbe9fd1a9de1054e7924baec3cc79ac488a01dfc6e6ae6aae97801650cfd85ac
test_acquire_libero_task_index.py SHA256:
  12fd32cc9d4e7e6b229af5a7726a3022cc01138d112bef273b7585f93c1bdca0
```

Current30 acquisition tests passed locally/remotely, including the new relative
cwd/output regression. Frozen selector9 tests also passed on both. Do not
attempt to resume the completed report with the later fixed code: its identity
intentionally binds the archived executed version.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_acquire_libero_task_index.py -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest discover -s tools -p test_acquire_libero_task_index.py -q"
```

Final metadata coverage1693 episodes/273465 rows, task39 count44, eligible42,
no missing identities. Frozen selection (do not replace after payload inspection):

```text
train:      1633,1674,1419,1312,1518,1520,1531,1690    1134 complete rows
validation: 1530,1476,1458,1566                       524 complete rows
```

Original report `cumulative_response_payload_bytes=19997025` counts the resume
attempt log only, because the initial log was empty. First failure status plus
four verified published files account for211143 additional bytes: reconstructed
total20208168. `conservative_request_reserved_bytes=20311019` additionally
includes102851 reserved for two0-byte failed requests. Keep these distinct and
preserve the original report. No successful transfer is repeated.

Local review archive115005B is SHA4569049c8a640d6e7f48af2f2cd22814d8415b360e4c7a44614e704106fd1df8;
it contains12 JSON/JSONL/Python metadata members, hash/path-checked before local
extraction. It excludes source Parquet and video payloads; do not claim a full
local source cache or run a full-source hash audit locally against this subset.

The selected video inventory totals14 files/273418911B. Two prior file030
videos were independently rehashed remotely against the old pinned source
report (43064421B total); the future missing inventory is12 files/230354490B.
None is individually above100MB, but a prepared combined archive would exceed
the100MB user-transfer rule. No new video was downloaded. Separate source
acquisition/auditing authorization is still required, followed by full video/
duplicate/family and new-versus-old development-overlap checks before features.
Metadata selection is not training readiness or verified family independence.

The companion audit below completed on the remote full cache (read-only source
access; exclusive new output). It verifies every new Parquet file, old cached
videos, pinned report/output/source hashes, two independent selection calculations
and the recovery byte accounting. It does not decode any Parquet columns or
images. The two empty `.part` files match the recorded zero-byte SSL failures.
Do not rerun against the same existing audit output.

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/audit_libero_task_index_result.py --root simulation_output/libero_action_study_task_index_v1 --source-root simulation_output/libero_public_source_audit_v1 --plan docs/libero-action-ablation-study-plan-v1.json --out simulation_output/libero_action_study_task_index_v1_recovery_audit.json"
```

```text
audit tool SHA256:
  f2de93a36f3c37a07038a05332fcb96d7d14c3d2b8bf30e0793c21b9dbfc57bb
audit output SHA256 (remote and local):
  31627b9234522812eac6d86f69119e4326d8829e92d12cb5ee2f9b1a8995bda8
status: passed_metadata_and_recovery_accounting_only
source_quality_audit_passed=false, training_ready=false, optimizer_steps=0
```

The completed acquisition's archived code is deliberately distinct from the
current repaired downloader; the audit verifies the archived b2a5...8505 bytes.
The original selector SHA51457b29735ee5aef46649b367ab9fc3572a283967a3507c3ecf4a229887ac2f
was separately checked locally/remotely. The audit uses the full remote cache;
local metadata readback alone cannot reproduce its entity-file hash checks.

## Fixed12 Plus Old Development Source Audit (2026-09-13)

Approved scope:12 missing videos (230,354,490B) plus integrity/duplicate auditing
of the frozen12 new trajectories and old1400/1402. No feature extraction,
normalization, sampler fitting, model/loss changes, training or rollout. The
source/selection/authorization and existing helper code are hash-bound before
requests. No large artifact is copied from Windows: the user-run command fetches
the12 original under100MB video shards directly from their pinned Hub revision.
Two old cached file030 videos are verified/reused, not downloaded again.

```text
docs/libero-action-study-source-authorization-v1.json
  792ace5e0df13f826154ff3a66330331fff3a532619daf6df7e64e695a54c0d6
tools/prepare_libero_action_study_sources.py
  5038e2ee27baef7062c3c5a1d90c4b6d298e35cf0ab6102c048a0163a1360a65
tools/libero_action_study_source_checks.py
  a7c83ca24a45bb7afcba50e946c678e2191a5d759094e67a5aafb5ca36d35c9f
tools/test_prepare_libero_action_study_sources.py
  f366cb345cec4232af03658d7dc7cb43861f5d00f138ad0293872731cecaeeac
tools/test_libero_action_study_source_checks.py
  abf8ee632e0051aa1d510fc53f098bb9ecf730e4d21fd58f5833e2cdc3caa19d
```

Offline preflight (hash/schema/dependency checks only, no source row/image
interpretation or network). Without `--preflight-out` this is safe to repeat;
specifying an existing preflight output is rejected rather than overwritten:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/prepare_libero_action_study_sources.py --stage preflight"
```

The first preflight was executed with
`--preflight-out simulation_output/libero_action_study_sources_preflight_v1.json`.
It passed, confirmed the11 native Parquet schemas and all source/code hashes,
and found the full-run output directory absent. Installed versions:
av15.1.0, pyarrow25.0.0, numpy2.2.6, Pillow12.3.0, requests2.34.2.
No dependency installation, downloads, action/state interpretation, images,
features or optimizer updates occurred in preflight.

Preflight SHA256 (remote/local readback):
`830b147ce7ac53f38063dbde2365109dbe5bbc2fed225ab9b27e2de4d9240248`.
All85 tests in the four-file suite below passed locally0.605s/remotely0.310s:
55 new guards plus30 unchanged native decoder/row tests. Mock execution pass/
rejection tests are not results on the actual source trajectories. The full
job output was absent after these checks.

Completed user-run command from Windows PowerShell (historical; do not rerun):

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/prepare_libero_action_study_sources.py --stage run --execute-authorized-source-audit"
```

It is also fine to enter the command after `cd /home/zsw/project_2026` in an
existing remote terminal. The whole job is assigned to the user under the runtime
rule: download variability plus roughly100k underlying prefix-decoded frames can
exceed five minutes. It does not split the computation into many sub-five-minute
jobs. Raw selected RGB count is3942, not the total codec work; expected generated
images.npy bytes are about775MB and should remain remote. Do not pack or SCP
those arrays as an over100MB archive; only later small reports/representative
PNGs should be retrieved by the agent.

Fixed output (never overwrite or create replacement IDs after inspection):

```text
simulation_output/libero_action_study_sources_v1/
  identity.json, preflight.json, status.json
  active_invocation.lock                         # only during an active call
  budget_reservations.jsonl, download_attempts.jsonl
  source/videos/...                             # only12 new MP4 files
  download_report.json
  episodes/episode_<id>/records.json, images.npy, decoded_frames.json
  episodes/episode_<id>/view<0|1>_frame<...>.png   # first/middle/last evidence
  duplicate_audit.json                           # all91 pair evidence/groups
  report.json
```

Exit0/report `passed_declared_source_checks` means only the declared technical
checks passed. `training_ready=false` and family/pretraining independence remain
unknown. Exit3/`completed_rejected` preserves a finished negative source audit;
inspect blocking pairs and do not replace IDs. Other failures preserve files
and phase in status. No automatic feature/training launch is reachable.

After a disconnect, inspect status and the listed PID before doing anything:

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/libero_action_study_sources_v1/status.json"
```

Only `status=failed`, `phase=download`, no active lock, same identity/code/input
hashes, complete byte accounting and no download_report/episodes/report permit
appending `--resume` to the full command. It rehashes/reuses verified videos and
preserves prior failure status and reservations; failed attempts never release
their full-file reservation. Two workers, at most2 attempts per file/invocation,
500MB cumulative conservative budget and100MB single-file cap remain enforced.
Corrupt caches fail closed, not overwritten. A running PID, hard-kill lock,
source_audit failure or partial finalization requires review; do not remove locks,
edit status, change code/output or delete artifacts to force another invocation.

Synthetic verification command (four files; no actual network/Parquet/video/model):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_*libero*source*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest discover -s tools -p 'test_*libero*source*.py' -q"
```

The full job was user-run and has now completed; the agent did not launch or
repeat it. Completed-result checks and representative-image review are recorded
below. No group is called an independent family just because its exact/quantized
windows differ. Leave the old native200, index and source reports unchanged.

### Completed Source Result Readback (2026-09-13)

Source report `passed_declared_source_checks`; final status `completed`.
Report runtime34.288926s, final-status runtime34.690626s. Fixed train8/1134 rows
and validation4/524 rows are unchanged. Old1400/1402 add313 audit-only rows;
total14 episodes/1971 rows/3942 selected viewframes. Twelve new video downloads
succeeded on the first attempt (230354490B actual and reserved each); no retry,
failure, residual partial or active lock. Two old cached videos were reused.
All91 pairs were recomputed;0 detected exact/quantized/two-view window matches,
0 blocking/grouping pairs,12 singleton descriptive groups (8/4), not families.

```text
source report SHA256:
  20d34377c636469e2d06a61cc7519db6fb0ee6c33379d9631b058870d101ef5c
duplicate_audit.json SHA256:
  0dafcedc14abc892145207b0ec17e1ba47aed7f244f8f41b039647d4040da6b4
tools/audit_libero_action_study_result.py SHA256:
  a7bde99214e77accecf44e7fdea8493e0207ddec2f1d0dd9b27074841949b63b
simulation_output/libero_action_study_sources_readback_v1.json SHA256:
  4032bf07578aba2502fa29c40eab2467539ee2ddfae9ae5c8deb7ffbd1f04eb7
```

The completed short companion invocation used the command below. It writes fresh
siblings only, never the source directory; do not repeat these output arguments.
It rehashes132 outputs/23 inputs/8 code files and all new videos; compares1971
source rows,3942 RGB hashes/PTS and84 representative PNGs; recomputes all91 pairs.
Observed readback1.396047s. No new video decode, features or optimizer updates.

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/audit_libero_action_study_result.py --out simulation_output/libero_action_study_sources_readback_v1.json --review-bundle simulation_output/libero_action_study_sources_review_v1.tar.gz"
```

For a future read-only recheck, omit both output options; this prints evidence
without overwriting the completed readback or source files. It requires the full
remote cache, not the local metadata-only subset:

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/audit_libero_action_study_result.py"
```

The retrieved review archive is4,480,961B, SHA
`c841fe44983cf051fd2eed1018e83f53c3fedc1c6d590510b01464d6f92ad433`.
It contains120 regular JSON/JSONL/PNG members and excludes MP4/NPY/Parquet.
Local member types/paths and SHA were verified before extraction to
`simulation_output/libero_action_study_sources_review_v1/`; present original
report-listed file hashes were then rechecked. Do not call this a full local
source cache or transfer the approximately775MB raw arrays as an agent archive.

Local Chinese first/middle/last sheets (84 representative PNGs, agent viewed):

```text
simulation_output/libero_action_study_sources_visual_review_v1/
  build_review.py                 # one-shot local extraction/render provenance
  representative_frames_1.png    # SHA33408bf3cb0ae642465b8ff50b5d9571fdc4149a2e0f79f554ee8a3967699d70
  representative_frames_2.png    # SHAdbe6a8b01f93dbd4faec8dc56121510352b8b4be80093b8864c62c59c8251c76
  review.json                    # viewed_not_accepted; no user approval inferred
```

No representative blank/broken/wrong-scene frame was seen. This is not full-video
manual review or success/label certification. The original report remains
`visual_status=not_viewed`. Technical checks pass, but family provenance and
checkpoint-training overlap are unknown; training_ready/formal_data_allowed are
false and features/optimizer remain0. Next proposed scope is fixed12 frozen
feature-cache preparation, not an implicit feature-extraction or training command.

## Fixed12 Frozen Feature Cache (2026-09-13, Complete)

The subsequent user approval covered fixed12 feature-cache generation and the
unchanged train-only normalization/window preparation; no optimization or rollout.
Old fixed-pair builders/guards, model/loss code and source/labels remain unchanged.
New plan and source/build tools were SCP-synchronized and remote SHA-checked before
execution. Source/selection/readback, checkpoint/config/protocol and installed/repo
implementation hashes are bound before and after; offline-only, CUDA, no fallback.

```text
docs/libero-action-study-feature-plan-v1.json
  ee85b9efbabf980f1aae550652a4db92c4f68b3b47b0bd34d25f3658cdefdf7a
tools/build_pi05_libero_action_study_features.py
  8f796c16903e684caa02691a0ff72d6661a03e60c84ee451d2795d12712fdb4c
tools/pi05_libero_action_study_feature_source.py
  d60422d1f450990be1247edd30bc923b5ac847a1e9986fe5a892a2d6313c07ed
tools/test_pi05_libero_action_study_feature_builder.py
  be8ca83f961fc530f0fb3a71a607b2b11199682250e107aa565dddf0dfd83469
tools/test_pi05_libero_action_study_feature_source.py
  e55f7a9184acbcc9b1beef7dc3e6bc655f18485810175199808f7dd29019a733
tools/audit_pi05_libero_action_study_features.py
  f2d6c693db3aff9afc46a8c5f8d73a69093c87bac35b4ce217e081f2d2a4a278
tools/test_audit_pi05_libero_action_study_features.py
  081c40dd12ff3a17eee633bf577e84fe9fd01a39f5d6f1fb7a375fc6bd77417d
```

Read-only preflight; omit `--preflight-out` for safe rechecks. It does not load a
policy or extract features. The completed first preflight added
`--preflight-out simulation_output/pi05_libero_action_study_feature_preflight_v1.json`
(fresh-only, do not overwrite). It passed in7.114677s, output absent,~23.5GB CUDA
free. SHA `fca96d41511202f350a6388ce33a2346c6f0be974d6da32f6f7218741d14ea42`.

```powershell
ssh project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/build_pi05_libero_action_study_features.py --stage preflight"
```

Historical completed full command below: **do not rerun**. The estimate from the
old313-row39.399s build was3-4minutes on the idle4090, so root executed under the
five-minute rule. Actual build excluding its own preflight67.661943s, final
status67.680854s; image encoding32.950807s, preparation/posthash4.103199s. No
optimizer, sampler fitting, world-model update or action/simulation step.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/build_pi05_libero_action_study_features.py --stage run --execute-frozen-feature-cache"
```

```text
simulation_output/pi05_libero_action_study_features_v1/
  preflight.json, status.json, report.json, run.log
  state.npy [1658,8], action.npy [1658,7]     # original native values
  visual_latent.npy [1658,2,2048]            # finite float32, image-only
  episode_index.npy, frame_index.npy, task_id.npy
  state_valid.npy, visual_valid.npy         # all observed; no imputation
  extraction_trace.jsonl                    #1658 preprocess/3316 real embeds
  extractor_provenance.json, manifest.json, split.json
  preparation/normalization.json           #1134 train state/1126 train actions
  preparation/train_windows.jsonl          #1086 windows
  preparation/validation_windows.jsonl     #500 windows
  preparation/report.json                 # generic schema scope unchanged
  episode*_source.png, episode*_processed.png  #96 first/last view comparisons
```

```text
report SHA:        9e0537b5d520f5e242a02be1f94c454febfec62854a8c43a517f7acdce05ac56
manifest SHA:      133894fb46d9d2920dc5393affa2d5fb07e5c0baf3fb4c12d7a6af25238ee56b
split SHA:         044ea50f188ff6f58e9680064a81e3970b15fde3c3c9ab5c024907aef2c662c7
normalization SHA: 1fdea3ce37c995b5de161b04fbf2b8567e24fbc7915786b29bf163fb147acf64
```

Fixed train8/1134 rows and val4/524 rows remain unchanged. Old1400/1402 are
audit-only and never extracted. Group IDs are descriptive leakage metadata,
not independent-family labels or policy inputs. Normalization excludes each
train final action, retains it in raw action.npy, uses no validation statistics,
and is shared for the future observed/zero arms. No normalized-zero action array
was written and no candidate action was generated during this cache job.

After interruption, inspect status/log/PID. Existing output is refused even if
failed; no automatic resume, overwrite, replacement IDs or retry. Preserve
failed artifacts and determine cause before a separately justified new attempt.
The successful cache is not a reason to rerun or bypass frozen two-pair guards.

```powershell
ssh project4090 "cat /home/zsw/project_2026/simulation_output/pi05_libero_action_study_features_v1/status.json"
```

Independent readback below is safe to repeat without `--out`; it does not load
the model, decode source videos, reopen rawsource images.npy/Parquet, or change
features/statistics. It checks all113 output hashes,25 source metadata files,
1658 raw rows/trace latent byte hashes,3316 source RGB links,48 source PNGs,
all1086/500 actual windows and independently recomputed1134/1126 train stats.

```powershell
.\.venv\Scripts\python.exe tools/audit_pi05_libero_action_study_features.py --feature-pack simulation_output/pi05_libero_action_study_features_v1 --source-root simulation_output/libero_action_study_sources_review_v1 --report-sha256 9e0537b5d520f5e242a02be1f94c454febfec62854a8c43a517f7acdce05ac56
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/audit_pi05_libero_action_study_features.py --feature-pack simulation_output/pi05_libero_action_study_features_v1 --source-root simulation_output/libero_action_study_sources_v1 --report-sha256 9e0537b5d520f5e242a02be1f94c454febfec62854a8c43a517f7acdce05ac56"
```

Completed local readback0.828s used fresh
`--out simulation_output/pi05_libero_action_study_features_readback_v1.json`, SHA
`6769fb11364851c248893b7b01c468bb4b5ee8e223fd535ca726269ed0bb36f2`.
Completed remote readback0.459116s used fresh
`--out simulation_output/pi05_libero_action_study_features_readback_remote_v1.json`,
SHA `1407f3133fada54e9ce20b30b95c2c2bec46068e5a63238eca8c52ca2516ad30`.
Both leave original reports untouched. Local uses the metadata-only source subset;
remote still opens only metadata, not live weights/installed code. Actual
before/after weight/code evidence belongs to the builder, not this readback.

Synthetic tests:25 adapter+20 builder passed locally and remotely (remote45 in
4.150s);7 readback helper tests passed locally0.267s/remotely0.023s. Old unchanged
builder19 tests passed locally11.446s. Synthetic/fake encoder tests are not real
PI0.5 evidence; actual extraction and independent readbacks above are separate.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action_study_feature_*.py' -q
.\.venv\Scripts\python.exe -m unittest discover -s tools -p test_audit_pi05_libero_action_study_features.py -q
```

The complete small feature review archive (no rawsource NPY/MP4/Parquet) is
29,863,631B, SHA `89733f72dd9f9e48af38eeb769af81894efba18b391bc3b34326a7ef137e5d79`.
All119 members/34,727,543B file contents were path/type/size checked before local
extraction. This is below100MB; no large-file user transfer was needed. Two
Chinese96-PNG contact sheets and `viewed_not_accepted` sidecar are under
`simulation_output/pi05_libero_action_study_features_visual_review_v1/`.

Next proposed scope: paired observed-action/normalized-zero-action entrypoint,
prespecified frozen group/episode/window draws and step0 checks only. No future
3seed x2arm x200update command is authorized or run by this feature milestone.
Family/pretraining overlap remain unknown; no quality, robustness, causal,
policy, formal-readiness or real-system claim follows from a passing cache.

## Fixed12 Shared Draws And Paired Step0 (2026-09-13, Complete)

User approval covered only shared sampling manifests, paired initialization and
random step0 evaluation. New runner has no training switch, optimizer construction,
backward, checkpoint or feature-extraction path. Old model/loss and all prior
artifacts stay unchanged. Runtime environment remains project2026-pi on4090;
local2.13dev torch is used only for synthetic checks, not the pinned real run.

```text
docs/libero-action-study-step0-plan-v1.json
  2716481652a40b783b55eaec4cd7ec54f6e7de845f0a249d54355b6ffedd21c9
tools/run_pi05_libero_action_study_step0.py
  eb1c56258492be157f33e3926dc0654d7b98b410d42836f3df472d62f50f24f1
tools/pi05_libero_action_study_sampling.py
  d36e8b43f8fd3c97cbd0b32c0d427d0e4c50d42e07f62861ab7f07f937d34d2f
tools/test_pi05_libero_action_study_sampling.py
  f9acf9d070c7324e913d648332aff1c4d591fa74375f60b41c807353c53984cc
tools/test_pi05_libero_action_study_step0.py
  46b12584621618109cfafa297f7d2863cf7e0dfa3ecc8dd1ef337e94916f58b6
```

Final changed small files were transferred by sequential SCP and SHA-checked
remotely before their execution. Preflight checks125 input/code/artifact hashes;
it is safe to repeat and does not construct a model or optimizer:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_action_study_step0.py --stage preflight --plan-sha256 2716481652a40b783b55eaec4cd7ec54f6e7de845f0a249d54355b6ffedd21c9"
```

Historical completed command, **do not rerun**:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_action_study_step0.py --stage step0 --plan-sha256 2716481652a40b783b55eaec4cd7ec54f6e7de845f0a249d54355b6ffedd21c9 --execute-step0"
```

Passed in13.029622s, well within the conservative2min estimate based on previous
same-CPU evaluation. Each of3 seeds owns one prospective3200-draw schedule,
shared by observed_action and normalized_zero_action; all9600 records precede
model initialization. Per seed71 train/33 validation batches cover1086/500
windows exactly once per arm.312 total batches/936 scored prediction batches,
including repeated persistence references; this excludes the unchanged common-target
helper's validation-only persistence construction. No optimizer or backward occurred. Both arm models own
disjoint storage with equal initial SHA; all modes/versions/parameter bytes remain
unchanged. Fixed all-valid target bytes and common support match; only normalized
candidate actions differ. Zero denotes train action mean, not physical hold.

```text
simulation_output/pi05_libero_action_study_step0_v1/
  preflight.json, started.json                # PID/time, exclusive fresh attempt
  draws_seed20260912.json                     #3200 future draw records,0 updates
  draws_seed20260913.json
  draws_seed20260914.json
  sampling_index.json                         # same file references for both arms
  evaluation_trace.jsonl                      #312 batch input/target/output hashes
  report.json, status.json                    # untrained reference only
```

```text
report SHA:
  1075355fc7ca0a519e40438aa6554cbf6fb3ec476b8be96c3abc8a9e480f3248
sampling index SHA:
  0952657872525d2374f2e64558532df3a3d1800cd471db711cc31feb674870ff
trace SHA:
  bde5618bc723f27f8d61727a0058d002f0ea9a82547dcdb3a5c5c1131690fa69
draw file SHAs, seed order20260912/20260913/20260914:
  12ee2519e5f40ec9cab0ef9f3de1b897c29a2fd21e609376951de7594670cdac
  deed27def86059b6b6cedf23cfb3314fb76d458088a3c060cd7675666f35b6d7
  434b18002996a660ea1b713f7ac586a9fda77b808791ee7a24c0dfccde2b183b
```

Full result file contents7307049B retrieved with hashes matched; no100MB transfer
was needed. On interruption inspect started.json PID/status/failure/stdout before
any action. Existing output is refused even if failed; preserve it. There is no
automatic resume/overwrite/retry, and a completed step0 is never rerun for tuning.

Final synthetic checks38 passed locally2.556s and remotely.952s (16 sampler,
22 entrypoint). Before actual execution the initial36 had passed remotely.812s;
the final two add cross-seed hash distinction and batch1/2/16 sufficient-statistic
invariance with masks. Tests remain synthetic, not source/model-quality evidence.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action_study_s*.py' -q
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest discover -s tools -p 'test_pi05_libero_action_study_s*.py' -q"
```

Validation random observed-minus-zero episode-macro MAE differences are
+.019792/+.027689/+.034586, mean+.027356 and descriptive population std.006044.
These random references cannot establish learned action value or select a seed.
The next training implementation/execution gates remain separate; no1200-update
command is authorized by this milestone. Keep family/pretraining overlap unknown
and quality/causal/robustness/policy/formal/real-system claims disabled.

Independent readback tool (new, not part of or a mutation to the completed runner):

```text
tools/audit_pi05_libero_action_study_step0.py
  8f48006545246bf74c83b8fe4aaab032db49d1454658b2aa7420e8623dfcdd2d
```

Sequential SCP/hash check preceded the complete remote audit. These commands are
safe read-only repeats, stdout only; neither imports/constructs a model nor runs
forward/backward or writes normalization/output. Local metadata-only passed.828s;
remote full replay passed1.104200s. Both validate125 input/7 output hashes and all
312 trace input/target/action-zero hashes reconstructed from immutable cache,
312 persistence hashes, exact coverage and saved macro/micro arithmetic. Only
remote replays all3 sampling manifests/9600 draws under the pinned torch version.
The624 arm prediction batches are not regenerated: error metrics and parameter
unchanged claims are checked against saved sums/hashes, not a new model forward.

```powershell
.\.venv\Scripts\python.exe tools/audit_pi05_libero_action_study_step0.py --metadata-only --report-sha256 1075355fc7ca0a519e40438aa6554cbf6fb3ec476b8be96c3abc8a9e480f3248
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/audit_pi05_libero_action_study_step0.py --report-sha256 1075355fc7ca0a519e40438aa6554cbf6fb3ec476b8be96c3abc8a9e480f3248"
```

## Fixed200 Training Implementation And Gradient Checks (2026-09-13, Complete)

Scope approved: implement and test training/gradient/checkpoint machinery, not
execute public training. Model/loss/old step0/optimizer/checkpoint helpers and
all cached data/draws remain unchanged. Final new files were sequentially SCP'd
and their remote SHA256 verified before execution:

```text
docs/libero-action-study-training-plan-v1.json
  34ab3e9c9aedd0da7a2dbecb8100cf448c7f5eb8fa358244b1340e7d68ef5302
tools/run_pi05_libero_action_study_training.py
  06850b23f14b5a40c24f86f13a2ce62561ed4eff6b3e9cd30fb7794054a60992
tools/pi05_libero_action_study_training.py
  d5f4430d769e1bae5d7741d3043bd1d662d82ab5ce833c95db1a24a682c9eb2f
tools/test_pi05_libero_action_study_training.py
  4b5526260aeebd4b025bc0f74a99919182eeb4b48a389f6daa92cc2c2c3ca797
tools/test_pi05_libero_action_study_training_runner.py
  cbf5f75b11c64bc2b128b2706876b68be061d842c9e3b3fa98c23871a1d29ab3
```

Safe repeated preflight:139 hashes, prior step0 readback/draw replay, no model
construction or optimizer. The current plan allows gradient-only checks but
does not authorize the train stage.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_action_study_training.py --stage preflight --plan-sha256 34ab3e9c9aedd0da7a2dbecb8100cf448c7f5eb8fa358244b1340e7d68ef5302"
```

Completed gradient-check command, **do not rerun**. It validates the new evaluator
against all624 frozen step0 prediction batches and complete1086/500 windows for
each seed/arm, then uses only the first16 train draws for each of6 backward calls.
No optimizer is created, no parameter is updated, no checkpoint is written.
Actual check11.883497s excluding preflight (conservative estimate30-60s).

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_action_study_training.py --stage gradient_check --plan-sha256 34ab3e9c9aedd0da7a2dbecb8100cf448c7f5eb8fa358244b1340e7d68ef5302 --execute-gradient-check"
```

```text
simulation_output/pi05_libero_action_study_gradient_check_v1/
  preflight.json  fe264489fe6fbae8317ef28336a7b1d9d33fff2f47e7e6581197587c893ac78f
  started.json    093800f63f0898bcd5839ccb5d758fb101a65c00f370e96bd4f199338736b997
  report.json     6eefb781f633e705d763af4191b374e61c931e48399b3d01b3803d6ac75f2ef6
  status.json     4d8dc7a5841bc9aef264deb612aa77e6f7d73d681ee4eff0cfe127db865f08bd
```

All23 parameter gradients present/finite per probe. Observed/zero global norms:
seed20260912 .0508694913/.0528886330;
seed20260913 .0531460458/.0396668772;
seed20260914 .0664293415/.0636352321.
All zero-arm action_projection.weight gradients exactly zero; positive global
norm remains valid. Parameter hashes/versions and input/target bytes unchanged,
gradients cleared. CPU float32,torch2.10.0+cu128,threads1,deterministic; CUDA never
initialized. Full117928B of JSON contents retrieved;4 output and139 input hashes,
first-draw indices, paired targets and gradient-norm arithmetic independently read
back locally. No100MB transfer or model feature regeneration was involved.

Tests:19 primitive+22 runner checks passed locally2.544s/remotely1.313s; prior38
sampler/step0 checks locally1.594s/remotely.899s. Synthetic checkpoint fixtures use
one synthetic update then counter200 ONLY to test serialization. Loop counters
use mocked updates; neither constitutes actual200-step public training.

Local torch2.13dev may initialize a visible GPU in AdamW's graph-capture health
check even for CPU tensors. This initially caused two combined-test CPU guards
to fail. Hide GPUs for the local test process; do not weaken production guards.
The fixture RNG is explicitly CPU-only too. Restore the caller's prior environment
after testing:

```powershell
$previousCuda = $env:CUDA_VISIBLE_DEVICES
try {
    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action_study_training*.py' -q
    if ($LASTEXITCODE -ne 0) { throw 'Training implementation tests failed' }
    .\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action_study_s*.py' -q
    if ($LASTEXITCODE -ne 0) { throw 'Step0 regression tests failed' }
} finally {
    $env:CUDA_VISIBLE_DEVICES = $previousCuda
}
ssh project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest discover -s tools -p 'test_pi05_libero_action_study_training*.py' -q"
```

Future training entrypoint is implemented, but its additional authorization file
`docs/libero-action-study-training-authorization-v1.json` is intentionally absent.
Only after separate user approval may that file set training_execution_allowed=true
and bind this plan SHA, the gradient report SHA above, exact200 per arm/seed,
total1200, and output `simulation_output/pi05_libero_action_study_training_v1`.
The runner then requires both its external --authorization-sha256 and
--execute-training with --stage train. A missing/false/drifted authorization fails
before models/optimizers/output creation; this milestone does not supply a live
training command or create that authorization.

Future output is six final200 checkpoints plus per-arm metadata/results, shared
step0/final evaluation traces, training trace and aggregate report/status. No
best checkpoint or resume; zero-arm actions remain zero in final evaluation.
Checkpoint binding includes actual auth and arm/seed; the legacy authorization
metadata field explicitly stores the binding digest instead of original auth SHA.
Original checkpoint parameter_sha and step0 parameter_hash are distinct formats.
Both externally expected binding and full metadata are mandatory during reload.

Recovery: inspect started.json PID, status/failure and partial trace/checkpoints.
Never overwrite/retry/resume automatically, infer0 updates from a failed train,
or refit norms/regenerate draws. Completed gradient output is preserved. No new
public training output or positive authorization existed after this check.

## Fixed200 Three-Seed Training Execution (2026-09-13, Complete)

The subsequent user approval authorized exactly3 seeds x2 arms x200 updates.
The following command is completed history, **not a rerun instruction**. Keep
the old frozen prospective/implementation plans unchanged. The only new gate
file was separately approved, sequentially SCP'd and hash-verified before use:

```text
docs/libero-action-study-training-authorization-v1.json
  6a00731976b5d5fb67d65fe7501f2031bb6276511e9b364179cff0c460778b9c
```

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_action_study_training.py --stage train --plan-sha256 34ab3e9c9aedd0da7a2dbecb8100cf448c7f5eb8fa358244b1340e7d68ef5302 --authorization-sha256 6a00731976b5d5fb67d65fe7501f2031bb6276511e9b364179cff0c460778b9c --execute-training"
```

One successful run,90.321776s excluding preflight,1200 actual updates. CPU
float32,torch2.10.0+cu128,threads1,deterministic,no CUDA init/AMP/TF32. Estimated
runtime was a few minutes. Individual200-update training10.9522-11.1971s;
save/reload/prediction verification0.0989-0.1159s per checkpoint. All6 initial
models reproduced frozen step0 before any optimizer; only final200 checkpoints
were saved and reloaded for the fixed final evaluation. No PI0.5 backbone update,
feature extraction, model/loss change, rollout, extra step or automatic retry.

```text
Remote directory: /home/zsw/project_2026/simulation_output/pi05_libero_action_study_training_v1/
Local reports:    D:/PycharmProjects/project_2026/simulation_output/pi05_libero_action_study_training_v1/
report.json SHA256:
  837cb7b8df45a3b7230e979f9378d91b0177242e98449f6a60ed1608ad2fe5bf
Reports-only archive (2,004,675 bytes), remote/local simulation_output/:
  pi05_libero_action_study_training_reports_v1.tar.gz
  eca785d4b080f6ab1fb0585963c828ba6382a04a8782d504eff52a11994a4d07
```

The slow raw-JSON SCP was stopped after identifying its owned process; training
had already completed. A reports-only compressed archive replaced partial local
copies after archive SHA/path verification. No experiment was retried. Only
JSON/JSONL reports were retrieved, not the6 checkpoints. Each remote checkpoint
is15,883,215B; full hashes are in report.output_sha256 and the six envelopes.
There was no file/archive over100MB transfer and no split to bypass that rule.

Independent remote readback verified141 input/code/auth and23 output hashes,
completed status and six weights_only checkpoint payloads. All23 AdamW parameter
states per checkpoint have step200; parameter/moment tensors are finite, exact
metadata/binding and paired initial bytes agree. Root's readback deserialized
only; the completed runner already verified reload prediction equality and full
final evaluation. Checkpoints remain final-only and not resumable.

Independent reviewer audit1.9598s, read-only/no model forward: three draw manifests
replayed,1200 training rows reconstructed from the cached inputs/targets, both
624-row step0/final traces checked for full coverage and unchanged supports.
All12 seed/arm/partition episode-to-macro/micro aggregations and comparisons
matched.141 input/23 output hashes were checked before and after that audit.

Safe repeated remote status/hash inspection, no training or model construction:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && cat simulation_output/pi05_libero_action_study_training_v1/status.json && sha256sum simulation_output/pi05_libero_action_study_training_v1/report.json"
```

Safe local report and retrieved-file integrity check (does not load checkpoints):

```powershell
$out = 'simulation_output/pi05_libero_action_study_training_v1'
if ((Get-FileHash "$out/report.json" -Algorithm SHA256).Hash.ToLower() -ne '837cb7b8df45a3b7230e979f9378d91b0177242e98449f6a60ed1608ad2fe5bf') { throw 'Report SHA drift' }
$report = Get-Content "$out/report.json" -Raw | ConvertFrom-Json
$status = Get-Content "$out/status.json" -Raw | ConvertFrom-Json
if ($status.status -ne 'completed' -or $status.optimizer_steps -ne 1200 -or $report.total_optimizer_steps -ne 1200) { throw 'Incomplete fixed budget' }
foreach ($entry in $report.output_sha256.PSObject.Properties) {
    if ($entry.Name -like '*.pt') { continue } # Remote-only; not a local checkpoint verification.
    if ((Get-FileHash "$out/$($entry.Name)" -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw "Output drift: $($entry.Name)" }
}
foreach ($entry in $report.input_sha256.PSObject.Properties) {
    if ((Get-FileHash $entry.Name -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw "Input drift: $($entry.Name)" }
}
$report.primary_paired_macro
```

Descriptive validation episode-macro state MAE, observed/zero/persistence:
seed20260912 .038559/.070074/.075486;
seed20260913 .037397/.067825/.075486;
seed20260914 .035425/.064827/.075486.
Seed means .037127/.067575/.075486: observed45.06% below zero and50.82% below
persistence. Paired difference mean-.030448, population std.000863. Mean visual
MAE .077483/.079344/.079822, combined objective .007554/.008757/.009965. No
visual episode-macro regression in any seed against zero or persistence.
Per-episode, micro, horizon/view/coordinate/same-unit metrics and all comparisons
remain in report.json; the handoff records all4x3 held-out state comparisons.

Decision: prespecified descriptive observed-action predictive association passes,
not causal action effects, counterfactual ranking, policy improvement or low-
visibility robustness. There are4 held-out episodes, not12 independent units;
source-family independence and checkpoint-pretraining overlap remain unverified.
Next proposed work is a separately specified frozen-checkpoint clean/degraded
diagnostic. No additional training, feature generation or rollout is authorized
by this completed run. Preserve outputs; inspect any future drift/failure before
proposing a new gated job, never overwrite/resume/retry this directory.

## Frozen Checkpoint Clean/Low-Contrast Evaluation (2026-09-13, Complete)

Subsequent user approval covered the fixed clean/degraded observation diagnostic,
including needed validation-only frozen features, not training or rollout. The
protocol was fixed before data generation/scoring; source/old models/losses/
checkpoints remain unchanged. New code was sequentially SCP'd and SHA-verified.

```text
docs/libero-action-study-low-contrast-plan-v1.json
  0558af2fc68e5cc8c0a6e869e34fda74782246a35120a871db9e86b3327ca307
tools/pi05_libero_action_study_low_contrast.py
  147914f6074cfd2de4aaa0e5f8b131641c5ffb062cacc1ca3bedaf39e86bf583
tools/run_pi05_libero_action_study_low_contrast.py
  31d365d6794cdad50e3ce9f920d32a4839155392140dace3959c5bb8e548c918
tools/pi05_libero_action_study_low_contrast_features.py
  4b6cb0989cfbd36881964784e0a5ecb3fe0bc10ab6d3fd45b0673e647496be99
tools/test_pi05_libero_action_study_low_contrast.py
  f61a154c86e13fc53f75aff4da5905abdf6fbd52e3504b0b0687316a47808225
tools/test_pi05_libero_action_study_low_contrast_features.py
  b57be4ac889f02bda6d9d1f33bc3322e8de7e8ca66fc527133a04bfdaf8ca792
```

Safe test/preflight repeats, no real feature extraction, optimizer or evaluation:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_pi05_libero_action_study_low_contrast*.py' -q
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest discover -s tools -p 'test_pi05_libero_action_study_low_contrast*.py' -q && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_action_study_low_contrast.py --stage preflight --plan-sha256 0558af2fc68e5cc8c0a6e869e34fda74782246a35120a871db9e86b3327ca307"
```

34 tests passed locally2.172s/remotely.841s;173-pin remote preflight passed.
Whole-job estimate3-4minutes including GPU loading/hashing and CPU evaluation.
The following actual commands are completed history; **do not rerun**:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_action_study_low_contrast.py --stage features --plan-sha256 0558af2fc68e5cc8c0a6e869e34fda74782246a35120a871db9e86b3327ca307 --execute"
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_action_study_low_contrast.py --stage evaluate --plan-sha256 0558af2fc68e5cc8c0a6e869e34fda74782246a35120a871db9e86b3327ca307 --feature-report-sha256 ebe3cf6d8e05a5e1bcc405db93c6f1ca7755f6348237766aac7b3a4df07564ad --execute"
```

Feature stage62.2536s excluding preflight:524 exact clean replays +524 low-contrast
rows,2096 actual real-view embedding calls,0 training rows encoded. Low contrast
is the unchanged alpha.45/severity2 raw uint8 per-channel mean transform. All1048
degraded images/524 latent rows changed. Usual vision-key warnings were nonfatal;
strict checkpoint load, frozen modes/versions and exact clean replay passed.

CPU evaluation9.7626s,0 updates,0 checkpoint saves,0 actions/rollout steps. All6
clean final200 predictions/statistics replayed exactly, then degraded historical
features only replaced. Same clean targets/support, task/state/action/norm/split;
zero-arm actions remain normalized0.396 learned forward batches,66 scored
persistence batches,462 trace rows;528 persistence helper calls including support
checks. Each model/condition covers all500 original validation windows.

```text
simulation_output/pi05_libero_action_study_low_contrast_features_v1/report.json
  ebe3cf6d8e05a5e1bcc405db93c6f1ca7755f6348237766aac7b3a4df07564ad
simulation_output/pi05_libero_action_study_low_contrast_eval_v1/report.json
  da68ad3a7253a4e04e10990ec4a6b1a2ce6dac7d7e336a85d1571c20c81c1784
Reports/64 preview PNGs archive (3,705,956 bytes), no NPY or weights:
simulation_output/pi05_libero_action_study_low_contrast_reports_v1.tar.gz
  6179307213932e454b5b08da4d6f745a696bb31af9b5dbd86d2120db4e4f55ac
```

Arrays remain remote: features directory degraded_visual_latent.npy[524,2,2048]
and row_indices.npy[524]. The local reports/previews are not a full executable
feature cache; do not claim local245-input verification without those arrays.
Checkpoint weights likewise remain remote. Reports contain all per-episode/
seed/condition macro/micro metrics, input/target/prediction traces and byte hashes.

Safe remote readback:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && cat simulation_output/pi05_libero_action_study_low_contrast_features_v1/status.json && cat simulation_output/pi05_libero_action_study_low_contrast_eval_v1/status.json && sha256sum simulation_output/pi05_libero_action_study_low_contrast_features_v1/report.json simulation_output/pi05_libero_action_study_low_contrast_eval_v1/report.json"
```

State MAE observed/zero/persistence: clean.037127/.067575/.075486, degraded
.037119/.067583/.075486. Observed state change-.0213% is effectively stable,
not meaningful improvement evidence. Visual MAE observed clean.077483 ->
degraded.117869 (+52.12%); zero+50.31%, persistence+49.97%. Observed combined
objective worsens89.92%. The narrow state action-association criterion remains
true in all3 seeds, but this is not an overall visual-robustness pass.

Preserve the protocol/checkpoints/output directories; do not tune severity,
repeat extraction/scoring or train automatically. Failed stages are exclusive
and non-resumable: inspect started.json PID/status/failure/partial traces rather
than overwrite or assume an SSH disconnect requires a new run. These two stages
completed successfully. Next proposed work is visual-dependence diagnosis before
choosing a robustness module; not new ablation/training authorization.

Independent read-only audit5.7867s verified394 unique files, all1048 raw RGB
corruption formulas,524 clean/degraded cache links and mappings,462 trace rows
and14 groups' metric arithmetic. It ran no additional embedding/model forward.
The local archive readback verified69 non-NPY feature outputs and all3 evaluation
outputs; report SHA/status matched. Remote NPY/checkpoint-only inputs were not
claimed as locally checked.

Chinese visual review tool was synchronized/hash-verified before local lightweight
Pillow rendering (no LeRobot needed):

```text
tools/render_pi05_libero_action_study_low_contrast.py
  a7822c1fd6787d5139bb723fd371bfc37ff4d3a4c95402f1432021ea51b9df23
```

Completed gallery command; output is exclusive, do not overwrite it:

```powershell
.\.venv\Scripts\python.exe tools/render_pi05_libero_action_study_low_contrast.py --features-dir simulation_output/pi05_libero_action_study_low_contrast_features_v1 --feature-report-sha256 ebe3cf6d8e05a5e1bcc405db93c6f1ca7755f6348237766aac7b3a4df07564ad --out simulation_output/pi05_libero_action_study_low_contrast_review_v1 --font C:/Windows/Fonts/msyh.ttc
```

64 preview hashes checked; source_paired_review.png and processed_paired_review.png
are1216x2370 with original-size pixels/aspect and Chinese labels. Both viewed,
including original-resolution processed sheet; no clipping/missing glyphs noted.
Visual status viewed_not_accepted, not user acceptance. This presentation-only
step changes no model, metric, source image, corruption or evaluation output.

## Frozen Visual-Dependence Diagnostic (2026-09-13, Complete)

User approval covered this no-training diagnostic. All6 final200 checkpoints,
same4 validation episodes/500 windows, original state/action/task/masks and clean
future targets remain fixed. Conditions: clean, repeat-current history, fixed
training episode1312/frame0 two-view anchor. Actual residual-head output is
captured by a temporary read-only hook; no model/loss changes or new embeddings.

Frozen files:

```text
docs/libero-visual-dependence-plan-v1.json
  b81ed7f66ee10625d66a9e34e582283690930833c91e0f2c7bdb77905f6d7aea
tools/run_pi05_libero_visual_dependence.py
  826742753e1c6c3aac4cb2abdc7b55b0584f0f7a0b6d7400b151f74981ed78f1
tools/pi05_libero_visual_dependence.py
  fe9dbe6c4f2f2311434b180952213a30bc600ac1cd8ea8945b918a05c42f44d5
tools/test_pi05_libero_visual_dependence.py
  e0199eddb213e60ac89d126be164450e8ed317896addaf852328bb2c5b06a86a
tools/test_run_pi05_libero_visual_dependence.py
  3de8a24964f10ef33f58a8b45d04b9f75ecc8bef9a8b156d24885de26a87fe0d
```

Sequential SCP/remote SHA verification completed before executing. Safe repeated
synthetic tests/preflight, with no real model evaluation or optimization:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONPATH='tools'
.\.venv\Scripts\python.exe -m unittest test_pi05_libero_visual_dependence test_run_pi05_libero_visual_dependence -q
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest test_pi05_libero_visual_dependence test_run_pi05_libero_visual_dependence -q && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_visual_dependence.py --stage preflight --plan-sha256 b81ed7f66ee10625d66a9e34e582283690930833c91e0f2c7bdb77905f6d7aea"
```

38 tests passed locally0.614s/remotely0.378s;176-pin preflight passed. The following
actual command is completed history, **do not rerun** or overwrite its directory:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_visual_dependence.py --stage evaluate --plan-sha256 b81ed7f66ee10625d66a9e34e582283690930833c91e0f2c7bdb77905f6d7aea --execute"
```

CPU evaluation36.1192s;594 learned forwards,99 scored persistence batches,
693 trace rows,792 persistence helper calls including support checks. Exact
saved clean predictions/statistics replayed;0 optimizer updates/checkpoint saves/
new embeddings/rollout steps. Hooks, parameter hashes and original input pins
unchanged. Batch traces store hashes and sufficient statistics, not large raw
prediction tensors. Complete report contains per-episode/macro/micro metrics,
paired target-error changes, actual-residual/skip/input/output drift and3-seed
descriptive means/std. Component magnitudes are not causal attribution shares.

```text
simulation_output/pi05_libero_visual_dependence_v1/report.json
  1849406c3a434b3dd43ed1d7859c0d401f69022acd4f9f1200a8af3951b25e41
simulation_output/pi05_libero_visual_dependence_reports_v1.tar.gz (3,218,984 bytes)
  f1d0bf944c22a361d43d80f889ed588055bd288d7c6d885884db783e6b13716c
```

Safe remote status readback:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && cat simulation_output/pi05_libero_visual_dependence_v1/status.json && sha256sum simulation_output/pi05_libero_visual_dependence_v1/report.json"
```

If SSH disconnects, inspect started.json PID/status/failure and partial traces;
do not infer it needs a retry. The entrypoint is exclusive and non-resumable.
Preserve all completed evidence and do not restart extraction/training. Observed
state MAE clean/repeat/anchor=.03712690/.03713091/.03731119; zero-arm anchor
state MAE worsens13.30%. Actual observed visual residual drift stays small while
the direct skip changes greatly. This is local input sensitivity, not a visual
robustness pass or a claim that vision is unnecessary. A read-only scale/
activation/fusion inspection is the next proposed question, not automatic
training or intervention-sweep authorization.

Independent read-only result audit passed on its first remote run in2.8760s.
It rechecks176 input/3 output hashes,693 trace rows/21 scoring groups,1485 raw
cache-derived drift fields and saved-statistic aggregation. No forward,
embedding, optimizer or checkpoint deserialization. Learned output/residual
drift is checked through saved hashes/sufficient statistics, not regenerated
predictions. The archive/report/status/all3 outputs also verified locally;
checkpoint weights remain remote. The stdout-only audit may safely be repeated:

```text
tools/audit_pi05_libero_visual_dependence.py
  d1c54c27550ce1ddb958a2c4d5c736f412e1f50874a084874f9f93c3f38db822
```

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_pi05_libero_visual_dependence.py --report-sha256 1849406c3a434b3dd43ed1d7859c0d401f69022acd4f9f1200a8af3951b25e41"
```

## Frozen Visual Activation / Scale Inspection (2026-09-13, Complete)

User-approved read-only inspection, not training or architecture repair. Same6
models at original step0/final200,500 clean validation windows each;396 real
forwards,0 backward/optimizer/embedding/persistence/objective/rollout calls.
Raw cache scale summaries use all1134/524 unique train/validation rows only as
descriptive evidence, never apply a newly fitted input normalization.

```text
docs/libero-activation-inspection-plan-v1.json
  3a0c79f69dffc998394795057ceb1ce2e1e8841cd55f46cefc8fba00f3a6c1ea
tools/pi05_libero_activation_inspection.py
  97775575f8ca8d5b9474afe6326a153a44d39f781b1d7f3364e71c2c02c67e88
tools/run_pi05_libero_activation_inspection.py
  d0a33c44f97fbfd68120464ffa48f398506eb3b859e69785bd8b7e9370c04e6a
tools/test_pi05_libero_activation_inspection.py
  250273d90c30c2ea22a0f7de04c718f16d173e46264d50510dfc0f29cc5a66d2
```

These final files were sequentially SCP'd and remote SHA-verified. Safe repeated
synthetic tests/preflight (no real model inspection):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONPATH='tools'
.\.venv\Scripts\python.exe -m unittest test_pi05_libero_activation_inspection test_pi05_libero_visual_dependence test_run_pi05_libero_visual_dependence -q
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools /home/zsw/miniconda3/envs/project2026-pi/bin/python -m unittest test_pi05_libero_activation_inspection test_pi05_libero_visual_dependence test_run_pi05_libero_visual_dependence -q && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_activation_inspection.py --stage preflight --plan-sha256 3a0c79f69dffc998394795057ceb1ce2e1e8841cd55f46cefc8fba00f3a6c1ea"
```

53 tests passed locally.925s/remotely.532s;183 input pins passed. Completed actual
command below is history, **do not rerun** or overwrite its exclusive output:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_activation_inspection.py --stage inspect --plan-sha256 3a0c79f69dffc998394795057ceb1ce2e1e8841cd55f46cefc8fba00f3a6c1ea --execute"
```

Completed22.1295s CPU;396 batches/12 model-phase groups, all original prediction
hashes replayed,17 activation tensors/five explicit tanh pairs captured read-only.
Two affine decompositions are float64 diagnostics, not extra model forwards or
causal shares. Actual float32 discrepancies are recorded separately. Parameter
hashes, hooks, source bytes and original artifacts unchanged. No new weights.

```text
simulation_output/pi05_libero_activation_inspection_v1/report.json
  963dad4c9450cf6c34e4be3b1a3be04e06861502c099f0e6188864aaaba8e980
simulation_output/pi05_libero_activation_inspection_reports_v1.tar.gz (2,510,489 bytes)
  d75b536bbbbe6ac5e0ec3e8a0bc28247932cc4026e9a6af59cf3af04c9c025a5
```

Safe status readback:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && cat simulation_output/pi05_libero_activation_inspection_v1/status.json && sha256sum simulation_output/pi05_libero_activation_inspection_v1/report.json"
```

After disconnect, inspect started.json PID/status/failure/partial trace; never
infer a retry or overwrite is required. Preserve the completed evidence. Final
observed view0/view1 tanh abs>=.99 fractions100%/99.7765%, versus step0
47.5405%/40.4977%; about90.3% of final outputs are exactly abs=1. This is strong
descriptive saturation evidence, not proof of the only cause or of a repair's
benefit. Downstream context/history are not universally saturated. Proposed next
work is a train-only preprojection visual centering/scaling branch plus schema/
forward checks; no repair or matched-budget training has been performed or
automatically authorized. Raw visual skip/targets and state/action normalization
must remain unchanged in that prospective comparison.

Independent stdout-only read-only result audit passed on its first execution
(0.4307s computation, about9.4s SSH):183 input/3 output hashes before/after,
396 traces,13,960 moment records,2,280 tanh records and2,520 equal-episode macro
fields plus seed summaries/paired changes checked. Four unique-cache raw and
coordinate summaries were recomputed directly. No new model forward, checkpoint
deserialization or file write occurred. Model activation tensors were not
regenerated; coordinate-M2 had energy-consistency checks only, since coordinate
vectors were not retained. The archive is now downloaded and safely extracted
locally; archive/report SHA, completed status and all3 output hashes passed.
Full183-input validation ran remotely, not against locally absent checkpoints.

## Train-Only Visual Normalization Comparison (2026-09-13)

User approved normalization plus matched-budget validation. New protocol SHA:
`18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5`.
Only raw-feature inputs before world-model visual Linear layers are normalized;
raw skip/targets, native state/action normalization, learned module dimensions,
tanh, loss, seeds and draws stay fixed. No PI0.5 backbone optimization.

Safe repeatable synthetic checks (128 passed locally/remotely):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'; $env:PYTHONPATH='tools'
.\.venv\Scripts\python.exe -m unittest test_pi05_libero_visual_normalization test_run_pi05_libero_visual_normalization test_pi05_libero_activation_inspection test_pi05_libero_visual_dependence test_run_pi05_libero_visual_dependence test_pi05_libero_action_study_training test_pi05_libero_action_study_training_runner -q
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python tools/run_pi05_libero_visual_normalization.py --stage preflight --plan-sha256 18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5"
```

Completed smoke command, history only; do not rerun/overwrite:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_visual_normalization.py --stage smoke --plan-sha256 18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5 --execute"
```

Smoke report SHA `ffbf8903ba2cd48c2a7a0646b8fe38f2b73e471974df0d0fb2ff9c8d0a55cf56`;
visual_normalization.json SHA `71c5cf5573d2aa5fc7615b8cfc753b2c80fe59d0f0d03eef6e05cdf2660a5e2f`.
Complete8-episode1134-row train-only fit;12 no-update gradient checks passed.

Completed authorized training command, history only; do not launch a duplicate:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -u tools/run_pi05_libero_visual_normalization.py --stage train --plan-sha256 18b1454341b16ac2b1fd7139216fca92ba0be5d8b21245f5fcee6433298ac9e5 --smoke-report-sha256 ffbf8903ba2cd48c2a7a0646b8fe38f2b73e471974df0d0fb2ff9c8d0a55cf56 --execute"
```

One logical job estimates3-5minutes from same-host prior timings. Exactly2400
updates:3 seeds x2 action arms x2 variants x200, not an open-ended training run.
Fresh baseline must exactly reproduce the old final parameter/clean predictions.
No automatic resume, retry, extra steps, model selection or normalization sweep.
If SSH disconnects, first inspect PID in started.json, current process, per-arm
result files and trace/failure/status under
`simulation_output/pi05_libero_visual_normalization_v1/`. Do not overwrite an
existing output. Failed exclusive checkpoint writes may be partial; no atomic
publication or resumable optimizer state is claimed. Report archives over100MB
remain user-transferred; do not transfer new checkpoints unnecessarily.

Completed on first execution in221.6186s,2400 updates/1584 evaluation forwards.
All6 baseline final parameters and clean predictions replayed exactly. Observed
state MAE -3.8581%, visual MAE +0.6171%, objective -0.6721%; all three seeds have
the same direction for each primary metric. Saturation abs>=.99 improves from
100%/99.7765% to65.8615%/67.7845%, but remains substantial. Predeclared decision:
mixed_or_no_stable_overall_prediction_gain. Keep baseline, no automatic next run.

```text
simulation_output/pi05_libero_visual_normalization_v1/report.json
  a06f6957a6575502782e646eb76e5b69f75d8a4d57c1ab9ad9277008c5817520
simulation_output/pi05_libero_visual_normalization_reports_v1.tar.gz (10,632,904 bytes; no .pt)
  b1787185ae650f306215a7f04e5f61253aa6a948db9e36fa0403886ea05e5913
tools/audit_pi05_libero_visual_normalization.py
  a1764475325244a135013e27c9d3af5b5d2c6224f2b18725bfe1ee75f8127166
```

Safe read-only independent audit, already passed once in5.43451s:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/audit_pi05_libero_visual_normalization.py --report simulation_output/pi05_libero_visual_normalization_v1/report.json --report-sha256 a06f6957a6575502782e646eb76e5b69f75d8a4d57c1ab9ad9277008c5817520"
```

Audit checks196 input/28 output hashes,2400 update records/1584 eval records,
1200 old baseline update details/792 baseline prediction hashes,48 score groups,
train-only1134-row statistics and12 checkpoint file hashes. No model forward,
backward, torch import or checkpoint deserialization. Activation sufficient
statistics only are checked; stored model activation tensors are not regenerated.

Reports-only archive was retrieved locally after size/SHA checks and a strict
member-path allowlist, then report/completed status/all16 non-checkpoint output
hashes were verified. The12 checkpoints remain remote; full196-input/28-output
audit is remote evidence. Normalized primary initial abs>=.99 was only
0.013758%/0.007061%, rising to65.8615%/67.7845% by final200. Actual visual-residual
response to repeat-current/fixed-anchor grows41.89x/79.14x, but normalized state
target error under those interventions rises0.5660%/54.4605% versus its clean
condition. Larger response is not robustness or a reason to adopt the checkpoint.

## Read-Only Normalization Saturation / Error Diagnosis (2026-09-13)

Completed diagnosis only; no training, backward, new encodings, architecture or
loss changes. Do not rerun the exclusive diagnosis output or invoke the old
training stage. Entry point/protocol/helper hashes, sequentially SCP'd and
verified before remote execution:

```text
docs/libero-normalization-diagnosis-plan-v1.json
  9c2e938a9655c89116b0479c4d659d3b984c05e934208fe601bd48619aa1ec37
tools/run_pi05_libero_normalization_diagnosis.py
  dd1cdb324f28b288b5c8d9a1a7a3d2a0fdb7101250348682fb4f8ebc0f61c49c
tools/test_run_pi05_libero_normalization_diagnosis.py
  307b2b73a6f4de71e087825c42cad502a5b1c956ce593f484ceabcd29d8216ff
tools/pi05_libero_normalization_trace_diagnosis.py
  9c15f5454119b29eda77eddbfc2a5511ea88ca0fcfbc1e7278913b43e529bdf1
tools/test_pi05_libero_normalization_trace_diagnosis.py
  5dda04e55f97e7ad0028dcf20efa36fd138268c32b156e3fa3eb4fc257d2ed46
tools/pi05_libero_projection_geometry.py
  0fda06ceb866a195fe69f4edf15c325b3be3abeaeac27b011f972cac6114b63b
tools/test_pi05_libero_projection_geometry.py
  56f2d553b123fe741d234ed28bf909984322014cff6f63e269dff5beac6afe93
```

Safe synthetic checks (no public-data training);70 tests passed locally2.958s,
remotely2.001s, including36 new and34 previous normalization tests:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= PYTHONPATH=tools OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest test_run_pi05_libero_normalization_diagnosis test_pi05_libero_normalization_trace_diagnosis test_pi05_libero_projection_geometry test_run_pi05_libero_visual_normalization test_pi05_libero_visual_normalization"
```

Safe read-only frozen input/provenance preflight, passed233 input pins; includes
the existing independent normalization audit, not a new training execution:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_pi05_libero_normalization_diagnosis.py --stage preflight --plan-sha256 9c2e938a9655c89116b0479c4d659d3b984c05e934208fe601bd48619aa1ec37"
```

Completed actual command below is **history only**, not another user-run step:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_pi05_libero_normalization_diagnosis.py --stage diagnose --plan-sha256 9c2e938a9655c89116b0479c4d659d3b984c05e934208fe601bd48619aa1ec37 --execute"
```

First execution completed in11.467436s after preflight;396 frozen forwards,
0 backward/updates/encodings/rollout steps. All48 old episode scores and clean
prediction hashes exactly replayed;48 unique-row affine geometry pairs.
Expected1-3minute job used CPU float32/one thread; NumPy affine algebra uses
float64 on the exact float32 inputs. No covariance matrix/SVD, normalization
refit, intermediate-checkpoint fabrication or model/input intervention.

If a connection drops, inspect started.json PID/current process and
status.json/failure.json/report.json/error_trace.jsonl first. Preserve any
partial output; exclusive claim has no automatic retry, resume or overwrite.
No checkpoints were copied. All5 output files total4,230,025B, retrieved locally
via sequential SCP; >100MB files remain user-transferred.

```text
simulation_output/pi05_libero_normalization_diagnosis_v1/report.json
  a840d53abfbaf8574cdb53bb2e9ce567984ecd9f85cf68677e06f6e5b3282427
simulation_output/pi05_libero_normalization_diagnosis_v1/error_trace.jsonl
  cbd70f9a8fc6c1772b40ab9e79f412b632601171e3f71d677fed37e731b68bec
```

Safe local report/readback checks; remote preflight remains the full input and
checkpoint provenance check, not a claim that old checkpoints exist locally:

```powershell
$d = 'simulation_output/pi05_libero_normalization_diagnosis_v1'
$r = Get-Content -Raw "$d/report.json" | ConvertFrom-Json
if ((Get-FileHash "$d/report.json" -Algorithm SHA256).Hash.ToLower() -ne 'a840d53abfbaf8574cdb53bb2e9ce567984ecd9f85cf68677e06f6e5b3282427') { throw 'Report hash differs' }
$status = Get-Content -Raw "$d/status.json" | ConvertFrom-Json
if ($status.status -ne 'completed' -or $status.optimizer_steps -ne 0 -or $status.report_sha256 -ne 'a840d53abfbaf8574cdb53bb2e9ce567984ecd9f85cf68677e06f6e5b3282427') { throw 'Completed status differs' }
foreach ($p in $r.output_sha256.PSObject.Properties) {
    if ((Get-FileHash "$d/$($p.Name)" -Algorithm SHA256).Hash.ToLower() -ne $p.Value) { throw "Output hash differs: $($p.Name)" }
}
```

Independent read-only local audit passed0.625s:2400 old training records,
1380 interval/parameter summaries,198 paired trace rows/396 old prediction
hashes,48 affine pairs,1368 bucket records and all48 old per-episode scores.
No torch/forward/backward/SSH/writes; it checked saved arithmetic, not tensors.
Primary normalized train preactivation RMS~0.59 to~5.76-6.02 despite valid
input normalization; correlated-coordinate energy amplification matters.
Primary mean visual SmoothL1+0.0000152673 is offset by weighted-state loss
-0.0000660372; total-0.0000507699. Step1 visual MAE+3.0462%, step2+0.1182%,
step3-0.3265%. No saturation-onset, unique-cause or capacity-shortage claim.
Keep baseline and preserve variants; any projection-output scale-control
implementation or matched-budget training is a separate proposal, not authorized
by this read-only diagnostic. No data/SOFA/shared/weekly document update.

## Projection-Output Scale Control: Zero-Update Gate (2026-09-13)

This approved step implements/checks only a fixed output scale after each visual
Linear and before the existing tanh, on top of unchanged input normalization.
No full training stage exists in this entrypoint. Its new schema/version marker
must not be retrofitted into an old checkpoint to relabel its computation.

Final files, sequentially SCP'd and remotely hash-verified before execution:

```text
docs/libero-projection-scale-control-smoke-plan-v1.json
  8976c32a2761c512220d1b6d102f0340411617cf213b9c4d1a48a9545b4aebe8
tools/pi05_libero_projection_scale_control.py
  4efd3a5d22eaac86a0fc0576bafff209a280d99980182ee624ddf02a349d3fd5
tools/test_pi05_libero_projection_scale_control.py
  c54ed47563d95f95e1360215433266a7937556a51b24d9053e98c213233765a6
tools/run_pi05_libero_projection_scale_control.py
  3be52470e9cc32994e16e2d75e5b831b7a2bb2f31b5bc08720a6fa7e3d3f5d91
tools/test_run_pi05_libero_projection_scale_control.py
  f58d668f86bac29481cf49a481563eac9c4acff82579307e1aebef0c716089cb
```

Safe synthetic checks:60 final tests passed locally2.520s and remotely1.088s.
No real training/feature extraction; synthetic backward checks are intentional.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= PYTHONPATH=tools OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest test_run_pi05_libero_projection_scale_control test_pi05_libero_projection_scale_control test_run_pi05_libero_visual_normalization test_pi05_libero_visual_normalization"
```

The first remote test suite stopped59/60 before preflight/smoke. A test's
contiguous clone selected a different float32 Linear path from the actual
strided input, giving1.1920929e-7 difference. Only test capture was repaired to
recompute in the actual layout/context; byte-exact assertions and production
model/runner were unchanged. Details and superseded pre-smoke hashes are in the
algorithm handoff. Do not reuse the first test/plan hashes above the final gate.

Safe read-only preflight (passed242 input pins; no model/gradient execution):

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_pi05_libero_projection_scale_control.py --stage preflight --plan-sha256 8976c32a2761c512220d1b6d102f0340411617cf213b9c4d1a48a9545b4aebe8"
```

Completed command, **history only; do not rerun or overwrite**:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_pi05_libero_projection_scale_control.py --stage smoke --plan-sha256 8976c32a2761c512220d1b6d102f0340411617cf213b9c4d1a48a9545b4aebe8 --execute"
```

First actual smoke completed1.224837s after preflight, within the<1minute
whole-job estimate. Exactly24 full-model forwards/12 backward checks across
3 seeds x2 action arms x2 variants;0 optimizer construction/updates,0 validation
forwards,0 trained checkpoint loads/saves,0 feature encodings or rollouts.
Additional affine/scale identity arithmetic in capture hooks is not counted as
full-model forwards. Each seed uses the old first shared16-window train batch;
all6 original input-normalized predictions/activations/gradient reports replay
exactly. No training split, label, loss or mean/std change. All parameters and
buffers remain unchanged, including23 parameter tensors/1,321,352 elements.

Fixed operator `z / sqrt(1 + mean(z**2, last_hidden_axis))` uses float64
intermediates and returns float32. No trainable affine/extra parameters. Initial
controlled vector-RMS maxima0.57402-0.66769; all controlled projection gradients
finite/nonzero in these batches. This is no-update interface evidence, not
proof of preventing saturation after training or improving prediction quality.

```text
simulation_output/pi05_libero_projection_scale_control_smoke_v1/report.json
  16a2d07a686bb16b248d84ebaca1797a97eeb9886c5fbded8a3e85b5dcb25d66
```

All4 reports total333,768B were transferred locally, no checkpoints. If SSH
disconnects, inspect output started.json PID/process and status/failure/report
first. Preserve partial outputs; no automatic retry, overwrite or resume.
Single files/archives>100MB remain user-transferred.

Safe local readback:

```powershell
$d = 'simulation_output/pi05_libero_projection_scale_control_smoke_v1'
$sha = '16a2d07a686bb16b248d84ebaca1797a97eeb9886c5fbded8a3e85b5dcb25d66'
if ((Get-FileHash "$d/report.json" -Algorithm SHA256).Hash.ToLower() -ne $sha) { throw 'Report hash differs' }
$r = Get-Content -Raw "$d/report.json" | ConvertFrom-Json
$s = Get-Content -Raw "$d/status.json" | ConvertFrom-Json
if ($s.status -ne 'completed' -or $s.report_sha256 -ne $sha -or $s.optimizer_steps -ne 0) { throw 'Completion status differs' }
foreach ($p in $r.output_sha256.PSObject.Properties) {
    if ((Get-FileHash "$d/$($p.Name)" -Algorithm SHA256).Hash.ToLower() -ne $p.Value) { throw "Output hash differs: $($p.Name)" }
}
```

Next is only a proposal for a separately authorized equal-budget comparison
with a new training/checkpoint entrypoint. That implementation/run has not
started. Preserve original baseline and prior diagnostic variants; no loss
retuning, scale sweep, capacity upgrade, PI0.5 backbone training, project-data
inspection or data/SOFA/shared/weekly document changes follow automatically.

## Projection Scale: Fixed Matched Training / Final Checkpoints (2026-09-13)

The next user approval authorized this new entrypoint and bounded comparison;
the original interface-only smoke command above remains frozen and is not a
training command. This new comparison has now completed. **Do not rerun, resume,
overwrite, extend steps or promote its checkpoints automatically.**

Final new files, sequentially SCP'd and remote-hash verified before execution:

```text
docs/libero-projection-scale-training-plan-v1.json
  a2e4856276e9b415c6044660b79a0c4aa342a6943c661eef1ac6d19eb8b7e7dd
tools/run_pi05_libero_projection_scale_training.py
  a755e222af79d34d201f8951f4320ae82e111f39edc5380e75043051742f68d2
tools/test_run_pi05_libero_projection_scale_training.py
  dfef28cb2119c6d1fb2764db2ae310365da05446f0d9f16e4e8821bc183b6f50
tools/pi05_libero_projection_scale_checkpoint.py
  2447c0e1fee331529c28419889f83e8899ff2b63f76a17c26d218e55d28e59ae
tools/test_pi05_libero_projection_scale_checkpoint.py
  ac683b1527a3699e1184cf483a2ceb6c2da1a1bf9d92fc5b2844a8a822ec140e
```

Safe synthetic checks:13 new runner +13 new checkpoint +60 frozen regression
tests,86 total; local6.089s, remote3.664s, all passed. Checkpoint tests fabricate
final optimizer counters/parameter perturbation only; they do not train data.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= PYTHONPATH=tools OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -m unittest test_run_pi05_libero_projection_scale_training test_pi05_libero_projection_scale_checkpoint test_run_pi05_libero_projection_scale_control test_pi05_libero_projection_scale_control test_run_pi05_libero_visual_normalization test_pi05_libero_visual_normalization"
```

Safe read-only preflight passed250 input pins, no training:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B tools/run_pi05_libero_projection_scale_training.py --stage preflight --plan-sha256 a2e4856276e9b415c6044660b79a0c4aa342a6943c661eef1ac6d19eb8b7e7dd"
```

Completed command, **history only, do not execute again**:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 project4090 "cd /home/zsw/project_2026 && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/zsw/miniconda3/envs/project2026-pi/bin/python -B -u tools/run_pi05_libero_projection_scale_training.py --stage train --plan-sha256 a2e4856276e9b415c6044660b79a0c4aa342a6943c661eef1ac6d19eb8b7e7dd --execute"
```

First actual run passed in166.388920s after preflight, within3-5minute estimate
(prior matched study221.619s with twice the eval forwards). CPU float32/one
thread, torch2.10.0+cu128, deterministic, no AMP/TF32/CUDA initialization; only
fixed scale arithmetic has float64 intermediates. Exactly3 seeds x2 arms x2
variants x200 train forwards/backwards/updates =2400; clean step0/final200
evaluation792 forwards;12 final checkpoint saves/reloads,0 new encodings/rollouts.
Original frozen split, draws, targets, loss, optimizer and parameter count stay
unchanged. Normalized reference1200 full training records,396 clean predictions/
scores and6 final model states replay exactly; all250 input hashes unchanged.

```text
simulation_output/pi05_libero_projection_scale_training_v1/report.json
  52476e5ff1d3eae68ae90d08758f3a33a9e989e6201119f175fab9d80ddaae4c
simulation_output/pi05_libero_projection_scale_training_v1/status.json
  32905a8f7ff77a4072e58241f4f8406e835ca972f7c5d1c793ac875871e0e1a2
```

Conclusion: primary state/visual MAE mean changes-0.71866%/-0.10665%, but
seed20260913 objective+0.44200%, h3 visual MAE+0.03833%, h3 state MAE+0.50992%.
The prospectively fixed primary gate fails. Final two-view tanh saturation
means fall65.86145%/67.78448% ->0.68265%/0.43484%; numerical desaturation is
verified, stable overall prediction benefit is not. Against original raw
baseline, visual MAE still worsens0.50982%. Do not claim policy/robustness/
architecture-innovation/formal/real-system benefit or automatically upgrade
capacity. See the handoff for all horizon means and interpretation.

Strict checkpoint APIs are `save_reload(path, model, optimizer, registry=...,
visual_stats=..., binding=...)` and `load_checkpoint(path, expected_sha256=...,
expected_binding=..., registry=..., visual_stats=...)`. Both return `(model,
evidence)`. Require the new final-only envelope and explicit normalization,
scale-formula/version, seeded initialization and final-state bindings. Old native/
normalization-only files are incompatible; never inject a version marker to
relabel an old checkpoint. Loader performs no optimization or resume.

Independent read-only post-run audit (0.826s compute) verified all28 output
hashes including12 checkpoints,250 input pins,2400/792 trace counts, complete
window coverage, exact reference replay and recomputed every reported summary/
delta/SD/raw comparison/guard. No checkpoint loads or model forwards in that
audit. Reports are evidence, not model promotion or visual acceptance.

The complete remote output directory is about103MiB; each checkpoint is about
5.2MiB. Keep all12 checkpoints and full traces remote; only report/status/
preflight/started JSON files are retrieved locally. No >100MB archive is built
or transferred. Larger single files/archives remain user-transferred.

Safe local report readback (not a claim all remote artifacts exist locally):

```powershell
$d = 'simulation_output/pi05_libero_projection_scale_training_v1'
$sha = '52476e5ff1d3eae68ae90d08758f3a33a9e989e6201119f175fab9d80ddaae4c'
if ((Get-FileHash "$d/report.json" -Algorithm SHA256).Hash.ToLower() -ne $sha) { throw 'Report hash differs' }
$r = Get-Content -Raw "$d/report.json" | ConvertFrom-Json
$s = Get-Content -Raw "$d/status.json" | ConvertFrom-Json
if ($s.status -ne 'completed' -or $s.report_sha256 -ne $sha -or $s.optimizer_steps -ne 2400) { throw 'Completion differs' }
foreach ($name in @('preflight.json','started.json')) {
    if ((Get-FileHash "$d/$name" -Algorithm SHA256).Hash.ToLower() -ne $r.output_sha256.$name) { throw "Hash differs: $name" }
}
$r.summary.primary_mean_MAE_guards
```

If SSH disconnects during a future authorized run, inspect output started.json
PID/process and status/failure/report first; a connection failure does not prove
the experiment failed. Preserve all partial artifacts, no automatic retry or
resume. This completed v1 needs no recovery/re-execution. The initial ad-hoc
remote readback had a PowerShell quoting SyntaxError, and an early local read
hit the in-progress SCP file lock; both were read-only inspection failures,
not training failures. Correct literal-stdin remote readback and completed-SCP
hash checks are used for final verification.
