# Real10 可见头段粗标：六窗入口

本轮只标 #4、#39、#62、#65、#77、#100，Side / Top 各首、中、末三帧，共36个帧位。
这是独立的诊断标注，不是新训练集；不需要重标100窗，不启动训练。

## 操作

1. 在页面选择窗口和首 / 中 / 末帧，分别检查 Side、Top。
2. **粗框**：拖动框住当前帧可辨认的红色头段，不必精确贴边。不要框整个血管或整根导丝。
3. **短折线**：沿可见头段点击至少两个点，再按“完成线段”。如中间被遮挡，用“另起一段”
   标另一段可见部分，不要跨遮挡连接。重新直接画线或框会替换当前帧几何；“另起一段”才是追加。
4. 有可靠几何后，选择“完整可辨 / 仅部分可辨 / 完整性不确定”。不知道完整长度时，不必硬选完整。
5. 无法画出可靠部分时，选“可见但无法可靠圈定”或“不可见 / 无可靠头段”。尚未检查的帧保持“未标”。
6. 填标注人，按“保存本窗口”，或“保存并到下一窗口”。同窗切帧保留草稿；换窗前必须保存。
   未完成的折线需先完成或取消。可以保存部分完成的窗口，之后继续。

“清空本帧”只清除该帧新几何和新可见状态，回到未标；不会修改旧点。只有按保存才追加到文件。
局部 / 全图与1:1 / 2倍只改变显示，保存坐标始终为原图1920×1080像素；图像没有增强。
每帧、每视角可选择不同标法。短折线无方向，其端点不是自动的尖端标签，也不承诺同一材料点。

## 标注边界

- 约25mm是用户提供的头段物理描述，不是已校准像素长度，不按固定长度补全。
- 粗框只代表大致范围，框内可能有背景；不是像素分割mask。需要跳过遮挡时优先断开的短折线。
- 能可靠标出一部分，但不知道是否完整：`visible + uncertain`；完全无法可靠定位：`ambiguous + null`。
- Side/Top可见性独立，不能用另一路替本路补画；模糊不自动等于不可见。
- 不要求连续匀速运动，也不把头段位置变化直接解释成沿管推进。#65尤其不能据此改原响应标签。
- 旧尖端 / 材料点为默认关闭的黄色只读叠加；既有窗口响应折叠显示，不在本页修改。
- 这六窗已被反复诊断，不是独立测试集；本轮没有模型收益、contact/tactile或实机结论。

## 文件与schema

入口：`tools/review_real10_head_segment_annotation.py`，界面：`tools/real10_head_segment_annotation.html`。
新包：`simulation_output/real10_head_segment_annotation_v1/`。

- `manifest.json`：固定案例、关键帧、原图路径、只读旧点/响应和来源。
- `source_manifest.json`、`source_points.jsonl`、`source_responses.jsonl`：准备时的原文件字节快照。
- `entrypoint_snapshot.py`、`interface_snapshot.html`：准备时入口快照。
- `head_annotations.jsonl`：首次人工保存后才创建；按窗口追加revision，不覆盖历史。

每个窗口独立revision，保存包含两视角的三个帧位，可以有未标帧。并行页面的旧revision会被拒绝；
只启动一个服务实例，不要同时用另一个进程编辑同一新包。

```json
{
  "status": "visible",
  "coverage": "partial",
  "geometry": {"kind": "polyline", "segments": [[[800, 740], [820, 735]], [[835, 730], [850, 725]]]},
  "notes": "示意坐标，不是真实标注；中间遮挡，不连线"
}
```

粗框使用 `{"kind":"bbox","xyxy":[x0,y0,x1,y1]}`。`coverage` 仅可为
`complete / partial / uncertain`；非visible帧的 `coverage` 与 `geometry` 都为JSON `null`。
`unreviewed`、`ambiguous`、`not_visible`相互不同，不能自动变成背景/静止标签。
所有新标注 `policy_input_allowed=false`、`formal_data_allowed=false`、`deployable=false`。

## 启动 / 恢复

在项目根目录，首次准备（已有包时禁止重复准备）：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --prepare
```

之后启动或关闭后恢复（不加 `--prepare`）：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --port 8795
```

访问 `http://127.0.0.1:8795/`，Ctrl+C关闭服务不删除已保存内容，未保存的页面草稿不会自动落盘。
本机原图在 `collected_data`；远端缺这些原图时不要在那里启动真人标注页，也不需要为本轮重传大图。

查看进度，不新增标签：

```powershell
.\.venv\Scripts\python.exe -B tools/review_real10_head_segment_annotation.py --summary
```

测试只能指定另外的含 `fixture` 名称的包，并在准备时加 `--fixture`；测试标注会记录
`annotation_source=ui_fixture_not_human`。不得把测试副本的框/线复制到人工包。
