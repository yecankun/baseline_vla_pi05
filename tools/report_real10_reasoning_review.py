"""Render a Chinese evidence table from completed offline review artifacts."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from review_real10_reasoning import ROOT, jsonl, read, write


def render(out):
    report, packets, reviews = read(out/"report.json"), jsonl(out/"packets.jsonl"), jsonl(out/"reviews.jsonl")
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = lambda size: ImageFont.truetype(font_path, size, index=2)
    canvas = Image.new("RGB", (1500, 1200), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 25), "Real10 推理审核第一版｜离线回溯", font=font(40), fill="#14263b")
    draw.text((48, 87), "流程已跑通；尚未证明推理优于简单规则或提高任务成功率", font=font(28), fill="#713f12")
    decisions = report["agent_decisions"]
    headline = f'{report["cases"]} 个候选 / {report["runs"]} 次运行     继续 {decisions.get("continue_candidate",0)} · 暂停 {decisions.get("pause_review",0)} · 补观察 {decisions.get("observe_again",0)}     与规则一致 {report["agreement_with_rule_count"]}/{report["cases"]}'
    draw.text((48, 142), headline, font=font(26), fill="#334155")
    columns = [58, 210, 480, 765, 1070]
    headers = ["候选", "参考方向投影（mm）", "推理审核", "简单规则", "原程序下发 / 到位"]
    draw.rounded_rectangle((40, 198, 1460, 242), radius=8, fill="#1e3a5f")
    for x, label in zip(columns, headers):
        draw.text((x, 203), label, font=font(24), fill="white")
    labels = {"continue_candidate":"继续候选", "pause_review":"暂停核查", "observe_again":"补充观察"}
    colors = {"continue_candidate":"#1d4ed8", "pause_review":"#92400e", "observe_again":"#6d28d9"}
    for i, (packet, decision) in enumerate(zip(packets, reviews)):
        y = 250 + i*43
        different = packet["rule_baseline"]["decision"] != decision["decision"]
        draw.rectangle((40,y-2,1460,y+39), fill="#ede9fe" if different else ("white" if i%2==0 else "#eaf0f6"))
        o = report["outcomes"][packet["id"]]
        values = [packet["id"], f'{packet["reference_alignment"]["projected_step_mm"]:+.3f}',
                  labels[decision["decision"]], labels[packet["rule_baseline"]["decision"]],
                  ("是" if o["move_attempted"] else "否")+" / "+("是" if o["target_reached"] else "否")]
        for j,(x,value) in enumerate(zip(columns, values)):
            color = colors[decision["decision"]] if j==2 else "#243447"
            draw.text((x,y),value,font=font(25),fill=color)
    notes = [
        "紫底：两处不同建议，均缺少任务正确性标签，不能判定推理更好。",
        "参考投影仅表示机械臂沿记录路径的方向；它不是导丝推进量或错误标签。",
        "历史下发15次、到位15次、递丝发包1次；任务成功与递丝物理完成仍未知。",
        "审核者已知历史结论并可见规则；当前/未来事后字段未进入逐步输入。",
        "全部16张动作前图已查看，但导丝尖端与推进量无法可靠判读。",
        "误拦率、漏检率、成功率收益：不可计算。没有连接硬件或运行自动在线闭环。",
    ]
    for i,note in enumerate(notes):
        draw.text((48,954+i*35),note,font=font(23),fill="#334155")
    canvas.save(out/"summary.png")
    page = (out/"index.html").read_text(encoding="utf-8")
    if 'src="summary.png"' not in page:
        page = page.replace("<section>", '<p><img src="summary.png" alt="离线审核结果总览"></p><section>', 1)
        (out/"index.html").write_text(page, encoding="utf-8")
    write(out/"figure-manifest.json", {"sources":["report.json","packets.jsonl","reviews.jsonl"],
          "figure":"summary.png","visual_status":"not_viewed","user_acceptance":"pending"})
    print(out/"summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"simulation_output/real10_reasoning_review_v1")
    render(parser.parse_args().out)
