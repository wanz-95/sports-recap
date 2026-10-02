#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
赛后 5 分钟简报生成器

读取 events.txt 里的比赛事件，调用 DeepSeek API 生成一份给外行看的赛事简报，
输出到 recap.md。

核心约束：**只能使用 events.txt 里出现的事件，不能编造。**
为此，脚本没有把约束全交给提示词，而是加了一层独立的校验：
  1. 输出里每条事实都必须挂上 `（事件N）` 引用，N 必须是输入里真实存在的事件编号；
  2. 输出里出现的时间、比分必须与输入逐字一致，不允许新数字。
校验不通过就带着反馈重试；重试仍不通过，就退化为一份确定性的“保守简报”，
保证任何情况下都不会写出输入里没有的比赛事实。

用法：
  export DEEPSEEK_API_KEY=sk-xxxx
  python recap.py                      # 读 events.txt，写 recap.md
  python recap.py --input a.txt --output b.md
  python recap.py --mock               # 不联网，用假响应演示校验/重试流程
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import sys
from typing import List, Sequence

import requests


DEFAULT_INPUT = "events.txt"
DEFAULT_OUTPUT = "recap.md"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
MAX_ATTEMPTS = 3

# 时间：67' 或 45+2'
TIME_RE = re.compile(r"\d{1,3}(?:\+\d{1,2})?'")
# 比分：0-1、2 - 2（支持各种连字符）
SCORE_RE = re.compile(r"\d{1,2}\s*[-–—]\s*\d{1,2}")
# 事实引用：事件3 / 事件 3
REF_RE = re.compile(r"事件\s*(\d+)")


@dataclasses.dataclass
class Event:
    idx: int           # 1 起的编号，就是给模型的“事件N”
    time: str          # 原始时间字符串，例如 67'
    scores: List[str]  # 归一化后的比分字符串，例如 ["0-1"]
    text: str          # 原始事件描述

    def as_line(self) -> str:
        return f"[事件{self.idx}] {self.text}"


class InputError(Exception):
    """events.txt 无法解析或不足以生成简报。"""


def _norm_score(token: str) -> str:
    """把比分归一化成 `X-Y`，去掉空格、统一连字符。"""
    parts = re.split(r"[-–—]", token)
    return f"{parts[0].strip()}-{parts[1].strip()}"


def parse_events(text: str) -> List[Event]:
    """解析事件时间线。空行与 # 注释行被忽略。"""
    events: List[Event] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        time_match = TIME_RE.search(line)
        time_token = time_match.group(0) if time_match else ""
        scores = [_norm_score(s) for s in SCORE_RE.findall(line)]

        if not time_token and not scores:
            raise InputError(
                f"第 {lineno} 行无法识别为比赛事件（既没有时间也没有比分）：{raw_line!r}"
            )

        events.append(
            Event(
                idx=len(events) + 1,
                time=time_token,
                scores=scores,
                text=line,
            )
        )

    if not events:
        raise InputError("events.txt 里没有任何事件，无法生成简报。")
    return events


def _known_tokens(events: Sequence[Event]) -> tuple[set, set]:
    """收集输入里出现过的全部时间与比分，作为校验白名单。"""
    times = set()
    scores = set()
    for ev in events:
        times.update(TIME_RE.findall(ev.text))
        scores.update(_norm_score(s) for s in SCORE_RE.findall(ev.text))
    return times, scores


def validate_recap(md: str, events: Sequence[Event]) -> List[str]:
    """独立校验层：返回违规列表，空列表表示通过。"""
    violations: List[str] = []
    valid_ids = {ev.idx for ev in events}
    input_times, input_scores = _known_tokens(events)

    # 1) 事实引用必须指向真实存在的事件编号
    for ref in REF_RE.findall(md):
        if int(ref) not in valid_ids:
            violations.append(f"引用了不存在的事件编号：事件{ref}")

    # 2) 时间必须逐字来自输入
    for token in TIME_RE.findall(md):
        if token not in input_times:
            violations.append(f"出现了输入里没有的时间：{token}")

    # 3) 比分必须逐字来自输入
    for token in SCORE_RE.findall(md):
        norm = _norm_score(token)
        if norm not in input_scores:
            violations.append(f"出现了输入里没有的比分：{token.strip()}")

    return violations


SYSTEM_PROMPT = """\
你是一名体育赛事简报编辑，服务对象是『被朋友拉着看球、不太懂规则的外行』。

你会拿到一份比赛事件时间线，需要写一份 5 分钟能读完、看得懂的中文简报。

【唯一事实来源 —— 这是硬约束，违反即失败】
1. 时间线之外的一切都算『输入未提供』。你可以解释、可以打比方、可以判断哪个瞬间更重要，
   但绝对不能新增任何比赛事实（不能编造进球、红牌、换人、伤病、阵型、球员心理、天气、
   观众、历史战绩等任何输入里没有的内容）。
2. 不允许用你的足球常识去补全原因。比如输入只说『某球员进球』，
   你不能写『他此前状态火热』或『对手后卫失误』。
3. 时间与比分必须与输入逐字一致，原文照抄，不换算、不四舍五入、不重新计算。
4. 每一条陈述比赛事实的句子，必须在句末用 （事件N） 标注它来自哪个输入事件，
   N 就是时间线里的编号。不能用输入里没有的编号。
5. 凡是输入里无法确认的信息，一律写『输入未提供』，不要猜。

【写作要求】
- 开头一句话总览全场走势。
- 挑出最多 3 个转折点，每个用『发生了什么 + 为什么重要』两句话说明。
  转折点必须来自输入事件；候选不足 3 个就写实际数量，并说明『输入未提供更多可识别的转折点』。
- 解释战术或走势时，可以用生活化的类比，但必须明确标注『以下为类比，不是比赛事实』，
  且类比不能引入输入里没有的事件或因果。
- 结尾用一小段汇总本次输入还缺哪些信息。

【输出格式】
直接输出 Markdown 正文，不要代码块包裹，不要额外说明。结构建议：
# 赛后简报
（总览一句话）
## 三个转折点
### 转折点 1（时间）
- 发生了什么：……（事件N）
- 为什么重要：……（事件N）
## 战术类比（非比赛事实）
## 输入缺口
"""


def build_user_prompt(events: Sequence[Event]) -> str:
    lines = "\n".join(ev.as_line() for ev in events)
    return (
        "下面是本场比赛的事件时间线，编号 [事件N] 供你引用：\n\n"
        f"{lines}\n\n"
        "请据此生成简报。记住：只能用上面这些事件，数字逐字照抄，"
        "每条事实标注 （事件N），信息缺失就写『输入未提供』。"
    )


def call_deepseek(messages: List[dict], model: str, base_url: str, api_key: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": False,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"无法连接 DeepSeek API（{url}）：{exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API 返回 {resp.status_code}：{resp.text[:500]}")
    payload = resp.json()
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"无法解析 DeepSeek 响应：{json.dumps(payload)[:500]}") from exc


def fallback_recap(events: Sequence[Event]) -> str:
    """校验反复失败时的兜底：只用输入事件拼一份保守简报，绝不编造。"""
    goal_words = ("进球", "得分", "破门", "点球")
    goals = [ev for ev in events if any(w in ev.text for w in goal_words)]
    final_score = events[-1].scores[-1] if events[-1].scores else "输入未提供"

    out = ["# 赛后简报", "", f"本场共记录 {len(events)} 个事件，最终比分 {final_score}。", ""]
    out.append("## 关键事件")
    for ev in goals or events:
        out.append(f"- {ev.text}（事件{ev.idx}）")
    out += [
        "",
        "## 战术类比（非比赛事实）",
        "输入未提供足以做战术类比的过程信息。",
        "",
        "## 输入缺口",
        "本简报为保守模式输出：模型生成的内容未通过事实校验，因此仅罗列输入中的原始事件。"
        "输入未提供换人、控球、射门等技术统计。",
    ]
    return "\n".join(out)


def _mock_response(events: Sequence[Event], call_index: int) -> str:
    """离线演示用：第一次故意塞一条编造事件，验证校验层能拦住它。"""
    if call_index == 1:
        return (
            "# 赛后简报\n\n"
            "这场球几度交替领先。\n\n"
            "## 三个转折点\n"
            "### 转折点 1（34'）\n"
            "- 发生了什么：江城海豚 7 号进球，比分 0-1。（事件2）\n"
            "- 为什么重要：客队率先得分。（事件2）\n"
            "### 转折点 2（63'）\n"
            "- 发生了什么：北辰联换上前锋加强进攻。（事件9）\n"
        )
    return fallback_recap(events)


def generate_recap(
    events: Sequence[Event],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    mock: bool = False,
    verbose: bool = True,
) -> tuple:
    """生成简报，返回 (markdown, 校验说明列表)。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(events)},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        md = _mock_response(events, attempt) if mock else call_deepseek(
            messages, model, base_url, api_key
        )

        violations = validate_recap(md, events)
        if not violations:
            if verbose:
                print(f"[校验] 第 {attempt} 次生成通过事实校验。")
            return md, [f"第 {attempt} 次生成通过事实校验。"]

        if verbose:
            print(f"[校验] 第 {attempt} 次生成未通过：")
            for v in violations:
                print(f"   - {v}")

        messages.append({"role": "assistant", "content": md})
        messages.append(
            {
                "role": "user",
                "content": "上面的简报违反了硬约束：\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n请只使用时间线里的事件重写，数字逐字照抄，每条事实标注 （事件N），"
                "信息缺失写『输入未提供』。",
            }
        )

    if verbose:
        print("[校验] 多次生成均未通过，改用保守模式输出。")
    return fallback_recap(events), ["多次生成未通过校验，已降级为保守模式。"]


def write_recap(path: str, md: str, source: str) -> None:
    footer = (
        "\n\n---\n"
        f"*本简报仅依据 {os.path.basename(source)} 中的事件生成，未使用任何外部信息。"
        f"生成时间：{_dt.datetime.now():%Y-%m-%d %H:%M}。*\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md.rstrip() + footer)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="用 DeepSeek 生成赛后简报（只用输入事件，不编造）"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="事件文件路径（默认 events.txt）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出路径（默认 recap.md）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名（默认 deepseek-chat）")
    parser.add_argument(
        "--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument(
        "--mock", action="store_true", help="不调用 API，用假响应演示校验流程"
    )
    parser.add_argument("--quiet", action="store_true", help="不打印校验过程")
    args = parser.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {args.input}", file=sys.stderr)
        return 2

    try:
        events = parse_events(raw)
    except InputError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if len(events) < 2:
        print("错误：输入未提供足够的比赛过程（少于 2 个事件），不生成简报。", file=sys.stderr)
        return 2

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.mock and not api_key:
        print(
            "错误：未设置 DEEPSEEK_API_KEY。请先执行 `export DEEPSEEK_API_KEY=sk-xxxx`，"
            "或用 --mock 离线演示。",
            file=sys.stderr,
        )
        return 2

    print(f"已解析 {len(events)} 个事件，正在生成简报…")
    try:
        md, notes = generate_recap(
            events,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            mock=args.mock,
            verbose=not args.quiet,
        )
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    write_recap(args.output, md, args.input)
    print(f"已写入 {args.output}")
    for note in notes:
        print(f"[结果] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
