#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛后简报 Web 界面。

做两件事：
  1. 读取 recap.md，把 Markdown 渲染成 HTML 显示在首页（保留标题、列表、表格）。
  2. 提供一个输入框：粘贴比赛事件 -> 交给 recap.py 生成简报 -> 写回
     events.txt / recap.md，并把生成结果和事实校验状态显示在页面上。

用法：
    python app.py
    浏览器打开 http://127.0.0.1:5000

调用 DeepSeek 需要先设置环境变量：export DEEPSEEK_API_KEY=sk-xxxx
没有 key 时，可以勾选页面上的「离线演示」用假响应跑通流程（不联网）。
"""

from __future__ import annotations

import os
from pathlib import Path

import markdown
from flask import Flask, render_template, request
from markdown.extensions import Extension

import recap

BASE_DIR = Path(__file__).resolve().parent
EVENTS_FILE = BASE_DIR / "events.txt"
RECAP_FILE = BASE_DIR / "recap.md"

MARKDOWN_EXTENSIONS = ["extra", "sane_lists"]

app = Flask(__name__)


class EscapeHTML(Extension):
    """关掉原始 HTML 直通，避免粘贴进来的内容变成可执行标签（XSS）。

    标题、列表、表格这些 Markdown 语法不受影响，只是不再放行 <script> 之类标签。
    """

    def extendMarkdown(self, md):  # noqa: N802 (Python-Markdown 要求的驼峰命名)
        md.preprocessors.deregister("html_block")
        md.inlinePatterns.deregister("html")


def render_markdown(text: str) -> str:
    """把 Markdown 转成 HTML，保留标题 / 列表 / 表格等格式。"""
    md = markdown.Markdown(extensions=MARKDOWN_EXTENSIONS + [EscapeHTML()])
    return md.convert(text)


def extract_title(md_text: str, fallback: str = "赛后简报") -> str:
    """取第一个一级标题作为网页标题。"""
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def generate_recap(events_text: str, offline: bool) -> dict:
    """把粘贴的事件交给 recap.py 生成简报。

    返回 dict：ok 表示是否成功产出新简报，status 是 ok / warn / error，
    另外带上校验说明 notes、违规明细 violations、错误提示 error 和生成的 markdown。
    """
    def fail(message: str) -> dict:
        return {
            "ok": False,
            "status": "error",
            "notes": [],
            "violations": [],
            "error": message,
            "recap_md": "",
        }

    if not events_text.strip():
        return fail("请先粘贴比赛事件文本，再点击生成。")

    # 1) 解析输入（复用 recap.py 的解析器）
    try:
        events = recap.parse_events(events_text)
    except recap.InputError as exc:
        return fail(f"无法解析输入：{exc}")

    if len(events) < 2:
        return fail("输入未提供足够的比赛过程（少于 2 个事件），不生成简报。")

    # 2) API key 检查（离线演示模式跳过）
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not offline and not api_key:
        return fail(
            "未设置 DEEPSEEK_API_KEY，无法调用 DeepSeek API。"
            "可在终端执行 export DEEPSEEK_API_KEY=sk-xxxx 后重启本应用，"
            "或勾选「离线演示」先用假响应跑通流程。"
        )

    # 3) 调用生成 + 事实校验（recap.py 内部会重试、失败则降级为保守简报）
    model = os.environ.get("DEEPSEEK_MODEL", recap.DEFAULT_MODEL)
    base_url = os.environ.get("DEEPSEEK_BASE_URL", recap.DEFAULT_BASE_URL)
    try:
        md, notes = recap.generate_recap(
            events,
            model=model,
            base_url=base_url,
            api_key=api_key,
            mock=offline,
            verbose=False,
        )
    except RuntimeError as exc:
        return fail(str(exc))

    # 4) 对最终稿再独立复核一遍，展示给用户看
    violations = recap.validate_recap(md, events)
    degraded = any("保守模式" in note for note in notes)
    status = "warn" if (degraded or violations) else "ok"

    # 5) 写回项目文件，保持 events.txt / recap.md 与页面一致
    EVENTS_FILE.write_text(events_text.rstrip() + "\n", encoding="utf-8")
    recap.write_recap(str(RECAP_FILE), md, str(EVENTS_FILE))

    return {
        "ok": True,
        "status": status,
        "notes": notes,
        "violations": violations,
        "error": "",
        "recap_md": md,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    events_text = read_text(EVENTS_FILE)
    recap_md = read_text(RECAP_FILE)

    offline = False
    status = None
    notes: list[str] = []
    violations: list[str] = []
    error = ""

    if request.method == "POST":
        events_text = request.form.get("events", "")
        offline = request.form.get("offline") == "1"

        result = generate_recap(events_text, offline)
        status = result["status"]
        notes = result["notes"]
        violations = result["violations"]
        error = result["error"]
        if result["ok"]:
            recap_md = result["recap_md"]

    return render_template(
        "index.html",
        page_title=extract_title(recap_md),
        recap_html=render_markdown(recap_md),
        has_recap=bool(recap_md.strip()),
        events_text=events_text,
        offline=offline,
        status=status,
        notes=notes,
        violations=violations,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
