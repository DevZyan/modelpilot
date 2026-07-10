"""Render a :class:`~modelpilot.report.DatasetReport` to a self-contained HTML page.

No templating engine, no external assets: the output is a single string with
inline CSS so it opens anywhere (browser, notebook, email) and needs no network.
"""
from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import DatasetReport, LeaderboardRow

_SEVERITY_COLOR = {"info": "#3b82f6", "warn": "#f59e0b", "high": "#ef4444"}


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _stat(label: str, value: str) -> str:
    return (f'<div class="stat"><div class="stat-label">{_esc(label)}</div>'
            f'<div class="stat-value">{_esc(value)}</div></div>')


def _patterns_html(report: "DatasetReport") -> str:
    if not report.patterns:
        return '<p class="empty">No notable patterns — clean, well-behaved dataset.</p>'
    items = []
    for p in report.patterns:
        color = _SEVERITY_COLOR.get(p.severity, "#6b7280")
        items.append(
            f'<div class="pattern" style="border-left-color:{color}">'
            f'<span class="badge" style="background:{color}">{_esc(p.severity.upper())}</span>'
            f'<span class="pattern-label">{_esc(p.label)}</span>'
            f'<span class="code">{_esc(p.code)}</span>'
            f'<div class="pattern-detail">{_esc(p.detail)}</div></div>'
        )
    return "\n".join(items)


def _recommendations_html(report: "DatasetReport") -> str:
    recs = report.recommendations
    max_score = max((r.score for r in recs), default=1.0) or 1.0
    min_score = min((r.score for r in recs), default=0.0)
    span = (max_score - min_score) or 1.0
    rows = []
    for i, r in enumerate(recs, 1):
        width = max(4.0, (r.score - min_score) / span * 100.0)
        reasons = "".join(
            f'<li class="{"pos" if reason.lstrip().startswith("+") else "neg"}">{_esc(reason)}</li>'
            for reason in r.reasons
        )
        reasons_block = f'<ul class="reasons">{reasons}</ul>' if reasons else ""
        rows.append(
            f'<div class="rec">'
            f'<div class="rec-head"><span class="rank">{i}</span>'
            f'<span class="rec-name">{_esc(r.name)}</span>'
            f'<span class="rec-score">{r.score:+.1f}</span></div>'
            f'<div class="bar"><div class="bar-fill" style="width:{width:.0f}%"></div></div>'
            f'{reasons_block}</div>'
        )
    return "\n".join(rows)


def _leaderboard_html(rows: "list[LeaderboardRow]") -> str:
    if not rows:
        return ""
    body = []
    for i, r in enumerate(rows, 1):
        if r.error:
            score_cell = f'<td class="failed" colspan="2">FAILED: {_esc(r.error)}</td>'
        else:
            score_cell = (f'<td class="num">{r.benchmark_score:.4f}</td>'
                          f'<td class="num muted">± {r.std:.3f}</td>')
        body.append(
            f'<tr><td class="num">{i}</td><td>{_esc(r.name)}</td>'
            f'<td class="num muted">#{r.heuristic_rank}</td>{score_cell}'
            f'<td class="num muted">{r.fit_seconds:.2f}s</td></tr>'
        )
    metric = next((r.metric for r in rows if not r.error), rows[0].metric)
    return (
        '<h2>Leaderboard <span class="sub">heuristic rank vs. measured score</span></h2>'
        '<table class="board"><thead><tr>'
        '<th>#</th><th>Model</th><th>Heuristic</th>'
        f'<th>{_esc(metric)}</th><th>std</th><th>fit</th>'
        '</tr></thead><tbody>' + "\n".join(body) + "</tbody></table>"
    )


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f7f8fa; color: #1a1d21; line-height: 1.5; }
.wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h1 .task { font-weight: 400; color: #6b7280; text-transform: capitalize; }
h2 { font-size: 1.15rem; margin: 32px 0 12px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }
h2 .sub { font-weight: 400; font-size: .8rem; color: #9ca3af; }
.tagline { color: #6b7280; margin: 0 0 24px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }
.stat { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; }
.stat-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #9ca3af; }
.stat-value { font-size: 1.15rem; font-weight: 600; margin-top: 2px; }
.pattern { background: #fff; border: 1px solid #e5e7eb; border-left: 4px solid; border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; }
.badge { color: #fff; font-size: .65rem; font-weight: 700; padding: 2px 7px; border-radius: 999px;
  letter-spacing: .03em; vertical-align: middle; }
.pattern-label { font-weight: 600; margin-left: 8px; }
.code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .72rem; color: #9ca3af;
  margin-left: 8px; }
.pattern-detail { color: #4b5563; font-size: .9rem; margin-top: 4px; }
.rec { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px;
  margin-bottom: 10px; }
.rec-head { display: flex; align-items: center; gap: 10px; }
.rank { width: 24px; height: 24px; border-radius: 50%; background: #1a1d21; color: #fff;
  display: inline-flex; align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; }
.rec-name { font-weight: 600; flex: 1; }
.rec-score { font-variant-numeric: tabular-nums; font-weight: 700; color: #2563eb; }
.bar { background: #eef0f3; border-radius: 999px; height: 6px; margin: 8px 0; overflow: hidden; }
.bar-fill { background: linear-gradient(90deg, #60a5fa, #2563eb); height: 100%; }
.reasons { margin: 8px 0 0; padding-left: 18px; font-size: .84rem; color: #4b5563; }
.reasons .pos { color: #15803d; }
.reasons .neg { color: #b91c1c; }
table.board { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb;
  border-radius: 10px; overflow: hidden; font-size: .9rem; }
.board th { text-align: left; background: #f3f4f6; padding: 8px 12px; font-size: .75rem;
  text-transform: uppercase; letter-spacing: .03em; color: #6b7280; }
.board td { padding: 8px 12px; border-top: 1px solid #eef0f3; }
.board td.num { text-align: right; font-variant-numeric: tabular-nums; }
.board td.muted { color: #9ca3af; }
.board td.failed { color: #b91c1c; }
.empty { color: #6b7280; font-style: italic; }
footer { margin-top: 40px; color: #9ca3af; font-size: .78rem; text-align: center; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #e5e7eb; }
  .stat, .pattern, .rec, table.board { background: #1a1d23; border-color: #2a2e37; }
  .board th { background: #22262f; color: #9ca3af; }
  .board td { border-top-color: #2a2e37; }
  h2 { border-bottom-color: #2a2e37; }
  .rank { background: #e5e7eb; color: #0f1115; }
  .bar { background: #2a2e37; }
  .pattern-detail { color: #9ca3af; }
}
"""


def render_html(report: "DatasetReport", *, include_leaderboard: bool = False,
                top_k: int = 5, cv: int = 5) -> str:
    """Return a full self-contained HTML document for ``report``."""
    p = report.profile
    stats = [
        _stat("Rows", f"{p.n_samples:,}"),
        _stat("Features", f"{p.n_features} ({p.n_numeric} num / {p.n_categorical} cat)"),
        _stat("Missing", f"{p.missing_fraction:.1%}"),
        _stat("Target signal", f"{p.target_signal:.2f}"),
    ]
    if p.task == "classification" and p.n_classes:
        imb = f" · {p.imbalance_ratio:.1f}x imb" if p.imbalance_ratio else ""
        stats.insert(2, _stat("Classes", f"{p.n_classes}{imb}"))

    leaderboard_section = ""
    if include_leaderboard:
        leaderboard_section = _leaderboard_html(report.leaderboard(top_k=top_k, cv=cv))

    best = report.recommendations[0].name if report.recommendations else "—"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modelpilot report</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>modelpilot report <span class="task">· {_esc(p.task)}</span></h1>
<p class="tagline">Best pick: <strong>{_esc(best)}</strong></p>
<div class="stats">{''.join(stats)}</div>
<h2>Patterns detected</h2>
{_patterns_html(report)}
<h2>Recommended models</h2>
{_recommendations_html(report)}
{leaderboard_section}
<footer>Generated by modelpilot · every recommendation shows its reasons</footer>
</div></body></html>"""
