#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, url_for

APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1])).resolve()
DOCS_PATH = Path(os.getenv("DOCS_PATH", str(APP_ROOT / "docs"))).resolve()
QUESTIONS_DIR = Path(os.getenv("QUESTIONS_DIR", str(APP_ROOT / "questions"))).resolve()
EDITOR_HOST = os.getenv("DOCS_EDITOR_HOST", "0.0.0.0")
EDITOR_PORT = int(os.getenv("DOCS_EDITOR_PORT", "8090"))
REINDEX_SCRIPT = os.getenv("REINDEX_SCRIPT", str(APP_ROOT / "scripts" / "reindex.py"))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python3")
ALLOWED_EXT = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv", ".html"}

QUESTIONS_LOG_FILE = Path(
    os.getenv("QUESTIONS_LOG_FILE", str(QUESTIONS_DIR / "questions.ndjson"))
)
NEW_QUESTIONS_LOG_FILE = Path(
    os.getenv("NEW_QUESTIONS_LOG_FILE", str(QUESTIONS_DIR / "new_questions.ndjson"))
)
UNANSWERED_LOG_FILE = Path(
    os.getenv("UNANSWERED_LOG_FILE", str(QUESTIONS_DIR / "unanswered_questions.ndjson"))
)
QUESTIONS_STATE_FILE = Path(
    os.getenv("QUESTIONS_STATE_FILE", str(QUESTIONS_DIR / "questions_state.json"))
)

DOCS_PATH.mkdir(parents=True, exist_ok=True)
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_LOG_FILE.touch(exist_ok=True)
NEW_QUESTIONS_LOG_FILE.touch(exist_ok=True)
UNANSWERED_LOG_FILE.touch(exist_ok=True)

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MaxKonvert KB Console</title>
  <style>
    body { font-family: sans-serif; margin: 0; background: #0b1020; color: #eaf0ff; }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 20px; }
    .top { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
    input, textarea, button { font: inherit; border-radius: 8px; border: 1px solid #2e3f66; background: #141d36; color: #eaf0ff; }
    input { padding: 10px 12px; min-width: 280px; }
    button { padding: 10px 14px; cursor: pointer; background: #1b66ff; border: none; }
    button:hover { opacity: .92; }
    .nav a { padding: 8px 12px; border: 1px solid #2e3f66; border-radius: 8px; background:#121a30; color:#9bc3ff; text-decoration:none; }
    .nav a.active { background: #1b66ff; color: white; border-color: transparent; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; }
    .panel { background: #121a30; border: 1px solid #2a3a60; border-radius: 12px; padding: 14px; }
    .files { max-height: 70vh; overflow: auto; }
    a { color: #9bc3ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    li { margin: 6px 0; word-break: break-word; }
    textarea { width: 100%; min-height: 68vh; padding: 12px; resize: vertical; line-height: 1.45; }
    .muted { color: #9eb0d8; font-size: 13px; }
    .ok { color: #73e2a7; }
    .err { color: #ff8c8c; }
    .stats { display:flex; gap:14px; flex-wrap: wrap; margin: 12px 0; }
    .stat { background:#0f1730; border:1px solid #2a3a60; border-radius:10px; padding:8px 12px; }
    table { width:100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid #25355a; text-align:left; padding:8px 6px; vertical-align: top; }
    code { white-space: pre-wrap; word-break: break-word; color:#c7d7ff; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>MaxKonvert KB Console</h2>
    <div class="top nav">
      <a class="{{ 'active' if view == 'docs' else '' }}" href="{{ url_for('home') }}">Docs</a>
      <a class="{{ 'active' if view == 'questions' else '' }}" href="{{ url_for('questions') }}">Questions</a>
      <form method="post" action="{{ url_for('reindex_now') }}">
        <button type="submit">Reindex Now</button>
      </form>
      <span class="muted">Docs: {{ docs_root }}</span>
      <span class="muted">Questions: {{ questions_root }}</span>
    </div>
    {% if msg %}<p class="ok">{{ msg }}</p>{% endif %}
    {% if err %}<p class="err">{{ err }}</p>{% endif %}

    {% if view == 'docs' %}
      <div class="top">
        <form method="post" action="{{ url_for('create') }}">
          <input name="path" placeholder="new-file.md or dir/new-file.md" required />
          <button type="submit">Create File</button>
        </form>
      </div>
      <div class="grid">
        <div class="panel files">
          <b>Files</b>
          <ul>
            {% for p in files %}
              <li><a href="{{ url_for('home', path=p) }}">{{ p }}</a></li>
            {% endfor %}
          </ul>
        </div>
        <div class="panel">
          {% if current_path %}
            <form method="post" action="{{ url_for('save') }}">
              <input type="hidden" name="path" value="{{ current_path }}" />
              <p><b>Editing:</b> {{ current_path }}</p>
              <textarea name="content">{{ content }}</textarea>
              <div style="margin-top:10px;"><button type="submit">Save</button></div>
            </form>
          {% else %}
            <p>Select a file on the left to edit.</p>
          {% endif %}
        </div>
      </div>
    {% else %}
      <div class="stats">
        <div class="stat">Total unique questions: <b>{{ unique_count }}</b></div>
        <div class="stat">Recent questions shown: <b>{{ all_questions|length }}</b></div>
        <div class="stat">Recent unanswered shown: <b>{{ unanswered|length }}</b></div>
        <div class="stat">Recent new shown: <b>{{ new_questions|length }}</b></div>
      </div>

      <div class="panel" style="margin-bottom:14px;">
        <h3 style="margin-top:0;">New Questions</h3>
        <table>
          <tr><th>ts</th><th>question</th><th>user</th><th>status</th></tr>
          {% for row in new_questions %}
            <tr>
              <td>{{ row.get('ts','') }}</td>
              <td><code>{{ row.get('question','') }}</code></td>
              <td>{{ row.get('user',{}).get('id','') }}</td>
              <td>{{ row.get('status','') }}</td>
            </tr>
          {% endfor %}
        </table>
      </div>

      <div class="panel" style="margin-bottom:14px;">
        <h3 style="margin-top:0;">Unanswered</h3>
        <table>
          <tr><th>ts</th><th>question</th><th>user</th><th>status</th></tr>
          {% for row in unanswered %}
            <tr>
              <td>{{ row.get('ts','') }}</td>
              <td><code>{{ row.get('question','') }}</code></td>
              <td>{{ row.get('user',{}).get('id','') }}</td>
              <td>{{ row.get('status','') }}</td>
            </tr>
          {% endfor %}
        </table>
      </div>

      <div class="panel">
        <h3 style="margin-top:0;">All Recent Questions</h3>
        <table>
          <tr><th>ts</th><th>question</th><th>user</th><th>best_score</th><th>status</th></tr>
          {% for row in all_questions %}
            <tr>
              <td>{{ row.get('ts','') }}</td>
              <td><code>{{ row.get('question','') }}</code></td>
              <td>{{ row.get('user',{}).get('id','') }}</td>
              <td>{{ row.get('best_score','') }}</td>
              <td>{{ row.get('status','') }}</td>
            </tr>
          {% endfor %}
        </table>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def read_json(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_ndjson(path: Path, limit: int = 200) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return list(reversed(rows[-limit:]))


def safe_rel(rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/")
    if not rel:
        raise ValueError("empty path")
    candidate = (DOCS_PATH / rel).resolve()
    if DOCS_PATH not in candidate.parents and candidate != DOCS_PATH:
        raise ValueError("path escape")
    if candidate.suffix.lower() not in ALLOWED_EXT:
        raise ValueError("unsupported extension")
    return candidate


def list_docs() -> list[str]:
    out = []
    for p in DOCS_PATH.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            out.append(str(p.relative_to(DOCS_PATH)))
    return sorted(out)


def render_docs(msg: str = "", err: str = "", rel: str = ""):
    content = ""
    current_path = ""
    if rel:
        try:
            full = safe_rel(rel)
            if full.exists():
                content = full.read_text(encoding="utf-8")
                current_path = rel
            else:
                err = f"File not found: {rel}"
        except Exception as e:
            err = str(e)
    return render_template_string(
        PAGE,
        view="docs",
        files=list_docs(),
        current_path=current_path,
        content=content,
        docs_root=str(DOCS_PATH),
        questions_root=str(QUESTIONS_DIR),
        msg=msg,
        err=err,
    )


def render_questions(msg: str = "", err: str = ""):
    state = read_json(QUESTIONS_STATE_FILE)
    return render_template_string(
        PAGE,
        view="questions",
        docs_root=str(DOCS_PATH),
        questions_root=str(QUESTIONS_DIR),
        unique_count=len(state),
        all_questions=read_ndjson(QUESTIONS_LOG_FILE, limit=300),
        unanswered=read_ndjson(UNANSWERED_LOG_FILE, limit=200),
        new_questions=read_ndjson(NEW_QUESTIONS_LOG_FILE, limit=200),
        msg=msg,
        err=err,
    )


@app.get("/")
def home():
    rel = request.args.get("path", "").strip()
    msg = request.args.get("msg", "").strip()
    err = request.args.get("err", "").strip()
    return render_docs(msg=msg, err=err, rel=rel)


@app.get("/questions")
def questions():
    msg = request.args.get("msg", "").strip()
    err = request.args.get("err", "").strip()
    return render_questions(msg=msg, err=err)


@app.post("/create")
def create():
    rel = request.form.get("path", "").strip()
    try:
        full = safe_rel(rel)
        full.parent.mkdir(parents=True, exist_ok=True)
        if not full.exists():
            full.write_text("# New Document\n", encoding="utf-8")
        return redirect(url_for("home", path=rel, msg=f"Created: {rel}"), code=303)
    except Exception as e:
        return redirect(url_for("home", err=str(e)), code=303)


@app.post("/save")
def save():
    rel = request.form.get("path", "").strip()
    content = request.form.get("content", "")
    try:
        full = safe_rel(rel)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return redirect(url_for("home", path=rel, msg=f"Saved: {rel}"), code=303)
    except Exception as e:
        return redirect(url_for("home", err=str(e), path=rel), code=303)


@app.post("/reindex-now")
def reindex_now():
    try:
        proc = subprocess.run(
            [PYTHON_BIN, REINDEX_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=240,
            check=False,
        )
        if proc.returncode == 0:
            return redirect(url_for("questions", msg="Reindex complete"), code=303)
        return redirect(url_for("questions", err=f"Reindex failed: rc={proc.returncode}"), code=303)
    except Exception as e:
        return redirect(url_for("questions", err=f"Reindex error: {e}"), code=303)


@app.get("/health")
def health():
    return {"status": "ok", "service": "maxkonvert-docs-editor"}


if __name__ == "__main__":
    app.run(host=EDITOR_HOST, port=EDITOR_PORT, debug=False)
