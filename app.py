"""
Stock Prediction Dashboard - Flask Web App (v2 - Fixed)
Chạy: python app.py
Truy cập: http://localhost:5000

Yêu cầu: pip install flask pandas
"""

from flask import Flask, render_template_string, jsonify
import json
import os
import glob
import pandas as pd
from datetime import datetime, date

app = Flask(__name__)

# ===================== CẤU HÌNH =====================
DATA_DIR = "/home/duckanh/airflow/data/stock"

# ===================== LOGIC ĐỌC DỮ LIỆU =====================

def load_actual_prices():
    """
    Merge TẤT CẢ processed_*.csv + processed.csv.
    Dedup + sort theo datetime thực (không phải string) → đúng thứ tự.
    Trả về (DataFrame, dict {date_str: close}).
    """
    all_dfs = []

    main_file = os.path.join(DATA_DIR, "processed.csv")
    if os.path.exists(main_file):
        try:
            df = pd.read_csv(main_file)
            df["Date"] = pd.to_datetime(df["Date"])
            all_dfs.append(df[["Date", "Close"]])
        except Exception as e:
            print(f"[WARN] processed.csv: {e}")

    for f in sorted(glob.glob(os.path.join(DATA_DIR, "processed_*.csv"))):
        try:
            df = pd.read_csv(f)
            df["Date"] = pd.to_datetime(df["Date"])
            all_dfs.append(df[["Date", "Close"]])
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=["Date", "Close", "date_str"]), {}

    combined = (pd.concat(all_dfs, ignore_index=True)
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")           # sort theo datetime object ← fix chính
                .reset_index(drop=True))
    combined["date_str"] = combined["Date"].dt.strftime("%Y-%m-%d")
    price_dict = dict(zip(combined["date_str"], combined["Close"].round(2)))
    return combined, price_dict


def load_all_predictions():
    """
    Đọc tất cả model_*_predictions.json.
    Sort theo last_actual_date DESC (run huấn luyện gần nhất lên đầu).
    Predictions trong mỗi run được sort theo datetime thực.
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "model_*_predictions.json")))
    future_file = os.path.join(DATA_DIR, "future_predictions.json")
    if os.path.exists(future_file):
        files.append(future_file)

    meta = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            # Sort predictions theo datetime object, không theo string
            raw_preds = data.get("predictions", {})
            sorted_preds = dict(sorted(
                raw_preds.items(),
                key=lambda kv: datetime.strptime(kv[0], "%Y-%m-%d")
            ))

            meta.append({
                "file":              os.path.basename(f),
                "prediction_date":   data.get("prediction_date", "")[:10],
                "last_actual_date":  data.get("last_actual_date", ""),
                "last_actual_close": data.get("last_actual_close"),
                "ticker":            data.get("ticker", "N/A"),
                "predictions":       sorted_preds,
            })
        except Exception as e:
            print(f"[WARN] {f}: {e}")

    # Sort DESC theo last_actual_date (dùng datetime để so sánh đúng)
    meta.sort(key=lambda x: x["last_actual_date"], reverse=True)
    return meta


def build_runs(meta_list, price_dict):
    """Ghép giá thực tế vào từng ngày dự đoán, tính MAPE/MAE."""
    runs = []
    for m in meta_list:
        rows = []
        for date_str, pred_val in m["predictions"].items():
            actual_val = price_dict.get(date_str)
            error = pct_error = None
            if actual_val is not None:
                error     = round(actual_val - pred_val, 2)
                pct_error = round((actual_val - pred_val) / actual_val * 100, 2)
            try:
                is_past = datetime.strptime(date_str, "%Y-%m-%d").date() <= date.today()
            except:
                is_past = False

            rows.append({
                "date":       date_str,
                "predicted":  pred_val,
                "actual":     actual_val,
                "error":      error,
                "pct_error":  pct_error,
                "has_actual": actual_val is not None,
                "is_past":    is_past,
            })

        compared = [r for r in rows if r["has_actual"]]
        summary = None
        if compared:
            mape = sum(abs(r["pct_error"]) for r in compared) / len(compared)
            mae  = sum(abs(r["error"])     for r in compared) / len(compared)
            summary = {
                "mape":       round(mape, 2),
                "mae":        round(mae, 2),
                "n_compared": len(compared),
                "n_total":    len(rows),
            }
        runs.append({**m, "rows": rows, "summary": summary})
    return runs


def build_chart_data(actual_df, selected_run, n_history=30):
    """
    Chart data cho một run:
    - n_history ngày thực tế gần nhất TRƯỚC last_actual_date (sort đúng)
    - Các ngày dự đoán của run đó (sort đúng)
    Đảm bảo trục X luôn tăng dần, không bị đảo.
    """
    last_actual = selected_run["last_actual_date"]

    # Lịch sử: actual_df đã sort tăng dần theo datetime
    hist = actual_df[actual_df["date_str"] <= last_actual].tail(n_history)

    pred_dates = set(selected_run["predictions"].keys())
    pred_map   = {r["date"]: r for r in selected_run["rows"]}

    labels, actuals, predicted = [], [], []

    for _, row in hist.iterrows():
        d = row["date_str"]
        labels.append(d)
        actuals.append(round(float(row["Close"]), 2))
        predicted.append(None)

    # Ngày dự đoán đã sort tăng dần trong build_runs
    for date_str in sorted(pred_dates, key=lambda s: datetime.strptime(s, "%Y-%m-%d")):
        r = pred_map[date_str]
        labels.append(date_str)
        actuals.append(r["actual"])
        predicted.append(r["predicted"])

    return {"labels": labels, "actual": actuals, "predicted": predicted}


def compute_global_stats(runs):
    all_compared = [r for run in runs for r in run["rows"] if r["has_actual"]]
    if not all_compared:
        return None
    mape = sum(abs(r["pct_error"]) for r in all_compared) / len(all_compared)
    mae  = sum(abs(r["error"])     for r in all_compared) / len(all_compared)
    return {
        "mape":           round(mape, 2),
        "mae":            round(mae, 2),
        "total_days":     len(all_compared),
        "total_runs":     len(runs),
        "runs_with_data": sum(1 for r in runs if r["summary"]),
    }


# ===================== HTML TEMPLATE =====================
HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Prediction Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#080c14;--s1:#0d1422;--s2:#111c2e;--s3:#162036;
  --border:#1c2d44;--accent:#00e5ff;--purple:#8b5cf6;
  --green:#22d3a0;--red:#ff4d6d;--yellow:#fbbf24;
  --text:#d4dff0;--muted:#4a637a;
  --mono:'JetBrains Mono',monospace;--sans:'Outfit',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}
body::after{
  content:'';position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,0.011) 2px,rgba(0,229,255,0.011) 4px);
  pointer-events:none;z-index:0
}
.wrap{max-width:1300px;margin:0 auto;padding:0 28px;position:relative;z-index:1}

/* HEADER */
header{
  border-bottom:1px solid var(--border);background:rgba(8,12,20,0.97);
  backdrop-filter:blur(16px);position:sticky;top:0;z-index:200;
}
.hdr{display:flex;align-items:center;justify-content:space-between;padding:15px 28px;gap:14px;flex-wrap:wrap}
.logo{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--accent);letter-spacing:1px;display:flex;align-items:center;gap:10px}
.logo-tag{font-size:10px;color:var(--muted);font-weight:300;letter-spacing:0}
.hdr-right{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.ticker-chip{
  font-family:var(--mono);font-size:12px;font-weight:700;color:var(--accent);
  background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.25);
  border-radius:6px;padding:4px 10px;letter-spacing:1px
}
.date-txt{font-family:var(--mono);font-size:11px;color:var(--muted)}
.live-dot{width:7px;height:7px;background:var(--green);border-radius:50%;animation:blink 2s infinite;flex-shrink:0}
@keyframes blink{0%,100%{opacity:1;box-shadow:0 0 6px var(--green)}50%{opacity:.3;box-shadow:none}}

/* GLOBAL STATS */
.gstats{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:13px;padding:24px 0 0}
.gc{
  background:var(--s1);border:1px solid var(--border);border-radius:12px;
  padding:17px 20px;position:relative;overflow:hidden;animation:up .4s ease both
}
.gc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--purple))}
@keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.gc-label{font-size:10px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px}
.gc-val{font-family:var(--mono);font-size:22px;font-weight:700;line-height:1;color:var(--accent)}
.gc-val.g{color:var(--green)}.gc-val.y{color:var(--yellow)}.gc-val.r{color:var(--red)}
.gc-sub{font-size:11px;color:var(--muted);margin-top:5px}

/* SECTION */
.section{background:var(--s1);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-top:18px;animation:up .4s ease .1s both}
.sec-head{display:flex;align-items:center;justify-content:space-between;padding:16px 22px;border-bottom:1px solid var(--border);flex-wrap:wrap;gap:10px;background:var(--s2)}
.sec-title{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px}
.sec-title::before{content:'';width:3px;height:14px;background:var(--accent);border-radius:2px;flex-shrink:0}

/* CHART */
.chart-wrap{padding:18px 22px;height:330px;position:relative}

/* CONTROLS */
.controls-bar{padding:14px 22px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--s2)}
.ctrl-label{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.run-select{
  flex:1;min-width:220px;background:var(--s3);border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-family:var(--mono);font-size:11px;padding:7px 32px 7px 11px;
  cursor:pointer;outline:none;appearance:none;-webkit-appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%234a637a'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 11px center;
}
.run-select:focus{border-color:var(--accent)}
.run-select option{background:var(--s3)}
.mape-chip{font-family:var(--mono);font-size:10px;border-radius:100px;padding:3px 9px;border:1px solid;white-space:nowrap}
.mc-g{color:var(--green);border-color:rgba(34,211,160,.35);background:rgba(34,211,160,.07)}
.mc-y{color:var(--yellow);border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.07)}
.mc-r{color:var(--red);border-color:rgba(255,77,109,.35);background:rgba(255,77,109,.07)}
.mc-n{color:var(--muted);border-color:var(--border)}

/* TIMELINE */
.timeline{padding:12px 22px;display:flex;gap:5px;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--s2)}
.tl-dot{
  font-family:var(--mono);font-size:10px;padding:4px 7px;border-radius:6px;
  border:1px solid var(--border);color:var(--muted);cursor:pointer;
  transition:all .15s;user-select:none;background:var(--s3)
}
.tl-dot:hover{border-color:var(--accent);color:var(--accent)}
.tl-dot.sel{background:rgba(0,229,255,.1);border-color:var(--accent);color:var(--accent)}
.tl-dot.has-cmp{border-color:rgba(34,211,160,.4)}

/* PILLS */
.pills-bar{padding:11px 22px;border-bottom:1px solid var(--border);background:rgba(13,20,34,.5);display:flex;gap:6px;flex-wrap:wrap}
.pill{font-family:var(--mono);font-size:10px;border-radius:100px;padding:3px 9px;border:1px solid var(--border);color:var(--muted);background:var(--s3)}
.pill.g{color:var(--green);border-color:rgba(34,211,160,.3)}
.pill.y{color:var(--yellow);border-color:rgba(251,191,36,.3)}
.pill.r{color:var(--red);border-color:rgba(255,77,109,.3)}

/* TABLE */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse}
thead tr{background:var(--s2);border-bottom:1px solid var(--border)}
th{padding:10px 18px;text-align:left;font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);font-weight:400;white-space:nowrap}
tbody tr{border-bottom:1px solid rgba(28,45,68,.5);transition:background .12s}
tbody tr:hover{background:rgba(255,255,255,.018)}
tbody tr:last-child{border-bottom:none}
td{padding:12px 18px;font-size:13px;white-space:nowrap}
.td-date{font-family:var(--mono);font-size:11px}
.td-n{font-family:var(--mono);font-size:11px}
.c-act{color:var(--green)}.c-pred{color:var(--accent)}
.c-pos{color:var(--green)}.c-neg{color:var(--red)}.c-mut{color:var(--muted)}
.badge{display:inline-flex;align-items:center;gap:3px;border-radius:5px;padding:2px 7px;font-size:10px;font-family:var(--mono)}
.b-g{background:rgba(34,211,160,.1);color:var(--green)}
.b-y{background:rgba(251,191,36,.1);color:var(--yellow)}
.b-r{background:rgba(255,77,109,.1);color:var(--red)}
.b-m{background:rgba(74,99,122,.1);color:var(--muted)}
.b-f{background:rgba(139,92,246,.1);color:#a78bfa}

.footer{text-align:center;padding:28px 0 44px;font-family:var(--mono);font-size:10px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <div class="hdr">
    <div class="logo">
      <div class="live-dot"></div>
      STOCK_PRED
      <span class="logo-tag">// Dashboard v2 · Backfill-ready</span>
    </div>
    <div class="hdr-right">
      <span class="ticker-chip">{{ ticker }}</span>
      <span class="date-txt">{{ today }}</span>
    </div>
  </div>
</header>

<div class="wrap">
{% if not runs %}
<div style="text-align:center;padding:80px;font-family:var(--mono);font-size:13px;color:var(--muted)">
  ⚠ Không tìm thấy file dự đoán trong <code>{{ data_dir }}</code>
</div>
{% else %}

<!-- GLOBAL STATS -->
<div class="gstats">
  <div class="gc" style="animation-delay:.0s">
    <div class="gc-label">Tổng runs</div>
    <div class="gc-val">{{ gstats.total_runs }}</div>
    <div class="gc-sub">{{ gstats.runs_with_data }} có dữ liệu thực tế</div>
  </div>
  <div class="gc" style="animation-delay:.05s">
    <div class="gc-label">Ngày so sánh</div>
    <div class="gc-val">{{ gstats.total_days }}</div>
    <div class="gc-sub">tổng toàn bộ backfill</div>
  </div>
  <div class="gc" style="animation-delay:.1s">
    <div class="gc-label">MAPE tổng thể</div>
    <div class="gc-val {% if gstats.mape < 2 %}g{% elif gstats.mape < 5 %}y{% else %}r{% endif %}">
      {{ gstats.mape }}%
    </div>
    <div class="gc-sub">trung bình tất cả ngày</div>
  </div>
  <div class="gc" style="animation-delay:.15s">
    <div class="gc-label">MAE tổng thể</div>
    <div class="gc-val">{{ gstats.mae|round(0)|int }}</div>
    <div class="gc-sub">VND sai số trung bình</div>
  </div>
  <div class="gc" style="animation-delay:.2s">
    <div class="gc-label">Giá cuối thực tế</div>
    <div class="gc-val" style="font-size:17px">{{ runs[0].last_actual_close|round(0)|int }}</div>
    <div class="gc-sub">{{ runs[0].last_actual_date }}</div>
  </div>
</div>

<!-- CHART -->
<div class="section">
  <div class="sec-head">
    <div class="sec-title">Biểu đồ — 30 ngày lịch sử + dự đoán</div>
    <span id="chart-run-lbl" style="font-family:var(--mono);font-size:10px;color:var(--muted)">
      run: {{ runs[0].last_actual_date }}
    </span>
  </div>
  <div class="chart-wrap"><canvas id="mainChart"></canvas></div>
</div>

<!-- DETAIL TABLE -->
<div class="section">
  <div class="sec-head">
    <div class="sec-title">Chi tiết từng lần chạy</div>
  </div>

  <!-- Dropdown + mape chip -->
  <div class="controls-bar">
    <span class="ctrl-label">Chọn run:</span>
    <select class="run-select" id="runSelect" onchange="selectRun(+this.value)">
      {% for run in runs %}
      <option value="{{ loop.index0 }}"
              data-last="{{ run.last_actual_date }}"
              data-mape="{% if run.summary %}{{ run.summary.mape }}{% endif %}"
              data-cls="{% if run.summary %}{% if run.summary.mape < 2 %}mc-g{% elif run.summary.mape < 5 %}mc-y{% else %}mc-r{% endif %}{% else %}mc-n{% endif %}">
        {{ run.last_actual_date }}
        → {{ run.rows[0].date if run.rows else '?' }} … {{ run.rows[-1].date if run.rows else '?' }}
        {% if run.summary %} | MAPE {{ run.summary.mape }}%{% endif %}
      </option>
      {% endfor %}
    </select>
    <span class="mape-chip" id="mapeChip"></span>
  </div>

  <!-- Timeline mini -->
  <div class="timeline" id="timeline">
    {% for run in runs %}
    <div class="tl-dot {% if loop.first %}sel{% endif %} {% if run.summary %}has-cmp{% endif %}"
         data-idx="{{ loop.index0 }}" onclick="selectRun({{ loop.index0 }})">
      {{ run.last_actual_date[5:] }}
    </div>
    {% endfor %}
  </div>

  <!-- Pills -->
  <div class="pills-bar" id="pillsBar"></div>

  <!-- Tables -->
  <div class="tbl-wrap">
    {% for run in runs %}
    <table id="tbl-{{ loop.index0 }}" class="run-table" style="{% if not loop.first %}display:none{% endif %}">
      <thead><tr>
        <th>#</th><th>Ngày GD</th>
        <th>Dự đoán (VND)</th><th>Thực tế (VND)</th>
        <th>Lệch (VND)</th><th>Lệch %</th><th>Đánh giá</th>
      </tr></thead>
      <tbody>
        {% for row in run.rows %}
        <tr>
          <td class="td-n c-mut">{{ loop.index }}</td>
          <td class="td-date">{{ row.date }}</td>
          <td class="td-n c-pred">{{ "{:,.0f}".format(row.predicted) }}</td>
          <td class="td-n c-act">
            {% if row.actual is not none %}{{ "{:,.0f}".format(row.actual) }}
            {% else %}<span class="c-mut">—</span>{% endif %}
          </td>
          <td class="td-n {% if row.error is not none %}{% if row.error >= 0 %}c-pos{% else %}c-neg{% endif %}{% endif %}">
            {% if row.error is not none %}{{ "{:+,.0f}".format(row.error) }}
            {% else %}<span class="c-mut">—</span>{% endif %}
          </td>
          <td>
            {% if row.pct_error is not none %}
            <span class="badge {% if row.pct_error|abs < 2 %}b-g{% elif row.pct_error|abs < 5 %}b-y{% else %}b-r{% endif %}">
              {{ "{:+.2f}".format(row.pct_error) }}%
            </span>
            {% else %}<span class="c-mut">—</span>{% endif %}
          </td>
          <td>
            {% if not row.is_past %}<span class="badge b-f">⌛ Chưa đến</span>
            {% elif row.actual is not none %}
              {% if row.pct_error|abs < 2 %}<span class="badge b-g">✓ Tốt</span>
              {% elif row.pct_error|abs < 5 %}<span class="badge b-y">~ Chấp nhận</span>
              {% else %}<span class="badge b-r">✗ Lệch nhiều</span>{% endif %}
            {% else %}<span class="badge b-m">⚠ Thiếu dữ liệu</span>{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% endfor %}
  </div>
</div>

{% endif %}

<div class="footer">STOCK_PRED DASHBOARD v2 &nbsp;·&nbsp; {{ data_dir }} &nbsp;·&nbsp; {{ now }}</div>
</div>

<script>
const ALL_CHART = {{ all_chart_data | safe }};
const RUNS_META = {{ runs_meta_js  | safe }};

// ── Chart ──
const ctx = document.getElementById('mainChart').getContext('2d');
let chart = null;

function buildChart(idx) {
  const d = ALL_CHART[idx];
  if (!d) return;
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [
        {
          label:'Thực tế',
          data: d.actual,
          borderColor:'#22d3a0',
          backgroundColor:'rgba(34,211,160,0.07)',
          borderWidth:2,
          pointRadius:3.5, pointHoverRadius:6,
          pointBackgroundColor:'#22d3a0',
          tension:0.25, fill:true,
          spanGaps:false,           // ← không nối qua null
        },
        {
          label:'Dự đoán',
          data: d.predicted,
          borderColor:'#00e5ff',
          backgroundColor:'rgba(0,229,255,0.05)',
          borderWidth:2, borderDash:[5,3],
          pointRadius:4, pointHoverRadius:7,
          pointStyle:'rectRot', pointBackgroundColor:'#00e5ff',
          tension:0.25, fill:false,
          spanGaps:false,
        }
      ]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction:{mode:'index', intersect:false},
      plugins:{
        legend:{labels:{color:'#7a93a8', font:{family:'JetBrains Mono',size:10}, usePointStyle:true, boxWidth:7}},
        tooltip:{
          backgroundColor:'#0d1422', borderColor:'#1c2d44', borderWidth:1,
          titleColor:'#d4dff0', bodyColor:'#7a93a8',
          titleFont:{family:'JetBrains Mono',size:10},
          bodyFont:{family:'JetBrains Mono',size:10},
          callbacks:{
            label: c => {
              const v = c.parsed.y;
              if (v === null || v === undefined) return null;
              return `${c.dataset.label}: ${v.toLocaleString('vi-VN')} VND`;
            }
          }
        }
      },
      scales:{
        x:{
          grid:{color:'rgba(28,45,68,0.85)'},
          ticks:{color:'#2e4460', font:{family:'JetBrains Mono',size:9}, maxRotation:45, autoSkip:true, maxTicksLimit:18}
        },
        y:{
          grid:{color:'rgba(28,45,68,0.85)'},
          ticks:{color:'#2e4460', font:{family:'JetBrains Mono',size:9}, callback:v=>v.toLocaleString('vi-VN')}
        }
      }
    }
  });
}

// ── Select run ──
function selectRun(idx) {
  // table
  document.querySelectorAll('.run-table').forEach(t => t.style.display = 'none');
  const tbl = document.getElementById('tbl-'+idx);
  if (tbl) tbl.style.display = '';

  // timeline
  document.querySelectorAll('.tl-dot').forEach(d => d.classList.remove('sel'));
  const dot = document.querySelector(`.tl-dot[data-idx="${idx}"]`);
  if (dot) { dot.classList.add('sel'); dot.scrollIntoView({behavior:'smooth',block:'nearest',inline:'center'}); }

  // dropdown
  document.getElementById('runSelect').value = idx;

  // chart
  buildChart(idx);
  const m = RUNS_META[idx];
  document.getElementById('chart-run-lbl').textContent = `run: ${m.last_actual_date}`;

  // pills
  const pb = document.getElementById('pillsBar');
  let h = '';
  h += `<span class="pill">${m.ticker}</span>`;
  h += `<span class="pill">Train đến: ${m.last_actual_date} | ${m.last_actual_close?.toLocaleString('vi-VN')} VND</span>`;
  if (m.summary) {
    const mc = m.summary.mape<2?'g':m.summary.mape<5?'y':'r';
    const ac = m.summary.mae<500?'g':m.summary.mae<1500?'y':'r';
    h += `<span class="pill ${mc}">MAPE ${m.summary.mape}%</span>`;
    h += `<span class="pill ${ac}">MAE ${Math.round(m.summary.mae).toLocaleString('vi-VN')} VND</span>`;
    h += `<span class="pill">${m.summary.n_compared}/${m.summary.n_total} ngày có thực tế</span>`;
  } else {
    h += `<span class="pill">Chưa có dữ liệu thực tế</span>`;
  }
  pb.innerHTML = h;

  // mape chip
  const opt = document.querySelector(`#runSelect option[value="${idx}"]`);
  const chip = document.getElementById('mapeChip');
  const mape = opt?.dataset.mape;
  chip.className = 'mape-chip ' + (opt?.dataset.cls || 'mc-n');
  chip.textContent = mape ? `MAPE ${mape}%` : 'Chưa có TT';
}

// init
buildChart(0);
selectRun(0);
</script>
</body>
</html>
"""


# ===================== ROUTES =====================
@app.route("/")
def index():
    actual_df, price_dict = load_actual_prices()
    meta_list = load_all_predictions()
    runs = build_runs(meta_list, price_dict)
    gstats = compute_global_stats(runs) or {
        "mape": 0, "mae": 0, "total_days": 0,
        "total_runs": len(runs), "runs_with_data": 0,
    }

    all_chart_data = [build_chart_data(actual_df, r, n_history=30) for r in runs]

    runs_meta_js = [{
        "ticker":            r["ticker"],
        "last_actual_date":  r["last_actual_date"],
        "last_actual_close": r["last_actual_close"],
        "summary":           r["summary"],
    } for r in runs]

    ticker = runs[0]["ticker"] if runs else "N/A"
    return render_template_string(
        HTML,
        runs=runs,
        gstats=gstats,
        all_chart_data=json.dumps(all_chart_data),
        runs_meta_js=json.dumps(runs_meta_js),
        today=date.today().strftime("%d/%m/%Y"),
        now=datetime.now().strftime("%d/%m/%Y %H:%M"),
        data_dir=DATA_DIR,
        ticker=ticker,
    )


@app.route("/api/data")
def api_data():
    _, price_dict = load_actual_prices()
    runs = build_runs(load_all_predictions(), price_dict)
    return jsonify([{k: v for k, v in r.items() if k != "predictions"} for r in runs])


if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║       STOCK PREDICTION DASHBOARD v2 (Fixed)          ║
╠══════════════════════════════════════════════════════╣
║  Data:  {DATA_DIR:<45}║
║  URL:   http://localhost:5000                        ║
║  API:   http://localhost:5000/api/data               ║
╚══════════════════════════════════════════════════════╝
""")
    app.run(debug=True, host="0.0.0.0", port=5000)
