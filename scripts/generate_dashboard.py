#!/usr/bin/env python3
"""
Generate a static HTML performance dashboard from benchmark history.

Usage:
    python scripts/generate_dashboard.py --data-dir benchmark-data --output-dir _site

Reads benchmark-results/history.jsonl and generates:
    - index.html   (overview with all machines/models)
    - per-machine pages with trend charts
"""

import argparse
import json
from pathlib import Path


def load_history(data_dir: Path) -> list[dict]:
    """Load all benchmark records from history.jsonl."""
    history_file = data_dir / "history.jsonl"
    records = []
    if history_file.exists():
        with history_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_records(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group records by (hostname, model)."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r.get("hostname", "unknown"), r.get("model", "unknown"))
        groups.setdefault(key, []).append(r)
    # Sort each group by timestamp
    for key in groups:
        groups[key].sort(key=lambda x: x.get("timestamp", ""))
    return groups


def compute_regression(current: dict, baseline: dict) -> dict:
    """Compute regression metrics vs baseline."""
    result = {}
    for metric in ["ttft_ms", "itl_ms", "tps"]:
        curr = current.get(metric)
        base = baseline.get(metric)
        if curr is not None and base is not None and base != 0:
            pct = (curr - base) / base * 100
            result[metric] = {
                "current": curr,
                "baseline": base,
                "delta_pct": round(pct, 1),
            }
    return result


def generate_machine_page(
    hostname: str,
    model: str,
    records: list[dict],
    output_dir: Path,
) -> None:
    """Generate a dashboard page for one (hostname, model) pair."""

    # Extract time series data
    dates = [r.get("timestamp", "")[:10] for r in records]
    commits = [r.get("git_commit", "")[:7] for r in records]
    ttft = [r.get("ttft_ms") for r in records]
    itl = [r.get("itl_ms") for r in records]
    tps = [r.get("tps") for r in records]

    # Build table rows
    rows = []
    baseline = records[0] if records else {}
    for r in records:
        reg = compute_regression(r, baseline)
        row = {
            "date": r.get("timestamp", "")[:10],
            "commit": r.get("git_commit", "")[:7],
            "branch": r.get("git_branch", ""),
            "ttft": r.get("ttft_ms"),
            "itl": r.get("itl_ms"),
            "tps": r.get("tps"),
        }
        # Color coding for regression
        for metric, label in [("ttft_ms", "ttft"), ("itl_ms", "itl"), ("tps", "tps")]:
            if metric in reg:
                pct = reg[metric]["delta_pct"]
                row[f"{label}_class"] = "regressed" if abs(pct) > 20 else "improved" if pct < -5 else "ok"
        rows.append(row)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Performance Dashboard — {hostname} / {model}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  h1 {{ margin-top: 0; color: #333; }}
  h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
  .meta {{ color: #888; font-size: 14px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
  th {{ background: #f0f0f0; padding: 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #fafafa; }}
  .ok {{ color: #228b22; }}
  .improved {{ color: #006400; font-weight: bold; }}
  .regressed {{ color: #dc143c; font-weight: bold; }}
  .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
  .nav {{ margin-bottom: 20px; }}
  .nav a {{ color: #0366d6; text-decoration: none; margin-right: 15px; }}
  .nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
  <div class="nav">
    <a href="index.html">← All Machines</a>
  </div>
  <h1>📊 Performance Dashboard</h1>
  <div class="meta">
    Machine: <strong>{hostname}</strong> | Model: <strong>{model}</strong> | Records: <strong>{len(records)}</strong>
  </div>

  <h2>Trends</h2>
  <div id="ttft-chart" class="chart"></div>
  <div id="itl-chart" class="chart"></div>
  <div id="tps-chart" class="chart"></div>

  <h2>History</h2>
  <table>
    <tr>
      <th>Date</th>
      <th>Commit</th>
      <th>Branch</th>
      <th>TTFT (ms)</th>
      <th>ITL (ms)</th>
      <th>TPS</th>
      <th>vs Baseline</th>
    </tr>
"""

    for row in rows:
        html += "    <tr>\n"
        html += f"      <td>{row['date']}</td>\n"
        html += f"      <td><code>{row['commit']}</code></td>\n"
        html += f"      <td>{row['branch']}</td>\n"

        for metric_key, label in [("ttft", "ttft"), ("itl", "itl"), ("tps", "tps")]:
            val = row.get(metric_key)
            cls = row.get(f"{label}_class", "")
            if val is not None:
                html += f'      <td class="{cls}">{val:.1f}</td>\n'
            else:
                html += '      <td>—</td>\n'

        # vs Baseline column
        reg = compute_regression(
            next(r for r in records if r.get("timestamp", "")[:10] == row["date"]),
            baseline
        )
        parts = []
        for m, name in [("ttft_ms", "TTFT"), ("itl_ms", "ITL"), ("tps", "TPS")]:
            if m in reg:
                pct = reg[m]["delta_pct"]
                sign = "+" if pct > 0 else ""
                parts.append(f"{name}: {sign}{pct:.1f}%")
        html += f'      <td>{"; ".join(parts)}</td>\n'
        html += "    </tr>\n"

    # Close table and add Plotly charts
    html += f"""  </table>
</div>
<script>
  const dates = {json.dumps(dates)};
  const commits = {json.dumps(commits)};
  const ttft = {json.dumps(ttft)};
  const itl = {json.dumps(itl)};
  const tps = {json.dumps(tps)};

  function makeChart(id, title, data, yLabel, lowerIsBetter) {{
    const trace = {{
      x: dates,
      y: data,
      mode: 'lines+markers',
      type: 'scatter',
      text: commits,
      hovertemplate: '%{{x}}<br>%{{text}}<br>%{{y:.1f}} ' + yLabel + '<extra></extra>',
      line: {{ color: lowerIsBetter ? '#dc143c' : '#228b22', width: 2 }},
      marker: {{ size: 8 }}
    }};
    const layout = {{
      title: title,
      xaxis: {{ title: 'Date' }},
      yaxis: {{ title: yLabel }},
      margin: {{ t: 40, b: 60 }},
      hovermode: 'closest'
    }};
    Plotly.newPlot(id, [trace], layout, {{responsive: true}});
  }}

  makeChart('ttft-chart', 'TTFT (Time To First Token)', ttft, 'ms', true);
  makeChart('itl-chart', 'ITL (Inter-Token Latency)', itl, 'ms', true);
  makeChart('tps-chart', 'Throughput', tps, 'tokens/sec', false);
</script>
</body>
</html>
"""

    filename = f"{hostname}_{model}.html".replace(" ", "_").replace("/", "_")
    output_file = output_dir / filename
    output_file.write_text(html)
    print(f"  Generated: {filename}")


def generate_index(
    groups: dict[tuple, list[dict]],
    output_dir: Path,
) -> None:
    """Generate the main index page listing all machines/models."""

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>dmlx Performance Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
  .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  h1 { margin-top: 0; color: #333; }
  h2 { color: #555; }
  .card { border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 12px 0; background: #fafbfc; }
  .card:hover { background: #f0f3f6; }
  .card a { text-decoration: none; color: #0366d6; font-weight: 600; font-size: 16px; }
  .card a:hover { text-decoration: underline; }
  .meta { color: #586069; font-size: 13px; margin-top: 6px; }
  .latest { color: #228b22; font-weight: 600; }
  .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #888; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
  <h1>📊 dmlx Performance Dashboard</h1>
  <p>Benchmark results for DeepSeek V4 inference. Auto-generated from <code>benchmark-results</code> branch.</p>

  <h2>Machines</h2>
"""

    for (hostname, model), records in sorted(groups.items()):
        latest = records[-1]
        filename = f"{hostname}_{model}.html".replace(" ", "_").replace("/", "_")
        ttft = latest.get("ttft_ms")
        itl = latest.get("itl_ms")
        tps = latest.get("tps")
        commit = latest.get("git_commit", "")[:7]
        date = latest.get("timestamp", "")[:10]

        html += f"""  <div class="card">
    <a href="{filename}">{hostname} — {model}</a>
    <div class="meta">
      Records: {len(records)} | Latest: <span class="latest">{date}</span> (@{commit})<br>
      TTFT: {ttft:.1f}ms | ITL: {itl:.1f}ms | TPS: {tps:.1f}
    </div>
  </div>
"""

    html += """  <div class="footer">
    Generated at deploy time. To add results, run <code>make benchmark && make upload-benchmark</code> locally.
  </div>
</div>
</body>
</html>
"""

    index_file = output_dir / "index.html"
    index_file.write_text(html)
    print(f"Generated: index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance dashboard")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to benchmark-results data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for HTML files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading history from {args.data_dir}/history.jsonl")
    records = load_history(args.data_dir)
    print(f"   Loaded {len(records)} records")

    if not records:
        print("⚠️  No records found. Creating empty dashboard.")
        # Create a placeholder index
        placeholder = args.output_dir / "index.html"
        placeholder.write_text(
            "<h1>dmlx Performance Dashboard</h1>"
            "<p>No benchmark results yet. Run <code>make benchmark && make upload-benchmark</code> to populate.</p>"
        )
        return

    groups = group_records(records)
    print(f"   Found {len(groups)} machine/model groups")

    for (hostname, model), group_records in groups.items():
        generate_machine_page(hostname, model, group_records, args.output_dir)

    generate_index(groups, args.output_dir)
    print(f"✅ Dashboard generated in {args.output_dir}")


if __name__ == "__main__":
    main()
