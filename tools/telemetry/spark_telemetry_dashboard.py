#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from . import spark_telemetry_common as telemetry_common
except ImportError:
    import spark_telemetry_common as telemetry_common


DEFAULT_SUMMARY_JSON = "/tmp/ds4_telemetry/mac/cluster_summary.json"
DEFAULT_NODES_DIR = "/tmp/ds4_telemetry/mac/nodes"
DEFAULT_HISTORY_LIMIT = 720
DEFAULT_REPO_ROOT = ""
DEFAULT_MODEL_LAYER_PARTITIONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_layer_partitions.json")
NODE_DOWN_ERROR_THRESHOLD = 3
NODE_ERROR_STREAKS: dict[str,dict[str,Any]] = {}
MODEL_LAYER_PARTITIONS: dict[str,list[int]] | None = None
REPO_ROOT_OVERRIDE = ""
MODEL_LAYER_PARTITIONS_JSON_OVERRIDE = ""
DISPLAY_CPU_CORES = 20
DISPLAY_CPU_PCT_MAX = DISPLAY_CPU_CORES * 100
STREAM_INTERVAL_S = 5.0
HISTORY_METRICS = [
    {"key": "gpu_pct", "label": "GPU", "field": "gpu_util_pct", "unit": "%"},
    {"key": "kv_pct", "label": "KV", "field": "vllm_kv_cache_pct", "unit": "%"},
    {"key": "cpu_pct", "label": "CPU", "field": "cpu_util_pct", "unit": "%"},
    {"key": "mem_pct", "label": "MEM", "field": "mem_used_pct", "unit": "%"},
    {"key": "temp_c", "label": "TEMP", "field": "gpu_temp_c", "unit": "C"},
    {"key": "vllm_running", "label": "RUN", "field": "vllm_requests_running", "unit": ""},
    {"key": "vllm_waiting", "label": "WAIT", "field": "vllm_requests_waiting", "unit": ""},
    {"key": "tok_s", "label": "TOK/S", "field": "vllm_tokens_per_s", "unit": ""},
    {"key": "input_tok_s", "label": "IN TOK/S", "field": "vllm_prompt_tokens_per_s", "unit": ""},
    {"key": "output_tok_s", "label": "OUT TOK/S", "field": "vllm_generation_tokens_per_s", "unit": ""},
    {"key": "cache_tok_s", "label": "CACHE TOK/S", "field": "vllm_prompt_tokens_cached_per_s", "unit": ""},
    {"key": "cache_hit_pct", "label": "CACHE HIT", "field": "vllm_prompt_cache_hit_pct", "unit": "%"},
    {"key": "external_hit_pct", "label": "EXT HIT", "field": "vllm_external_prefix_cache_hit_pct", "unit": "%"},
]
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spark Telemetry</title>
<style>
:root{color-scheme:dark;--bg:#111316;--panel:#1b1f24;--line:#313943;--text:#f2f5f8;--muted:#a8b1bb;--ok:#53d18a;--busy:#63b3ff;--warn:#f4bf5f;--bad:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;letter-spacing:0}
main{max-width:1280px;margin:0 auto;padding:18px}.top{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:14px}
h1{font-size:22px;line-height:1.1;margin:0}.meta{color:var(--muted);text-align:right;line-height:1.5}.summary{display:grid;grid-template-columns:repeat(6,minmax(110px,1fr));gap:10px;margin-bottom:10px}.models{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:14px;overflow:hidden}.model-table{width:100%;border-collapse:collapse;table-layout:fixed}.model-table th,.model-table td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}.model-table th{color:var(--muted);font-size:11px;text-transform:uppercase}.model-table td:first-child,.model-table th:first-child{text-align:left;width:38%}.model-table tr:last-child td{border-bottom:0}.model-name{font-weight:700;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.model-empty{color:var(--muted);padding:10px}
.metric,.card{background:var(--panel);border:1px solid var(--line);border-radius:8px}.metric{padding:12px}.label{color:var(--muted);font-size:12px}.value{font-size:23px;font-weight:700;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-auto-rows:minmax(150px,auto);gap:10px}.card{padding:12px;min-height:150px;cursor:pointer}.card.selected{border-color:var(--busy);box-shadow:0 0 0 1px var(--busy)}.card header{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.node{font-size:18px;font-weight:700}.pill{border-radius:999px;padding:3px 8px;font-size:12px;font-weight:700;color:#111316;background:var(--muted)}.busy .pill{background:var(--busy)}.idle .pill{background:var(--ok)}.warn .pill,.hot .pill{background:var(--warn)}.down .pill{background:var(--bad)}
.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:54px 1fr 48px;align-items:center;gap:8px;color:var(--muted)}.track{height:8px;background:#0d0f12;border-radius:999px;overflow:hidden}.fill{height:100%;width:0;background:var(--ok)}.busy .gpu .fill,.busy .kv .fill{background:var(--busy)}.warn .fill,.hot .fill{background:var(--warn)}.down .fill{background:var(--bad)}
.details{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-top:10px;color:var(--muted)}.details b{color:var(--text);font-weight:600}.error{margin-top:8px;color:var(--bad);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.history{margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}.history-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}.history-title{font-size:17px;font-weight:700}.modes{display:flex;gap:6px}.modes button{appearance:none;border:1px solid var(--line);background:#101318;color:var(--muted);border-radius:7px;padding:5px 10px;font:inherit;font-size:12px;font-weight:700;cursor:pointer}.modes button.active{background:var(--busy);border-color:var(--busy);color:#071018}.legend{display:flex;flex-wrap:wrap;gap:8px 14px;color:var(--muted);font-size:12px}.legend span{white-space:nowrap}.swatch{display:inline-block;width:9px;height:9px;border-radius:999px;margin-right:5px}.chart-wrap{height:270px;min-height:270px}.chart-wrap canvas{display:block;width:100%;height:100%}.empty{color:var(--muted);padding:24px 0;text-align:center}
@media (max-width:1100px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media (max-width:720px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.meta{text-align:left}.summary{grid-template-columns:repeat(2,minmax(120px,1fr))}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body><main>
<div class="top"><h1>Spark Telemetry</h1><div class="meta"><div id="updated">loading</div><div id="source"></div></div></div>
<section class="summary" id="summary"></section>
<section class="models" id="models"></section>
<section class="grid" id="nodes"></section>
<section class="history" id="history"><div class="empty">select a spark</div></section>
</main>
<script>
const fmt=n=>Number.isFinite(Number(n))?Number(n).toFixed(0):"";
const pct=n=>Number.isFinite(Number(n))?Number(n).toFixed(0)+"%":"n/a";
const val=(n,s="")=>Number.isFinite(Number(n))?Number(n).toFixed(1)+s:"n/a";
const rate=n=>{let x=Number(n);if(!Number.isFinite(x))return "n/a";return (Math.abs(x)<10?x.toFixed(2):x.toFixed(1))}
let selectedNode="";
let lastHistory=null;
let selectedMode="queue";
let telemetryStream=null;
const CPU_PCT_MAX=2000;
const metricModes={queue:["input_tok_s","output_tok_s","cache_tok_s","cache_hit_pct","vllm_running","vllm_waiting","cpu_pct"],gpu:["gpu_pct","kv_pct","cpu_pct","mem_pct","temp_c"]};
const modeLabels={queue:"Queue",gpu:"GPU"};
const modeColors={queue:["#00e5ff","#ff4d4d","#ffe156","#53d18a","#a78bfa","#f4bf5f"],gpu:["#2f80ed","#ff7a00","#f4bf5f","#00c853","#e040fb","#f4d35e"]};
function metric(label,value){return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`}
function bar(label,value,cls,known=true,text=""){let width=known?Math.max(0,Math.min(100,Number(value)||0)):0;return `<div class="barrow ${cls}"><span>${label}</span><div class="track"><div class="fill" style="width:${width}%"></div></div><span>${known?(text||pct(value)):"n/a"}</span></div>`}
function workKnown(n){return n.vllm_metrics_up||Number(n.local_q_depth)>0||Number(n.input_tok_s)>0||Number(n.output_tok_s)>0||Number(n.vllm_running)>0||Number(n.vllm_waiting)>0}
function workVal(n,key,unit=""){return workKnown(n)?rate(n[key])+unit:"n/a"}
function workRun(n){return workKnown(n)?`${fmt(n.vllm_running)}/${fmt(n.vllm_waiting)}`:"n/a"}
function tokenLabel(n,label){return n.token_scope==="allocated"?`Layer ${label}`:(n.token_scope==="pipeline"?`Pipe ${label}`:label)}
function tokenScope(n){return n.token_scope==="allocated"?"layer":(n.token_scope==="pipeline"?`PP${fmt(n.pipeline_parallel_size)}`:(n.vllm_metrics_up?"node":"n/a"))}
function queueVal(n){return n.local_queue_known?fmt(n.local_q_depth):"n/a"}
function modelHint(n){let a=n.model_allocations||[];if(!a.length)return "";return a.map(m=>`${m.model.replace("-pp8","")}: ${rate(m.output_tok_s)}`).join(" · ")}
function card(n){let err=n.error||n.fetch_error||"";let hint=modelHint(n);return `<article class="card ${n.state} ${n.node===selectedNode?"selected":""}" data-node="${n.node}"><header><div class="node">${n.node}</div><div class="pill">${n.state_label}</div></header><div class="bars">${bar("GPU",n.gpu_pct,"gpu")}${bar("KV",n.kv_pct,"kv",n.kv_known,n.kv_label)}${bar("MEM",n.mem_pct,"mem")}</div><div class="details"><span>${tokenLabel(n,"In")} <b>${workVal(n,"input_tok_s")}</b></span><span>${tokenLabel(n,"Out")} <b>${workVal(n,"output_tok_s")}</b></span><span>Run <b>${workRun(n)}</b></span><span>Tok <b>${tokenScope(n)}</b></span><span>CPU <b>${pct(n.cpu_pct)}</b></span><span>Svc <b>${n.ds_service_id||"n/a"}</b></span><span>Temp <b>${fmt(n.gpu_temp_c)}C</b></span><span>Queue <b>${queueVal(n)}</b></span></div>${hint?`<div class="label">${hint}</div>`:""}${err?`<div class="error">${err}</div>`:""}</article>`}
function wireCards(){document.querySelectorAll(".card[data-node]").forEach(el=>el.onclick=()=>{selectedNode=el.dataset.node;document.querySelectorAll(".card").forEach(c=>c.classList.toggle("selected",c.dataset.node===selectedNode));startTelemetryStream()})}
function modeButtons(){return `<div class="modes">${Object.keys(metricModes).map(k=>`<button class="${k===selectedMode?"active":""}" data-mode="${k}">${modeLabels[k]}</button>`).join("")}</div>`}
function wireModes(){document.querySelectorAll(".modes button").forEach(el=>el.onclick=()=>{selectedMode=el.dataset.mode;drawHistory(lastHistory)})}
function activeMetrics(data){let allowed=new Set(metricModes[selectedMode]||metricModes.queue);return data.metrics.filter(m=>allowed.has(m.key))}
function metricLast(metric,points){let p=points[points.length-1]||{};let v=Number(p[metric.key]);return Number.isFinite(v)?v:null}
function metricScale(metric,points){if(["tok_s","input_tok_s","output_tok_s","cache_tok_s"].includes(metric.key))return Math.max(50,...points.map(p=>Number(p[metric.key])||0))*1.2;let fixed={gpu_pct:100,kv_pct:100,cpu_pct:CPU_PCT_MAX,mem_pct:100,temp_c:100,vllm_running:64,vllm_waiting:64,queue_depth:128,cache_hit_pct:100,external_hit_pct:100};return fixed[metric.key]||100}
function emaValues(points,key){let out=[],acc=null,alpha=0.34;points.forEach(p=>{let v=Number(p[key]);v=Number.isFinite(v)?v:0;acc=acc===null?v:((alpha*v)+((1-alpha)*acc));out.push(acc)});return out}
function drawHistory(data){lastHistory=data;let el=document.getElementById("history");if(!data||!data.ok||!data.points.length){el.innerHTML=`<div class="history-head"><div class="history-title">${selectedNode||"spark"}</div>${modeButtons()}</div><div class="empty">no history</div>`;wireModes();return}let metrics=activeMetrics(data);let colors=modeColors[selectedMode]||modeColors.queue;let legend=metrics.map((m,i)=>{let v=metricLast(m,data.points);return `<span><i class="swatch" style="background:${colors[i%colors.length]}"></i>${m.label} <b>${v===null?"n/a":val(v,m.unit)}</b></span>`}).join("");el.innerHTML=`<div class="history-head"><div><div class="history-title">${data.node}</div><div class="label">last hour · ${data.points.length} samples · EMA</div></div>${modeButtons()}</div><div class="legend">${legend}</div><div class="chart-wrap"><canvas id="chart"></canvas></div>`;wireModes();paintChart(data,metrics,colors)}
function paintChart(data,metrics,colors){let canvas=document.getElementById("chart");if(!canvas)return;metrics=metrics||activeMetrics(data);colors=colors||modeColors[selectedMode]||modeColors.queue;let rect=canvas.getBoundingClientRect();let dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.floor(rect.width*dpr));canvas.height=Math.max(1,Math.floor(rect.height*dpr));let ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);let w=rect.width,h=rect.height,pad=28;ctx.clearRect(0,0,w,h);ctx.strokeStyle="#313943";ctx.lineWidth=1;for(let i=0;i<=4;i++){let y=pad+((h-(pad*2))*i/4);ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke()}let points=data.points;metrics.forEach((m,i)=>{let scale=metricScale(m,points);let vals=emaValues(points,m.key);ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=2.2;ctx.beginPath();vals.forEach((v,idx)=>{let x=pad+((w-(pad*2))*idx/Math.max(1,points.length-1));let y=h-pad-((h-(pad*2))*Math.max(0,Math.min(scale,v))/scale);if(idx===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke()});ctx.fillStyle="#a8b1bb";ctx.font="12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif";ctx.fillText("now",w-pad-24,h-8);ctx.fillText("then",pad,h-8)}
async function refreshHistory(){if(!selectedNode)return;try{let r=await fetch(`/api/history?node=${encodeURIComponent(selectedNode)}`,{cache:"no-store"});drawHistory(await r.json())}catch(e){document.getElementById("history").innerHTML=`<div class="empty">history read failed</div>`}}
function renderModels(d){let models=d.models||[];let el=document.getElementById("models");if(!models.length){el.innerHTML=`<div class="model-empty">no active model token rate</div>`;return}el.innerHTML=`<table class="model-table"><thead><tr><th>Model</th><th>In/s</th><th>Out/s</th><th>Total/s</th><th>Run</th><th>Wait</th></tr></thead><tbody>${models.map(m=>`<tr><td><div class="model-name">${m.model}</div></td><td>${rate(m.input_tok_s)}</td><td>${rate(m.output_tok_s)}</td><td>${rate(m.tok_s)}</td><td>${rate(m.running)}</td><td>${rate(m.waiting)}</td></tr>`).join("")}</tbody></table>`}
function renderSummary(d){if(!selectedNode&&d.nodes&&d.nodes.length)selectedNode=d.selected_node||d.nodes[0].node;document.getElementById("updated").textContent="updated "+(d.updated_iso||"unknown");document.getElementById("source").textContent=d.summary_path||"";document.getElementById("summary").innerHTML=[metric("Active",`${fmt(d.active_nodes)}/${d.reachable_nodes}`),metric("GPU Avg",d.gpu_known?pct(d.avg_gpu_pct):"n/a"),metric("Run/Wait",`${fmt(d.vllm_running)}/${fmt(d.vllm_waiting)}`),metric("Live In/Out",`${val(d.input_tok_s)} / ${val(d.output_tok_s)}`),metric("Active Svc",d.ds_services_known?`${fmt(d.ds_service_count)} svc`:"n/a"),metric("Queue Depth",fmt(d.queue_depth))].join("");renderModels(d);document.getElementById("nodes").innerHTML=d.nodes.map(card).join("");wireCards()}
async function refreshOnce(){try{let r=await fetch("/api/summary",{cache:"no-store"});let d=await r.json();renderSummary(d);await refreshHistory()}catch(e){document.getElementById("updated").textContent="dashboard read failed: "+e}}
function startTelemetryStream(){if(telemetryStream)telemetryStream.close();if(!window.EventSource){refreshOnce();return}let node=encodeURIComponent(selectedNode||"");telemetryStream=new EventSource(`/api/stream?node=${node}`);telemetryStream.addEventListener("telemetry",event=>{try{let payload=JSON.parse(event.data);if(payload.summary){renderSummary(payload.summary)}if(payload.history){drawHistory(payload.history)}}catch(e){document.getElementById("updated").textContent="stream parse failed: "+e}});telemetryStream.onerror=()=>{document.getElementById("updated").textContent="stream reconnecting"}}
window.addEventListener("resize",()=>{if(lastHistory)paintChart(lastHistory)});
startTelemetryStream();
</script></body></html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    p.add_argument("--nodes-dir", default=DEFAULT_NODES_DIR)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--repo-root", default=os.environ.get("DS4_TELEMETRY_REPO_ROOT",DEFAULT_REPO_ROOT))
    p.add_argument("--layer-partitions-json", default=os.environ.get("DS4_TELEMETRY_LAYER_PARTITIONS_JSON",DEFAULT_MODEL_LAYER_PARTITIONS_JSON))
    return(p.parse_args())


def fnum(value: Any) -> float:
    try:
        return(float(value))
    except Exception:
        return(0.0)


def display_cpu_pct(value: Any) -> float:
    return(round(fnum(value) * DISPLAY_CPU_CORES,2))


def valid_node_name(node: str) -> bool:
    return((node.startswith("spark") or node in telemetry_common.AUXILIARY_NODES) and all(ch.isalnum() or ch in "-_" for ch in node))


def reset_node_error_streaks() -> None:
    NODE_ERROR_STREAKS.clear()


def node_down_error(row: dict[str,Any]) -> str:
    error = str(row.get("error",""))
    if int(fnum(row.get("sample_count"))) <= 0:
        return(error or "no samples")
    return(error)


def node_error_streak(node: str, row: dict[str,Any], summary_id: Any) -> int:
    error = node_down_error(row)
    if error == "":
        NODE_ERROR_STREAKS.pop(node,None)
        return(0)
    observed = "%s|%s|%s" % (summary_id,row.get("last_iso_ts",""),error)
    previous = NODE_ERROR_STREAKS.get(node,{})
    if previous.get("observed") == observed:
        return(int(fnum(previous.get("count"))))
    count = int(fnum(previous.get("count"))) + 1
    NODE_ERROR_STREAKS[node] = {"observed":observed,"count":count}
    return(count)


def node_state(row: dict[str,Any], error_streak: int = NODE_DOWN_ERROR_THRESHOLD) -> tuple[str,str]:
    if node_down_error(row) != "":
        if error_streak >= NODE_DOWN_ERROR_THRESHOLD:
            return("down","down")
        return("warn","checking")
    if fnum(row.get("stale_data")) > 0.0 or str(row.get("fetch_error","")) != "":
        return("warn","stale")
    if fnum(row.get("last_gpu_temp_c")) >= 80.0 or fnum(row.get("last_thermal_max_c")) >= 85.0:
        return("hot","hot")
    if fnum(row.get("last_vllm_requests_waiting")) > 0.0 or fnum(row.get("last_vllm_kv_cache_pct")) >= 90.0:
        return("warn","queued")
    if fnum(row.get("last_vllm_requests_running")) > 0.0 or fnum(row.get("last_vllm_prompt_tokens_per_s")) > 0.0 or fnum(row.get("last_vllm_generation_tokens_per_s")) > 0.0:
        return("busy","busy")
    return("idle","idle")


def normalize_node(node: str, row: dict[str,Any], error_streak: int = NODE_DOWN_ERROR_THRESHOLD) -> dict[str,Any]:
    state,label = node_state(row,error_streak)
    gpu_power_known = fnum(row.get("last_gpu_power_known")) > 0.0
    return({
        "node": node,
        "state": state,
        "state_label": label,
        "error_streak": int(error_streak),
        "down_after_errors": NODE_DOWN_ERROR_THRESHOLD,
        "sample_count": int(fnum(row.get("sample_count"))),
        "last_iso_ts": row.get("last_iso_ts",""),
        "last_sample_age_s": fnum(row.get("last_sample_age_s")),
        "gpu_pct": fnum(row.get("last_gpu_util_pct")),
        "gpu_temp_c": fnum(row.get("last_gpu_temp_c")),
        "gpu_power_w": fnum(row.get("last_gpu_power_w")) if gpu_power_known else 0.0,
        "gpu_power_raw_w": fnum(row.get("last_gpu_power_raw_w")),
        "gpu_power_limit_w": fnum(row.get("last_gpu_power_limit_w")),
        "gpu_power_known": gpu_power_known,
        "gpu_power_source": str(row.get("last_gpu_power_source","")),
        "gpu_power_reason": str(row.get("last_gpu_power_reason","")),
        "thermal_max_c": fnum(row.get("last_thermal_max_c")),
        "cpu_pct": display_cpu_pct(row.get("last_cpu_util_pct")),
        "mem_pct": fnum(row.get("last_mem_used_pct")),
        "vllm_running": fnum(row.get("last_vllm_requests_running")),
        "vllm_waiting": fnum(row.get("last_vllm_requests_waiting")),
        "vllm_metrics_up": fnum(row.get("last_vllm_metrics_up")) > 0.0,
        "token_scope": str(row.get("last_vllm_metrics_scope","")) or ("local" if fnum(row.get("last_vllm_metrics_up")) > 0.0 else ""),
        "pipeline_parallel_size": fnum(row.get("last_vllm_pipeline_parallel_size")),
        "pipeline_node_rank": fnum(row.get("last_vllm_pipeline_node_rank")),
        "pipeline_stage_models": node_text_list(row.get("last_vllm_pipeline_stage_models","")),
        "pipeline_stage_pp_by_model": node_metric_map(row.get("last_vllm_pipeline_stage_pp_by_model","")),
        "pipeline_stage_rank_by_model": node_metric_map(row.get("last_vllm_pipeline_stage_rank_by_model","")),
        "model_input_tok_s": node_metric_map(row.get("last_vllm_prompt_tokens_per_s_by_model","")),
        "model_output_tok_s": node_metric_map(row.get("last_vllm_generation_tokens_per_s_by_model","")),
        "model_running": node_metric_map(row.get("last_vllm_requests_running_by_model","")),
        "model_waiting": node_metric_map(row.get("last_vllm_requests_waiting_by_model","")),
        "local_queue_known": False,
        "kv_pct": fnum(row.get("last_vllm_kv_cache_pct")),
        "kv_known": fnum(row.get("last_vllm_metrics_up")) > 0.0,
        "kv_label": "",
        "ds_service_id": "",
        "ds_kv_known": False,
        "tok_s": fnum(row.get("last_vllm_tokens_per_s")),
        "input_tok_s": fnum(row.get("last_vllm_prompt_tokens_per_s")),
        "output_tok_s": fnum(row.get("last_vllm_generation_tokens_per_s")),
        "cache_tok_s": fnum(row.get("last_vllm_prompt_tokens_cached_per_s")),
        "cache_hit_pct": fnum(row.get("last_vllm_prompt_cache_hit_pct")),
        "prefix_hit_pct": fnum(row.get("last_vllm_prefix_cache_hit_pct")),
        "external_hit_pct": fnum(row.get("last_vllm_external_prefix_cache_hit_pct")),
        "gateway_up": fnum(row.get("last_ds4_gateway_up")) > 0.0,
        "gateway_active": fnum(row.get("last_ds4_gateway_active")) > 0.0,
        "local_q_depth": 0.0,
        "error": str(row.get("error","")),
        "fetch_error": str(row.get("fetch_error","")),
    })


def node_metric_map(raw: Any) -> dict[str,float]:
    out: dict[str,float] = {}
    for item in str(raw or "").split(";"):
        if ":" not in item:
            continue
        key,value = item.split(":",1)
        key = key.strip()
        if key:
            out[key] = fnum(value)
    return(out)


def node_text_map(raw: Any) -> dict[str,str]:
    out: dict[str,str] = {}
    for item in str(raw or "").split(";"):
        if ":" not in item:
            continue
        key,value = item.split(":",1)
        key = key.strip()
        value = value.strip()
        if key:
            out[key] = value
    return(out)


def node_text_list(raw: Any) -> list[str]:
    return([item.strip() for item in str(raw or "").split(";") if item.strip()])


def repo_root() -> Path:
    configured = REPO_ROOT_OVERRIDE or DEFAULT_REPO_ROOT
    if configured:
        return(Path(configured).expanduser().resolve())
    return(Path(__file__).resolve().parents[1])


def load_installed_model_layer_partitions(path: Path) -> dict[str,list[int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return({})
    source = raw.get("model_layer_partitions",raw) if isinstance(raw,dict) else {}
    if not isinstance(source,dict):
        return({})
    out: dict[str,list[int]] = {}
    for model,partition in source.items():
        if isinstance(partition,list):
            values = [int(item) for item in partition]
            if values:
                out[str(model)] = values
    return(out)


def load_model_layer_partitions() -> dict[str,list[int]]:
    global MODEL_LAYER_PARTITIONS
    if MODEL_LAYER_PARTITIONS is not None:
        return(MODEL_LAYER_PARTITIONS)
    installed_path = Path(MODEL_LAYER_PARTITIONS_JSON_OVERRIDE or DEFAULT_MODEL_LAYER_PARTITIONS_JSON).expanduser()
    installed = load_installed_model_layer_partitions(installed_path)
    if installed:
        MODEL_LAYER_PARTITIONS = installed
        return(installed)
    root = repo_root()
    partitions_by_service: dict[str,list[int]] = {}
    budget_path = root / "v2" / "profiles" / "production" / "first3_resident_memory_budget.json"
    try:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        raw_partitions = budget.get("layer_partitions",{})
        if isinstance(raw_partitions,dict):
            for service_id,partition in raw_partitions.items():
                if isinstance(partition,list):
                    values = [int(item) for item in partition]
                    if values:
                        partitions_by_service[str(service_id)] = values
    except Exception:
        pass
    model_to_service: dict[str,str] = {}
    for path in (root / "v2" / "profiles" / "models").glob("*.json"):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        routing = profile.get("routing",{}) if isinstance(profile.get("routing"),dict) else {}
        pipeline = routing.get("pipeline",{}) if isinstance(routing.get("pipeline"),dict) else {}
        model = str(pipeline.get("served_model_name") or routing.get("served_model_name") or "")
        service_id = str(pipeline.get("service_id") or routing.get("pipeline_service_id") or "")
        if model and service_id:
            model_to_service[model] = service_id
    out: dict[str,list[int]] = {}
    for model,service_id in model_to_service.items():
        if service_id in partitions_by_service:
            out[model] = partitions_by_service[service_id]
    for path in (root / "v2" / "profiles" / "kv_cache").glob("*.json"):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = str(profile.get("served_model_name") or "")
        partition = profile.get("layer_partition")
        if model and model not in out and isinstance(partition,list):
            values = [int(item) for item in partition]
            if values:
                out[model] = values
    MODEL_LAYER_PARTITIONS = out
    return(out)


def layer_share(model: str, rank: int, pp_size: int, partitions: dict[str,list[int]]) -> float:
    partition = partitions.get(model)
    if partition and 0 <= rank < len(partition):
        total = sum(partition)
        if total > 0:
            return(float(partition[rank]) / float(total))
    return(0.0)


def layer_count(model: str, rank: int, partitions: dict[str,list[int]]) -> tuple[int,int]:
    partition = partitions.get(model)
    if partition and 0 <= rank < len(partition):
        return(int(partition[rank]),int(sum(partition)))
    return(0,0)


def build_snapshot(summary_path: str) -> dict[str,Any]:
    path = Path(summary_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return({"ok":False,"summary_path":str(path),"updated_iso":"","error":str(exc),"nodes":[]})
    queue = raw.get("queue",{}) if isinstance(raw.get("queue"),dict) else {}
    queue_kv_by_node = node_text_map(queue.get("local_queue_kv_by_node",""))
    stage_service_by_node = node_text_map(queue.get("local_queue_stage_service_by_node",""))
    pending_by_service = node_metric_map(queue.get("local_queue_pending_by_service",""))
    derived_active_services = {service for service in list(queue_kv_by_node.values()) + list(stage_service_by_node.values()) if service}
    for service_id,count in pending_by_service.items():
        if count > 0.0:
            derived_active_services.add(service_id)
    active_ds_services = str(queue.get("local_queue_active_services","")).strip()
    if active_ds_services == "" and derived_active_services:
        active_ds_services = ",".join(sorted(derived_active_services))
    active_ds_service_count = fnum(queue.get("local_queue_active_service_count",0.0))
    if active_ds_service_count <= 0.0 and active_ds_services != "":
        active_ds_service_count = len([service for service in active_ds_services.split(",") if service.strip()])
    summary_id = raw.get("updated_unix") or raw.get("updated_iso","")
    raw_nodes = raw.get("nodes",{}) if isinstance(raw.get("nodes",{}),dict) else {}
    node_names = set(str(node) for node,row in raw_nodes.items() if isinstance(row,dict))
    node_names.update(stage_service_by_node)
    node_names.update(queue_kv_by_node)
    node_names.update(telemetry_common.SPARK_NODES)
    rows: dict[str,dict[str,Any]] = {}
    for name in sorted(node_names):
        base = raw_nodes.get(name,{})
        row = dict(base) if isinstance(base,dict) else {}
        rows[name] = row
    nodes = [normalize_node(node,row,node_error_streak(node,row,summary_id)) for node,row in sorted(rows.items())]
    for node in nodes:
        name = str(node.get("node",""))
        service_id = queue_kv_by_node.get(name) or stage_service_by_node.get(name) or ""
        if service_id:
            node["ds_service_id"] = service_id
            node["ds_kv_known"] = True
            if not node.get("kv_known"):
                node["kv_known"] = True
                node["kv_label"] = "api"
        node["tok_s"] = max(fnum(node.get("tok_s")),fnum(node.get("input_tok_s")) + fnum(node.get("output_tok_s")))
    model_rates: dict[str,dict[str,float]] = {}
    for node in nodes:
        for model,value in dict(node.get("model_input_tok_s",{})).items():
            model_rates.setdefault(model,{"input_tok_s":0.0,"output_tok_s":0.0,"running":0.0,"waiting":0.0})
            model_rates[model]["input_tok_s"] += value
        for model,value in dict(node.get("model_output_tok_s",{})).items():
            model_rates.setdefault(model,{"input_tok_s":0.0,"output_tok_s":0.0,"running":0.0,"waiting":0.0})
            model_rates[model]["output_tok_s"] += value
        for model,value in dict(node.get("model_running",{})).items():
            model_rates.setdefault(model,{"input_tok_s":0.0,"output_tok_s":0.0,"running":0.0,"waiting":0.0})
            model_rates[model]["running"] += value
        for model,value in dict(node.get("model_waiting",{})).items():
            model_rates.setdefault(model,{"input_tok_s":0.0,"output_tok_s":0.0,"running":0.0,"waiting":0.0})
            model_rates[model]["waiting"] += value
    partitions = load_model_layer_partitions()
    for node in nodes:
        alloc_input = 0.0
        alloc_output = 0.0
        alloc_running = 0.0
        alloc_waiting = 0.0
        active_models: list[str] = []
        model_allocations: list[dict[str,Any]] = []
        for model in list(node.get("pipeline_stage_models",[])):
            rates = model_rates.get(model)
            if rates is None:
                continue
            if max(rates.get("input_tok_s",0.0),rates.get("output_tok_s",0.0),rates.get("running",0.0),rates.get("waiting",0.0)) <= 0.0:
                continue
            pp_size = int(fnum(dict(node.get("pipeline_stage_pp_by_model",{})).get(model,0.0)))
            rank = int(fnum(dict(node.get("pipeline_stage_rank_by_model",{})).get(model,0.0)))
            share = layer_share(model,rank,pp_size,partitions)
            if share <= 0.0:
                continue
            layers,total_layers = layer_count(model,rank,partitions)
            alloc_input += rates.get("input_tok_s",0.0) * share
            alloc_output += rates.get("output_tok_s",0.0) * share
            alloc_running += rates.get("running",0.0) * share
            alloc_waiting += rates.get("waiting",0.0) * share
            model_allocations.append({
                "model": model,
                "rank": rank,
                "pp_size": pp_size,
                "layers": layers,
                "total_layers": total_layers,
                "share": round(share,6),
                "share_pct": round(share * 100.0,2),
                "input_tok_s": round(rates.get("input_tok_s",0.0) * share,3),
                "output_tok_s": round(rates.get("output_tok_s",0.0) * share,3),
                "tok_s": round((rates.get("input_tok_s",0.0) + rates.get("output_tok_s",0.0)) * share,3),
                "running": round(rates.get("running",0.0) * share,3),
                "waiting": round(rates.get("waiting",0.0) * share,3),
            })
            active_models.append(model)
        if active_models:
            node["input_tok_s"] = round(alloc_input,3)
            node["output_tok_s"] = round(alloc_output,3)
            node["tok_s"] = round(alloc_input + alloc_output,3)
            node["vllm_running"] = round(alloc_running,3)
            node["vllm_waiting"] = round(alloc_waiting,3)
            node["token_scope"] = "allocated"
            node["token_models"] = ",".join(active_models)
            node["model_allocations"] = model_allocations
            if node["state"] == "idle":
                node["state"] = "busy"
                node["state_label"] = "stage"
    reachable = [node for node in nodes if node["state"] != "down"]
    gpu_nodes = [node for node in reachable if int(fnum(node.get("sample_count"))) > 0]
    known_kv = [node["kv_pct"] for node in reachable if node.get("kv_known")]
    known_cache = [node["cache_hit_pct"] for node in reachable if node.get("vllm_metrics_up")]
    model_items = [
        {
            "model": model,
            "input_tok_s": round(values.get("input_tok_s",0.0),3),
            "output_tok_s": round(values.get("output_tok_s",0.0),3),
            "tok_s": round(values.get("input_tok_s",0.0) + values.get("output_tok_s",0.0),3),
            "running": round(values.get("running",0.0),3),
            "waiting": round(values.get("waiting",0.0),3),
        }
        for model,values in sorted(model_rates.items(), key=lambda item: item[1].get("output_tok_s",0.0), reverse=True)
        if max(values.get("input_tok_s",0.0),values.get("output_tok_s",0.0),values.get("running",0.0),values.get("waiting",0.0)) > 0.0
    ]
    if model_items:
        input_tok_s = round(sum(item["input_tok_s"] for item in model_items),3)
        output_tok_s = round(sum(item["output_tok_s"] for item in model_items),3)
        tok_s = round(input_tok_s + output_tok_s,3)
    else:
        input_tok_s = sum(fnum(node.get("input_tok_s")) for node in reachable)
        output_tok_s = sum(fnum(node.get("output_tok_s")) for node in reachable)
        tok_s = max(sum(fnum(node.get("tok_s")) for node in reachable),input_tok_s + output_tok_s)
    running = max(sum(node["vllm_running"] for node in reachable),fnum(queue.get("local_queue_running",0.0)))
    waiting = max(sum(node["vllm_waiting"] for node in reachable),fnum(queue.get("local_queue_queued",0.0)))
    power_nodes = [node for node in reachable if node.get("gpu_power_known")]
    total_gpu_power_w = round(sum(fnum(node.get("gpu_power_w")) for node in power_nodes),2)
    avg_gpu_pct = round(sum(fnum(node.get("gpu_pct")) for node in gpu_nodes) / max(1,len(gpu_nodes)),2)
    ds_catalog_services = str(queue.get("local_queue_ds_services",""))
    return({
        "ok": True,
        "summary_path": str(path),
        "updated_iso": raw.get("updated_iso",""),
        "updated_unix": raw.get("updated_unix",0),
        "age_s": max(0,int(time.time()) - int(fnum(raw.get("updated_unix")))),
        "nodes": nodes,
        "models": model_items,
        "reachable_nodes": len(reachable),
        "gpu_known": len(gpu_nodes) > 0,
        "avg_gpu_pct": avg_gpu_pct,
        "busy_gpu_nodes": sum(1 for node in reachable if node["gpu_pct"] >= 90.0),
        "active_nodes": sum(1 for node in reachable if node["state"] in ("busy","hot")),
        "hot_nodes": sum(1 for node in reachable if node["state"] == "hot"),
        "vllm_running": running,
        "vllm_waiting": waiting,
        "kv_known": len(known_kv) > 0,
        "max_kv_pct": max(known_kv or [0.0]),
        "cache_known": len(known_cache) > 0,
        "max_cache_hit_pct": max(known_cache or [0.0]),
        "ds_services_known": active_ds_services != "",
        "ds_services": active_ds_services,
        "ds_service_count": active_ds_service_count,
        "ds_catalog_services": ds_catalog_services,
        "ds_catalog_service_count": fnum(queue.get("local_queue_ds_service_count",0.0)),
        "ds_model_count": fnum(queue.get("local_queue_ds_model_count",0.0)),
        "ds_last_service": str(queue.get("local_queue_last_service","")),
        "ds_kv_shards": fnum(queue.get("local_queue_kv_shards",0.0)),
        "queue_depth": fnum(queue.get("local_queue_depth",0.0)),
        "power_known": len(power_nodes) > 0,
        "power_known_node_count": len(power_nodes),
        "power_node_count": len(gpu_nodes),
        "total_gpu_power_w": total_gpu_power_w,
        "tok_s": tok_s,
        "input_tok_s": input_tok_s,
        "output_tok_s": output_tok_s,
    })


def history_limit(value: str | None) -> int:
    try:
        limit = int(value or str(DEFAULT_HISTORY_LIMIT))
    except ValueError:
        limit = DEFAULT_HISTORY_LIMIT
    return(max(1,min(2000,limit)))


def build_history(nodes_dir: str, node: str, limit: int = 360) -> dict[str,Any]:
    if not valid_node_name(node):
        return({"ok":False,"node":node,"error":"invalid node","metrics":HISTORY_METRICS,"points":[]})
    path = Path(nodes_dir) / ("%s.csv" % node)
    rows: deque[dict[str,str]] = deque(maxlen=limit)
    try:
        with path.open("r",encoding="utf-8",newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("iso_ts","") not in ("","iso_ts"):
                    rows.append(row)
    except Exception as exc:
        return({"ok":False,"node":node,"history_path":str(path),"error":str(exc),"metrics":HISTORY_METRICS,"points":[]})
    points = []
    for row in rows:
        point: dict[str,Any] = {"iso_ts": row.get("iso_ts",""), "unix_ts": fnum(row.get("unix_ts"))}
        for metric in HISTORY_METRICS:
            value = fnum(row.get(metric["field"]))
            if metric["key"] == "cpu_pct":
                value = display_cpu_pct(value)
            point[metric["key"]] = value
        points.append(point)
    return({"ok":True,"node":node,"history_path":str(path),"metrics":HISTORY_METRICS,"points":points})


def stream_payload(summary_path: str, nodes_dir: str, node: str, limit: int) -> dict[str,Any]:
    summary = build_snapshot(summary_path)
    selected = node if valid_node_name(node) else ""
    if not selected and summary.get("nodes"):
        selected = str(summary["nodes"][0].get("node",""))
    summary["selected_node"] = selected
    history = build_history(nodes_dir,selected,limit) if selected else {"ok":False,"node":"","metrics":HISTORY_METRICS,"points":[]}
    return({"summary":summary,"history":history})


def make_handler(summary_path: str, nodes_dir: str) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def handle(self) -> None:
            try:
                super().handle()
            except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                return
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/","/index.html"):
                self._send(200,"text/html; charset=utf-8",DASHBOARD_HTML.encode("utf-8"))
            elif path == "/api/summary":
                payload = json.dumps(build_snapshot(summary_path),sort_keys=True).encode("utf-8")
                self._send(200,"application/json",payload)
            elif path == "/api/history":
                qs = parse_qs(parsed.query)
                node = (qs.get("node") or [""])[0]
                limit = history_limit((qs.get("limit") or [""])[0])
                payload = json.dumps(build_history(nodes_dir,node,limit),sort_keys=True).encode("utf-8")
                self._send(200,"application/json",payload)
            elif path == "/api/stream":
                qs = parse_qs(parsed.query)
                node = (qs.get("node") or [""])[0]
                limit = history_limit((qs.get("limit") or [""])[0])
                self._stream(node,limit)
            elif path == "/healthz":
                self._send(200,"text/plain; charset=utf-8",b"ok\n")
            else:
                self._send(404,"text/plain; charset=utf-8",b"not found\n")
        def log_message(self, fmt: str, *args: Any) -> None:
            return
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def _stream(self, node: str, limit: int) -> None:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.send_header("Cache-Control","no-store")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            while True:
                try:
                    payload = json.dumps(stream_payload(summary_path,nodes_dir,node,limit),sort_keys=True)
                    self.wfile.write(("event: telemetry\ndata: %s\n\n" % payload).encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(STREAM_INTERVAL_S)
                except (BrokenPipeError,ConnectionResetError,OSError):
                    return
    return(DashboardHandler)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    global MODEL_LAYER_PARTITIONS,MODEL_LAYER_PARTITIONS_JSON_OVERRIDE,REPO_ROOT_OVERRIDE
    args = parse_args()
    REPO_ROOT_OVERRIDE = str(args.repo_root or "")
    MODEL_LAYER_PARTITIONS_JSON_OVERRIDE = str(args.layer_partitions_json or "")
    MODEL_LAYER_PARTITIONS = None
    partitions = load_model_layer_partitions()
    server = ReusableThreadingHTTPServer((args.host,args.port),make_handler(args.summary_json,args.nodes_dir))
    print("serving Spark telemetry dashboard on http://%s:%d repo_root=%s layer_partitions=%d" % (args.host,args.port,repo_root(),len(partitions)),flush=True)
    server.serve_forever()
    return(0)


if __name__ == "__main__":
    raise SystemExit(main())
