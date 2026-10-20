#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

try:
    from . import spark_telemetry_common as telemetry
except ImportError:
    import spark_telemetry_common as telemetry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", default=telemetry.DEFAULT_NODES)
    p.add_argument("--remote-dir", default=telemetry.TELEMETRY_DIR)
    p.add_argument("--out-dir", default=telemetry.MAC_TELEMETRY_DIR)
    p.add_argument("--tail-lines", type=int, default=17280)
    p.add_argument("--loop-interval", type=float, default=0.0)
    p.add_argument("--ssh-timeout", type=float, default=8.0)
    p.add_argument("--ssh-config", default="")
    p.add_argument("--stale-ok-seconds", type=float, default=300.0)
    p.add_argument("--ds4-api-url", default=os.environ.get("DS4_API_URL",telemetry.DEFAULT_DS4_API_URL))
    p.add_argument("--ds4-api-timeout", type=float, default=1.5)
    p.add_argument("--ssh-control-dir", default="")
    p.add_argument("--ssh-control-persist", type=int, default=0)
    p.add_argument("--fetch-workers", type=int, default=telemetry.SPARK_NODE_COUNT)
    return(p.parse_args())


def ssh_options(timeout: float, control_dir: str = "", control_persist: int = 0) -> List[str]:
    opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=%d" % max(1,int(timeout)),
    ]
    if control_dir:
        os.makedirs(control_dir,exist_ok=True)
        opts.extend([
            "-o",
            "ServerAliveInterval=2",
            "-o",
            "ServerAliveCountMax=1",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=%ds" % max(1,int(control_persist or 1)),
            "-o",
            "ControlPath=%s" % os.path.join(control_dir,"t-%C"),
        ])
    return(opts)


def fetch_node(node: str, remote_dir: str, timeout: float, lines: int, target: str | None = None, control_dir: str = "", control_persist: int = 0, ssh_config: str = "") -> Tuple[str,str,str]:
    target = target or node
    remote_path = shlex.quote(telemetry.node_csv_path(remote_dir))
    cmd = [
        "ssh",
        *(["-F",ssh_config] if ssh_config else []),
        *ssh_options(timeout,control_dir,control_persist),
        target,
        "if [ -r %s ]; then head -n 1 %s; tail -n %d %s; else exit 1; fi" % (remote_path,remote_path,lines,remote_path),
    ]
    try:
        p = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    except Exception as e:
        return(node,"","%s" % e)
    if p.returncode != 0:
        return(node,"",p.stderr.strip() or ("ssh exited %d" % p.returncode))
    return(node,p.stdout,"")


def read_rows(text: str) -> List[Dict[str,str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) == 0:
        return([])
    while lines and not lines[0].startswith("unix_ts,"):
        lines.pop(0)
    if len(lines) == 0:
        return([])
    rows: List[Dict[str,str]] = []
    for row in csv.DictReader(lines):
        row.pop(None,None)
        if row.get("unix_ts","") == "unix_ts":
            continue
        rows.append(row)
    return(rows)


def row_unix_ts(row: Dict[str,str]) -> float:
    try:
        return(float(row.get("unix_ts","0") or 0.0))
    except Exception:
        pass
    try:
        return(dt.datetime.fromisoformat(row.get("iso_ts","").replace("Z","+00:00")).timestamp())
    except Exception:
        return(0.0)


def labelled_csv(text: str, node: str) -> str:
    rows = read_rows(text)
    if not rows:
        return(text)
    output = io.StringIO()
    writer = csv.DictWriter(output,fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        row["node"] = node
        writer.writerow(row)
    return(output.getvalue())


def read_cached_rows(path: str, max_age_s: float) -> List[Dict[str,str]]:
    try:
        with open(path,"r",encoding="utf-8",newline="") as fp:
            rows = read_rows(fp.read())
    except Exception:
        return([])
    if len(rows) == 0:
        return([])
    age_s = time.time() - row_unix_ts(rows[-1])
    return(rows if age_s <= max_age_s else [])


def summarize_node(rows: List[Dict[str,str]], error: str, fetch_error: str = "", stale_data: bool = False) -> Dict[str,object]:
    if len(rows) == 0:
        return({"sample_count":0,"error":error})
    good = [r for r in rows if not r.get("error")]
    latest = rows[-1]
    latest_gpu = latest
    for row in reversed(good):
        if telemetry.gpu_index(row) >= 0:
            latest_gpu = row
            break
    latest_age_s = max(0.0,time.time() - row_unix_ts(latest))
    gpu_vals = [telemetry.fnum(r,"gpu_util_pct") for r in good if telemetry.gpu_index(r) >= 0]
    gpu_temps = [telemetry.fnum(r,"gpu_temp_c") for r in good if telemetry.gpu_index(r) >= 0 and telemetry.fnum(r,"gpu_temp_c") > 0.0]
    hot = [v for v in gpu_vals if v >= 90.0]
    hot_temps = [v for v in gpu_temps if v >= 80.0]
    return({
        "sample_count": len(rows),
        "first_iso_ts": rows[0].get("iso_ts",""),
        "last_iso_ts": latest.get("iso_ts",""),
        "last_sample_age_s": round(latest_age_s,2),
        "stale_data": 1 if stale_data else 0,
        "fetch_error": fetch_error,
        "last_cpu_util_pct": telemetry.fnum(latest,"cpu_util_pct"),
        "last_mem_used_pct": telemetry.fnum(latest,"mem_used_pct"),
        "last_thermal_avg_c": telemetry.fnum(latest,"thermal_avg_c"),
        "last_thermal_max_c": telemetry.fnum(latest,"thermal_max_c"),
        "last_root_disk_used_pct": telemetry.fnum(latest,"root_disk_used_pct"),
        "last_net_rx_mbps": telemetry.fnum(latest,"net_rx_mbps"),
        "last_net_tx_mbps": telemetry.fnum(latest,"net_tx_mbps"),
        "last_proc_count": telemetry.fnum(latest,"proc_count"),
        "last_thread_count": telemetry.fnum(latest,"thread_count"),
        "last_uptime_s": telemetry.fnum(latest,"uptime_s"),
        "last_ds4_gateway_up": telemetry.fnum(latest,"ds4_gateway_up"),
        "last_ds4_gateway_active": telemetry.fnum(latest,"ds4_gateway_active"),
        "last_ds4_gateway_idle_s": telemetry.fnum(latest,"ds4_gateway_idle_s"),
        "last_ds4_gateway_current_model": latest.get("ds4_gateway_current_model",""),
        "last_ds4_gateway_cpu_pending": telemetry.fnum(latest,"ds4_gateway_cpu_pending"),
        "last_ds4_gateway_cpu_active": telemetry.fnum(latest,"ds4_gateway_cpu_active"),
        "last_ds4_gateway_cpu_completed": telemetry.fnum(latest,"ds4_gateway_cpu_completed"),
        "last_ds4_gateway_cpu_failed": telemetry.fnum(latest,"ds4_gateway_cpu_failed"),
        "last_vllm_metrics_up": telemetry.fnum(latest,"vllm_metrics_up"),
        "last_vllm_requests_running": telemetry.fnum(latest,"vllm_requests_running"),
        "last_vllm_requests_waiting": telemetry.fnum(latest,"vllm_requests_waiting"),
        "last_vllm_requests_running_by_model": latest.get("vllm_requests_running_by_model",""),
        "last_vllm_requests_waiting_by_model": latest.get("vllm_requests_waiting_by_model",""),
        "last_vllm_requests_total": telemetry.fnum(latest,"vllm_requests_total"),
        "last_vllm_requests_per_s": telemetry.fnum(latest,"vllm_requests_per_s"),
        "last_vllm_kv_cache_pct": telemetry.fnum(latest,"vllm_kv_cache_pct"),
        "last_vllm_prompt_tokens_total": telemetry.fnum(latest,"vllm_prompt_tokens_total"),
        "last_vllm_generation_tokens_total": telemetry.fnum(latest,"vllm_generation_tokens_total"),
        "last_vllm_prompt_tokens_total_by_model": latest.get("vllm_prompt_tokens_total_by_model",""),
        "last_vllm_generation_tokens_total_by_model": latest.get("vllm_generation_tokens_total_by_model",""),
        "last_vllm_prompt_tokens_local_compute_total": telemetry.fnum(latest,"vllm_prompt_tokens_local_compute_total"),
        "last_vllm_prompt_tokens_local_cache_hit_total": telemetry.fnum(latest,"vllm_prompt_tokens_local_cache_hit_total"),
        "last_vllm_prompt_tokens_external_kv_transfer_total": telemetry.fnum(latest,"vllm_prompt_tokens_external_kv_transfer_total"),
        "last_vllm_prompt_tokens_cached_total": telemetry.fnum(latest,"vllm_prompt_tokens_cached_total"),
        "last_vllm_prefix_cache_queries_total": telemetry.fnum(latest,"vllm_prefix_cache_queries_total"),
        "last_vllm_prefix_cache_hits_total": telemetry.fnum(latest,"vllm_prefix_cache_hits_total"),
        "last_vllm_external_prefix_cache_queries_total": telemetry.fnum(latest,"vllm_external_prefix_cache_queries_total"),
        "last_vllm_external_prefix_cache_hits_total": telemetry.fnum(latest,"vllm_external_prefix_cache_hits_total"),
        "last_vllm_tokens_total": telemetry.fnum(latest,"vllm_tokens_total"),
        "last_vllm_tokens_per_s": telemetry.fnum(latest,"vllm_tokens_per_s"),
        "last_vllm_prompt_tokens_per_s": telemetry.fnum(latest,"vllm_prompt_tokens_per_s"),
        "last_vllm_generation_tokens_per_s": telemetry.fnum(latest,"vllm_generation_tokens_per_s"),
        "last_vllm_prompt_tokens_per_s_by_model": latest.get("vllm_prompt_tokens_per_s_by_model",""),
        "last_vllm_generation_tokens_per_s_by_model": latest.get("vllm_generation_tokens_per_s_by_model",""),
        "last_vllm_prompt_tokens_cached_per_s": telemetry.fnum(latest,"vllm_prompt_tokens_cached_per_s"),
        "last_vllm_prompt_tokens_local_compute_per_s": telemetry.fnum(latest,"vllm_prompt_tokens_local_compute_per_s"),
        "last_vllm_prompt_tokens_local_cache_hit_per_s": telemetry.fnum(latest,"vllm_prompt_tokens_local_cache_hit_per_s"),
        "last_vllm_prompt_tokens_external_kv_transfer_per_s": telemetry.fnum(latest,"vllm_prompt_tokens_external_kv_transfer_per_s"),
        "last_vllm_prompt_cache_hit_pct": telemetry.fnum(latest,"vllm_prompt_cache_hit_pct"),
        "last_vllm_prefix_cache_hit_pct": telemetry.fnum(latest,"vllm_prefix_cache_hit_pct"),
        "last_vllm_external_prefix_cache_hit_pct": telemetry.fnum(latest,"vllm_external_prefix_cache_hit_pct"),
        "last_vllm_metrics_sources": latest.get("vllm_metrics_sources",""),
        "last_vllm_metrics_scope": latest.get("vllm_metrics_scope",""),
        "last_vllm_pipeline_parallel_size": telemetry.fnum(latest,"vllm_pipeline_parallel_size"),
        "last_vllm_pipeline_node_rank": telemetry.fnum(latest,"vllm_pipeline_node_rank"),
        "last_vllm_pipeline_stage_models": latest.get("vllm_pipeline_stage_models",""),
        "last_vllm_pipeline_stage_pp_by_model": latest.get("vllm_pipeline_stage_pp_by_model",""),
        "last_vllm_pipeline_stage_rank_by_model": latest.get("vllm_pipeline_stage_rank_by_model",""),
        "last_gpu_util_pct": telemetry.fnum(latest_gpu,"gpu_util_pct"),
        "last_gpu_temp_c": telemetry.fnum(latest_gpu,"gpu_temp_c"),
        "last_gpu_fan_pct": telemetry.fnum(latest_gpu,"gpu_fan_pct"),
        "last_gpu_clock_sm_mhz": telemetry.fnum(latest_gpu,"gpu_clock_sm_mhz"),
        "last_gpu_clock_mem_mhz": telemetry.fnum(latest_gpu,"gpu_clock_mem_mhz"),
        "last_gpu_power_w": telemetry.fnum(latest_gpu,"gpu_power_w"),
        "last_gpu_power_raw_w": telemetry.fnum(latest_gpu,"gpu_power_raw_w"),
        "last_gpu_power_limit_w": telemetry.fnum(latest_gpu,"gpu_power_limit_w"),
        "last_gpu_power_known": telemetry.fnum(latest_gpu,"gpu_power_known"),
        "last_gpu_power_source": latest_gpu.get("gpu_power_source",""),
        "last_gpu_power_reason": latest_gpu.get("gpu_power_reason",""),
        "last_gpu_pstate": latest_gpu.get("gpu_pstate",""),
        "cpu_util_pct": telemetry.stats(telemetry.fnum(r,"cpu_util_pct") for r in rows),
        "mem_used_pct": telemetry.stats(telemetry.fnum(r,"mem_used_pct") for r in rows),
        "thermal_max_c": telemetry.stats(telemetry.fnum(r,"thermal_max_c") for r in rows if telemetry.fnum(r,"thermal_max_c") > 0.0),
        "root_disk_used_pct": telemetry.stats(telemetry.fnum(r,"root_disk_used_pct") for r in rows),
        "net_rx_mbps": telemetry.stats(telemetry.fnum(r,"net_rx_mbps") for r in rows),
        "net_tx_mbps": telemetry.stats(telemetry.fnum(r,"net_tx_mbps") for r in rows),
        "ds4_gateway_cpu_pending": telemetry.stats(telemetry.fnum(r,"ds4_gateway_cpu_pending") for r in rows),
        "ds4_gateway_cpu_active": telemetry.stats(telemetry.fnum(r,"ds4_gateway_cpu_active") for r in rows),
        "vllm_requests_running": telemetry.stats(telemetry.fnum(r,"vllm_requests_running") for r in rows),
        "vllm_requests_waiting": telemetry.stats(telemetry.fnum(r,"vllm_requests_waiting") for r in rows),
        "vllm_requests_per_s": telemetry.stats(telemetry.fnum(r,"vllm_requests_per_s") for r in rows),
        "vllm_kv_cache_pct": telemetry.stats(telemetry.fnum(r,"vllm_kv_cache_pct") for r in rows),
        "vllm_tokens_per_s": telemetry.stats(telemetry.fnum(r,"vllm_tokens_per_s") for r in rows),
        "vllm_prompt_tokens_per_s": telemetry.stats(telemetry.fnum(r,"vllm_prompt_tokens_per_s") for r in rows),
        "vllm_generation_tokens_per_s": telemetry.stats(telemetry.fnum(r,"vllm_generation_tokens_per_s") for r in rows),
        "vllm_prompt_tokens_cached_per_s": telemetry.stats(telemetry.fnum(r,"vllm_prompt_tokens_cached_per_s") for r in rows),
        "vllm_prompt_cache_hit_pct": telemetry.stats(telemetry.fnum(r,"vllm_prompt_cache_hit_pct") for r in rows),
        "vllm_prefix_cache_hit_pct": telemetry.stats(telemetry.fnum(r,"vllm_prefix_cache_hit_pct") for r in rows),
        "vllm_external_prefix_cache_hit_pct": telemetry.stats(telemetry.fnum(r,"vllm_external_prefix_cache_hit_pct") for r in rows),
        "gpu_util_pct": telemetry.stats(gpu_vals),
        "gpu_temp_c": telemetry.stats(gpu_temps),
        "gpu_samples_ge_90": len(hot),
        "pct_gpu_samples_ge_90": round(100.0 * len(hot) / len(gpu_vals),2) if gpu_vals else 0.0,
        "gpu_temp_samples_ge_80": len(hot_temps),
        "pct_gpu_temp_samples_ge_80": round(100.0 * len(hot_temps) / len(gpu_temps),2) if gpu_temps else 0.0,
        "error": error,
    })


def write_combined(out_dir: str, all_rows: Dict[str,List[Dict[str,str]]], errors: Dict[str,str], queue: Dict[str,object], fetch_errors: Dict[str,str] | None = None, stale_nodes: set[str] | None = None) -> Dict[str,object]:
    fetch_errors = fetch_errors or {}
    stale_nodes = stale_nodes or set()
    os.makedirs(out_dir,exist_ok=True)
    combined_path = os.path.join(out_dir,"combined_latest.csv")
    summary_path = os.path.join(out_dir,"cluster_summary.json")
    md_path = os.path.join(out_dir,"cluster_summary.md")
    fieldnames: List[str] = []
    seen = set()
    for rows in all_rows.values():
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(combined_path + ".tmp","w",encoding="utf-8",newline="") as f:
        if fieldnames:
            w = csv.DictWriter(f,fieldnames=fieldnames)
            w.writeheader()
            for node in sorted(all_rows):
                for row in all_rows[node]:
                    w.writerow({key:row.get(key,"") for key in fieldnames})
    os.replace(combined_path + ".tmp",combined_path)
    summary = telemetry.summary_base()
    summary.update({"combined_csv":combined_path,"queue":queue,"nodes":{}})
    for node in sorted(set(all_rows) | set(errors)):
        summary["nodes"][node] = summarize_node(all_rows.get(node,[]),errors.get(node,""),fetch_errors.get(node,""),node in stale_nodes)
    telemetry.write_json_atomic(summary_path,summary)
    lines = ["# Spark telemetry summary",""]
    if str(queue.get("local_queue_source","")):
        lines.append("DS API queue: depth=%s queued=%s running=%s services=%s kv_shards=%s source=%s" % (
            queue.get("local_queue_depth",0),
            queue.get("local_queue_queued",0),
            queue.get("local_queue_running",0),
            queue.get("local_queue_active_services",""),
            queue.get("local_queue_kv_shards",0),
            queue.get("local_queue_source",""),
        ))
        lines.append("")
    global_queue_known = str(queue.get("local_queue_source","")) != ""
    lines.append("| node | samples | gpu % | gpu C | gpu W | vLLM run/wait | KV % | api q | gateway cpu q | disk % | rx Mbps | tx Mbps | cpu % | mem % | model | error | req/s | in tok/s | out tok/s | cache hit % | ext hit % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for node,row in summary["nodes"].items():
        sample_known = int(float(row.get("sample_count",0))) > 0
        vllm_known = float(row.get("last_vllm_metrics_up",0.0)) > 0.0
        queue_known = global_queue_known
        work_known = vllm_known or queue_known
        gateway_known = float(row.get("last_ds4_gateway_up",0.0)) > 0.0
        vllm_running = float(row.get("last_vllm_requests_running",0.0))
        vllm_waiting = float(row.get("last_vllm_requests_waiting",0.0))
        local_q = float(queue.get("local_queue_depth",0.0)) if queue_known else 0.0
        if not vllm_known and queue_known:
            vllm_running = max(vllm_running,float(queue.get("local_queue_running",0.0)))
            vllm_waiting = max(vllm_waiting,float(queue.get("local_queue_queued",0.0)))
        kv_text = "%.2f" % float(row.get("last_vllm_kv_cache_pct",0.0)) if vllm_known else "n/a"
        req_s = float(row.get("last_vllm_requests_per_s",0.0))
        in_tok_s = float(row.get("last_vllm_prompt_tokens_per_s",0.0))
        out_tok_s = float(row.get("last_vllm_generation_tokens_per_s",0.0))
        cache_hit_pct = float(row.get("last_vllm_prompt_cache_hit_pct",0.0))
        ext_hit_pct = float(row.get("last_vllm_external_prefix_cache_hit_pct",0.0))
        gpu_util_text = "%.2f" % float(row.get("last_gpu_util_pct",0.0)) if sample_known else "n/a"
        gpu_temp_text = "%.2f" % float(row.get("last_gpu_temp_c",0.0)) if sample_known else "n/a"
        power_known = float(row.get("last_gpu_power_known",0.0)) > 0.0
        gpu_power_text = "%.2f" % float(row.get("last_gpu_power_w",0.0)) if sample_known and power_known else "n/a"
        work_text = "%.0f/%.0f" % (vllm_running,vllm_waiting) if work_known else "n/a"
        local_q_text = "%.0f" % local_q if queue_known else "n/a"
        gateway_text = "%.0f/%.0f" % (float(row.get("last_ds4_gateway_cpu_pending",0.0)),float(row.get("last_ds4_gateway_cpu_active",0.0))) if gateway_known else "n/a"
        disk_text = "%.2f" % float(row.get("last_root_disk_used_pct",0.0)) if sample_known else "n/a"
        rx_text = "%.4f" % float(row.get("last_net_rx_mbps",0.0)) if sample_known else "n/a"
        tx_text = "%.4f" % float(row.get("last_net_tx_mbps",0.0)) if sample_known else "n/a"
        cpu_text = "%.2f" % float(row.get("last_cpu_util_pct",0.0)) if sample_known else "n/a"
        mem_text = "%.2f" % float(row.get("last_mem_used_pct",0.0)) if sample_known else "n/a"
        req_s_text = "%.1f" % req_s if work_known else "n/a"
        in_tok_s_text = "%.3f" % in_tok_s if work_known else "n/a"
        out_tok_s_text = "%.3f" % out_tok_s if work_known else "n/a"
        cache_text = "%.2f" % cache_hit_pct if vllm_known else "n/a"
        ext_text = "%.2f" % ext_hit_pct if vllm_known else "n/a"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            node,
            row.get("sample_count",0),
            gpu_util_text,
            gpu_temp_text,
            gpu_power_text,
            work_text,
            kv_text,
            local_q_text,
            gateway_text,
            disk_text,
            rx_text,
            tx_text,
            cpu_text,
            mem_text,
            str(row.get("last_ds4_gateway_current_model","")).replace("|","/")[:40],
            str(row.get("error","")).replace("|","/"),
            req_s_text,
            in_tok_s_text,
            out_tok_s_text,
            cache_text,
            ext_text,
        ))
    telemetry.write_text_atomic(md_path,"\n".join(lines) + "\n")
    return(summary)


def collect_once(args: argparse.Namespace) -> Dict[str,object]:
    raw_dir = os.path.join(args.out_dir,"nodes")
    os.makedirs(raw_dir,exist_ok=True)
    all_rows: Dict[str,List[Dict[str,str]]] = {}
    errors: Dict[str,str] = {}
    fetch_errors: Dict[str,str] = {}
    stale_nodes: set[str] = set()
    targets = telemetry.parse_node_targets(args.nodes)
    workers = max(1,min(len(targets),int(args.fetch_workers)))
    fetched: List[Tuple[str,str,str]] = []
    if workers <= 1:
        for node,target in targets:
            fetched.append(fetch_node(node,args.remote_dir,args.ssh_timeout,args.tail_lines,target,args.ssh_control_dir,args.ssh_control_persist,args.ssh_config))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_node,node,args.remote_dir,args.ssh_timeout,args.tail_lines,target,args.ssh_control_dir,args.ssh_control_persist,args.ssh_config) for node,target in targets]
            for future in as_completed(futures):
                fetched.append(future.result())
    for name,text,error in fetched:
        if error:
            cached = read_cached_rows(os.path.join(raw_dir,name + ".csv"),args.stale_ok_seconds)
            if cached:
                fetch_errors[name] = error
                stale_nodes.add(name)
                all_rows[name] = cached
            else:
                errors[name] = error
                all_rows[name] = []
            continue
        text = labelled_csv(text,name)
        telemetry.write_text_atomic(os.path.join(raw_dir,name + ".csv"),text)
        all_rows[name] = read_rows(text)
    queue = telemetry.read_ds4_api_queue(args.ds4_api_url,args.ds4_api_timeout)
    return(write_combined(args.out_dir,all_rows,errors,queue,fetch_errors,stale_nodes))


def main() -> int:
    args = parse_args()
    while True:
        summary = collect_once(args)
        print("wrote %s" % summary["combined_csv"],flush=True)
        if args.loop_interval <= 0.0:
            break
        time.sleep(args.loop_interval)
    return(0)


if __name__ == "__main__":
    raise SystemExit(main())
