#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Deque, Dict, Iterable, List, Optional, Tuple

try:
    from . import spark_telemetry_common as telemetry
except ImportError:
    import spark_telemetry_common as telemetry


BASE_GPU_FIELDS = telemetry.BASE_GPU_FIELDS
GPU_FIELDS = telemetry.GPU_FIELDS
CSV_FIELDS = telemetry.CSV_FIELDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", default=socket.gethostname())
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--out-dir", default=telemetry.TELEMETRY_DIR)
    p.add_argument("--summary-samples", type=int, default=720)
    p.add_argument("--nvidia-smi-timeout", type=float, default=3.0)
    p.add_argument("--gateway-url", default="http://127.0.0.1:8700")
    p.add_argument("--metrics-urls", default="auto,http://127.0.0.1:8102/metrics,http://127.0.0.1:8101/metrics")
    p.add_argument("--http-timeout", type=float, default=1.5)
    return(p.parse_args())


def read_cpu_times() -> Optional[Tuple[int,int]]:
    try:
        with open("/proc/stat","r",encoding="utf-8") as f:
            parts = f.readline().split()
    except Exception:
        return(None)
    if len(parts) < 8 or parts[0] != "cpu":
        return(None)
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return(sum(vals),idle)


def cpu_pct(prev: Optional[Tuple[int,int]], cur: Optional[Tuple[int,int]]) -> float:
    if prev is None or cur is None:
        return(0.0)
    total = cur[0] - prev[0]
    idle = cur[1] - prev[1]
    if total <= 0:
        return(0.0)
    return(round(100.0 * (total - idle) / total,2))


def read_loadavg() -> Tuple[float,float,float]:
    try:
        with open("/proc/loadavg","r",encoding="utf-8") as f:
            vals = f.read().split()
        return(float(vals[0]),float(vals[1]),float(vals[2]))
    except Exception:
        return(0.0,0.0,0.0)


def read_meminfo() -> Dict[str,float]:
    info: Dict[str,float] = {}
    try:
        with open("/proc/meminfo","r",encoding="utf-8") as f:
            for line in f:
                parts = line.replace(":","").split()
                if len(parts) >= 2:
                    info[parts[0]] = float(parts[1]) / 1024.0
    except Exception:
        pass
    total = info.get("MemTotal",0.0)
    avail = info.get("MemAvailable",0.0)
    used = max(0.0,total - avail)
    swap_total = info.get("SwapTotal",0.0)
    swap_free = info.get("SwapFree",0.0)
    swap_used = max(0.0,swap_total - swap_free)
    return({
        "mem_total_mib": round(total,2),
        "mem_available_mib": round(avail,2),
        "mem_used_mib": round(used,2),
        "mem_used_pct": round(100.0 * used / total,2) if total > 0.0 else 0.0,
        "swap_total_mib": round(swap_total,2),
        "swap_used_mib": round(swap_used,2),
        "swap_used_pct": round(100.0 * swap_used / swap_total,2) if swap_total > 0.0 else 0.0,
    })


def read_thermal() -> Dict[str,object]:
    vals: List[float] = []
    labels: List[str] = []
    roots = ["/sys/class/thermal","/sys/class/hwmon"]
    for root in roots:
        try:
            stack = [os.path.join(root,name) for name in os.listdir(root)]
        except Exception:
            stack = []
        for path in stack:
            if os.path.basename(path).startswith("thermal_zone"):
                temp_path = os.path.join(path,"temp")
                try:
                    with open(temp_path,"r",encoding="utf-8") as f:
                        raw = f.read().strip()
                    val = float(raw)
                    if val > 1000.0:
                        val /= 1000.0
                    if val < -40.0 or val > 130.0:
                        continue
                    label = os.path.basename(path)
                    type_path = os.path.join(path,"type")
                    try:
                        with open(type_path,"r",encoding="utf-8") as f:
                            label = f.read().strip() or label
                    except Exception:
                        pass
                    vals.append(val)
                    labels.append(label)
                except Exception:
                    pass
            try:
                names = [name for name in os.listdir(path) if name.startswith("temp") and name.endswith("_input")]
            except Exception:
                continue
            for name in names:
                full = os.path.join(path,name)
                try:
                    with open(full,"r",encoding="utf-8") as f:
                        raw = f.read().strip()
                    val = float(raw)
                    if val > 1000.0:
                        val /= 1000.0
                    if val < -40.0 or val > 130.0:
                        continue
                except Exception:
                    continue
                label = name.replace("_input","")
                label_path = os.path.join(path,name.replace("_input","_label"))
                type_path = os.path.join(path,"type")
                try:
                    with open(label_path,"r",encoding="utf-8") as f:
                        label = f.read().strip() or label
                except Exception:
                    try:
                        with open(type_path,"r",encoding="utf-8") as f:
                            label = f.read().strip() or label
                    except Exception:
                        pass
                vals.append(val)
                labels.append(label)
    return({
        "thermal_avg_c": round(sum(vals) / len(vals),2) if vals else 0.0,
        "thermal_max_c": round(max(vals),2) if vals else 0.0,
        "thermal_sources": ";".join(sorted(set(labels)))[:240],
    })


def read_disk(path: str = "/") -> Dict[str,float]:
    try:
        s = os.statvfs(path)
    except Exception:
        return({"root_disk_total_gib":0.0,"root_disk_used_gib":0.0,"root_disk_available_gib":0.0,"root_disk_used_pct":0.0})
    total = float(s.f_blocks * s.f_frsize)
    avail = float(s.f_bavail * s.f_frsize)
    used = max(0.0,total - avail)
    return({
        "root_disk_total_gib": round(total / (1024.0 ** 3),2),
        "root_disk_used_gib": round(used / (1024.0 ** 3),2),
        "root_disk_available_gib": round(avail / (1024.0 ** 3),2),
        "root_disk_used_pct": round(100.0 * used / total,2) if total > 0.0 else 0.0,
    })


def read_netdev() -> Tuple[int,int]:
    rx = 0
    tx = 0
    try:
        with open("/proc/net/dev","r",encoding="utf-8") as f:
            lines = f.readlines()[2:]
    except Exception:
        return(0,0)
    for line in lines:
        if ":" not in line:
            continue
        iface,rest = line.split(":",1)
        iface = iface.strip()
        if iface == "lo":
            continue
        parts = rest.split()
        if len(parts) < 16:
            continue
        try:
            rx += int(parts[0])
            tx += int(parts[8])
        except Exception:
            pass
    return(rx,tx)


def net_rates(prev: Optional[Tuple[int,int]], cur: Tuple[int,int], elapsed: float) -> Dict[str,float]:
    if prev is None or elapsed <= 0.0:
        return({"net_rx_bytes":cur[0],"net_tx_bytes":cur[1],"net_rx_mbps":0.0,"net_tx_mbps":0.0})
    rx_delta = max(0,cur[0] - prev[0])
    tx_delta = max(0,cur[1] - prev[1])
    return({
        "net_rx_bytes": cur[0],
        "net_tx_bytes": cur[1],
        "net_rx_mbps": round((float(rx_delta) * 8.0) / (elapsed * 1000000.0),4),
        "net_tx_mbps": round((float(tx_delta) * 8.0) / (elapsed * 1000000.0),4),
    })


def read_proc_counts() -> Dict[str,int]:
    procs = 0
    threads = 0
    try:
        names = os.listdir("/proc")
    except Exception:
        return({"proc_count":0,"thread_count":0})
    for name in names:
        if not name.isdigit():
            continue
        procs += 1
        try:
            with open(os.path.join("/proc",name,"status"),"r",encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Threads:"):
                        threads += int(line.split()[1])
                        break
        except Exception:
            pass
    return({"proc_count":procs,"thread_count":threads})


def read_uptime_s() -> float:
    try:
        with open("/proc/uptime","r",encoding="utf-8") as f:
            return(round(float(f.read().split()[0]),2))
    except Exception:
        return(0.0)


def read_system(prev_net: Optional[Tuple[int,int]], elapsed: float) -> Tuple[Dict[str,object],Tuple[int,int]]:
    cur_net = read_netdev()
    out: Dict[str,object] = {}
    out.update(read_thermal())
    out.update(read_disk("/"))
    out.update(net_rates(prev_net,cur_net,elapsed))
    out.update(read_proc_counts())
    out["uptime_s"] = read_uptime_s()
    return(out,cur_net)


def read_json_url(url: str, timeout: float) -> Tuple[Dict[str,object],str]:
    try:
        with urllib.request.urlopen(url,timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8",errors="replace")[:300]
        except Exception:
            detail = str(e)
        return({},detail)
    except Exception as e:
        return({},"%s" % e)
    return(data if isinstance(data,dict) else {}, "")


def read_text_url(url: str, timeout: float) -> Tuple[str,str]:
    try:
        with urllib.request.urlopen(url,timeout=timeout) as resp:
            return(resp.read().decode("utf-8",errors="replace"),"")
    except Exception as e:
        return("","%s" % e)


def empty_gateway() -> Dict[str,object]:
    return({
        "ds4_gateway_up": 0,
        "ds4_gateway_active": 0,
        "ds4_gateway_idle_s": 0.0,
        "ds4_gateway_current_model": "",
        "ds4_gateway_cpu_pending": 0,
        "ds4_gateway_cpu_active": 0,
        "ds4_gateway_cpu_completed": 0,
        "ds4_gateway_cpu_failed": 0,
    })


def queue_counts(data: Dict[str,object]) -> Tuple[int,int,int,int]:
    counts_raw = data.get("state_counts",{})
    if not isinstance(counts_raw,dict):
        return(0,0,0,0)
    counts = {str(k):int(telemetry.num(v)) for k,v in counts_raw.items()}
    pending = int(counts.get("queued",0) + counts.get("prefilling",0) + counts.get("ready",0))
    active = int(counts.get("running",0))
    completed = int(counts.get("completed",0) + counts.get("completed_with_failures",0) + counts.get("completed_with_cancelled",0))
    failed = int(counts.get("failed",0))
    return(pending,active,completed,failed)


def read_coordinator_gateway(base_url: str, timeout: float) -> Dict[str,object]:
    out = empty_gateway()
    base = base_url.rstrip("/")
    health,error = read_json_url(base + "/health",timeout)
    dispatcher,dispatcher_error = read_json_url(base + "/ds4/dispatcher/status",timeout)
    queue,queue_error = read_json_url(base + "/ds4/queue/status",timeout)
    if error != "" or dispatcher_error != "" or queue_error != "":
        return(out)
    if not bool(health.get("ok",False)) or str(queue.get("format","")) != "ds4-inference-queue-v1":
        return(out)
    pending,active,completed,failed = queue_counts(queue)
    out["ds4_gateway_up"] = 1
    out["ds4_gateway_active"] = 1 if bool(dispatcher.get("running",False)) or active > 0 else 0
    out["ds4_gateway_current_model"] = str(dispatcher.get("last_claimed_service_id") or "")
    out["ds4_gateway_cpu_pending"] = pending
    out["ds4_gateway_cpu_active"] = active
    out["ds4_gateway_cpu_completed"] = completed
    out["ds4_gateway_cpu_failed"] = failed
    last_work_at = telemetry.num(dispatcher.get("last_work_at",0.0))
    if last_work_at > 0.0:
        out["ds4_gateway_idle_s"] = round(max(0.0,time.time() - last_work_at),2)
    return(out)


def read_gateway(raw_urls: str, timeout: float) -> Dict[str,object]:
    base_url = str(raw_urls or "").strip()
    return(read_coordinator_gateway(base_url,timeout) if base_url else empty_gateway())


def prometheus_value(line: str) -> float:
    try:
        return(float(line.rsplit(" ",1)[1]))
    except Exception:
        return(0.0)


def prometheus_name(line: str) -> str:
    text = line.strip()
    if text == "" or text.startswith("#"):
        return("")
    head = text.split(None,1)[0]
    return(head.split("{",1)[0])


def prometheus_label(line: str, key: str) -> str:
    marker = key + "=\""
    start = line.find(marker)
    if start < 0:
        return("")
    start += len(marker)
    end = line.find("\"",start)
    return(line[start:end] if end >= start else "")


def metric_map_text(values: Dict[str,float]) -> str:
    parts: List[str] = []
    for key in sorted(values):
        safe_key = str(key).replace(";","_").replace(":","_").strip()
        if safe_key:
            parts.append("%s:%.6f" % (safe_key,float(values[key])))
    return(";".join(parts)[:480])


def parse_metric_map(raw: object) -> Dict[str,float]:
    out: Dict[str,float] = {}
    for item in str(raw or "").split(";"):
        if ":" not in item:
            continue
        key,value = item.split(":",1)
        key = key.strip()
        try:
            out[key] = float(value)
        except Exception:
            continue
    return(out)


def add_metric_value(values: Dict[str,float], key: str, value: float) -> None:
    safe_key = str(key or "").strip()
    if safe_key:
        values[safe_key] = values.get(safe_key,0.0) + value


def prometheus_model(line: str) -> str:
    return(prometheus_label(line,"model_name") or prometheus_label(line,"model"))


def argv_value(argv: List[str], flag: str) -> Optional[str]:
    for i,arg in enumerate(argv):
        if arg == flag and (i + 1) < len(argv):
            return(argv[i + 1])
        if arg.startswith(flag + "="):
            return(arg.split("=",1)[1])
    return(None)


def argv_int(argv: List[str], flag: str, default: int) -> int:
    try:
        return(int(argv_value(argv,flag) or default))
    except Exception:
        return(default)


def vllm_metrics_target_from_argv(argv: List[str]) -> Optional[Dict[str,object]]:
    if "serve" not in argv or "--headless" in argv:
        return(None)
    joined = " ".join(argv)
    if "vllm.entrypoints.cli.main" not in joined and os.path.basename(argv[0] if argv else "") != "vllm":
        return(None)
    port = argv_int(argv,"--port",8000)
    pipeline_parallel_size = argv_int(argv,"--pipeline-parallel-size",1)
    node_rank = argv_int(argv,"--node-rank",0)
    return({
        "url": "http://127.0.0.1:%d/metrics" % port,
        "scope": "pipeline" if pipeline_parallel_size > 1 else "local",
        "pipeline_parallel_size": pipeline_parallel_size,
        "pipeline_node_rank": node_rank,
    })


def vllm_metrics_port_from_argv(argv: List[str]) -> Optional[int]:
    target = vllm_metrics_target_from_argv(argv)
    if target is None:
        return(None)
    url = str(target.get("url",""))
    try:
        return(int(url.rsplit(":",1)[1].split("/",1)[0]))
    except Exception:
        return(None)


def discover_vllm_metrics_targets(timeout: float) -> List[Dict[str,object]]:
    try:
        p = subprocess.run(["ps","-eo","args"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=max(1.0,timeout))
    except Exception:
        return([])
    if p.returncode != 0:
        return([])
    targets: List[Dict[str,object]] = []
    seen: Dict[str,object] = {}
    for line in p.stdout.splitlines():
        try:
            argv = shlex.split(line)
        except Exception:
            continue
        target = vllm_metrics_target_from_argv(argv)
        if target is None:
            continue
        url = str(target.get("url",""))
        if url not in seen:
            seen[url] = True
            targets.append(target)
    return(targets)


def discover_vllm_metrics_urls(timeout: float) -> List[str]:
    urls: List[str] = []
    for target in discover_vllm_metrics_targets(timeout):
        urls.append(str(target.get("url","")))
    return(urls)


def discover_vllm_pipeline_stage_models(timeout: float) -> Tuple[Dict[str,int],Dict[str,int]]:
    try:
        p = subprocess.run(["ps","-eo","args"],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=max(1.0,timeout))
    except Exception:
        return(({},{}))
    if p.returncode != 0:
        return(({},{}))
    models: Dict[str,int] = {}
    ranks: Dict[str,int] = {}
    for line in p.stdout.splitlines():
        try:
            argv = shlex.split(line)
        except Exception:
            continue
        if "serve" not in argv:
            continue
        joined = " ".join(argv)
        if "vllm.entrypoints.cli.main" not in joined and os.path.basename(argv[0] if argv else "") != "vllm":
            continue
        pipeline_parallel_size = argv_int(argv,"--pipeline-parallel-size",1)
        if pipeline_parallel_size <= 1:
            continue
        model = argv_value(argv,"--served-model-name") or ""
        if model:
            models[model] = pipeline_parallel_size
            ranks[model] = argv_int(argv,"--node-rank",0)
    return((models,ranks))


def metrics_targets(raw_urls: str, timeout: float) -> List[Dict[str,object]]:
    targets: List[Dict[str,object]] = []
    seen: Dict[str,object] = {}
    for item in [part.strip() for part in raw_urls.split(",") if part.strip()]:
        expanded = discover_vllm_metrics_targets(timeout) if item == "auto" else [{"url": item, "scope": "local", "pipeline_parallel_size": 1, "pipeline_node_rank": 0}]
        for target in expanded:
            url = str(target.get("url",""))
            if url not in seen:
                seen[url] = True
                targets.append(target)
            elif str(target.get("scope","local")) == "pipeline":
                for existing in targets:
                    if str(existing.get("url","")) == url:
                        existing.update(target)
    return(targets)


def metrics_urls(raw_urls: str, timeout: float) -> List[str]:
    urls: List[str] = []
    for target in metrics_targets(raw_urls,timeout):
        urls.append(str(target.get("url","")))
    return(urls)


def read_vllm_metrics(raw_urls: str, timeout: float, prev: Optional[Dict[str,object]] = None, now: Optional[float] = None) -> Dict[str,object]:
    out: Dict[str,object] = {
        "vllm_metrics_up": 0,
        "vllm_requests_running": 0.0,
        "vllm_requests_waiting": 0.0,
        "vllm_requests_running_by_model": "",
        "vllm_requests_waiting_by_model": "",
        "vllm_requests_total": 0.0,
        "vllm_requests_per_s": 0.0,
        "vllm_kv_cache_pct": 0.0,
        "vllm_prompt_tokens_total": 0.0,
        "vllm_generation_tokens_total": 0.0,
        "vllm_prompt_tokens_total_by_model": "",
        "vllm_generation_tokens_total_by_model": "",
        "vllm_prompt_tokens_local_compute_total": 0.0,
        "vllm_prompt_tokens_local_cache_hit_total": 0.0,
        "vllm_prompt_tokens_external_kv_transfer_total": 0.0,
        "vllm_prompt_tokens_cached_total": 0.0,
        "vllm_prefix_cache_queries_total": 0.0,
        "vllm_prefix_cache_hits_total": 0.0,
        "vllm_external_prefix_cache_queries_total": 0.0,
        "vllm_external_prefix_cache_hits_total": 0.0,
        "vllm_tokens_total": 0.0,
        "vllm_tokens_per_s": 0.0,
        "vllm_prompt_tokens_per_s": 0.0,
        "vllm_generation_tokens_per_s": 0.0,
        "vllm_prompt_tokens_per_s_by_model": "",
        "vllm_generation_tokens_per_s_by_model": "",
        "vllm_prompt_tokens_cached_per_s": 0.0,
        "vllm_prompt_tokens_local_compute_per_s": 0.0,
        "vllm_prompt_tokens_local_cache_hit_per_s": 0.0,
        "vllm_prompt_tokens_external_kv_transfer_per_s": 0.0,
        "vllm_prompt_cache_hit_pct": 0.0,
        "vllm_prefix_cache_hit_pct": 0.0,
        "vllm_external_prefix_cache_hit_pct": 0.0,
        "vllm_metrics_sources": "",
        "vllm_metrics_scope": "",
        "vllm_pipeline_parallel_size": 0,
        "vllm_pipeline_node_rank": 0,
        "vllm_pipeline_stage_models": "",
        "vllm_pipeline_stage_pp_by_model": "",
        "vllm_pipeline_stage_rank_by_model": "",
    }
    sources: List[str] = []
    source_scopes: List[str] = []
    pipeline_parallel_size = 0
    pipeline_node_rank = 0
    kv_vals: List[float] = []
    running_by_model: Dict[str,float] = {}
    waiting_by_model: Dict[str,float] = {}
    prompt_by_model: Dict[str,float] = {}
    generation_by_model: Dict[str,float] = {}
    source_prompt_total = 0.0
    request_counter_total = 0.0
    http_request_counter_total = 0.0
    for target in metrics_targets(raw_urls,timeout):
        url = str(target.get("url",""))
        text,error = read_text_url(url,timeout)
        if error != "":
            continue
        found = False
        for line in text.splitlines():
            name = prometheus_name(line)
            if name == "vllm:num_requests_running":
                value = prometheus_value(line)
                out["vllm_requests_running"] = float(out["vllm_requests_running"]) + value
                add_metric_value(running_by_model,prometheus_model(line),value)
                found = True
            elif name == "vllm:num_requests_waiting":
                value = prometheus_value(line)
                out["vllm_requests_waiting"] = float(out["vllm_requests_waiting"]) + value
                add_metric_value(waiting_by_model,prometheus_model(line),value)
                found = True
            elif name in ("vllm:request_success_total","vllm:request_failure_total","vllm:requests_total"):
                request_counter_total += prometheus_value(line)
                found = True
            elif name == "http_requests_total":
                handler = prometheus_label(line,"handler")
                method = prometheus_label(line,"method")
                if method == "POST" and handler in ("/v1/completions","/v1/chat/completions"):
                    http_request_counter_total += prometheus_value(line)
                    found = True
            elif name in ("vllm:kv_cache_usage_perc","vllm:gpu_cache_usage_perc"):
                value = prometheus_value(line)
                kv_vals.append(value * 100.0 if value <= 1.0 else value)
                found = True
            elif name in ("vllm:prompt_tokens_total","vllm:prompt_tokens"):
                value = prometheus_value(line)
                out["vllm_prompt_tokens_total"] = float(out["vllm_prompt_tokens_total"]) + value
                add_metric_value(prompt_by_model,prometheus_model(line),value)
                found = True
            elif name in ("vllm:generation_tokens_total","vllm:generation_tokens"):
                value = prometheus_value(line)
                out["vllm_generation_tokens_total"] = float(out["vllm_generation_tokens_total"]) + value
                add_metric_value(generation_by_model,prometheus_model(line),value)
                found = True
            elif name == "vllm:prompt_tokens_cached_total":
                out["vllm_prompt_tokens_cached_total"] = float(out["vllm_prompt_tokens_cached_total"]) + prometheus_value(line)
                found = True
            elif name == "vllm:prefix_cache_queries_total":
                out["vllm_prefix_cache_queries_total"] = float(out["vllm_prefix_cache_queries_total"]) + prometheus_value(line)
                found = True
            elif name == "vllm:prefix_cache_hits_total":
                out["vllm_prefix_cache_hits_total"] = float(out["vllm_prefix_cache_hits_total"]) + prometheus_value(line)
                found = True
            elif name == "vllm:external_prefix_cache_queries_total":
                out["vllm_external_prefix_cache_queries_total"] = float(out["vllm_external_prefix_cache_queries_total"]) + prometheus_value(line)
                found = True
            elif name == "vllm:external_prefix_cache_hits_total":
                out["vllm_external_prefix_cache_hits_total"] = float(out["vllm_external_prefix_cache_hits_total"]) + prometheus_value(line)
                found = True
            elif name in ("vllm:prompt_tokens_by_source","vllm:prompt_tokens_by_source_total"):
                value = prometheus_value(line)
                source = prometheus_label(line,"source")
                if source == "local_compute":
                    out["vllm_prompt_tokens_local_compute_total"] = float(out["vllm_prompt_tokens_local_compute_total"]) + value
                    source_prompt_total += value
                elif source == "local_cache_hit":
                    out["vllm_prompt_tokens_local_cache_hit_total"] = float(out["vllm_prompt_tokens_local_cache_hit_total"]) + value
                    source_prompt_total += value
                elif source == "external_kv_transfer":
                    out["vllm_prompt_tokens_external_kv_transfer_total"] = float(out["vllm_prompt_tokens_external_kv_transfer_total"]) + value
                    source_prompt_total += value
                found = True
        if found:
            sources.append(url)
            scope = str(target.get("scope","local"))
            source_scopes.append(scope)
            if scope == "pipeline":
                pipeline_parallel_size = max(pipeline_parallel_size,int(target.get("pipeline_parallel_size",0) or 0))
                pipeline_node_rank = int(target.get("pipeline_node_rank",0) or 0)
    out["vllm_metrics_up"] = 1 if sources else 0
    out["vllm_requests_running_by_model"] = metric_map_text(running_by_model)
    out["vllm_requests_waiting_by_model"] = metric_map_text(waiting_by_model)
    out["vllm_prompt_tokens_total_by_model"] = metric_map_text(prompt_by_model)
    out["vllm_generation_tokens_total_by_model"] = metric_map_text(generation_by_model)
    out["vllm_requests_total"] = request_counter_total if request_counter_total > 0.0 else http_request_counter_total
    out["vllm_kv_cache_pct"] = round(max(kv_vals),2) if kv_vals else 0.0
    if float(out["vllm_prompt_tokens_total"]) <= 0.0 and source_prompt_total > 0.0:
        out["vllm_prompt_tokens_total"] = source_prompt_total
    source_cached_total = float(out["vllm_prompt_tokens_local_cache_hit_total"]) + float(out["vllm_prompt_tokens_external_kv_transfer_total"])
    if float(out["vllm_prompt_tokens_cached_total"]) <= 0.0 and source_cached_total > 0.0:
        out["vllm_prompt_tokens_cached_total"] = source_cached_total
    out["vllm_tokens_total"] = float(out["vllm_prompt_tokens_total"]) + float(out["vllm_generation_tokens_total"])
    def pct(num: float, den: float) -> float:
        return(round((100.0 * num / den),2) if den > 0.0 else 0.0)
    out["vllm_prompt_cache_hit_pct"] = pct(float(out["vllm_prompt_tokens_cached_total"]),float(out["vllm_prompt_tokens_total"]))
    out["vllm_prefix_cache_hit_pct"] = pct(float(out["vllm_prefix_cache_hits_total"]),float(out["vllm_prefix_cache_queries_total"]))
    out["vllm_external_prefix_cache_hit_pct"] = pct(float(out["vllm_external_prefix_cache_hits_total"]),float(out["vllm_external_prefix_cache_queries_total"]))
    if prev is not None and now is not None:
        elapsed = now - float(prev.get("unix_ts",0.0))
        if elapsed > 0.0:
            def delta(key: str) -> float:
                return(max(0.0,float(out.get(key,0.0)) - float(prev.get(key,0.0))))
            total_delta = delta("vllm_tokens_total")
            prompt_delta = delta("vllm_prompt_tokens_total") if "vllm_prompt_tokens_total" in prev else max(0.0,total_delta - delta("vllm_generation_tokens_total"))
            gen_delta = delta("vllm_generation_tokens_total")
            cached_delta = delta("vllm_prompt_tokens_cached_total")
            prefix_query_delta = delta("vllm_prefix_cache_queries_total")
            prefix_hit_delta = delta("vllm_prefix_cache_hits_total")
            external_query_delta = delta("vllm_external_prefix_cache_queries_total")
            external_hit_delta = delta("vllm_external_prefix_cache_hits_total")
            prev_prompt_by_model = parse_metric_map(prev.get("vllm_prompt_tokens_total_by_model",""))
            prev_generation_by_model = parse_metric_map(prev.get("vllm_generation_tokens_total_by_model",""))
            prompt_rate_by_model: Dict[str,float] = {}
            generation_rate_by_model: Dict[str,float] = {}
            for model,total in prompt_by_model.items():
                prompt_rate_by_model[model] = max(0.0,total - prev_prompt_by_model.get(model,total)) / elapsed
            for model,total in generation_by_model.items():
                generation_rate_by_model[model] = max(0.0,total - prev_generation_by_model.get(model,total)) / elapsed
            out["vllm_requests_per_s"] = round(delta("vllm_requests_total") / elapsed,3)
            out["vllm_tokens_per_s"] = round(total_delta / elapsed,3)
            out["vllm_prompt_tokens_per_s"] = round(prompt_delta / elapsed,3)
            out["vllm_generation_tokens_per_s"] = round(gen_delta / elapsed,3)
            out["vllm_prompt_tokens_per_s_by_model"] = metric_map_text(prompt_rate_by_model)
            out["vllm_generation_tokens_per_s_by_model"] = metric_map_text(generation_rate_by_model)
            out["vllm_prompt_tokens_cached_per_s"] = round(cached_delta / elapsed,3)
            out["vllm_prompt_tokens_local_compute_per_s"] = round(delta("vllm_prompt_tokens_local_compute_total") / elapsed,3)
            out["vllm_prompt_tokens_local_cache_hit_per_s"] = round(delta("vllm_prompt_tokens_local_cache_hit_total") / elapsed,3)
            out["vllm_prompt_tokens_external_kv_transfer_per_s"] = round(delta("vllm_prompt_tokens_external_kv_transfer_total") / elapsed,3)
            if prompt_delta > 0.0:
                out["vllm_prompt_cache_hit_pct"] = pct(cached_delta,prompt_delta)
            if prefix_query_delta > 0.0:
                out["vllm_prefix_cache_hit_pct"] = pct(prefix_hit_delta,prefix_query_delta)
            if external_query_delta > 0.0:
                out["vllm_external_prefix_cache_hit_pct"] = pct(external_hit_delta,external_query_delta)
    out["vllm_metrics_sources"] = ";".join(sources)[:240]
    out["vllm_metrics_scope"] = "pipeline" if "pipeline" in source_scopes else ("local" if sources else "")
    out["vllm_pipeline_parallel_size"] = pipeline_parallel_size
    out["vllm_pipeline_node_rank"] = pipeline_node_rank
    stage_models,stage_ranks = discover_vllm_pipeline_stage_models(timeout)
    out["vllm_pipeline_stage_models"] = ";".join(sorted(stage_models))[:480]
    out["vllm_pipeline_stage_pp_by_model"] = metric_map_text({model:float(pp) for model,pp in stage_models.items()})
    out["vllm_pipeline_stage_rank_by_model"] = metric_map_text({model:float(rank) for model,rank in stage_ranks.items()})
    return(out)


def poll_gpu_fields(fields: List[str], timeout: float) -> Tuple[List[Dict[str,str]],str]:
    query = telemetry.nvidia_smi_query(fields)
    try:
        p = subprocess.run(query.split(),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    except Exception as e:
        return([],"%s" % e)
    if p.returncode != 0:
        return([],p.stderr.strip() or ("nvidia-smi exited %d" % p.returncode))
    rows: List[Dict[str,str]] = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        rows.append(telemetry.parse_gpu_line(line,fields))
    return(rows,"")


def poll_gpus(timeout: float) -> Tuple[List[Dict[str,str]],str]:
    last_error = ""
    field_sets = [GPU_FIELDS,BASE_GPU_FIELDS + ["temperature.gpu"],BASE_GPU_FIELDS]
    for fields in field_sets:
        rows,error = poll_gpu_fields(fields,timeout)
        if error == "":
            return(rows,"")
        last_error = error
        if "timed out" in error.lower():
            break
    return([],last_error)


def write_summary(path: str, rows: Deque[Dict[str,object]], total_samples: int) -> None:
    system_rows = list(rows)
    good = [r for r in system_rows if not r.get("error")]
    latest = system_rows[-1] if system_rows else {}
    gpu_vals = [float(r.get("gpu_util_pct",0.0)) for r in good if int(r.get("gpu_index",-1)) >= 0]
    gpu_temps = [float(r.get("gpu_temp_c",0.0)) for r in good if int(r.get("gpu_index",-1)) >= 0 and float(r.get("gpu_temp_c",0.0)) > 0.0]
    hot = [v for v in gpu_vals if v >= 90.0]
    hot_temps = [v for v in gpu_temps if v >= 80.0]
    out = {
        "updated_unix": int(time.time()),
        "total_samples": total_samples,
        "window_samples": len(rows),
        "latest": latest,
        "window": {
            "cpu_util_pct": telemetry.stats(float(r.get("cpu_util_pct",0.0)) for r in system_rows),
            "mem_used_pct": telemetry.stats(float(r.get("mem_used_pct",0.0)) for r in system_rows),
            "thermal_max_c": telemetry.stats(float(r.get("thermal_max_c",0.0)) for r in system_rows if float(r.get("thermal_max_c",0.0)) > 0.0),
            "root_disk_used_pct": telemetry.stats(float(r.get("root_disk_used_pct",0.0)) for r in system_rows),
            "net_rx_mbps": telemetry.stats(float(r.get("net_rx_mbps",0.0)) for r in system_rows),
            "net_tx_mbps": telemetry.stats(float(r.get("net_tx_mbps",0.0)) for r in system_rows),
            "ds4_gateway_cpu_pending": telemetry.stats(float(r.get("ds4_gateway_cpu_pending",0.0)) for r in system_rows),
            "ds4_gateway_cpu_active": telemetry.stats(float(r.get("ds4_gateway_cpu_active",0.0)) for r in system_rows),
            "vllm_requests_running": telemetry.stats(float(r.get("vllm_requests_running",0.0)) for r in system_rows),
            "vllm_requests_waiting": telemetry.stats(float(r.get("vllm_requests_waiting",0.0)) for r in system_rows),
            "vllm_requests_per_s": telemetry.stats(float(r.get("vllm_requests_per_s",0.0)) for r in system_rows),
            "vllm_kv_cache_pct": telemetry.stats(float(r.get("vllm_kv_cache_pct",0.0)) for r in system_rows),
            "vllm_tokens_per_s": telemetry.stats(float(r.get("vllm_tokens_per_s",0.0)) for r in system_rows),
            "vllm_prompt_tokens_per_s": telemetry.stats(float(r.get("vllm_prompt_tokens_per_s",0.0)) for r in system_rows),
            "vllm_generation_tokens_per_s": telemetry.stats(float(r.get("vllm_generation_tokens_per_s",0.0)) for r in system_rows),
            "vllm_prompt_tokens_cached_per_s": telemetry.stats(float(r.get("vllm_prompt_tokens_cached_per_s",0.0)) for r in system_rows),
            "vllm_prompt_cache_hit_pct": telemetry.stats(float(r.get("vllm_prompt_cache_hit_pct",0.0)) for r in system_rows),
            "vllm_prefix_cache_hit_pct": telemetry.stats(float(r.get("vllm_prefix_cache_hit_pct",0.0)) for r in system_rows),
            "vllm_external_prefix_cache_hit_pct": telemetry.stats(float(r.get("vllm_external_prefix_cache_hit_pct",0.0)) for r in system_rows),
            "gpu_util_pct": telemetry.stats(gpu_vals),
            "gpu_temp_c": telemetry.stats(gpu_temps),
            "gpu_samples_ge_90": len(hot),
            "pct_gpu_samples_ge_90": round(100.0 * len(hot) / len(gpu_vals),2) if gpu_vals else 0.0,
            "gpu_temp_samples_ge_80": len(hot_temps),
            "pct_gpu_temp_samples_ge_80": round(100.0 * len(hot_temps) / len(gpu_temps),2) if gpu_temps else 0.0,
        },
    }
    telemetry.write_json_atomic(path,out)


def build_rows(args: argparse.Namespace, prev_cpu: Optional[Tuple[int,int]], prev_net: Optional[Tuple[int,int]], prev_ts: float, prev_vllm: Optional[Dict[str,object]] = None) -> Tuple[List[Dict[str,object]],Optional[Tuple[int,int]],Tuple[int,int],float,Optional[Dict[str,object]]]:
    now = time.time()
    iso = dt.datetime.fromtimestamp(now,dt.timezone.utc).isoformat()
    cur_cpu = read_cpu_times()
    load1,load5,load15 = read_loadavg()
    mem = read_meminfo()
    system,cur_net = read_system(prev_net,max(0.0,now - prev_ts) if prev_ts > 0.0 else 0.0)
    gateway = read_gateway(args.gateway_url,args.http_timeout)
    vllm = read_vllm_metrics(args.metrics_urls,args.http_timeout,prev_vllm,now)
    next_vllm = prev_vllm
    if int(vllm.get("vllm_metrics_up",0)) != 0:
        next_vllm = {
            "unix_ts": now,
            "vllm_requests_total": float(vllm.get("vllm_requests_total",0.0)),
            "vllm_tokens_total": float(vllm.get("vllm_tokens_total",0.0)),
            "vllm_prompt_tokens_total": float(vllm.get("vllm_prompt_tokens_total",0.0)),
            "vllm_generation_tokens_total": float(vllm.get("vllm_generation_tokens_total",0.0)),
            "vllm_prompt_tokens_total_by_model": str(vllm.get("vllm_prompt_tokens_total_by_model","")),
            "vllm_generation_tokens_total_by_model": str(vllm.get("vllm_generation_tokens_total_by_model","")),
            "vllm_prompt_tokens_local_compute_total": float(vllm.get("vllm_prompt_tokens_local_compute_total",0.0)),
            "vllm_prompt_tokens_local_cache_hit_total": float(vllm.get("vllm_prompt_tokens_local_cache_hit_total",0.0)),
            "vllm_prompt_tokens_external_kv_transfer_total": float(vllm.get("vllm_prompt_tokens_external_kv_transfer_total",0.0)),
            "vllm_prompt_tokens_cached_total": float(vllm.get("vllm_prompt_tokens_cached_total",0.0)),
            "vllm_prefix_cache_queries_total": float(vllm.get("vllm_prefix_cache_queries_total",0.0)),
            "vllm_prefix_cache_hits_total": float(vllm.get("vllm_prefix_cache_hits_total",0.0)),
            "vllm_external_prefix_cache_queries_total": float(vllm.get("vllm_external_prefix_cache_queries_total",0.0)),
            "vllm_external_prefix_cache_hits_total": float(vllm.get("vllm_external_prefix_cache_hits_total",0.0)),
        }
    gpus,error = poll_gpus(args.nvidia_smi_timeout)
    base: Dict[str,object] = {
        "unix_ts": int(now),
        "iso_ts": iso,
        "node": args.node,
        "hostname": socket.gethostname(),
        "cpu_util_pct": cpu_pct(prev_cpu,cur_cpu),
        "load1": load1,
        "load5": load5,
        "load15": load15,
        **mem,
        **system,
        **gateway,
        **vllm,
    }
    if len(gpus) == 0:
        row = dict(base)
        row.update({"gpu_index":-1,"gpu_name":"","gpu_util_pct":0.0,"gpu_mem_util_pct":0.0,"gpu_mem_used_mib":0.0,"gpu_mem_total_mib":0.0,"gpu_power_w":0.0,"gpu_power_raw_w":0.0,"gpu_power_limit_w":0.0,"gpu_power_known":0,"gpu_power_source":"","gpu_power_reason":"nvidia-smi-unavailable","gpu_temp_c":0.0,"gpu_fan_pct":0.0,"gpu_clock_sm_mhz":0.0,"gpu_clock_mem_mhz":0.0,"gpu_pstate":"","error":error})
        return([row],cur_cpu,cur_net,now,next_vllm)
    rows: List[Dict[str,object]] = []
    for gpu in gpus:
        gpu_util_pct = telemetry.num(gpu.get("utilization.gpu","0"))
        power = telemetry.gpu_power_status(telemetry.num(gpu.get("power.draw","0")),telemetry.num(gpu.get("power.limit","0")),gpu_util_pct)
        row = dict(base)
        row.update({
            "gpu_index": int(telemetry.num(gpu.get("index","-1"))),
            "gpu_name": gpu.get("name",""),
            "gpu_util_pct": gpu_util_pct,
            "gpu_mem_util_pct": telemetry.num(gpu.get("utilization.memory","0")),
            "gpu_mem_used_mib": telemetry.num(gpu.get("memory.used","0")),
            "gpu_mem_total_mib": telemetry.num(gpu.get("memory.total","0")),
            **power,
            "gpu_temp_c": telemetry.num(gpu.get("temperature.gpu","0")),
            "gpu_fan_pct": telemetry.num(gpu.get("fan.speed","0")),
            "gpu_clock_sm_mhz": telemetry.num(gpu.get("clocks.gr","0")),
            "gpu_clock_mem_mhz": telemetry.num(gpu.get("clocks.mem","0")),
            "gpu_pstate": gpu.get("pstate",""),
            "error": "",
        })
        rows.append(row)
    return(rows,cur_cpu,cur_net,now,next_vllm)


def csv_needs_header(path: str) -> bool:
    if not os.path.exists(path):
        return(True)
    try:
        with open(path,"r",encoding="utf-8",newline="") as f:
            first = f.readline().strip()
    except Exception:
        return(True)
    if first == "":
        return(True)
    if first == ",".join(CSV_FIELDS):
        return(False)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.replace(path,"%s.schema-%s.bak" % (path,stamp))
    return(True)


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    csv_path = telemetry.node_csv_path(args.out_dir)
    summary_path = telemetry.node_summary_path(args.out_dir)
    window: Deque[Dict[str,object]] = collections.deque(maxlen=max(1,args.summary_samples))
    prev_cpu = read_cpu_times()
    prev_net = read_netdev()
    prev_ts = time.time()
    prev_vllm: Optional[Dict[str,object]] = None
    total_samples = 0
    start = time.time()
    new_file = csv_needs_header(csv_path)
    with open(csv_path,"a",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        while True:
            rows,prev_cpu,prev_net,prev_ts,prev_vllm = build_rows(args,prev_cpu,prev_net,prev_ts,prev_vllm)
            for row in rows:
                writer.writerow(row)
                window.append(row)
                total_samples += 1
            f.flush()
            write_summary(summary_path,window,total_samples)
            if args.duration > 0.0 and (time.time() - start) >= args.duration:
                break
            time.sleep(args.interval)
    return(0)


if __name__ == "__main__":
    raise SystemExit(main())
