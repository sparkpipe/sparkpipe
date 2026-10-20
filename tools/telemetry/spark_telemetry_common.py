#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Tuple


SPARK_NODES = (
    "spark0",
    "spark1",
    "spark2",
    "spark3",
    "spark4",
    "spark5",
    "spark6",
    "spark7",
    "spark8",
    "spark9",
    "sparka",
    "sparkb",
    "sparkc",
    "sparkd",
    "sparke",
    "sparkf",
)
AUXILIARY_NODES = ("lenovo7","lenovo8","lenovo9")
SPARK_NODE_COUNT = len(SPARK_NODES)
DEFAULT_NODES = ",".join(SPARK_NODES + tuple("%s=%s" % (node,node) for node in AUXILIARY_NODES))
TELEMETRY_DIR = "/tmp/ds4_telemetry"
MAC_TELEMETRY_DIR = os.path.join(TELEMETRY_DIR,"mac")
NODE_TELEMETRY_CSV = "node_telemetry.csv"
NODE_TELEMETRY_SUMMARY = "node_telemetry.summary.json"
QUEUE_RATE_WINDOW_S = float(os.environ.get("DS4_QUEUE_RATE_WINDOW_S","1800"))
DEFAULT_DS4_API_URL = "http://10.20.0.10:8700"

BASE_GPU_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "pstate",
]

EXTRA_GPU_FIELDS = [
    "temperature.gpu",
    "fan.speed",
    "clocks.gr",
    "clocks.mem",
]

GPU_FIELDS = BASE_GPU_FIELDS + EXTRA_GPU_FIELDS

CSV_FIELDS = [
    "unix_ts",
    "iso_ts",
    "node",
    "hostname",
    "cpu_util_pct",
    "load1",
    "load5",
    "load15",
    "mem_total_mib",
    "mem_available_mib",
    "mem_used_mib",
    "mem_used_pct",
    "swap_total_mib",
    "swap_used_mib",
    "swap_used_pct",
    "thermal_avg_c",
    "thermal_max_c",
    "thermal_sources",
    "root_disk_total_gib",
    "root_disk_used_gib",
    "root_disk_available_gib",
    "root_disk_used_pct",
    "net_rx_bytes",
    "net_tx_bytes",
    "net_rx_mbps",
    "net_tx_mbps",
    "proc_count",
    "thread_count",
    "uptime_s",
    "ds4_gateway_up",
    "ds4_gateway_active",
    "ds4_gateway_idle_s",
    "ds4_gateway_current_model",
    "ds4_gateway_cpu_pending",
    "ds4_gateway_cpu_active",
    "ds4_gateway_cpu_completed",
    "ds4_gateway_cpu_failed",
    "vllm_metrics_up",
    "vllm_requests_running",
    "vllm_requests_waiting",
    "vllm_requests_running_by_model",
    "vllm_requests_waiting_by_model",
    "vllm_requests_total",
    "vllm_requests_per_s",
    "vllm_kv_cache_pct",
    "vllm_prompt_tokens_total",
    "vllm_generation_tokens_total",
    "vllm_prompt_tokens_total_by_model",
    "vllm_generation_tokens_total_by_model",
    "vllm_prompt_tokens_local_compute_total",
    "vllm_prompt_tokens_local_cache_hit_total",
    "vllm_prompt_tokens_external_kv_transfer_total",
    "vllm_prompt_tokens_cached_total",
    "vllm_prefix_cache_queries_total",
    "vllm_prefix_cache_hits_total",
    "vllm_external_prefix_cache_queries_total",
    "vllm_external_prefix_cache_hits_total",
    "vllm_tokens_total",
    "vllm_tokens_per_s",
    "vllm_prompt_tokens_per_s",
    "vllm_generation_tokens_per_s",
    "vllm_prompt_tokens_per_s_by_model",
    "vllm_generation_tokens_per_s_by_model",
    "vllm_prompt_tokens_cached_per_s",
    "vllm_prompt_tokens_local_compute_per_s",
    "vllm_prompt_tokens_local_cache_hit_per_s",
    "vllm_prompt_tokens_external_kv_transfer_per_s",
    "vllm_prompt_cache_hit_pct",
    "vllm_prefix_cache_hit_pct",
    "vllm_external_prefix_cache_hit_pct",
    "vllm_metrics_sources",
    "vllm_metrics_scope",
    "vllm_pipeline_parallel_size",
    "vllm_pipeline_node_rank",
    "vllm_pipeline_stage_models",
    "vllm_pipeline_stage_pp_by_model",
    "vllm_pipeline_stage_rank_by_model",
    "gpu_index",
    "gpu_name",
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "gpu_mem_used_mib",
    "gpu_mem_total_mib",
    "gpu_power_w",
    "gpu_power_raw_w",
    "gpu_power_limit_w",
    "gpu_power_known",
    "gpu_power_source",
    "gpu_power_reason",
    "gpu_temp_c",
    "gpu_fan_pct",
    "gpu_clock_sm_mhz",
    "gpu_clock_mem_mhz",
    "gpu_pstate",
    "error",
]


def parse_nodes(raw: str) -> List[str]:
    text = (raw or "").strip()
    if text == "" or text.lower() in ("all","8x","spark8","sparks"):
        return(list(SPARK_NODES))
    return([node.strip() for node in text.split(",") if node.strip()])


def parse_node_targets(raw: str) -> List[Tuple[str,str]]:
    out: List[Tuple[str,str]] = []
    for item in parse_nodes(raw):
        if "=" in item:
            label,target = item.split("=",1)
            label = label.strip()
            target = target.strip()
        else:
            label = item
            target = item
        if label and target:
            out.append((label,target))
    return(out)


def node_csv_path(root: str = TELEMETRY_DIR) -> str:
    return(os.path.join(root.rstrip("/"),NODE_TELEMETRY_CSV))


def node_summary_path(root: str = TELEMETRY_DIR) -> str:
    return(os.path.join(root.rstrip("/"),NODE_TELEMETRY_SUMMARY))


def utc_iso() -> str:
    return(dt.datetime.now(dt.timezone.utc).isoformat())


def write_text_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp,path)


def write_json_atomic(path: str, data: Dict[str,Any]) -> None:
    write_text_atomic(path,json.dumps(data,indent=2,sort_keys=True) + "\n")


def stats(vals: Iterable[float]) -> Dict[str,float]:
    data = list(vals)
    if len(data) == 0:
        return({"avg":0.0,"max":0.0,"min":0.0})
    return({"avg":round(sum(data) / len(data),2),"max":round(max(data),2),"min":round(min(data),2)})


def num(value: object) -> float:
    try:
        s = str(value).replace(" W","").replace(" %","").replace(" MHz","").strip()
        if s in ("","N/A","[N/A]","Not Supported"):
            return(0.0)
        return(float(s))
    except Exception:
        return(0.0)


def fnum(row: Dict[str,str], key: str) -> float:
    return(num(row.get(key,"0")))


def gpu_index(row: Dict[str,str]) -> int:
    try:
        return(int(float(row.get("gpu_index","-1") or -1)))
    except Exception:
        return(-1)


def parse_gpu_line(line: str, fields: List[str]) -> Dict[str,str]:
    vals = [value.strip() for value in line.split(",")]
    return({key:value for key,value in zip(fields,vals)})


def nvidia_smi_query(fields: List[str]) -> str:
    return("nvidia-smi --query-gpu=%s --format=csv,noheader,nounits" % ",".join(fields))


def gpu_power_status(raw_power_w: float, power_limit_w: float, gpu_util_pct: float) -> Dict[str,object]:
    source = "nvml.power.draw"
    if raw_power_w <= 0.0:
        return({"gpu_power_w":0.0,"gpu_power_raw_w":round(raw_power_w,2),"gpu_power_limit_w":round(power_limit_w,2),"gpu_power_known":0,"gpu_power_source":source,"gpu_power_reason":"nvml-power-missing"})
    if power_limit_w <= 0.0:
        return({"gpu_power_w":0.0,"gpu_power_raw_w":round(raw_power_w,2),"gpu_power_limit_w":0.0,"gpu_power_known":0,"gpu_power_source":source,"gpu_power_reason":"nvml-power-limit-unavailable"})
    if gpu_util_pct >= 90.0 and raw_power_w < 25.0:
        return({"gpu_power_w":0.0,"gpu_power_raw_w":round(raw_power_w,2),"gpu_power_limit_w":round(power_limit_w,2),"gpu_power_known":0,"gpu_power_source":source,"gpu_power_reason":"nvml-power-sanity-failed"})
    return({"gpu_power_w":round(raw_power_w,2),"gpu_power_raw_w":round(raw_power_w,2),"gpu_power_limit_w":round(power_limit_w,2),"gpu_power_known":1,"gpu_power_source":source,"gpu_power_reason":""})


def summary_base() -> Dict[str,object]:
    return({"updated_unix":int(time.time()),"updated_iso":utc_iso()})


def node_metric_map(raw: object) -> Dict[str,float]:
    out: Dict[str,float] = {}
    for item in str(raw or "").split(";"):
        if ":" not in item:
            continue
        key,value = item.split(":",1)
        key = key.strip()
        if key:
            out[key] = num(value)
    return(out)


def format_node_map(values: Dict[str,float]) -> str:
    parts: List[str] = []
    for key in sorted(values):
        value = values[key]
        text = str(int(value)) if float(value).is_integer() else ("%.3f" % value).rstrip("0").rstrip(".")
        parts.append("%s:%s" % (key,text))
    return(";".join(parts)[:240])


def format_text_map(values: Dict[str,str]) -> str:
    parts: List[str] = []
    for key in sorted(values):
        value = str(values[key]).replace(";",",").strip()
        if key and value:
            parts.append("%s:%s" % (key,value[:80]))
    return(";".join(parts)[:240])


def add_service_id(values: Dict[str,object], raw: Any) -> None:
    service_id = str(raw or "").strip()
    if service_id:
        values[service_id] = True


def add_lease_service_ids(values: Dict[str,object], raw: Any) -> None:
    if isinstance(raw,dict):
        for key in ("service_id","selected_service_id","ds4_service_id"):
            add_service_id(values,raw.get(key))
        for value in raw.values():
            if isinstance(value,(dict,list,tuple)):
                add_lease_service_ids(values,value)
    elif isinstance(raw,(list,tuple)):
        for item in raw:
            add_lease_service_ids(values,item)


def empty_queue_summary() -> Dict[str,object]:
    return({
        "local_queue_source": "",
        "local_queue_total": 0,
        "local_queue_depth": 0,
        "local_queue_queued": 0,
        "local_queue_running": 0,
        "local_queue_completed": 0,
        "local_queue_failed": 0,
        "local_queue_pending_by_service": "",
        "local_queue_resident_service_targets": "",
        "local_queue_ds_services": "",
        "local_queue_ds_service_count": 0,
        "local_queue_ds_model_count": 0,
        "local_queue_active_services": "",
        "local_queue_active_service_count": 0,
        "local_queue_prompt_tok_s": 0.0,
        "local_queue_completion_tok_s": 0.0,
        "local_queue_total_tok_s": 0.0,
        "local_queue_token_rate_window_s": 0.0,
        "local_queue_token_rate_completed": 0,
        "local_queue_last_service": "",
        "local_queue_resident_multimodel": 0,
        "local_queue_kv_shards": 0,
        "local_queue_kv_entries": 0,
        "local_queue_kv_bytes": 0,
        "local_queue_kv_services": "",
        "local_queue_kv_by_node": "",
        "local_queue_stage_service_by_node": "",
    })


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


def ds4_api_token_rates(base: str, newest_event_id: int, timeout: float, rate_window_s: float) -> Dict[str,object]:
    del newest_event_id
    window = max(1.0,float(rate_window_s))
    usage,error = read_json_url("%s/ds4/queue/usage?window_s=%.3f" % (base,window),timeout)
    if error != "" or str(usage.get("format","")) != "ds4-inference-queue-v1":
        return({})
    return({
        "local_queue_prompt_tok_s": num(usage.get("prompt_tok_s",0.0)),
        "local_queue_completion_tok_s": num(usage.get("completion_tok_s",0.0)),
        "local_queue_total_tok_s": num(usage.get("total_tok_s",0.0)),
        "local_queue_token_rate_window_s": num(usage.get("window_s",window)),
        "local_queue_token_rate_completed": int(num(usage.get("completed_count",0))),
    })


def ds4_api_queue_from_status(data: Dict[str,object], source: str, dispatcher: Dict[str,object] | None = None, models: Dict[str,object] | None = None, token_rates: Dict[str,object] | None = None) -> Dict[str,object]:
    out = empty_queue_summary()
    counts_raw = data.get("state_counts",{})
    if not isinstance(counts_raw,dict):
        return(out)
    dispatcher = dispatcher or {}
    models = models or {}
    token_rates = token_rates or {}
    counts = {str(k):int(num(v)) for k,v in counts_raw.items()}
    queued = int(counts.get("queued",0) + counts.get("prefilling",0) + counts.get("ready",0))
    running = int(counts.get("running",0))
    completed = int(counts.get("completed",0) + counts.get("completed_with_failures",0) + counts.get("completed_with_cancelled",0))
    failed = int(counts.get("failed",0))
    pending_by_service_raw = dispatcher.get("pending_by_service",{})
    pending_by_service = {str(k):num(v) for k,v in pending_by_service_raw.items()} if isinstance(pending_by_service_raw,dict) else {}
    resident_targets_raw = dispatcher.get("resident_service_targets",{})
    resident_targets = {str(k):num(v) for k,v in resident_targets_raw.items()} if isinstance(resident_targets_raw,dict) else {}
    service_ids: Dict[str,object] = {}
    active_service_ids: Dict[str,object] = {}
    model_count = 0
    for item in models.get("data",[]) if isinstance(models.get("data",[]),list) else []:
        if not isinstance(item,dict):
            continue
        model_count += 1
        service_id = str(item.get("ds4_service_id") or "")
        if service_id:
            service_ids[service_id] = True
    for service_id in list(pending_by_service) + list(resident_targets):
        if service_id:
            service_ids[service_id] = True
    for service_id,count in pending_by_service.items():
        if service_id and count > 0.0:
            active_service_ids[service_id] = True
    last_service = str(dispatcher.get("last_claimed_service_id") or "")
    if last_service:
        service_ids[last_service] = True
    pipeline = data.get("pipeline_status",{})
    kv_shards = pipeline.get("kv_shards",[]) if isinstance(pipeline,dict) else []
    stages = pipeline.get("stages",[]) if isinstance(pipeline,dict) else []
    if isinstance(pipeline,dict):
        add_service_id(active_service_ids,pipeline.get("service_id"))
        add_lease_service_ids(active_service_ids,pipeline.get("active_compute_leases",[]))
    add_lease_service_ids(active_service_ids,data.get("active_compute_leases",[]))
    kv_services: Dict[str,object] = {}
    kv_by_node: Dict[str,str] = {}
    stage_services: Dict[str,str] = {}
    kv_entries = 0
    kv_bytes = 0
    if isinstance(kv_shards,list):
        for shard in kv_shards:
            if not isinstance(shard,dict):
                continue
            service_id = str(shard.get("service_id") or "")
            node_id = str(shard.get("node_id") or "")
            if service_id:
                kv_services[service_id] = True
                service_ids[service_id] = True
                active_service_ids[service_id] = True
            if node_id and service_id:
                kv_by_node[node_id] = service_id
            kv_entries += int(num(shard.get("entries",0)))
            kv_bytes += int(num(shard.get("bytes",0)))
    if isinstance(stages,list):
        for stage in stages:
            if not isinstance(stage,dict):
                continue
            node_id = str(stage.get("node_id") or "")
            service_id = str(stage.get("service_id") or "")
            if not node_id:
                continue
            if service_id:
                stage_services[node_id] = service_id
                service_ids[service_id] = True
                active_service_ids[service_id] = True
                if node_id not in kv_by_node:
                    kv_by_node[node_id] = service_id
    out.update({
        "local_queue_source": source,
        "local_queue_api_up": 1,
        "local_queue_total": sum(counts.values()),
        "local_queue_depth": queued + running,
        "local_queue_queued": queued,
        "local_queue_running": running,
        "local_queue_completed": completed,
        "local_queue_failed": failed,
        "local_queue_pending_by_service": format_node_map(pending_by_service),
        "local_queue_resident_service_targets": format_node_map(resident_targets),
        "local_queue_ds_services": ",".join(sorted(service_ids)),
        "local_queue_ds_service_count": len(service_ids),
        "local_queue_ds_model_count": model_count,
        "local_queue_active_services": ",".join(sorted(active_service_ids)),
        "local_queue_active_service_count": len(active_service_ids),
        "local_queue_prompt_tok_s": num(token_rates.get("local_queue_prompt_tok_s",0.0)),
        "local_queue_completion_tok_s": num(token_rates.get("local_queue_completion_tok_s",0.0)),
        "local_queue_total_tok_s": num(token_rates.get("local_queue_total_tok_s",0.0)),
        "local_queue_token_rate_window_s": num(token_rates.get("local_queue_token_rate_window_s",0.0)),
        "local_queue_token_rate_completed": int(num(token_rates.get("local_queue_token_rate_completed",0))),
        "local_queue_last_service": last_service,
        "local_queue_resident_multimodel": 1 if bool(dispatcher.get("resident_multimodel",False)) else 0,
        "local_queue_kv_shards": len(kv_shards) if isinstance(kv_shards,list) else 0,
        "local_queue_kv_entries": kv_entries,
        "local_queue_kv_bytes": kv_bytes,
        "local_queue_kv_services": ",".join(sorted(kv_services)),
        "local_queue_kv_by_node": format_text_map(kv_by_node),
        "local_queue_stage_service_by_node": format_text_map(stage_services),
    })
    return(out)


def read_ds4_api_queue(raw_url: str, timeout: float, rate_window_s: float = QUEUE_RATE_WINDOW_S) -> Dict[str,object]:
    base = str(raw_url or "").strip().rstrip("/")
    if base == "":
        return(empty_queue_summary())
    data,error = read_json_url(base + "/ds4/queue/status",timeout)
    dispatcher,dispatcher_error = read_json_url(base + "/ds4/dispatcher/status",timeout)
    models,models_error = read_json_url(base + "/v1/models",timeout)
    if error != "" or dispatcher_error != "" or models_error != "":
        return(empty_queue_summary())
    if str(data.get("format","")) != "ds4-inference-queue-v1":
        return(empty_queue_summary())
    token_rates = ds4_api_token_rates(base,int(num(data.get("newest_event_id",0))),timeout,rate_window_s)
    return(ds4_api_queue_from_status(data,"ds4-api:%s" % base,dispatcher,models,token_rates))
