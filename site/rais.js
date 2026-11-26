(function () {
  var ko = document.documentElement.lang === "ko";
  var T = ko ? {
    loading: "RAIS 데이터를 불러오는 중…",
    err: "rais-data.json을 불러오지 못했습니다.",
    hwSummary: function (h) {
      var sysPart = (h.nvme_bw_gb_s_per_rank != null && h.ram_gib_per_rank != null)
        ? (" · NVMe " + h.nvme_bw_gb_s_per_rank + " GB/s · RAM " + h.ram_gib_per_rank + " GiB (OS " + h.os_reserve_gib_per_rank + " GiB)")
        : "";
      return h.ranks + "랭크 · 랭크당 메모리 " + h.mem_bw_gb_s_per_rank + " GB/s (합계 " + Math.round(h.ranks * h.mem_bw_gb_s_per_rank).toLocaleString("en-US") + " GB/s) · BF16/FP8/FP4 " + h.bf16_tflops_per_rank + "/" + h.fp8_tflops_per_rank + "/" + h.fp4_tflops_per_rank + " TFLOPS · 네트워크 " + h.net_bw_gb_s_per_rank + " GB/s (" + h.net_hop_us + " µs)" + sysPart;
    },
    statusLabels: {
      VALID: "인증 완료",
      INVALID: "무효 (게이트 실패)",
      UNVERIFIED: "부분 실측 · 전체 RAIS 스윕 대기",
      PENDING: "전체 RAIS 스윕 대기"
    },
    sweepBadge: function (m) {
      if (m.status === "VALID") return "인증 완료 (" + (m.measured_count || 12) + "/15)";
      if (m.status === "INVALID") return "무효 (" + ((m.failed_gates && m.failed_gates.length) ? m.failed_gates.join(", ") : "게이트 실패") + ")";
      if (m.sweep_state === "partial_multi") return "부분 실측 " + m.measured_count + "/15 · 전체 RAIS 스윕 대기";
      if (m.sweep_state === "partial_qual") return "릴리스 검증 " + m.measured_count + "/15 · 전체 RAIS 스윕 대기";
      return "전체 RAIS 스윕 대기";
    },
    kpiHw: "선택한 하드웨어",
    kpiCoverage: "인증된 플릿 모델",
    kpiRais: "플릿 RAIS (인증 모델)",
    kpiRaisPending: "부분 RAIS (미인증)",
    kpiRoi: "최우선 최적화 여지",
    pendingShort: "스윕 대기",
    pendingCell: "–",
    noMeasured: "이 하드웨어 프로필의 실측 데이터가 아직 없습니다. 물리적 루프라인 하한만 표시합니다.",
    selectModel: "행의 모델을 클릭하면 5개 축과 15개 세부 지표, 3대 루프라인 및 시스템 물리 하한을 볼 수 있습니다.",
    detailTitle: function (m, hw) { return m.model + " · " + hw.name; },
    detailSub: function (m) {
      var r = m.receipt ? ("근거: " + m.receipt) : "전체 RAIS 스윕 대기 중입니다.";
      return "티어 " + m.tier + " (" + m.fleet_weight_pct + "%) · " + m.dtype + " · 활성 " + m.active_params_b + "B · 컨텍스트 " + fmtCtx(m.context_tokens) + " · " + r;
    },
    pillars: [
      ["P1", "단일 스트림 디코드 (22.5%)"],
      ["P2", "추측 디코딩 (22.5%)"],
      ["P3", "배치·동시성 확장 (22.5%)"],
      ["P4", "프리필·멀티턴 에이전트 (22.5%)"],
      ["P5", "운영·시스템 물리 (10%)"]
    ],
    rooflineHead: "물리적 하드웨어 루프라인 (B1 짧은 컨텍스트 · 8K 콜드 프리필)",
    b1Floor: "B1 루프라인 하한",
    pf8kFloor: "8K 프리필 하한",
    overlapMax: "최대 중첩 이득",
    measuredB1: "실측 B1",
    measured8k: "실측 8K 프리필",
    measuredB8: "실측 B8 합계",
    measuredWarm: "실측 웜 TTFT",
    subMeasured: "실측됨",
    subPending: "스윕 대기",
    bottlenecks: {
      memory: "메모리 대역폭",
      network: "네트워크 집합통신",
      compute: "연산",
      speculation: "추측 디코딩",
      jitter: "지터",
      capacity_cap: "동시성 상한",
      storage: "NVMe 부팅",
      memory_footprint: "KV 메모리 점유",
      determinism: "배치 비트 일치",
      unknown: "–"
    },
    submetricLabels: {
      p1a_short_b1: "1a 짧은 컨텍스트 B1 (C=256)",
      p1b_long_b1: "1b 긴 컨텍스트 B1 (64K–128K)",
      p1c_tail_jitter: "1c 꼬리 지터 (p50/p99 ITL)",
      p2a_spec_acceptance: "2a 추측 수락률 (α vs 4.0)",
      p2b_verify_b1: "2b-i B1 검증·드래프트 효율 (1/V)",
      p2b_verify_b8: "2b-ii B8 배치 검증 효율 (1/V)",
      p3a_interactive_batch: "3a 대화형 배치 (B=4,8,16 + SLA)",
      p3b_high_concurrency: "3b 고동시성 포화 (B=32,64)",
      p3c_long_ctx_batch: "3c 긴 컨텍스트 배치 (B=8 @ 32K–64K)",
      p4a_cold_prefill_8k: "4a 콜드 프리필 (8K)",
      p4b_long_cold_prefill: "4b 긴 콜드 프리필 (32K/128K+)",
      p4c_multiturn_prefix_ttft: "4c 멀티턴 프리픽스·상태 재사용 TTFT",
      p5a_cold_boot: "5a 콜드 부트·스왑 효율 (NVMe 하한)",
      p5b_memory_efficiency: "5b 가용 KV 메모리 효율",
      p5c_batch_invariance: "5c 배치 크기 비트 불변성"
    },
    noRoi: "이 하드웨어 프로필에 등록된 실측 결과가 아직 없어 ROI 순위가 비어 있습니다.",
    modelsShown: function (n, total) { return total + "개 모델 중 " + n + "개 표시"; }
  } : {
    loading: "Loading RAIS scorecard data…",
    err: "Failed to load rais-data.json.",
    hwSummary: function (h) {
      var sysPart = (h.nvme_bw_gb_s_per_rank != null && h.ram_gib_per_rank != null)
        ? (" · NVMe " + h.nvme_bw_gb_s_per_rank + " GB/s · RAM " + h.ram_gib_per_rank + " GiB (OS " + h.os_reserve_gib_per_rank + " GiB)")
        : "";
      return h.ranks + " rank" + (h.ranks === 1 ? "" : "s") + " · " + h.mem_bw_gb_s_per_rank + " GB/s/rank (" + Math.round(h.ranks * h.mem_bw_gb_s_per_rank).toLocaleString("en-US") + " GB/s agg) · BF16/FP8/FP4 " + h.bf16_tflops_per_rank + "/" + h.fp8_tflops_per_rank + "/" + h.fp4_tflops_per_rank + " TFLOPS · Net " + h.net_bw_gb_s_per_rank + " GB/s (" + h.net_hop_us + " µs hop)" + sysPart;
    },
    statusLabels: {
      VALID: "Certified VALID",
      INVALID: "INVALID (gate failed)",
      UNVERIFIED: "Partial receipts · Pending full RAIS sweep",
      PENDING: "Pending full RAIS sweep"
    },
    sweepBadge: function (m) {
      if (m.status === "VALID") return "VALID (" + (m.measured_count || 12) + "/15)";
      if (m.status === "INVALID") return "INVALID (" + ((m.failed_gates && m.failed_gates.length) ? m.failed_gates.join(", ") : "gate failed") + ")";
      if (m.sweep_state === "partial_multi") return "Partial " + m.measured_count + "/15 · Pending full RAIS sweep";
      if (m.sweep_state === "partial_qual") return "Qual " + m.measured_count + "/15 · Pending full RAIS sweep";
      return "Pending full RAIS sweep";
    },
    kpiHw: "Selected hardware",
    kpiCoverage: "Certified fleet models",
    kpiRais: "Valid Fleet RAIS",
    kpiRaisPending: "Partial RAIS (unverified)",
    kpiRoi: "Top ROI headroom",
    pendingShort: "Pending sweep",
    pendingCell: "–",
    noMeasured: "No committed receipts on this hardware profile yet. Exact physical rooflines are shown for all 19 fleet models.",
    selectModel: "Select any model row to inspect its 5 pillars, 15 sub-metrics, 3-roofline breakdown, and systems physics floors.",
    detailTitle: function (m, hw) { return m.model + " · " + hw.name; },
    detailSub: function (m) {
      var r = m.receipt ? ("Run: " + m.receipt) : "Pending full RAIS sweep on this profile.";
      return "Tier " + m.tier + " (" + m.fleet_weight_pct + "% fleet wt) · " + m.dtype + " · " + m.active_params_b + "B active · " + fmtCtx(m.context_tokens) + " ctx · " + r;
    },
    pillars: [
      ["P1", "Single-Stream Decode (22.5%)"],
      ["P2", "Speculation (22.5%)"],
      ["P3", "Batch & Concurrency (22.5%)"],
      ["P4", "Prefill & Multi-Turn (22.5%)"],
      ["P5", "Operational & Systems (10%)"]
    ],
    rooflineHead: "Physical hardware rooflines (B1 short ctx · 8K cold prefill)",
    b1Floor: "B1 roofline floor",
    pf8kFloor: "8K prefill floor",
    overlapMax: "Max overlap gain",
    measuredB1: "Measured B1",
    measured8k: "Measured 8K prefill",
    measuredB8: "Measured B8 agg",
    measuredWarm: "Measured warm TTFT",
    subMeasured: "Measured",
    subPending: "Pending sweep",
    bottlenecks: {
      memory: "memory",
      network: "network",
      compute: "compute",
      speculation: "speculation",
      jitter: "jitter",
      capacity_cap: "capacity_cap",
      storage: "storage",
      memory_footprint: "memory_footprint",
      determinism: "determinism",
      unknown: "–"
    },
    submetricLabels: {
      p1a_short_b1: "1a Short-ctx B1 (C=256)",
      p1b_long_b1: "1b Long-ctx B1 (64K–128K)",
      p1c_tail_jitter: "1c Tail Jitter (p50/p99 ITL)",
      p2a_spec_acceptance: "2a Spec Acceptance (α vs 4.0)",
      p2b_verify_b1: "2b-i B1 Verify & Draft Eff (1/V)",
      p2b_verify_b8: "2b-ii B8 Batched Verify Eff (1/V)",
      p3a_interactive_batch: "3a Interactive Batch (B=4,8,16 + SLA)",
      p3b_high_concurrency: "3b High-Concurrency (B=32,64)",
      p3c_long_ctx_batch: "3c Long-Ctx Batch (B=8 @ 32K–64K)",
      p4a_cold_prefill_8k: "4a Cold Prefill (8K)",
      p4b_long_cold_prefill: "4b Long Cold Prefill (32K/128K+)",
      p4c_multiturn_prefix_ttft: "4c Multi-Turn Prefix Reuse TTFT",
      p5a_cold_boot: "5a Cold Boot / Model Swap Eff",
      p5b_memory_efficiency: "5b Usable KV Memory Eff",
      p5c_batch_invariance: "5c Batch-Size Bit-Invariance"
    },
    noRoi: "No measured runs on this profile yet.",
    modelsShown: function (n, total) { return "Showing " + n + " of " + total + " fleet models"; }
  };

  function el(id) { return document.getElementById(id); }
  function mk(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function fmtCtx(c) {
    if (c >= 1048576) return Math.round(c / 1048576) + "M";
    return Math.round(c / 1024) + "K";
  }
  function fmtScore(v) {
    return (v == null || isNaN(v)) ? T.pendingCell : Number(v).toFixed(1);
  }
  function fmtTok(v) {
    if (v == null || isNaN(v)) return T.pendingCell;
    return v >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v).toFixed(1);
  }
  function bnLabel(code) {
    return T.bottlenecks[code] || code || "–";
  }

  var DATA = null;
  var state = { hw: "16x-gb10-1x100g", tier: "all", filter: "all", selected: "kimi-k3.1-pro", histView: "fleet", histModel: "kimi-k3.1-pro", histIdx: -1 };

  function currentProfile() {
    if (!DATA) return null;
    if (state.hw === DATA.hardware.name) {
      return {
        hardware: DATA.hardware,
        fleet: DATA.fleet,
        models: DATA.models,
        roi_top: DATA.roi_top
      };
    }
    return (DATA.profiles && DATA.profiles[state.hw]) || null;
  }

  function filteredModels(prof) {
    return prof.models.filter(function (m) {
      if (state.tier !== "all" && String(m.tier) !== state.tier) return false;
      if (state.filter === "multi" && m.status !== "VALID" && m.sweep_state !== "partial_multi") return false;
      if (state.filter === "measured" && m.sweep_state === "pending") return false;
      if (state.filter === "pending" && m.sweep_state !== "pending") return false;
      return true;
    });
  }

  function renderSummary(prof) {
    var hwEl = el("rais-hw-desc");
    if (hwEl) hwEl.textContent = T.hwSummary(prof.hardware);
    var kpi = el("rais-kpi");
    if (!kpi) return;
    kpi.textContent = "";
    var measuredModels = prof.models.filter(function (m) { return m.sweep_state !== "pending"; });
    var validModels = prof.models.filter(function (m) { return m.status === "VALID"; });
    var topRoi = prof.roi_top && prof.roi_top.length ? prof.roi_top[0] : null;
    var hasValid = validModels.length > 0 && prof.fleet.rais_covered != null;
    var covVal = hasValid ? (validModels.length + " / " + prof.models.length + " VALID") : (measuredModels.length + " / " + prof.models.length);
    var covSub = hasValid ? ((prof.fleet.valid_weight_pct != null ? prof.fleet.valid_weight_pct.toFixed(1) : prof.fleet.coverage_pct.toFixed(1)) + (ko ? "% 플릿 비중 인증 · " + measuredModels.length + "/" + prof.models.length + " 실측" : "% fleet weight certified · " + measuredModels.length + "/" + prof.models.length + " swept")) : (T.statusLabels[prof.fleet.status] || prof.fleet.status);
    var raisLbl = hasValid ? T.kpiRais : T.kpiRaisPending;
    var raisNum = hasValid ? (prof.fleet.rais_covered.toFixed(1) + " / 100") : (prof.fleet.rais_raw_covered != null ? prof.fleet.rais_raw_covered.toFixed(1) + " / 100" : T.pendingCell);
    var raisSub = hasValid ? ((ko ? "엄격 플릿 " : "Strict fleet ") + (prof.fleet.rais_strict != null ? prof.fleet.rais_strict.toFixed(1) : "–") + (ko ? " · 19종 원시 " : " · 19-model raw ") + (prof.fleet.rais_raw_covered != null ? prof.fleet.rais_raw_covered.toFixed(1) : "–")) : (prof.fleet.coverage_pct > 0 ? (prof.fleet.coverage_pct.toFixed(1) + "% weight covered") : T.pendingShort);
    var stackBadge = prof.fleet && prof.fleet.stack_badge;
    var items = [
      [T.kpiHw, prof.hardware.name, prof.hardware.description, stackBadge],
      [T.kpiCoverage, covVal, covSub, null],
      [raisLbl, raisNum, raisSub, null],
      [T.kpiRoi, topRoi ? ("+" + topRoi.fleet_headroom_pts.toFixed(2) + " pts") : T.pendingCell,
        topRoi ? (topRoi.model + " · " + (T.submetricLabels[topRoi.submetric] || topRoi.label)) : T.pendingShort, null]
    ];
    items.forEach(function (it) {
      var box = mk("div");
      box.appendChild(mk("p", "label", it[0]));
      box.appendChild(mk("strong", "num", it[1]));
      if (it[3]) {
        box.appendChild(mk("span", "rais-stack-badge", it[3]));
      }
      box.appendChild(mk("span", "", it[2]));
      kpi.appendChild(box);
    });
  }

  function renderDetail(prof) {
    var box = el("rais-detail");
    if (!box) return;
    box.textContent = "";
    var model = prof.models.find(function (m) { return m.model === state.selected; }) || prof.models[0];
    if (!model) return;
    state.selected = model.model;

    var head = mk("div", "rais-detail-head");
    var left = mk("div");
    left.appendChild(mk("h3", "mono", T.detailTitle(model, prof.hardware)));
    left.appendChild(mk("p", "rais-detail-sub", T.detailSub(model)));
    head.appendChild(left);
    var rightBadges = mk("div", "rais-badge-group");
    var mBadge = model.stack_badge || (prof.fleet && prof.fleet.stack_badge);
    if (mBadge) {
      rightBadges.appendChild(mk("span", "rais-stack-badge", mBadge));
    }
    var badgeCls = model.status === "VALID" ? "rais-badge valid" : (model.sweep_state === "partial_multi" ? "rais-badge partial" : "rais-badge pending");
    rightBadges.appendChild(mk("span", badgeCls, T.sweepBadge(model)));
    head.appendChild(rightBadges);
    box.appendChild(head);

    var rfGrid = mk("div", "rais-rf-grid");
    var rf = model.rooflines;
    var raw = model.raw_metrics || {};
    var rfCards = [
      [T.b1Floor, fmtTok(rf.b1_floor_tok_s) + " tok/s", bnLabel(rf.b1_bottleneck) + " · Tc=" + rf.b1_t_c_ms + " Tm=" + rf.b1_t_m_ms + " Tn=" + rf.b1_t_n_ms + " ms"],
      [T.pf8kFloor, fmtTok(rf.pf8k_floor_tok_s) + " tok/s", bnLabel(rf.pf8k_bottleneck) + " · " + T.overlapMax + " " + rf.pf8k_overlap_max + "×"],
      [T.measuredB1, raw.b1_short_tok_s != null ? (fmtTok(raw.b1_short_tok_s) + " tok/s") : T.pendingCell,
        raw.b8_tok_s != null ? (T.measuredB8 + ": " + fmtTok(raw.b8_tok_s) + " tok/s") : T.pendingShort],
      [T.measured8k, raw.prefill_8k_tok_s != null ? (fmtTok(raw.prefill_8k_tok_s) + " tok/s") : (raw.prefill_32k_tok_s != null ? (fmtTok(raw.prefill_32k_tok_s) + " tok/s (33K)") : T.pendingCell),
        raw.warm_ttft_ms != null ? (T.measuredWarm + ": " + Math.round(raw.warm_ttft_ms) + " ms") : T.pendingShort]
    ];
    rfCards.forEach(function (c) {
      var d = mk("div", "rais-rf-card");
      d.appendChild(mk("span", "label", c[0]));
      d.appendChild(mk("strong", "num", c[1]));
      d.appendChild(mk("small", "mono", c[2]));
      rfGrid.appendChild(d);
    });
    box.appendChild(rfGrid);

    var pBars = mk("div", "rais-pillars-grid");
    T.pillars.forEach(function (p, idx) {
      var val = model.pillars && model.pillars[idx];
      var card = mk("div", "rais-pillar-card");
      var top = mk("div", "rais-pillar-top");
      top.appendChild(mk("span", "label", p[0] + " · " + p[1]));
      top.appendChild(mk("strong", "num", val != null ? val.toFixed(1) : T.pendingCell));
      card.appendChild(top);
      var track = mk("div", "rais-bar-track");
      var fill = mk("i", "rais-bar-fill" + (val != null && val > 0 ? " on" : ""));
      fill.style.width = (val != null ? Math.max(0, Math.min(100, val)) : 0) + "%";
      track.appendChild(fill);
      card.appendChild(track);
      pBars.appendChild(card);
    });
    box.appendChild(pBars);

    var subList = mk("div", "rais-sub-grid");
    var measuredSet = {};
    (model.measured_submetrics || []).forEach(function (k) { measuredSet[k] = true; });
    (DATA.submetrics_meta || []).forEach(function (meta) {
      var k = meta.key;
      var hasMeas = !!measuredSet[k];
      var score = model.submetric_scores ? model.submetric_scores[k] : null;
      var bn = (model.submetric_bottlenecks && model.submetric_bottlenecks[k]) || "";
      var row = mk("div", "rais-sub-row" + (hasMeas ? " measured" : " unmeasured"));
      row.appendChild(mk("span", "rais-sub-name", T.submetricLabels[k] || meta.label));
      row.appendChild(mk("span", "mono rais-sub-wt", meta.weight_pct + "%"));
      row.appendChild(mk("span", "mono rais-sub-bn", bn ? bnLabel(bn) : T.pendingShort));
      row.appendChild(mk("strong", "num rais-sub-val", hasMeas && score != null ? score.toFixed(1) : T.subPending));
      subList.appendChild(row);
    });
    box.appendChild(subList);
  }

  function renderTable(prof) {
    var tbody = el("rais-rows");
    var countEl = el("rais-count");
    if (!tbody) return;
    tbody.textContent = "";
    var list = filteredModels(prof);
    if (countEl) countEl.textContent = T.modelsShown(list.length, prof.models.length);

    if (!list.length) {
      var trEmpty = mk("tr");
      var tdEmpty = mk("td", "muted", T.noMeasured);
      tdEmpty.colSpan = 10;
      trEmpty.appendChild(tdEmpty);
      tbody.appendChild(trEmpty);
      return;
    }

    list.forEach(function (m) {
      var tr = mk("tr", "rais-row" + (m.model === state.selected ? " active" : ""));
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-pressed", m.model === state.selected ? "true" : "false");

      var tdModel = mk("td");
      var nameWrap = mk("div", "rais-model-cell");
      nameWrap.appendChild(mk("b", "mono", m.model));
      nameWrap.appendChild(mk("span", "rais-tier-pill t" + m.tier, "T" + m.tier + " · " + m.fleet_weight_pct + "%"));
      var mStack = m.stack_badge || (prof.fleet && prof.fleet.stack_badge);
      if (mStack && m.sweep_state !== "pending") {
        nameWrap.appendChild(mk("span", "rais-stack-badge compact", mStack));
      }
      tdModel.appendChild(nameWrap);
      tdModel.appendChild(mk("div", "rais-model-meta mono", m.dtype + " · " + m.active_params_b + "B · " + fmtCtx(m.context_tokens)));
      tr.appendChild(tdModel);

      var tdStatus = mk("td");
      var badgeCls = m.status === "VALID" ? "rais-badge valid" : ((m.status === "INVALID" || m.sweep_state === "partial_multi") ? "rais-badge partial" : "rais-badge pending");
      tdStatus.appendChild(mk("span", badgeCls, T.sweepBadge(m)));
      tr.appendChild(tdStatus);

      var hasScore = m.sweep_state !== "pending" && m.raw_rais != null;
      tr.appendChild(mk("td", "num rais-score-cell", hasScore ? (m.status === "INVALID" ? (fmtScore(m.raw_rais) + "*") : fmtScore(m.raw_rais)) : T.pendingCell));
      tr.appendChild(mk("td", "num", hasScore ? fmtScore(m.base_r) : T.pendingCell));
      tr.appendChild(mk("td", "num", hasScore ? fmtScore(m.pillars[0]) : T.pendingCell));

      var p2Text = T.pendingCell;
      var p2HasSpec = hasScore && m.spec && m.spec.alpha != null && m.spec.alpha > 1.001;
      if (p2HasSpec) {
        p2Text = fmtScore(m.pillars[1]) + " (" + m.spec.alpha.toFixed(2) + " / " + m.spec.inv_v_b1.toFixed(2) + ")";
      } else if (hasScore) {
        p2Text = T.pendingShort;
      }
      tr.appendChild(mk("td", p2HasSpec ? "num" : "num muted", p2Text));
      var hasP3 = hasScore && (m.measured_submetrics || []).some(function (k) { return k.indexOf("p3") === 0; });
      tr.appendChild(mk("td", "num", hasP3 ? fmtScore(m.pillars[2]) : T.pendingCell));
      tr.appendChild(mk("td", "num", hasScore ? fmtScore(m.pillars[3]) : T.pendingCell));
      var hasP5 = hasScore && (m.measured_submetrics || []).some(function (k) { return k.indexOf("p5") === 0; });
      tr.appendChild(mk("td", hasP5 ? "num" : "num muted", hasScore ? fmtScore(m.pillars[4]) : T.pendingCell));

      var tdRf = mk("td", "num mono rais-rf-cell");
      tdRf.textContent = fmtTok(m.rooflines.b1_floor_tok_s) + " / " + fmtTok(m.rooflines.pf8k_floor_tok_s);
      tr.appendChild(tdRf);

      function pick() {
        state.selected = m.model;
        renderTable(prof);
        renderDetail(prof);
      }
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
      });
      tbody.appendChild(tr);
    });
  }

  function renderRoi(prof) {
    var tbody = el("rais-roi-rows");
    if (!tbody) return;
    tbody.textContent = "";
    var list = (prof.roi_top || []).slice(0, 10);
    if (!list.length) {
      var tr0 = mk("tr");
      var td0 = mk("td", "muted", T.noRoi);
      td0.colSpan = 6;
      tr0.appendChild(td0);
      tbody.appendChild(tr0);
      return;
    }
    list.forEach(function (item, idx) {
      var tr = mk("tr");
      tr.appendChild(mk("td", "num mono", String(idx + 1)));
      tr.appendChild(mk("td", "mono", item.model));
      tr.appendChild(mk("td", "", T.submetricLabels[item.submetric] || item.label));
      tr.appendChild(mk("td", "num", item.current_score.toFixed(1)));
      tr.appendChild(mk("td", "num signal", "+" + item.fleet_headroom_pts.toFixed(2)));
      tr.appendChild(mk("td", "mono", bnLabel(item.bottleneck)));
      tbody.appendChild(tr);
    });
  }

  function historySeries(snaps) {
    var view = state.histView;
    if (view === "fleet") {
      return [
        { key: "rais", label: ko ? "플릿 RAIS (전체)" : "Fleet RAIS (19-model)", color: "#f97316", width: 3.0, get: function (s) { return s.fleet.rais_unpenalized; } },
        { key: "valid", label: ko ? "인증 가중 RAIS" : "Valid-Weighted RAIS", color: "#4ade80", width: 2.2, dash: "5 4", get: function (s) { return s.fleet.rais_valid_weighted; } },
        { key: "base_r", label: ko ? "기본 루프라인 R" : "Base Roofline R", color: "#e2e8f0", width: 2.0, get: function (s) { return s.fleet.base_r; } },
        { key: "P1", label: "P1 Decode", color: "#38bdf8", width: 1.8, get: function (s) { return s.fleet.pillars.P1; } },
        { key: "P2", label: "P2 Spec", color: "#c084fc", width: 1.8, get: function (s) { return s.fleet.pillars.P2; } },
        { key: "P3", label: "P3 Batch", color: "#fbbf24", width: 1.8, get: function (s) { return s.fleet.pillars.P3; } },
        { key: "P4", label: "P4 Prefill", color: "#fb7185", width: 1.8, get: function (s) { return s.fleet.pillars.P4; } },
        { key: "P5", label: "P5 Systems", color: "#2dd4bf", width: 1.8, get: function (s) { return s.fleet.pillars.P5 != null ? s.fleet.pillars.P5 : 0; } }
      ];
    }
    if (view === "model") {
      var mn = state.histModel;
      return [
        { key: "rais", label: mn + " RAIS", color: "#f97316", width: 3.0, get: function (s) { var m = s.models[mn]; return m ? m.raw_rais : null; } },
        { key: "base_r", label: "Base R", color: "#e2e8f0", width: 2.0, get: function (s) { var m = s.models[mn]; return m ? m.base_r : null; } },
        { key: "P1", label: "P1 Decode", color: "#38bdf8", width: 1.8, get: function (s) { var m = s.models[mn]; return m ? m.pillars.P1 : null; } },
        { key: "P2", label: "P2 Spec", color: "#c084fc", width: 1.8, get: function (s) { var m = s.models[mn]; return m ? m.pillars.P2 : null; } },
        { key: "P3", label: "P3 Batch", color: "#fbbf24", width: 1.8, get: function (s) { var m = s.models[mn]; return m ? m.pillars.P3 : null; } },
        { key: "P4", label: "P4 Prefill", color: "#fb7185", width: 1.8, get: function (s) { var m = s.models[mn]; return m ? m.pillars.P4 : null; } },
        { key: "P5", label: "P5 Systems", color: "#2dd4bf", width: 1.8, get: function (s) { var m = s.models[mn]; return m && m.pillars.P5 != null ? m.pillars.P5 : null; } }
      ];
    }
    var groupMap = {
      p1: [["p1a_short_b1", "#38bdf8"], ["p1b_long_b1", "#0ea5e9"], ["p1c_tail_jitter", "#a7f3d0"]],
      p2: [["p2a_spec_acceptance", "#c084fc"], ["p2b_verify_b1", "#f97316"], ["p2b_verify_b8", "#fbbf24"]],
      p3: [["p3a_interactive_batch", "#fbbf24"], ["p3b_high_concurrency", "#f97316"], ["p3c_long_ctx_batch", "#38bdf8"]],
      p4: [["p4a_cold_prefill_8k", "#fb7185"], ["p4b_long_cold_prefill", "#f97316"], ["p4c_multiturn_prefix_ttft", "#4ade80"]],
      p5: [["p5a_cold_boot", "#2dd4bf"], ["p5b_memory_efficiency", "#38bdf8"], ["p5c_batch_invariance", "#a7f3d0"]]
    };
    return (groupMap[view] || []).map(function (pair) {
      return {
        key: pair[0],
        label: T.submetricLabels[pair[0]] || pair[0],
        color: pair[1],
        width: 2.4,
        get: function (s) { return s.fleet.submetrics ? s.fleet.submetrics[pair[0]] : null; }
      };
    });
  }

  function fmtDelta(a, b) {
    if (a == null || b == null) return { text: "–", cls: "flat" };
    var d = a - b;
    if (Math.abs(d) < 0.05) return { text: "0.0", cls: "flat" };
    return { text: (d > 0 ? "+" : "") + d.toFixed(2), cls: d > 0 ? "up" : "down" };
  }

  function renderHistory() {
    var chartWrap = el("rais-hist-chart");
    var legWrap = el("rais-hist-legend");
    var snapEl = el("rais-hist-snapshot");
    var subGrid = el("rais-hist-subgrid");
    var rowsEl = el("rais-hist-rows");
    var modelWrap = el("rais-hist-model-wrap");
    var modelSel = el("rais-hist-model");
    if (!chartWrap || !DATA || !DATA.history || !DATA.history.length) return;

    var snaps = DATA.history;
    if (state.histIdx < 0 || state.histIdx >= snaps.length) state.histIdx = snaps.length - 1;

    if (modelWrap) modelWrap.style.display = state.histView === "model" ? "" : "none";
    if (modelSel && !modelSel.options.length && DATA.models) {
      DATA.models.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m.model;
        opt.textContent = m.model + " (" + m.fleet_weight_pct + "%)";
        if (m.model === state.histModel) opt.selected = true;
        modelSel.appendChild(opt);
      });
    }

    var series = historySeries(snaps);
    if (legWrap) {
      legWrap.textContent = "";
      series.forEach(function (sr) {
        var sp = mk("span");
        var dot = mk("i", "rais-hist-dot");
        dot.style.background = sr.color;
        sp.appendChild(dot);
        var lastVal = sr.get(snaps[state.histIdx]);
        sp.appendChild(document.createTextNode(sr.label + (lastVal != null ? (": " + lastVal.toFixed(1)) : "")));
        legWrap.appendChild(sp);
      });
    }

    var W = 820, H = 260, padL = 44, padR = 26, padT = 18, padB = 42;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var vals = [];
    series.forEach(function (sr) {
      snaps.forEach(function (s) {
        var v = sr.get(s);
        if (v != null && !isNaN(v)) vals.push(v);
      });
    });
    var minV = vals.length ? Math.floor(Math.max(0, Math.min.apply(null, vals) - 5) / 5) * 5 : 0;
    var maxV = vals.length ? Math.ceil(Math.min(100, Math.max.apply(null, vals) + 5) / 5) * 5 : 100;
    if (maxV - minV < 15) { minV = Math.max(0, minV - 5); maxV = Math.min(100, maxV + 5); }

    function xPos(i) { return snaps.length === 1 ? padL + plotW / 2 : padL + (i / (snaps.length - 1)) * plotW; }
    function yPos(v) { return padT + (1 - (v - minV) / (maxV - minV)) * plotH; }

    var svgNS = "http://www.w3.org/2000/svg";
    function sEl(tag, attrs) {
      var n = document.createElementNS(svgNS, tag);
      Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
      return n;
    }

    var svg = sEl("svg", { viewBox: "0 0 " + W + " " + H, class: "rais-hist-svg", role: "img", "aria-label": "RAIS Progress Over Time" });
    var ticks = 5;
    for (var t = 0; t <= ticks; t++) {
      var tv = minV + (t / ticks) * (maxV - minV);
      var ty = yPos(tv);
      svg.appendChild(sEl("line", { x1: padL, y1: ty, x2: W - padR, y2: ty, stroke: "rgba(148,163,184,0.18)", "stroke-width": 1 }));
      var tl = sEl("text", { x: padL - 8, y: ty + 4, "text-anchor": "end", fill: "#94a3b8", "font-size": 11, "font-family": "var(--mono)" });
      tl.textContent = tv.toFixed(0);
      svg.appendChild(tl);
    }

    var activeX = xPos(state.histIdx);
    svg.appendChild(sEl("line", { x1: activeX, y1: padT, x2: activeX, y2: H - padB, stroke: "rgba(249,115,22,0.45)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));

    series.forEach(function (sr) {
      var pts = [];
      snaps.forEach(function (s, idx) {
        var v = sr.get(s);
        if (v != null && !isNaN(v)) pts.push(xPos(idx).toFixed(1) + "," + yPos(v).toFixed(1));
      });
      if (pts.length > 1) {
        var pathAttrs = { fill: "none", stroke: sr.color, "stroke-width": sr.width, "stroke-linejoin": "round", "stroke-linecap": "round", points: pts.join(" ") };
        if (sr.dash) pathAttrs["stroke-dasharray"] = sr.dash;
        svg.appendChild(sEl("polyline", pathAttrs));
      }
      snaps.forEach(function (s, idx) {
        var v = sr.get(s);
        if (v == null || isNaN(v)) return;
        var isCert = s.fleet.status === "VALID";
        var isSel = idx === state.histIdx;
        var c = sEl("circle", {
          cx: xPos(idx).toFixed(1),
          cy: yPos(v).toFixed(1),
          r: isSel ? 5.5 : (isCert ? 4.2 : 3.2),
          fill: isCert ? sr.color : "#0f172a",
          stroke: sr.color,
          "stroke-width": isSel ? 2.5 : 1.8,
          style: "cursor:pointer"
        });
        c.addEventListener("click", function () { state.histIdx = idx; renderHistory(); });
        svg.appendChild(c);
      });
    });

    snaps.forEach(function (s, idx) {
      var x = xPos(idx);
      var lbl1 = sEl("text", { x: x, y: H - padB + 16, "text-anchor": "middle", fill: idx === state.histIdx ? "#f8fafc" : "#94a3b8", "font-size": 11, "font-weight": idx === state.histIdx ? "700" : "500", "font-family": "var(--mono)", style: "cursor:pointer" });
      lbl1.textContent = s.label + (s.fleet.status === "VALID" ? " ✓" : "");
      lbl1.addEventListener("click", function () { state.histIdx = idx; renderHistory(); });
      svg.appendChild(lbl1);
      var tStr = (s.utc || "").slice(5, 16).replace("T", " ") + "Z";
      var lbl2 = sEl("text", { x: x, y: H - padB + 30, "text-anchor": "middle", fill: "#64748b", "font-size": 10, "font-family": "var(--mono)", style: "cursor:pointer" });
      lbl2.textContent = tStr;
      lbl2.addEventListener("click", function () { state.histIdx = idx; renderHistory(); });
      svg.appendChild(lbl2);
    });

    chartWrap.textContent = "";
    chartWrap.appendChild(svg);

    var cur = snaps[state.histIdx];
    var base = snaps[0];
    if (snapEl) {
      var tag = cur.label + " (" + cur.release + ") · " + cur.utc + " · " + cur.fleet.valid_models + "/" + cur.fleet.total_models + " VALID";
      snapEl.textContent = cur.note ? (tag + " — " + cur.note) : tag;
    }

    if (subGrid) {
      subGrid.textContent = "";
      var curSubs = state.histView === "model" && cur.models[state.histModel] ? cur.models[state.histModel].submetrics : cur.fleet.submetrics;
      var baseSubs = state.histView === "model" && base.models[state.histModel] ? base.models[state.histModel].submetrics : base.fleet.submetrics;
      (DATA.submetrics_meta || []).forEach(function (meta) {
        var k = meta.key;
        var v = curSubs ? curSubs[k] : null;
        var bv = baseSubs ? baseSubs[k] : null;
        var dl = fmtDelta(v, bv);
        var card = mk("div", "rais-hist-subcard");
        card.appendChild(mk("span", "k", (T.submetricLabels[k] || meta.label).split(" (")[0]));
        var vrow = mk("div", "v");
        vrow.appendChild(mk("span", "", v != null ? v.toFixed(1) : "–"));
        vrow.appendChild(mk("span", "d " + dl.cls, dl.text));
        card.appendChild(vrow);
        subGrid.appendChild(card);
      });
    }

    if (rowsEl) {
      rowsEl.textContent = "";
      snaps.forEach(function (s, idx) {
        var tr = mk("tr", "rais-row" + (idx === state.histIdx ? " active" : ""));
        tr.tabIndex = 0;
        var dl = fmtDelta(s.fleet.rais_unpenalized, base.fleet.rais_unpenalized);
        var tdRel = mk("td");
        tdRel.appendChild(mk("b", "mono", s.label + " (" + s.release + ")"));
        tdRel.appendChild(mk("div", "rais-model-meta mono", (s.utc || "") + (s.note ? (" · " + s.note) : "")));
        tr.appendChild(tdRel);
        var tdVal = mk("td");
        var bCls = s.fleet.status === "VALID" ? "rais-badge valid" : "rais-badge partial";
        tdVal.appendChild(mk("span", bCls, s.fleet.valid_models + "/" + s.fleet.total_models + (s.fleet.status === "VALID" ? " VALID" : "")));
        tr.appendChild(tdVal);
        tr.appendChild(mk("td", "num rais-score-cell", s.fleet.rais_unpenalized.toFixed(2)));
        tr.appendChild(mk("td", "num " + (dl.cls === "up" ? "signal" : ""), idx === 0 ? "base" : dl.text));
        tr.appendChild(mk("td", "num", s.fleet.base_r.toFixed(2)));
        tr.appendChild(mk("td", "num", s.fleet.pillars.P1.toFixed(1)));
        tr.appendChild(mk("td", "num", s.fleet.pillars.P2.toFixed(1)));
        tr.appendChild(mk("td", "num", s.fleet.pillars.P3.toFixed(1)));
        tr.appendChild(mk("td", "num", s.fleet.pillars.P4.toFixed(1)));
        tr.appendChild(mk("td", "num", (s.fleet.pillars.P5 != null ? s.fleet.pillars.P5 : 0).toFixed(1)));
        tr.addEventListener("click", function () { state.histIdx = idx; renderHistory(); });
        rowsEl.appendChild(tr);
      });
    }
  }

  function renderHwGraphic() {
    var wrap = el("rais-hw-graphic");
    var updatedEl = el("rais-updated-utc");
    if (updatedEl && DATA && DATA.updated_utc) {
      updatedEl.textContent = (ko ? "최근 갱신: " : "Last updated: ") + DATA.updated_utc;
    }
    if (!wrap || !DATA || !DATA.hw_scoreboard || !DATA.hw_scoreboard.length) return;
    var items = DATA.hw_scoreboard;
    var svgNS = "http://www.w3.org/2000/svg";
    function sEl(tag, attrs) {
      var n = document.createElementNS(svgNS, tag);
      Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, String(attrs[k])); });
      return n;
    }
    var W = 820, rowH = 66, padL = 248, padR = 136, padT = 26, padB = 26;
    var plotW = W - padL - padR;
    var H = padT + items.length * rowH + padB;
    var svg = sEl("svg", { viewBox: "0 0 " + W + " " + H, class: "rais-hist-svg", role: "img", "aria-label": ko ? "하드웨어 프로필별 인증 RAIS v1.1 진척 그래픽" : "Certified RAIS v1.1 Progress Across Hardware Profiles" });

    for (var g = 0; g <= 100; g += 20) {
      var gx = padL + (g / 100) * plotW;
      svg.appendChild(sEl("line", { x1: gx, y1: padT - 6, x2: gx, y2: H - padB + 4, stroke: "rgba(148,163,184,0.16)", "stroke-width": 1 }));
      var gl = sEl("text", { x: gx, y: H - 8, "text-anchor": "middle", fill: "#94a3b8", "font-size": 10.5, "font-family": "var(--mono)" });
      gl.textContent = String(g);
      svg.appendChild(gl);
    }

    var hdr = sEl("text", { x: padL, y: 14, fill: "#94a3b8", "font-size": 10.5, "font-family": "var(--mono)" });
    hdr.textContent = ko ? "기준선(회색) → 최신 VALID 인증 RAIS v1.1(주황) · 비교선(점선)" : "Baseline (slate) -> Latest Certified VALID RAIS v1.1 (orange) · Ref marker (dashed)";
    svg.appendChild(hdr);

    items.forEach(function (it, idx) {
      var y = padT + idx * rowH;
      var barY = y + 18, barH = 18;
      var baseW = Math.max(2, (Math.min(100, it.baseline) / 100) * plotW);
      var curW = Math.max(2, (Math.min(100, it.current) / 100) * plotW);
      var delta = it.current - it.baseline;

      var t1 = sEl("text", { x: 8, y: y + 20, fill: "#f8fafc", "font-size": 12, "font-weight": "700", "font-family": "var(--mono)" });
      t1.textContent = ko ? (it.title_ko || it.title) : it.title;
      svg.appendChild(t1);

      var t2 = sEl("text", { x: 8, y: y + 36, fill: "#5eead4", "font-size": 10.5, "font-family": "var(--mono)" });
      t2.textContent = it.stack + " · Base R " + it.base_r.toFixed(1) + " · P5 " + it.p5.toFixed(1);
      svg.appendChild(t2);

      var t3 = sEl("text", { x: padL, y: y + 51, fill: "#94a3b8", "font-size": 10, "font-family": "var(--mono)" });
      t3.textContent = ko ? (it.note_ko || it.note) : it.note;
      svg.appendChild(t3);

      svg.appendChild(sEl("rect", { x: padL, y: barY, width: plotW, height: barH, rx: 3, fill: "rgba(15,23,42,0.75)", stroke: "rgba(148,163,184,0.22)", "stroke-width": 1 }));
      svg.appendChild(sEl("rect", { x: padL, y: barY, width: curW.toFixed(1), height: barH, rx: 3, fill: "#f97316", opacity: "0.9" }));
      svg.appendChild(sEl("rect", { x: padL, y: barY, width: baseW.toFixed(1), height: barH, rx: 3, fill: "#64748b", opacity: "0.75" }));

      if (it.ref_score != null) {
        var rx = padL + (Math.min(100, it.ref_score) / 100) * plotW;
        svg.appendChild(sEl("line", { x1: rx.toFixed(1), y1: barY - 3, x2: rx.toFixed(1), y2: barY + barH + 3, stroke: "#38bdf8", "stroke-width": 2, "stroke-dasharray": "3 2" }));
        var rl = sEl("text", { x: rx.toFixed(1), y: barY - 6, "text-anchor": "middle", fill: "#38bdf8", "font-size": 9.5, "font-family": "var(--mono)" });
        rl.textContent = it.ref_label + " " + it.ref_score.toFixed(1);
        svg.appendChild(rl);
      }

      var sv = sEl("text", { x: W - padR + 10, y: barY + 13, fill: "#f8fafc", "font-size": 13, "font-weight": "700", "font-family": "var(--mono)" });
      sv.textContent = it.current.toFixed(2);
      svg.appendChild(sv);

      var dv = sEl("text", { x: W - padR + 68, y: barY + 13, fill: "#86efac", "font-size": 11.5, "font-weight": "700", "font-family": "var(--mono)" });
      dv.textContent = (delta >= 0 ? "+" : "") + delta.toFixed(2);
      svg.appendChild(dv);
    });

    wrap.textContent = "";
    wrap.appendChild(svg);
  }

  function renderAll() {
    var prof = currentProfile();
    if (!prof) return;
    renderHwGraphic();
    renderHistory();
    renderSummary(prof);
    renderTable(prof);
    renderDetail(prof);
    renderRoi(prof);
  }

  function wireControls() {
    var histBtns = document.querySelectorAll("[data-hist-view]");
    Array.prototype.forEach.call(histBtns, function (btn) {
      btn.addEventListener("click", function () {
        state.histView = btn.getAttribute("data-hist-view");
        Array.prototype.forEach.call(histBtns, function (b) { b.setAttribute("aria-pressed", b === btn ? "true" : "false"); });
        renderHistory();
      });
    });
    var histModelSel = el("rais-hist-model");
    if (histModelSel) {
      histModelSel.addEventListener("change", function () {
        state.histModel = histModelSel.value;
        renderHistory();
      });
    }
    var hwSel = el("rais-hw");
    if (hwSel) {
      hwSel.addEventListener("change", function () {
        state.hw = hwSel.value;
        var prof = currentProfile();
        if (prof) {
          var firstMeasured = prof.models.find(function (m) { return m.sweep_state !== "pending"; });
          if (firstMeasured) state.selected = firstMeasured.model;
        }
        renderAll();
      });
    }
    var tierBtns = document.querySelectorAll("[data-rais-tier]");
    Array.prototype.forEach.call(tierBtns, function (btn) {
      btn.addEventListener("click", function () {
        state.tier = btn.getAttribute("data-rais-tier");
        Array.prototype.forEach.call(tierBtns, function (b) { b.setAttribute("aria-pressed", b === btn ? "true" : "false"); });
        renderAll();
      });
    });
    var filterBtns = document.querySelectorAll("[data-rais-filter]");
    Array.prototype.forEach.call(filterBtns, function (btn) {
      btn.addEventListener("click", function () {
        state.filter = btn.getAttribute("data-rais-filter");
        Array.prototype.forEach.call(filterBtns, function (b) { b.setAttribute("aria-pressed", b === btn ? "true" : "false"); });
        renderAll();
      });
    });
  }

  var statusEl = el("rais-status");
  if (statusEl) statusEl.textContent = T.loading;
  wireControls();
  fetch("rais-data.json?v=10")
    .then(function (r) {
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    })
    .then(function (json) {
      DATA = json;
      if (statusEl) statusEl.textContent = T.selectModel;
      renderAll();
    })
    .catch(function () {
      if (statusEl) statusEl.textContent = T.err;
    });
})();
