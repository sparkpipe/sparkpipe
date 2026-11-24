(function () {
  var ko = document.documentElement.lang === "ko";
  var T = ko ? {
    loading: "RAIS 데이터를 불러오는 중…",
    err: "rais-data.json을 불러오지 못했습니다.",
    hwSummary: function (h) {
      return h.ranks + "랭크 · 랭크당 메모리 " + h.mem_bw_gb_s_per_rank + " GB/s (합계 " + Math.round(h.ranks * h.mem_bw_gb_s_per_rank).toLocaleString("en-US") + " GB/s) · BF16/FP8/FP4 " + h.bf16_tflops_per_rank + "/" + h.fp8_tflops_per_rank + "/" + h.fp4_tflops_per_rank + " TFLOPS · 네트워크 " + h.net_bw_gb_s_per_rank + " GB/s (" + h.net_hop_us + " µs)";
    },
    statusLabels: {
      VALID: "인증 완료",
      INVALID: "무효 (게이트 실패)",
      UNVERIFIED: "부분 실측 · 전체 RAIS 스윕 대기",
      PENDING: "전체 RAIS 스윕 대기"
    },
    sweepBadge: function (m) {
      if (m.status === "VALID") return "인증 완료 (" + (m.measured_count || 11) + "/12)";
      if (m.status === "INVALID") return "무효 (" + ((m.failed_gates && m.failed_gates.length) ? m.failed_gates.join(", ") : "게이트 실패") + ")";
      if (m.sweep_state === "partial_multi") return "부분 실측 " + m.measured_count + "/12 · 전체 RAIS 스윕 대기";
      if (m.sweep_state === "partial_qual") return "릴리스 검증 " + m.measured_count + "/12 · 전체 RAIS 스윕 대기";
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
    selectModel: "행의 모델을 클릭하면 4개 축과 12개 세부 지표, 3대 루프라인 분해를 볼 수 있습니다.",
    detailTitle: function (m, hw) { return m.model + " · " + hw.name; },
    detailSub: function (m) {
      var r = m.receipt ? ("근거: " + m.receipt) : "아직 전체 rais_bench.py 스윕 전입니다.";
      return "티어 " + m.tier + " (" + m.fleet_weight_pct + "%) · " + m.dtype + " · 활성 " + m.active_params_b + "B · 컨텍스트 " + fmtCtx(m.context_tokens) + " · " + r;
    },
    pillars: [
      ["P1", "단일 스트림 디코드 (25%)"],
      ["P2", "추측 디코딩 (25%)"],
      ["P3", "배치·동시성 확장 (25%)"],
      ["P4", "프리필·멀티턴 에이전트 (25%)"]
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
      p4c_multiturn_prefix_ttft: "4c 멀티턴 프리픽스·상태 재사용 TTFT"
    },
    noRoi: "이 하드웨어 프로필에 등록된 실측 결과가 아직 없어 ROI 순위가 비어 있습니다.",
    modelsShown: function (n, total) { return total + "개 모델 중 " + n + "개 표시"; }
  } : {
    loading: "Loading RAIS scorecard data…",
    err: "Failed to load rais-data.json.",
    hwSummary: function (h) {
      return h.ranks + " rank" + (h.ranks === 1 ? "" : "s") + " · " + h.mem_bw_gb_s_per_rank + " GB/s/rank (" + Math.round(h.ranks * h.mem_bw_gb_s_per_rank).toLocaleString("en-US") + " GB/s agg) · BF16/FP8/FP4 " + h.bf16_tflops_per_rank + "/" + h.fp8_tflops_per_rank + "/" + h.fp4_tflops_per_rank + " TFLOPS · Net " + h.net_bw_gb_s_per_rank + " GB/s (" + h.net_hop_us + " µs hop)";
    },
    statusLabels: {
      VALID: "Certified VALID",
      INVALID: "INVALID (gate failed)",
      UNVERIFIED: "Partial receipts · Pending full RAIS sweep",
      PENDING: "Pending full RAIS sweep"
    },
    sweepBadge: function (m) {
      if (m.status === "VALID") return "VALID (" + (m.measured_count || 11) + "/12)";
      if (m.status === "INVALID") return "INVALID (" + ((m.failed_gates && m.failed_gates.length) ? m.failed_gates.join(", ") : "gate failed") + ")";
      if (m.sweep_state === "partial_multi") return "Partial " + m.measured_count + "/12 · Pending full RAIS sweep";
      if (m.sweep_state === "partial_qual") return "Qual " + m.measured_count + "/12 · Pending full RAIS sweep";
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
    selectModel: "Select any model row to inspect its 4 pillars, 12 sub-metrics, 3-roofline breakdown, and source receipt.",
    detailTitle: function (m, hw) { return m.model + " · " + hw.name; },
    detailSub: function (m) {
      var r = m.receipt ? ("Receipt: " + m.receipt) : "Pending full rais_bench.py sweep on this profile.";
      return "Tier " + m.tier + " (" + m.fleet_weight_pct + "% fleet wt) · " + m.dtype + " · " + m.active_params_b + "B active · " + fmtCtx(m.context_tokens) + " ctx · " + r;
    },
    pillars: [
      ["P1", "Single-Stream Decode (25%)"],
      ["P2", "Speculation (25%)"],
      ["P3", "Batch & Concurrency (25%)"],
      ["P4", "Prefill & Multi-Turn (25%)"]
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
      p4c_multiturn_prefix_ttft: "4c Multi-Turn Prefix Reuse TTFT"
    },
    noRoi: "No receipted measurements on this profile yet; run tools/rais_bench.py to populate ROI headroom.",
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
  var state = { hw: "16x-gb10-1x100g", tier: "all", filter: "all", selected: "kimi-k3.1-pro" };

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
    var covSub = hasValid ? ((prof.fleet.valid_weight_pct != null ? prof.fleet.valid_weight_pct.toFixed(1) : prof.fleet.coverage_pct.toFixed(1)) + (ko ? "% 플릿 비중 인증 · 19/19 실측" : "% fleet weight certified · 19/19 swept")) : (T.statusLabels[prof.fleet.status] || prof.fleet.status);
    var raisLbl = hasValid ? T.kpiRais : T.kpiRaisPending;
    var raisNum = hasValid ? (prof.fleet.rais_covered.toFixed(1) + " / 100") : (prof.fleet.rais_raw_covered != null ? prof.fleet.rais_raw_covered.toFixed(1) + " / 100" : T.pendingCell);
    var raisSub = hasValid ? ((ko ? "엄격 플릿 " : "Strict fleet ") + (prof.fleet.rais_strict != null ? prof.fleet.rais_strict.toFixed(1) : "–") + (ko ? " · 19종 원시 " : " · 19-model raw ") + (prof.fleet.rais_raw_covered != null ? prof.fleet.rais_raw_covered.toFixed(1) : "–")) : (prof.fleet.coverage_pct > 0 ? (prof.fleet.coverage_pct.toFixed(1) + "% weight covered") : T.pendingShort);
    var items = [
      [T.kpiHw, prof.hardware.name, prof.hardware.description],
      [T.kpiCoverage, covVal, covSub],
      [raisLbl, raisNum, raisSub],
      [T.kpiRoi, topRoi ? ("+" + topRoi.fleet_headroom_pts.toFixed(2) + " pts") : T.pendingCell,
        topRoi ? (topRoi.model + " · " + (T.submetricLabels[topRoi.submetric] || topRoi.label)) : T.pendingShort]
    ];
    items.forEach(function (it) {
      var box = mk("div");
      box.appendChild(mk("p", "label", it[0]));
      box.appendChild(mk("strong", "num", it[1]));
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
    var badgeCls = model.status === "VALID" ? "rais-badge valid" : (model.sweep_state === "partial_multi" ? "rais-badge partial" : "rais-badge pending");
    head.appendChild(mk("span", badgeCls, T.sweepBadge(model)));
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
      tdEmpty.colSpan = 9;
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

  function renderAll() {
    var prof = currentProfile();
    if (!prof) return;
    renderSummary(prof);
    renderTable(prof);
    renderDetail(prof);
    renderRoi(prof);
  }

  function wireControls() {
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
  fetch("rais-data.json?v=1")
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
