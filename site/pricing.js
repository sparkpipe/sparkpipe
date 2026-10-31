window.SP_PRICING = {
  "vatRate": 0.10,
  "phaseADays": 30,
  "hourlyEarliestStart": "2026-12-01",
  "fleets": [
    { "name": "Fleet 1", "kind": "16 Sparks", "state": "serving", "rate": 1 },
    { "name": "Fleet 2", "kind": "M5U", "state": "building", "rate": 1 },
    { "name": "Fleet 3", "kind": "16 Sparks", "state": "building", "rate": 1 }
  ],
  "accelerators": [],
  "windows": [
    { "id": "americas", "name": "Americas", "from": 0, "to": 8, "localDay": -1 },
    { "id": "asia", "name": "Asia", "from": 8, "to": 16, "localDay": 0 },
    { "id": "europe", "name": "Europe", "from": 16, "to": 24, "localDay": 0 }
  ],
  "models": [
    { "id": "kimi-k3", "name": "Kimi K3", "decode": 16.2, "prefill": 590 },
    { "id": "glm-5.3", "name": "GLM 5.3", "decode": 28.9, "prefill": 1162 },
    { "id": "glm-5.3-flash", "name": "GLM 5.3 Flash", "decode": 55.0, "prefill": 876 },
    { "id": "deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash", "decode": 42.5, "prefill": 1290 },
    { "id": "k-exaone-2.0", "name": "K-EXAONE 2.0", "decode": 27.3, "prefill": 324 },
    { "id": "a.x-k2", "name": "A.X K2", "decode": 23.8, "prefill": 1274 },
    { "id": "solar-open-2", "name": "Solar Open 2", "decode": 61.2, "prefill": 1025 },
    { "id": "mimo-v2.6-pro", "name": "MiMo V2.6 Pro", "decode": 19.0, "prefill": 55 },
    { "id": "mimo-v2.6-flash", "name": "MiMo V2.6 Flash", "decode": 40.9, "prefill": 157 },
    { "id": "gemma-4-31b", "name": "Gemma 4 31B", "decode": 34.4, "prefill": 834 },
    { "id": "gemma-4-26b-a4b", "name": "Gemma 4 26B A4B", "decode": 89.5, "prefill": 1848 },
    { "id": "ling-3.0-flash", "name": "Ling 3.0 Flash", "decode": 80.4, "prefill": 1514 },
    { "id": "ling-3.0-flash-fin", "name": "Ling 3.0 Flash Finance", "decode": 79.2, "prefill": 1550 },
    { "id": "laguna-s-2.1", "name": "Laguna S 2.1", "decode": 51.0, "prefill": 773 },
    { "id": "muse-glimmer-30b", "name": "Muse Glimmer 30B", "decode": 37.9, "prefill": 1186 },
    { "id": "hy4-preview", "name": "Hunyuan Hy4 Preview", "decode": 14.2, "prefill": 53 },
    { "id": "qwen3.8-27b", "name": "Qwen 3.8 27B", "decode": 57.4, "prefill": 1852 },
    { "id": "qwen3.8-max", "name": "Qwen 3.8 Max", "decode": 10.4, "prefill": 35 }
  ],
  "plans": {
    "earliestStart": "2026-12-01",
    "startWithinDays": 60,
    "terms": [
      { "months": 3, "hourPrice": 17.5, "hourPriceKRW": 23000, "payment": "upfront" },
      { "months": 12, "hourPrice": 16.5, "hourPriceKRW": 22000, "payment": "gates" },
      { "months": 24, "hourPrice": 15, "hourPriceKRW": 20000, "payment": "gates" },
      { "months": 36, "hourPrice": 14, "hourPriceKRW": 18500, "payment": "gates" }
    ],
    "lockPremium": 1,
    "lockPremiumKRW": 1300,
    "maxFleets": 3,
    "maintenanceHoursPerWeek": 3
  },
  "market": {
    "hcPrice": 20,
    "hcPriceKRW": 26000,
    "minPurchase": 10,
    "validityMonths": 12,
    "releaseFee": 0.10,
    "advanceDays": 31,
    "lastCallHours": 6,
    "maxBidDays": 31,
    "trancheDemoHours": 40,
    "accelerator": {
      "maxBid": 10,
      "defaultWait": 300,
      "maxWait": 600
    },
    "pool": {
      "seats": 8,
      "minMultiplier": 0.5,
      "maxMultiplier": 3
    }
  },
  "gates": [
    { "name": "Gate 1", "tranches": 6, "dueDay": 30 },
    { "name": "Gate 2", "tranches": 12, "dueDay": 60 },
    { "name": "Gate 3", "tranches": null, "dueDay": 90 }
  ]
};
