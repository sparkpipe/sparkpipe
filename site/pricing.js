window.SP_PRICING = {
  "vatRate": 0.1,
  "demoFrom": "2026-12-01",
  "weeksFrom": "2027-01-04",
  "fleets": [
    {
      "id": "F1",
      "name": "Fleet 1",
      "kind": "16 Sparks",
      "state": "serving",
      "rate": 1,
      "demo": true,
      "storageTB": 4
    },
    {
      "id": "F2",
      "name": "Fleet 2",
      "kind": "M5U",
      "state": "building",
      "rate": 1,
      "demo": false,
      "storageTB": 4
    },
    {
      "id": "F3",
      "name": "Fleet 3",
      "kind": "16 Sparks",
      "state": "building",
      "rate": 1,
      "demo": false,
      "storageTB": 4
    }
  ],
  "accelerators": [],
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
  "hc": {
    "prices": [
      {
        "from": 1,
        "usd": 20,
        "krw": 26000
      },
      {
        "from": 144,
        "usd": 15,
        "krw": 19500
      }
    ],
    "min": 1,
    "validityMonths": 12
  },
  "bookings": {
    "hourHC": 1,
    "weekHC": 144,
    "discounts": [
      {
        "weeks": 4,
        "off": 0.05
      },
      {
        "weeks": 13,
        "off": 0.1
      },
      {
        "weeks": 26,
        "off": 0.15
      },
      {
        "weeks": 52,
        "off": 0.2
      }
    ],
    "maxWeeksAhead": 52,
    "demoDaysAhead": 7,
    "demoHoursPerWeek": 24,
    "maintenanceHoursPerWeek": 3
  },
  "accelerator": {
    "maxBid": 10,
    "defaultWait": 300,
    "maxWait": 600
  }
};
