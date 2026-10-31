window.SP_PRICING = {
  "committedPrice": 15,
  "committedPriceKRW": 20000,
  "vatRate": 0.10,
  "termMonths": 24,
  "phaseADays": 30,
  "hourlyEarliestStart": "2026-12-01",
  "committedEarliestStart": "2026-12-01",
  "fleets": [
    { "name": "Fleet 1", "kind": "16 Sparks", "state": "serving", "rate": 1 },
    { "name": "Fleet 2", "kind": "M5U", "state": "building", "rate": 1 },
    { "name": "Fleet 3", "kind": "16 Sparks", "state": "building", "rate": 1 }
  ],
  "windows": [
    { "id": "americas", "name": "Americas", "from": 0, "to": 8 },
    { "id": "asia", "name": "Asia", "from": 8, "to": 16 },
    { "id": "europe", "name": "Europe", "from": 16, "to": 24 }
  ],
  "market": {
    "hcPrice": 20,
    "hcPriceKRW": 26000,
    "minPurchase": 10,
    "validityMonths": 12,
    "releaseFee": 0.10,
    "horizonDays": 7,
    "advanceDays": 31,
    "advanceMultiplier": 1.25,
    "lastCallHours": 6,
    "maxBidDays": 31,
    "trancheDemoHours": 40,
    "pool": {
      "seats": 8,
      "minMultiplier": 0.5,
      "maxMultiplier": 3,
      "models": [
        { "id": "kimi-k3", "name": "Kimi K3" },
        { "id": "glm-5.3", "name": "GLM 5.3" }
      ]
    }
  },
  "gates": [
    { "name": "Gate 1", "tranches": 6, "dueDay": 30 },
    { "name": "Gate 2", "tranches": 12, "dueDay": 60 },
    { "name": "Gate 3", "tranches": null, "dueDay": 90 }
  ]
};
