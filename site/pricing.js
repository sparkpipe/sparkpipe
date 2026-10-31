window.SP_PRICING = {
  "vatRate": 0.10,
  "phaseADays": 30,
  "hourlyEarliestStart": "2026-12-01",
  "fleets": [
    { "name": "Fleet 1", "kind": "16 Sparks", "state": "serving", "rate": 1 },
    { "name": "Fleet 2", "kind": "M5U", "state": "building", "rate": 1 },
    { "name": "Fleet 3", "kind": "16 Sparks", "state": "building", "rate": 1 }
  ],
  "windows": [
    { "id": "americas", "name": "Americas", "from": 0, "to": 8, "localDay": -1 },
    { "id": "asia", "name": "Asia", "from": 8, "to": 16, "localDay": 0 },
    { "id": "europe", "name": "Europe", "from": 16, "to": 24, "localDay": 0 }
  ],
  "plans": {
    "earliestStart": "2026-12-01",
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
