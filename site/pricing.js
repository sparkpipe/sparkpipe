window.SP_PRICING = {
  hourlyPrice: 20,
  committedPrice: 15,
  vatRate: 0.10,
  hourlyMaxHours: 168,
  verifiedMaxHours: 720,
  hourlyValidityMonths: 12,
  displaceNoticeHours: 6,
  termMonths: 24,
  phaseADays: 30,
  hourlyEarliestStart: "2026-12-01",
  committedEarliestStart: "2026-12-01",
  fleets: [
    { name: "Fleet 1", kind: "16 Sparks", state: "serving", rate: 1 },
    { name: "Fleet 2", kind: "M5U", state: "building", rate: 1 },
    { name: "Fleet 3", kind: "16 Sparks", state: "building", rate: 1 }
  ],
  gates: [
    { name: "Gate 1", tranches: 6, dueDay: 30 },
    { name: "Gate 2", tranches: 12, dueDay: 60 },
    { name: "Gate 3", tranches: null, dueDay: 90 }
  ]
};
