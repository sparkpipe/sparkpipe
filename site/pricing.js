window.SP_PRICING = {
  hourlyPrice: 20,
  committedPrice: 15,
  vatRate: 0.10,
  hourlyMaxHours: 24,
  hourlyValidityMonths: 12,
  displaceNoticeHours: 6,
  committedMinHours: 1,
  committedMaxHours: 15000,
  termMonths: 24,
  earliestStart: "2027-01-01",
  fleets: [
    { name: "Fleet 1", kind: "16 Sparks", state: "serving", creditRate: 1 },
    { name: "Fleet 2", kind: "M5U", state: "building", creditRate: 1 },
    { name: "Fleet 3", kind: "16 Sparks", state: "building", creditRate: 1 }
  ],
  gates: [
    { name: "Gate 1", tranches: 6, dueDay: 0 },
    { name: "Gate 2", tranches: 12, dueDay: 60 },
    { name: "Gate 3", tranches: null, dueDay: 90 }
  ]
};
