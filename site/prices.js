(function () {
  var P = window.SP_PRICING;
  if (!P) return;
  var ko = document.documentElement.lang === "ko";
  var usd = function (n) { return "$" + (Math.round(n * 100) % 100 === 0 ? Math.round(n).toLocaleString("en-US") : n.toFixed(2)); };
  var krw = function (n) { return "₩" + Math.round(n).toLocaleString("en-US"); };
  var months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  var day = function (text) {
    var p = text.split("-");
    return ko ? p[0] + "년 " + Number(p[1]) + "월 " + Number(p[2]) + "일" : Number(p[2]) + " " + months[Number(p[1]) - 1] + " " + p[0];
  };
  var H = P.hc.prices, small = H[0], bulk = H[H.length - 1], B = P.bookings, D = B.discounts;
  var pct = function (x) { return Math.round(x * 100) + "%"; };
  var list = D.map(function (d, i) {
    if (ko) return d.weeks + "주" + (i === D.length - 1 ? " " : " 이상 ") + pct(d.off);
    return pct(d.off) + (i === D.length - 1 ? " at " : " from ") + d.weeks;
  }).join(", ");
  var table = ko
    ? "예약하는 주와 현재 주를 포함하여 " + D.map(function (d, i) { return d.weeks + "주" + (i === D.length - 1 ? "" : " 이상") + " 보유 시 새로 예약하는 각 주 " + pct(d.off) + " 할인"; }).join(", ")
    : D.map(function (d, i) { return pct(d.off) + " off each new week when you hold " + d.weeks + (i === D.length - 1 ? "" : " or more") + " weeks"; }).join(", ") + ", counting the current week and the weeks you are booking";
  var values = {
    start: { demo: day(P.demoFrom), weeks: day(P.weeksFrom) },
    week: { hc: String(B.weekHC), usd: usd(B.weekHC * bulk.usd), krw: krw(B.weekHC * bulk.krw), effective: usd(B.weekHC * bulk.usd / 168) },
    off: { max: pct(D[D.length - 1].off), list: list, table: table },
    hc: { "small-usd": usd(small.usd), "small-krw": krw(small.krw), "bulk-usd": usd(bulk.usd), "bulk-krw": krw(bulk.krw), "bulk-from": String(bulk.from), months: String(P.hc.validityMonths) },
    hour: { hc: String(B.hourHC) },
    demo: { days: String(B.demoDaysAhead), hours: String(B.demoHoursPerWeek) },
    fleets: { pin: String(P.fleets.filter(function (f) { return !f.demo; }).length), demo: String(P.fleets.filter(function (f) { return f.demo; }).length) }
  };
  Object.keys(values).forEach(function (key) {
    document.querySelectorAll("[data-" + key + "]").forEach(function (el) {
      var text = values[key][el.getAttribute("data-" + key)];
      if (text !== undefined) el.textContent = text;
    });
  });
  document.querySelectorAll("[data-storage]").forEach(function (el) { el.textContent = Math.max.apply(null, P.fleets.map(function (f) { return f.storageTB; })) + " TB"; });
  document.querySelectorAll("[data-ahead]").forEach(function (el) { el.textContent = String(B.maxWeeksAhead); });
  document.querySelectorAll("[data-off-week]").forEach(function (el) {
    var d = D.filter(function (x) { return String(x.weeks) === el.getAttribute("data-off-week"); })[0];
    if (d) el.textContent = usd(B.weekHC * (1 - d.off) * bulk.usd);
  });
})();
