import { afterAll, beforeAll, expect, it, vi } from "vitest";
import type PlotlyType from "plotly.js-dist-min";

let Plotly: typeof PlotlyType;

beforeAll(async () => {
  // jsdom has no canvas/WebGL implementation. SVG/schema tests only need text
  // measurement; these stubs do not test canvas pixels or browser rendering.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => ({
    measureText: (text: string) => ({ width: text.length * 6 }),
  }) as unknown as CanvasRenderingContext2D);
  vi.stubGlobal("URL", class extends URL { static createObjectURL() { return "blob:plotly-fixture"; } });
  Plotly = (await import("plotly.js-dist-min")).default;
});

afterAll(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("loads the exact locked runtime and validates supported trace families", () => {
  expect(Reflect.get(Plotly, "version")).toBe("4.1.2");
  for (const data of [
    [{ type: "scatter", x: [1, 2], y: [100, 102] }],
    [{ type: "bar", x: ["A", "B"], y: [0.4, 0.6] }],
    [{ type: "histogram", x: [0.01, -0.02, 0.03] }],
    [{ type: "heatmap", z: [[0.1, 0.2]], colorbar: { title: { text: "MAE" } } }],
    [{ type: "pie", labels: ["A", "B"], values: [0.4, 0.6] }],
  ]) {
    const layout = data[0].type === "pie" ? {} : {
      xaxis: { title: { text: "Period" } }, yaxis: { title: { text: "Value" } },
    };
    expect(Plotly.validate(data as Parameters<typeof Plotly.validate>[0], layout)).toBeUndefined();
  }
});

it("renders SVG portfolio values without a cloud-upload control", async () => {
  const element = document.createElement("div");
  document.body.append(element);
  await Plotly.newPlot(element, [{ type: "scatter", x: [1, 2], y: [100, 102] }], {
    width: 600, height: 400, xaxis: { title: { text: "Period" } },
    yaxis: { title: { text: "Portfolio value" } },
  }, { responsive: false, displayModeBar: true, showSendToCloud: false });
  expect(element.querySelector(".scatterlayer path.js-line")).not.toBeNull();
  expect(element.textContent).toContain("Portfolio value");
  expect(element.querySelectorAll(".modebar-btn").length).toBeGreaterThan(0);
  expect(element.querySelector('[data-title*="cloud" i]')).toBeNull();
  Plotly.purge(element);
  element.remove();
});
