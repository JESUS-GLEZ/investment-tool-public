import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import PlotlyChart from "@/components/ui/PlotlyChart";

const { plot, purge } = vi.hoisted(() => ({ plot: vi.fn(), purge: vi.fn() }));
vi.mock("plotly.js-dist-min", () => ({ default: { react: plot, purge } }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("local Plotly chart", () => {
  it("renders server values with local controls and disables cloud upload", () => {
    const figure = {
      data: [{ type: "scatter", x: ["2024-01-01"], y: [123.45] }],
      layout: { title: { text: "Portfolio value" }, yaxis: { title: { text: "USD" } } },
    };
    const { unmount } = render(<PlotlyChart figure={figure} />);
    expect(screen.getByText("Portfolio value")).toBeInTheDocument();
    const [element, data, layout, config] = plot.mock.calls[0];
    expect(data).toEqual(figure.data);
    expect(layout.yaxis).toEqual(figure.layout.yaxis);
    expect(layout.title).toBeUndefined();
    expect(config).toMatchObject({ showSendToCloud: false, displayModeBar: true, responsive: true });
    unmount();
    expect(purge).toHaveBeenCalledWith(element);
  });

  it("redraws changed figures and purges the previous plot", () => {
    const { rerender } = render(<PlotlyChart figure={{ data: [], layout: { title: "First" } }} />);
    rerender(<PlotlyChart figure={{ data: [{ type: "bar", y: [1, 2] }], layout: { title: "Second" } }} />);
    expect(screen.getByText("Second")).toBeInTheDocument();
    expect(screen.queryByText("First")).not.toBeInTheDocument();
    expect(plot).toHaveBeenCalledTimes(2);
    expect(purge).toHaveBeenCalledTimes(1);
    expect(plot.mock.calls[1][1][0].y).toEqual([1, 2]);
  });
});
