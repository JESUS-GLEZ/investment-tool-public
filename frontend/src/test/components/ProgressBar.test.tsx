import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgressBar from "@/components/ui/ProgressBar";

describe("ProgressBar", () => {
  it("renders the progress percentage", () => {
    render(<ProgressBar fraction={0.5} message="Loading..." isConnected={true} />);
    expect(screen.getByText(/50%/)).toBeInTheDocument();
  });

  it("clamps fraction above 1 to 100%", () => {
    render(<ProgressBar fraction={1.5} message="" isConnected={false} />);
    expect(screen.getByText(/100%/)).toBeInTheDocument();
  });

  it("shows connected indicator when connected", () => {
    render(<ProgressBar fraction={0.3} message="" isConnected={true} />);
    expect(screen.getByText("connected")).toBeInTheDocument();
  });

  it("shows disconnected indicator when not connected", () => {
    render(<ProgressBar fraction={0.3} message="" isConnected={false} />);
    expect(screen.getByText("disconnected")).toBeInTheDocument();
  });

  it("shows custom message", () => {
    render(<ProgressBar fraction={0.1} message="Fetching data" isConnected={true} />);
    expect(screen.getByText("Fetching data")).toBeInTheDocument();
  });

  it("shows fallback message when message is empty", () => {
    render(<ProgressBar fraction={0} message="" isConnected={false} />);
    expect(screen.getByText(/Waiting for progress/)).toBeInTheDocument();
  });

  it("sets correct width style on progress bar", () => {
    const { container } = render(
      <ProgressBar fraction={0.75} message="" isConnected={false} />
    );
    const bar = container.querySelector(".bg-blue-600") as HTMLElement;
    expect(bar.style.width).toBe("75%");
  });
});
