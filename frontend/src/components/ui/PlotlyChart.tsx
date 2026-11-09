/** Render server-side Plotly figure JSON with the locked local runtime. */
import { useEffect, useRef } from "react";
import Plotly, { type Data } from "plotly.js-dist-min";

interface Props {
  figure: Record<string, unknown>;
}

export default function PlotlyChart({ figure }: Props) {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = divRef.current;
    if (!element) return;

    const data = (figure.data ?? []) as Data[];
    const layoutRaw = (figure.layout ?? {}) as Record<string, unknown>;

    const layout = {
      ...layoutRaw,
      autosize: true,
      margin: { l: 55, r: 30, t: 20, b: 50 },
      title: undefined, // rendered in our own header strip
    };

    Plotly.react(element, data, layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      showSendToCloud: false,
    });

    return () => {
      Plotly.purge(element);
    };
  }, [figure]);

  // Extract title for the header strip
  const layoutObj = figure.layout as Record<string, unknown> | undefined;
  const rawTitle = layoutObj?.title;
  const titleText =
    typeof rawTitle === "string"
      ? rawTitle
      : typeof rawTitle === "object" && rawTitle !== null
        ? ((rawTitle as { text?: string }).text ?? "")
        : "";

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      {titleText && (
        <div className="border-b border-gray-100 px-4 py-2">
          <p className="text-sm font-medium text-gray-700">{titleText}</p>
        </div>
      )}
      <div ref={divRef} className="h-[480px] w-full" />
    </div>
  );
}
