import { lazy, Suspense, useState } from "react";
import { record } from "@/lib/runPresentation";
const PlotlyChart = lazy(() => import("@/components/ui/PlotlyChart"));
/** Mount exactly one retained figure. Filters use only recorded titles. */
export default function SavedChartBrowser({figures, configurations}: {figures: Record<string, unknown>[]; configurations: string[]}) {
  const [strategy, setStrategy] = useState("");
  const [month, setMonth] = useState("");
  const [selected, setSelected] = useState("");
  const charts = figures.map((figure, index) => {
    const rawTitle = record(figure.layout).title;
    const title = typeof rawTitle === "string" ? rawTitle : typeof record(rawTitle).text === "string" ? String(record(rawTitle).text) : `Chart ${index + 1}`;
    const name = [...configurations].sort((a, b) => b.length - a.length).find(c => title.startsWith(`${c} — `));
    return {figure, index: String(index), title, strategy: name, month: title.match(/\b\d{4}-\d{2}(?:-\d{2})?\b/)?.[0].slice(0, 7)};
  });
  const months = [...new Set(charts.flatMap(c => c.month ? [c.month] : []))].sort();
  const visible = charts.filter(c => (!strategy || c.strategy === strategy) && (!month || c.month === month));
  const current = visible.find(c => c.index === selected);
  const field = "mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
  return <div className="space-y-4"><div className="grid gap-3 sm:grid-cols-2">{configurations.some(c => charts.some(chart => chart.strategy === c)) && <label className="text-sm font-medium">Chart configuration<select className={field} value={strategy} onChange={e => {setStrategy(e.target.value); setSelected("");}}><option value="">All configurations</option>{configurations.filter(c => charts.some(chart => chart.strategy === c)).map(c => <option key={c}>{c}</option>)}</select></label>}{months.length > 0 && <label className="text-sm font-medium">Chart month<select className={field} value={month} onChange={e => {setMonth(e.target.value); setSelected("");}}><option value="">All recorded months</option>{months.map(m => <option key={m}>{m}</option>)}</select></label>}</div><label className="block text-sm font-medium">Saved chart<select className={field} value={current?.index ?? ""} onChange={e => setSelected(e.target.value)}><option value="">Choose one of {visible.length} saved charts</option>{visible.map(c => <option key={c.index} value={c.index}>{c.title}</option>)}</select></label>{current && <Suspense fallback={<p role="status">Loading chart…</p>}><PlotlyChart key={current.index} figure={current.figure} /></Suspense>}</div>;
}
