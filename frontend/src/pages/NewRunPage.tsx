import { useQuery } from "@tanstack/react-query";
import { listCampaigns } from "@/api/endpoints";
import { CampaignSelect } from "@/components/ui/Campaigns";
import { useSearchParams } from "react-router-dom";
import StrategyExperimentForm from "@/components/ui/StrategyExperimentForm";
import ModelComparisonForm from "@/components/ui/ModelComparisonForm";

const RUN_TASKS = {
  investment: { label: "Generate an investment portfolio", description: "Calculate a portfolio from the latest completed data within your chosen historical window. This produces suggested allocations; it does not place orders." },
  "strategy-comparison": { label: "Backtest investment strategies", description: "Backtest one or more strategy configurations over the same historical period, optionally against a benchmark." },
  predictors: { label: "Compare return predictors", description: "Evaluate forecasts against subsequent observed returns. This measures prediction accuracy, not investment performance." },
  solvers: { label: "Benchmark portfolio optimization solvers", description: "Compare search algorithms on one shared portfolio problem. This is not a backtest over time." },
};
type Task = keyof typeof RUN_TASKS;

export default function NewRunPage({ defaultTask = "investment" }: { defaultTask?: Task }) {
  const [params, setParams] = useSearchParams();
  const campaignId = params.get("campaign") || null;
  const campaigns = useQuery({queryKey: ["campaigns"], queryFn: listCampaigns});
  const campaignName = campaigns.data?.find(c => c.id === campaignId)?.name ?? (campaignId ? "Unavailable campaign" : "Ungrouped");
  const requested = params.get("task") === "backtest" ? "strategy-comparison" : params.get("task");
  const task: Task = requested && Object.hasOwn(RUN_TASKS, requested) ? requested as Task : defaultTask;
  function choose(next: Task) {
    const search = new URLSearchParams(params);
    search.set("task", next);
    setParams(search);
  }
  return <div className="mx-auto max-w-5xl space-y-6">
    <div>
      <h1 className="text-2xl font-semibold text-gray-900">New Run</h1>
      <p className="mt-1 text-sm text-gray-500">Choose the task, configure its inputs, then review before launching.</p>
    </div>
    {task !== "investment" && <section className="rounded-lg border border-gray-200 bg-white p-5"><CampaignSelect value={campaignId ?? ""} onChange={id => {const next = new URLSearchParams(params); if (id) next.set("campaign", id); else next.delete("campaign"); setParams(next);}} label="Campaign (optional)" /></section>}
    <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-5" aria-label="Run purpose">
      <label className="block text-sm font-medium">What would you like to run?
        <select className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm" value={task} onChange={e => choose(e.target.value as Task)}>
          {Object.entries(RUN_TASKS).map(([key, value]) =>
            <option key={key} value={key}>{value.label}</option>)}
        </select>
      </label>
      <p className="text-sm text-gray-600">{RUN_TASKS[task].description}</p>
      <p className="text-xs text-gray-500">Each task keeps its draft while you switch within this page. Nothing runs until you review and launch.</p>
    </section>
    <StrategyExperimentForm campaignId={campaignId} campaignName={campaignName} mode="execution" active={task === "investment"} executionPurpose="investing" />
    <StrategyExperimentForm campaignId={campaignId} campaignName={campaignName} mode="strategy_comparison" active={task === "strategy-comparison"} />
    <ModelComparisonForm campaignId={campaignId} campaignName={campaignName} mode="prediction" active={task === "predictors"} />
    <ModelComparisonForm campaignId={campaignId} campaignName={campaignName} mode="solver" active={task === "solvers"} />
  </div>;
}
