import {useState} from "react";
import {cleanup, fireEvent, render, screen, waitFor} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";
import {MemoryRouter} from "react-router-dom";
import {afterEach, beforeEach, expect, it, vi} from "vitest";
import * as api from "@/api/endpoints";
import {CampaignPanel, CampaignSelect, RunCampaignAssignment} from "@/components/ui/Campaigns";
import CompareResultsPage from "@/pages/CompareResultsPage";
import DashboardPage from "@/pages/DashboardPage";
import type {Campaign, RunSummary} from "@/types/api";
import {backtest} from "../fixtures/savedRuns";
vi.mock("@/api/endpoints",()=>({listCampaigns:vi.fn(),saveCampaign:vi.fn(),assignRunCampaign:vi.fn(),listRuns:vi.fn(),getRun:vi.fn(),getRunManifest:vi.fn()}));
vi.mock("@/hooks",()=>({useAuth:()=>({logout:vi.fn()})}));
const campaign:Campaign={id:"aaaaaaaaaaaa",name:"Annual validation",hypothesis:"Does the signal persist?",notes:"Compare years",created_at:"2026-10-01T00:00:00Z",updated_at:"2026-10-01T00:00:00Z",run_count:2};
function show(element:React.ReactNode,url="/") {
 const client=new QueryClient({defaultOptions:{queries:{retry:false},mutations:{retry:false}}});
 render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[url]}>{element}</MemoryRouter></QueryClientProvider>);return client;
}
function Panel(){const [value,setValue]=useState("all");return <CampaignPanel value={value} onChange={setValue}/>;}
beforeEach(()=>{vi.resetAllMocks();vi.mocked(api.listCampaigns).mockResolvedValue([campaign]);});
afterEach(cleanup);
it("creates and edits one campaign with a hypothesis and notes, without launching runs",async()=>{
 const user=userEvent.setup();show(<Panel/>);
 await user.click(screen.getByRole("button",{name:"New campaign"}));
 await user.type(screen.getByLabelText("Campaign name"),"Annual validation");
 await user.type(screen.getByLabelText("Hypothesis or objective"),campaign.hypothesis);
 await user.type(screen.getByLabelText("Notes and conclusions"),campaign.notes);
 vi.mocked(api.saveCampaign).mockRejectedValueOnce(new Error("Try again")).mockResolvedValue(campaign);
 await user.click(screen.getByRole("button",{name:"Save campaign"}));
 expect(await screen.findByRole("alert")).toHaveTextContent("Try again");
 expect(screen.getByLabelText("Campaign name")).toHaveValue(campaign.name);
 await user.click(screen.getByRole("button",{name:"Save campaign"}));
 expect(await screen.findByRole("link",{name:"New Run in this campaign"})).toHaveAttribute("href",`/runs/new?campaign=${campaign.id}`);
 expect(screen.getByRole("link",{name:"Compare campaign runs"})).toHaveAttribute("href",`/results/compare?campaign=${campaign.id}`);
 expect(screen.getByRole("combobox",{name:"Filter by campaign"})).toHaveDisplayValue("Annual validation (2)");
 await user.click(screen.getByText("Hypothesis and notes"));
 expect(screen.getByText(campaign.hypothesis)).toBeVisible();
 await user.click(screen.getByRole("button",{name:"Edit campaign"}));
 await user.clear(screen.getByLabelText("Notes and conclusions"));await user.type(screen.getByLabelText("Notes and conclusions"),"Conclusion");
 await user.click(screen.getByRole("button",{name:"Save campaign"}));
 await waitFor(()=>expect(api.saveCampaign).toHaveBeenLastCalledWith({name:campaign.name,hypothesis:campaign.hypothesis,notes:"Conclusion"},campaign.id));
 expect(api.getRun).not.toHaveBeenCalled();
});
it("preserves a failed assignment, then assigns and removes membership only",async()=>{
 const run={...backtest("cccccccccccc"),campaign_id:campaign.id};
 const client=show(<RunCampaignAssignment run={run}/>);
 const select=await screen.findByRole("combobox",{name:`Campaign for ${run.id}`});await waitFor(()=>expect(select).toBeEnabled());
 vi.mocked(api.assignRunCampaign).mockRejectedValueOnce(new Error("Assignment unavailable"));
 fireEvent.change(select,{target:{value:""}});
 expect(await screen.findByRole("alert")).toHaveTextContent("Assignment unavailable");expect(select).toHaveValue(campaign.id);
 vi.mocked(api.assignRunCampaign).mockResolvedValue({...run,campaign_id:null} as RunSummary);
 const invalidate=vi.spyOn(client,"invalidateQueries");fireEvent.change(select,{target:{value:""}});
 await waitFor(()=>expect(invalidate).toHaveBeenCalledWith({queryKey:["run",run.id]}));
 expect(api.assignRunCampaign).toHaveBeenLastCalledWith(run.id,null);expect(api.getRun).not.toHaveBeenCalled();
});
it("recovers the campaign catalog while retaining an unavailable selection",async()=>{
 vi.mocked(api.listCampaigns).mockRejectedValueOnce(new Error("offline"));const change=vi.fn();
 show(<CampaignSelect value={campaign.id} onChange={change}/>);
 await screen.findByRole("alert");expect(screen.getByRole("combobox")).toHaveDisplayValue("Unavailable campaign");
 vi.mocked(api.listCampaigns).mockResolvedValue([campaign]);fireEvent.click(screen.getByRole("button",{name:"Retry campaigns"}));
 await waitFor(()=>expect(screen.getByRole("combobox")).toHaveDisplayValue(campaign.name));expect(change).not.toHaveBeenCalled();
});
it("filters the dashboard by campaign and ungrouped without loading result payloads",async()=>{
 const grouped={...backtest("cccccccccccc"),campaign_id:campaign.id};const ungrouped={...backtest("dddddddddddd"),campaign_id:null};
 vi.mocked(api.listRuns).mockResolvedValue([grouped,ungrouped]);show(<DashboardPage/>);
 const filter=await screen.findByRole("combobox",{name:"Filter by campaign"});await waitFor(()=>expect(filter).toBeEnabled());
 fireEvent.change(filter,{target:{value:campaign.id}});
 await waitFor(()=>expect(screen.queryByText(ungrouped.id)).not.toBeInTheDocument());expect(screen.getByText(grouped.id)).toBeVisible();
 fireEvent.change(filter,{target:{value:"ungrouped"}});
 await waitFor(()=>expect(screen.queryByText(grouped.id)).not.toBeInTheDocument());expect(screen.getByText(ungrouped.id)).toBeVisible();
 expect(api.getRun).not.toHaveBeenCalled();
});

it("campaign filtering keeps existing result selections but clears hidden bulk selections",async()=>{
 const grouped={...backtest("cccccccccccc"),campaign_id:campaign.id},other={...backtest("dddddddddddd"),campaign_id:null};
 vi.mocked(api.listRuns).mockResolvedValue([grouped,other]);vi.mocked(api.getRun).mockResolvedValue(other);
 show(<CompareResultsPage/>,"/results/compare?item=dddddddddddd%3ASharpe");
 const filter=await screen.findByRole("combobox",{name:"Filter runs by campaign"});await waitFor(()=>expect(filter).toBeEnabled());
 await screen.findByRole("region",{name:"Backtest results"});
 fireEvent.click(screen.getByRole("checkbox",{name:`Select run ${other.id}`}));
 fireEvent.change(filter,{target:{value:campaign.id}});
 await waitFor(()=>expect(screen.queryByRole("button",{name:`Inspect ${other.id}`})).not.toBeInTheDocument());
 expect(screen.getByRole("button",{name:`Inspect ${grouped.id}`})).toBeVisible();
 expect(screen.getByRole("region",{name:"Backtest results"})).toBeInTheDocument();
 expect(api.getRun).toHaveBeenCalledTimes(1);
});
