/// <reference types="node" />
import {readFileSync} from "node:fs";
import {cleanup, fireEvent, render, screen, within} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {afterEach, expect, it, vi} from "vitest";
import ParameterHelp from "@/components/ui/ParameterHelp";
import {parameterHelp} from "@/lib/parameterHelp";
import PortfolioSelectionFields from "@/components/ui/PortfolioSelectionFields";
import ModelConfigurationDrawer from "@/components/ui/ModelConfigurationDrawer";
import {createModelVariant} from "@/lib/modelVariants";
afterEach(cleanup);
it("covers all exposed engine default fields without confusing prediction and covariance spans",()=>{
 const source=readFileSync("../src/investment_tool/defaults.py","utf8");
 const keys=[...source.matchAll(/^ {4}(\w+): /gm)].map(match=>match[1]);
 expect(keys.length).toBeGreaterThan(50);
 for(const key of keys)expect(parameterHelp(key),`Missing explanation: ${key}`).toBeTruthy();
 expect(parameterHelp("ewm_span","covariance")).toMatch(/separate from the predictor/);
 expect(parameterHelp("ewm_span","prediction")).toMatch(/not a hard lookback/);
 expect(parameterHelp("random_seed")).toMatch(/period label and restart index/);
 expect(parameterHelp("refinement_supports")).toMatch(/one shared cap/);
 expect(parameterHelp("risk_free_rate")).toMatch(/does not change its ranking/);
 expect(parameterHelp("selection")).toMatch(/zero risk-free baseline/);
 expect(parameterHelp("population_size")).toBe(parameterHelp("pop_size"));
});
it("opens a compact help card by keyboard without submitting, and returns focus on close",async()=>{
 const user=userEvent.setup(),submit=vi.fn(e=>e.preventDefault());
 render(<form onSubmit={submit}><ParameterHelp name="transaction_cost_rate" label="Trading cost rate"/><button>Launch</button></form>);
 const trigger=screen.getByRole("button",{name:"Explain Trading cost rate"});
 expect(screen.queryByRole("dialog")).not.toBeInTheDocument();expect(trigger).toHaveAttribute("aria-expanded","false");
 await user.tab();expect(trigger).toHaveFocus();await user.keyboard("{Enter}");
 const card=screen.getByRole("dialog",{name:"Trading cost rate"});
 expect(card).toHaveAttribute("aria-modal","true");expect(trigger).toHaveAttribute("aria-expanded","true");
 expect(within(card).getByText(/Trading \$200 costs \$0.20/)).toBeVisible();expect(submit).not.toHaveBeenCalled();
 await user.click(within(card).getByRole("button",{name:"Close parameter help"}));
 expect(screen.queryByRole("dialog")).not.toBeInTheDocument();expect(trigger).toHaveFocus();
 await user.keyboard(" ");fireEvent(screen.getByRole("dialog"),new Event("cancel",{bubbles:true,cancelable:true}));
 expect(screen.queryByRole("dialog")).not.toBeInTheDocument();expect(trigger).toHaveFocus();expect(submit).not.toHaveBeenCalled();
});
it("explains the return-target fraction without changing the selected rule or input",()=>{
 const change=vi.fn();render(<PortfolioSelectionFields prefix="selection" method="return_target" fraction={0.75} increase={0.2} onMethod={change} onFraction={change} onIncrease={change}/>);
 fireEvent.click(screen.getByLabelText("Explain Position in feasible return range"));
 expect(screen.getByText(/threshold is 4%/)).toBeVisible();expect(screen.getByText(/neither a 75% return forecast/)).toBeVisible();
 expect(screen.getByRole("spinbutton")).toHaveValue(75);expect(change).not.toHaveBeenCalled();
});
it("dismisses help inside a predictor drawer without closing or saving the configuration",async()=>{
 const user=userEvent.setup(),save=vi.fn(),close=vi.fn();
 const value=createModelVariant("FuzzyThesis",[],false);
 render(<ModelConfigurationDrawer value={value} others={[]} frequency="monthly" shared={{population_size:300,n_generations:6000,n_restarts:3}} focusFallbackId="unused" onSave={save} onClose={close}/>);
 const drawer=screen.getByRole("dialog");
 const trigger=within(drawer).getByRole("button",{name:"Explain Gaussian width (sigma)"});
 await user.click(trigger);
 const card=screen.getByRole("dialog",{name:"Gaussian width (sigma)"});
 expect(within(card).getByText(/percentage-point return units/)).toBeVisible();
 fireEvent(card,new Event("cancel",{bubbles:true,cancelable:true}));
 expect(card).not.toBeInTheDocument();expect(drawer).toBeVisible();expect(trigger).toHaveFocus();
 expect(close).not.toHaveBeenCalled();expect(save).not.toHaveBeenCalled();
});
it("keeps both epsilon labels associated with their own numeric input",()=>{
 const value={...createModelVariant("EpsNSGAII",[],true),epsilons:[0.001,0.002]};
 render(<ModelConfigurationDrawer value={value} others={[]} frequency="monthly" shared={{population_size:300,n_generations:6000,n_restarts:3}} focusFallbackId="unused" onSave={vi.fn()} onClose={vi.fn()}/>);
 const variance=screen.getByLabelText("Variance epsilon"),ret=screen.getByLabelText("Negative-return epsilon");
 expect(variance).toHaveValue(0.001);expect(ret).toHaveValue(0.002);expect(variance.id).not.toBe(ret.id);
});
