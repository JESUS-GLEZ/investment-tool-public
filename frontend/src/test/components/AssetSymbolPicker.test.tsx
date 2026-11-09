import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it } from "vitest";
import AssetSymbolPicker from "@/components/ui/AssetSymbolPicker";
import CurrentHoldingsFields from "@/components/ui/CurrentHoldingsFields";
import type { HoldingInput } from "@/types/api";
function Picker({symbols,initial = ""}: {symbols:string[];initial?:string}) {
  const [value,setValue] = useState(initial);
  return <AssetSymbolPicker id="symbol" label="Asset" value={value} symbols={symbols} onChange={setValue} />;
}
afterEach(cleanup);
it("sorts matches alphabetically and searches case-insensitively with keyboard selection", () => {
  render(<Picker symbols={["ZZZ","NVDA","AAPL","MSFT","BTC/USD"]} />);
  const input=screen.getByRole("combobox",{name:"Asset"});
  fireEvent.focus(input);
  expect(screen.getAllByRole("option").map(o=>o.textContent)).toEqual(["AAPL","BTC/USD","MSFT","NVDA","ZZZ"]);
  fireEvent.change(input,{target:{value:"nv"}});
  expect(screen.getAllByRole("option").map(o=>o.textContent)).toEqual(["NVDA"]);
  fireEvent.keyDown(input,{key:"Enter"});
  expect(input).toHaveValue("NVDA"); expect(input).toHaveAttribute("aria-expanded","false");
});
it("finds the last symbol in a thousand-asset universe and permits browsing every alphabetical result", () => {
  render(<Picker symbols={Array.from({length:1000},(_,i)=>`SYM${String(i).padStart(4,"0")}`).reverse()} />);
  const input=screen.getByRole("combobox",{name:"Asset"}); fireEvent.focus(input);
  expect(screen.getAllByRole("option")).toHaveLength(1000);
  expect(screen.getByRole("status")).toHaveTextContent("1000 alphabetical matches");
  fireEvent.change(input,{target:{value:"0999"}});
  fireEvent.click(screen.getByRole("option",{name:"SYM0999"}));
  expect(input).toHaveValue("SYM0999");
});
it("supports Escape and never offers a symbol already selected in another holding", () => {
  function Holdings() {
    const [value,setValue] = useState<HoldingInput[]>([{symbol:"AAPL",quantity:2},{symbol:"",quantity:1}]);
    return <CurrentHoldingsFields value={value} symbols={["NVDA","MSFT","AAPL"]} loading={false} failed={false} onRetry={()=>{}} onChange={setValue} />;
  }
  render(<Holdings />);
  const input=screen.getByRole("combobox",{name:"Asset 2"}); fireEvent.focus(input);
  expect(within(screen.getByRole("listbox")).queryByRole("option",{name:"AAPL"})).not.toBeInTheDocument();
  fireEvent.keyDown(input,{key:"ArrowDown"}); fireEvent.keyDown(input,{key:"Enter"});
  expect(input).toHaveValue("NVDA");
  fireEvent.focus(input); fireEvent.keyDown(input,{key:"Escape"});
  expect(input).toHaveAttribute("aria-expanded","false");
});

it("keeps edits separate, suppresses premature errors, and restores the holding on Escape", () => {
  render(<Picker initial="NVDA" symbols={["NVDA","MSFT","AAPL"]} />);
  const input = screen.getByRole("combobox",{name:"Asset"});
  fireEvent.focus(input);
  expect((input as HTMLInputElement).selectionStart).toBe(0);
  expect((input as HTMLInputElement).selectionEnd).toBe(4);
  expect(screen.getAllByRole("option")).toHaveLength(3);
  fireEvent.keyDown(input,{key:"Enter"});
  expect(input).toHaveValue("NVDA");
  fireEvent.focus(input); fireEvent.change(input,{target:{value:"MS"}});
  expect(screen.queryByText("Choose an eligible symbol from this universe.")).not.toBeInTheDocument();
  fireEvent.keyDown(input,{key:"Escape"}); fireEvent.blur(input);
  expect(input).toHaveValue("NVDA");
  fireEvent.focus(input); fireEvent.change(input,{target:{value:"MSFT"}}); fireEvent.blur(input);
  expect(input).toHaveValue("MSFT"); expect(input).toHaveAttribute("aria-invalid","false");
});
it("commits an invalid draft on blur so it cannot silently submit the previous holding", () => {
  render(<Picker initial="NVDA" symbols={["NVDA","MSFT"]} />);
  const input=screen.getByRole("combobox",{name:"Asset"});
  fireEvent.focus(input); fireEvent.change(input,{target:{value:"BOGUS"}}); fireEvent.blur(input);
  expect(input).toHaveValue("BOGUS"); expect(input).toHaveAttribute("aria-invalid","true");
  expect(input).toHaveAccessibleDescription("Choose an eligible symbol from this universe.");
});

it("does not change the holding when its existing symbol is retyped and confirmed", () => {
  render(<Picker initial="NVDA" symbols={["NVDA","MSFT","AAPL"]} />);
  const input=screen.getByRole("combobox",{name:"Asset"});
  fireEvent.focus(input); fireEvent.change(input,{target:{value:"N"}});
  fireEvent.change(input,{target:{value:"NVDA"}});
  expect(screen.getAllByRole("option").map(e=>e.textContent)).toEqual(["NVDA"]);
  fireEvent.keyDown(input,{key:"Enter"});
  expect(input).toHaveValue("NVDA");
});
