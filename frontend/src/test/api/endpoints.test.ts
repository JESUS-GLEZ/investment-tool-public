import { describe, it, expect, vi, beforeEach } from "vitest";
import * as endpoints from "@/api/endpoints";

// Mock the axios client so no real HTTP requests are made
vi.mock("@/api/client", () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
  },
}));

import client from "@/api/client";

const mockPost = vi.mocked(client.post);
const mockGet = vi.mocked(client.get);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("auth endpoints", () => {
  it("login posts credentials and returns token", async () => {
    mockPost.mockResolvedValueOnce({ data: { access_token: "tok123" } });
    const result = await endpoints.login("user@test.com", "pass");
    expect(mockPost).toHaveBeenCalledWith("/auth/login", {
      email: "user@test.com",
      password: "pass",
    });
    expect(result.access_token).toBe("tok123");
  });

  it("register posts credentials and returns token", async () => {
    mockPost.mockResolvedValueOnce({ data: { access_token: "newtoken" } });
    const result = await endpoints.register("new@test.com", "secret");
    expect(mockPost).toHaveBeenCalledWith("/auth/register", {
      email: "new@test.com",
      password: "secret",
    });
    expect(result.access_token).toBe("newtoken");
  });
});

describe("runs endpoints", () => {
  it("deletes a run through the authenticated client", async () => {
    vi.mocked(client.delete).mockResolvedValueOnce({status: 204});
    await endpoints.deleteRun("aaaaaaaaaaaa");
    expect(client.delete).toHaveBeenCalledWith("/runs/aaaaaaaaaaaa");
  });
  it("listRuns calls GET /runs and returns array", async () => {
    const runs = [{ id: "abc", status: "completed", run_type: "execution" }];
    mockGet.mockResolvedValueOnce({ data: runs });
    const result = await endpoints.listRuns();
    expect(mockGet).toHaveBeenCalledWith("/runs");
    expect(result).toEqual(runs);
  });

  it("getRun calls GET /runs/:id", async () => {
    const run = { id: "abc123", status: "completed", result: {} };
    mockGet.mockResolvedValueOnce({ data: run });
    const result = await endpoints.getRun("abc123");
    expect(mockGet).toHaveBeenCalledWith("/runs/abc123");
    expect(result).toEqual(run);
  });

  it("createRun posts to /runs and returns summary", async () => {
    const summary = { id: "xyz", status: "queued", run_type: "execution" };
    mockPost.mockResolvedValueOnce({ data: summary });
    const body = { data_source: "local" } as Parameters<typeof endpoints.createRun>[0];
    const result = await endpoints.createRun(body);
    expect(mockPost).toHaveBeenCalledWith("/runs", body);
    expect(result).toEqual(summary);
  });
});

describe("comparison endpoints", () => {
  it("createStrategyComparison posts to /comparisons/strategy", async () => {
    mockPost.mockResolvedValueOnce({ data: { id: "s1", status: "queued" } });
    const body = {} as Parameters<typeof endpoints.createStrategyComparison>[0];
    await endpoints.createStrategyComparison(body);
    expect(mockPost).toHaveBeenCalledWith("/comparisons/strategy", body);
  });

  it("createSolverComparison posts to /comparisons/solver", async () => {
    mockPost.mockResolvedValueOnce({ data: { id: "s2", status: "queued" } });
    const body = {} as Parameters<typeof endpoints.createSolverComparison>[0];
    await endpoints.createSolverComparison(body);
    expect(mockPost).toHaveBeenCalledWith("/comparisons/solver", body);
  });

  it("createPredictionComparison posts to /comparisons/prediction", async () => {
    mockPost.mockResolvedValueOnce({ data: { id: "s3", status: "queued" } });
    const body = {} as Parameters<typeof endpoints.createPredictionComparison>[0];
    await endpoints.createPredictionComparison(body);
    expect(mockPost).toHaveBeenCalledWith("/comparisons/prediction", body);
  });
});

describe("reference data endpoints", () => {
  it("getDefaults calls GET /defaults", async () => {
    mockGet.mockResolvedValueOnce({ data: { frequency: "daily" } });
    await endpoints.getDefaults();
    expect(mockGet).toHaveBeenCalledWith("/defaults");
  });

  it("getSymbols calls GET /symbols", async () => {
    mockGet.mockResolvedValueOnce({ data: { symbols: ["AAPL"] } });
    await endpoints.getSymbols();
    expect(mockGet).toHaveBeenCalledWith("/symbols");
  });
});

describe("saved run manifest",()=>{
  it("reads only the owned manifest JSON endpoint",async()=>{
    mockGet.mockResolvedValueOnce({data:{format_version:1}});
    expect(await endpoints.getRunManifest("abc")).toEqual({format_version:1});
    expect(mockGet).toHaveBeenCalledWith("/runs/abc/files/run_manifest.json",{responseType:"json"});
  });
  it("treats a removed manifest as unknown but preserves permission and network errors",async()=>{
    mockGet.mockRejectedValueOnce({isAxiosError:true,response:{status:404}});
    expect(await endpoints.getRunManifest("abc")).toBeNull();
    for(const error of [{isAxiosError:true,response:{status:403}},new Error("network")]){
      mockGet.mockRejectedValueOnce(error);
      await expect(endpoints.getRunManifest("abc")).rejects.toBe(error);
    }
  });
});

it("requests the selected universe's symbols explicitly",async()=>{
  mockGet.mockResolvedValueOnce({data:{stocks:["EXAMPLE"],crypto:[]}});
  await endpoints.getSymbols("expanded.json");
  expect(mockGet).toHaveBeenCalledWith("/symbols",{params:{file:"expanded.json"}});
});
