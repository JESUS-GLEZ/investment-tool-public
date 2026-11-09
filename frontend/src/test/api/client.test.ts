import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  AxiosAdapter,
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosStatic,
  InternalAxiosRequestConfig,
} from "axios";
import { getAccessToken, getRefreshToken, setTokens } from "@/lib/tokens";

let axios: AxiosStatic;
let client: AxiosInstance;
let originalAdapter: AxiosRequestConfig["adapter"];
let adapter: ReturnType<typeof vi.fn<AxiosAdapter>>;

function response(config: InternalAxiosRequestConfig, data: unknown): AxiosResponse {
  return { config, data, headers: {}, status: 200, statusText: "OK" };
}

function unauthorized(config: InternalAxiosRequestConfig) {
  return new axios.AxiosError(
    "Unauthorized",
    axios.AxiosError.ERR_BAD_REQUEST,
    config,
    undefined,
    { ...response(config, {}), status: 401, statusText: "Unauthorized" },
  );
}

beforeEach(async () => {
  vi.resetModules();
  const values = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    removeItem: (key: string) => { values.delete(key); },
    clear: () => values.clear(),
  });
  axios = (await import("axios")).default;
  originalAdapter = axios.defaults.adapter;
  adapter = vi.fn<AxiosAdapter>();
  // Exercise real Axios transforms and interceptors without network requests.
  axios.defaults.adapter = adapter;
  client = (await import("@/api/client")).default;
});

afterEach(() => {
  axios.defaults.adapter = originalAdapter;
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe("authenticated Axios client", () => {
  it("preserves the API base, bearer token and JSON request body", async () => {
    setTokens("access-before", "refresh-before");
    adapter.mockImplementation(async (config) => response(config, { id: "run-1" }));

    const result = await client.post("/runs", { data_source: "local" });

    const config = adapter.mock.calls[0][0];
    expect(config.baseURL).toBe("/api");
    expect(config.url).toBe("/runs");
    expect(config.method).toBe("post");
    expect(config.headers.get("Authorization")).toBe("Bearer access-before");
    expect(config.headers.get("Content-Type")).toBe("application/json");
    expect(JSON.parse(config.data)).toEqual({ data_source: "local" });
    expect(result.data).toEqual({ id: "run-1" });
  });

  it("refreshes a rejected token and retries with the rotated credentials", async () => {
    setTokens("access-before", "refresh-before");
    const requests: Array<{ url?: string; authorization: unknown }> = [];
    adapter.mockImplementation(async (config) => {
      requests.push({ url: config.url, authorization: config.headers.get("Authorization") });
      if (config.url === "/api/auth/refresh") {
        expect(JSON.parse(config.data)).toEqual({ refresh_token: "refresh-before" });
        return response(config, { access_token: "access-after", refresh_token: "refresh-after" });
      }
      if (config.headers.get("Authorization") === "Bearer access-before") {
        throw unauthorized(config);
      }
      return response(config, [{ id: "run-1" }]);
    });

    const result = await client.get("/runs");

    expect(requests).toEqual([
      { url: "/runs", authorization: "Bearer access-before" },
      { url: "/api/auth/refresh", authorization: undefined },
      { url: "/runs", authorization: "Bearer access-after" },
    ]);
    expect(getAccessToken()).toBe("access-after");
    expect(getRefreshToken()).toBe("refresh-after");
    expect(result.data).toEqual([{ id: "run-1" }]);
  });

  it("shares one refresh request between concurrent unauthorized requests", async () => {
    setTokens("access-before", "refresh-before");
    let resolveRefresh!: (value: AxiosResponse) => void;
    let refreshConfig!: InternalAxiosRequestConfig;
    const pendingRefresh = new Promise<AxiosResponse>((resolve) => { resolveRefresh = resolve; });
    let refreshCount = 0;
    let authorizedCount = 0;
    adapter.mockImplementation(async (config) => {
      if (config.url === "/api/auth/refresh") {
        refreshCount += 1;
        refreshConfig = config;
        return pendingRefresh;
      }
      if (config.headers.get("Authorization") === "Bearer access-before") {
        throw unauthorized(config);
      }
      expect(config.headers.get("Authorization")).toBe("Bearer access-after");
      authorizedCount += 1;
      return response(config, { url: config.url });
    });

    const requests = Promise.all([client.get("/runs"), client.get("/defaults")]);
    await vi.waitFor(() => expect(refreshCount).toBe(1));
    resolveRefresh(response(refreshConfig, {
      access_token: "access-after", refresh_token: "refresh-after",
    }));
    const results = await requests;

    expect(refreshCount).toBe(1);
    expect(authorizedCount).toBe(2);
    expect(results.map((result) => result.data.url)).toEqual(["/runs", "/defaults"]);
    expect(getRefreshToken()).toBe("refresh-after");
  });

  it("rejects a second unauthorized response instead of refreshing repeatedly", async () => {
    setTokens("access-before", "refresh-before");
    let refreshCount = 0;
    let requestCount = 0;
    adapter.mockImplementation(async (config) => {
      if (config.url === "/api/auth/refresh") {
        refreshCount += 1;
        return response(config, { access_token: "access-after", refresh_token: "refresh-after" });
      }
      requestCount += 1;
      throw unauthorized(config);
    });

    await expect(client.get("/runs")).rejects.toMatchObject({ response: { status: 401 } });

    expect(refreshCount).toBe(1);
    expect(requestCount).toBe(2);
  });
});
