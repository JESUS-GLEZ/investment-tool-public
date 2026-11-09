/**
 * Axios instance pre-configured with auth interceptors.
 *
 * - Adds `Authorization: Bearer <access_token>` to every request.
 * - On 401, attempts a token refresh and retries the original request once.
 */
import axios from "axios";
import {
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
} from "@/lib/tokens";
import type { TokenResponse } from "@/types/api";

const client = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

/* ---- Request interceptor: attach access token ---- */
client.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/* ---- Response interceptor: auto-refresh on 401 ---- */
let refreshPromise: Promise<TokenResponse> | null = null;

client.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config;
    if (
      error.response?.status === 401 &&
      !original._retry &&
      getRefreshToken()
    ) {
      original._retry = true;

      // Deduplicate concurrent refresh attempts
      if (!refreshPromise) {
        refreshPromise = axios
          .post<TokenResponse>("/api/auth/refresh", {
            refresh_token: getRefreshToken(),
          })
          .then((r) => r.data)
          .finally(() => {
            refreshPromise = null;
          });
      }

      try {
        const tokens = await refreshPromise;
        setTokens(tokens.access_token, tokens.refresh_token);
        original.headers.Authorization = `Bearer ${tokens.access_token}`;
        return client(original);
      } catch {
        clearTokens();
        window.location.href = "/login";
        return Promise.reject(error);
      }
    }
    return Promise.reject(error);
  },
);

export default client;
