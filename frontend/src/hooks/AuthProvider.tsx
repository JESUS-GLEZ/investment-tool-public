import { useCallback, useMemo, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getAccessToken, setTokens, clearTokens } from "@/lib/tokens";
import * as api from "@/api/endpoints";
import { AuthContext } from "./auth-context";

/**
 * Auth provider — exposes auth state and actions to the app tree.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!getAccessToken());

  const login = useCallback(async (email: string, password: string) => {
    const tokens = await api.login(email, password);
    queryClient.clear();
    setTokens(tokens.access_token, tokens.refresh_token);
    setIsAuthenticated(true);
  }, [queryClient]);

  const register = useCallback(async (email: string, password: string) => {
    const tokens = await api.register(email, password);
    queryClient.clear();
    setTokens(tokens.access_token, tokens.refresh_token);
    setIsAuthenticated(true);
  }, [queryClient]);

  const logout = useCallback(() => {
    queryClient.clear();
    clearTokens();
    setIsAuthenticated(false);
  }, [queryClient]);

  const value = useMemo(
    () => ({ isAuthenticated, isLoading: false, login, register, logout }),
    [isAuthenticated, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
