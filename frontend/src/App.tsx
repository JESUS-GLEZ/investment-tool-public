import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "@/hooks";
import { AppShell, ProtectedRoute } from "@/components/layout";
import LoginPage from "@/pages/LoginPage";
import RegisterPage from "@/pages/RegisterPage";
import DashboardPage from "@/pages/DashboardPage";
import NewRunPage from "@/pages/NewRunPage";
import RunDetailPage from "@/pages/RunDetailPage";
import CompareResultsPage from "@/pages/CompareResultsPage";
import DataLibraryPage from "@/pages/DataLibraryPage";
import SettingsPage from "@/pages/SettingsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 10_000,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            {/* Authenticated routes */}
            <Route element={<ProtectedRoute />}>
              <Route element={<AppShell />}>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/runs/new" element={<NewRunPage />} />
                <Route path="/runs/:runId" element={<RunDetailPage />} />
                <Route path="/comparisons/new" element={<Navigate to="/runs/new?task=strategy-comparison" replace />} />
                <Route path="/results/compare" element={<CompareResultsPage />} />
                <Route path="/datasets" element={<DataLibraryPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Route>
            </Route>

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
