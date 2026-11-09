/**
 * Shell layout — top nav + content area.
 * Used for all authenticated pages.
 */
import { Link, Outlet, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "@/hooks";

export default function AppShell() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const returnSearch: unknown = location.state?.dashboardSearch;
  // Carry library state through a detail visit without persisting account data.
  const dashboardTarget = {
    pathname: "/",
    search: location.pathname === "/" ? location.search
      : typeof returnSearch === "string" && returnSearch.startsWith("?") ? returnSearch : "",
  };

  const compareSearch: unknown = location.state?.comparisonSearch;
  const compareTarget = {pathname: "/results/compare", search: location.pathname === "/results/compare" ? location.search : typeof compareSearch === "string" && compareSearch.startsWith("?") ? compareSearch : ""};

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* ---- Top nav ---- */}
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex min-h-14 max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 lg:h-14 lg:flex-nowrap lg:py-0">
          <div className="flex items-center gap-6">
            <Link to={dashboardTarget} className="text-lg font-semibold text-gray-900">
              Investment Tool
            </Link>
            <nav className="hidden items-center gap-4 lg:flex">
              <Link
                to={dashboardTarget}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Dashboard
              </Link>
              <Link to={compareTarget} className="text-sm text-gray-600 hover:text-gray-900">Compare Runs</Link>
              <Link
                to="/runs/new"
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                New Run
              </Link>
              <Link to="/datasets" className="text-sm text-gray-600 hover:text-gray-900">Data library</Link>
              <Link
                to="/settings"
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Settings
              </Link>
            </nav>
          </div>

          <button
            onClick={handleLogout}
            className="text-sm text-gray-500 hover:text-gray-700 lg:ml-4"
          >
            Log out
          </button>
          <nav className="flex w-full items-center gap-4 overflow-x-auto text-sm lg:hidden">
            <Link to={dashboardTarget} className="whitespace-nowrap text-gray-600 hover:text-gray-900">
              Dashboard
            </Link>
            <Link to={compareTarget} className="whitespace-nowrap text-gray-600 hover:text-gray-900">Compare Runs</Link>
            <Link to="/runs/new" className="whitespace-nowrap text-gray-600 hover:text-gray-900">
              New Run
            </Link>
            <Link to="/datasets" className="whitespace-nowrap text-gray-600 hover:text-gray-900">Data library</Link>
            <Link to="/settings" className="whitespace-nowrap text-gray-600 hover:text-gray-900">
              Settings
            </Link>
          </nav>
        </div>
      </header>

      {/* ---- Content ---- */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
