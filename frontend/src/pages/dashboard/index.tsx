import * as React from "react";
import { useNavigate } from "react-router-dom";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import {
  Search,
  Eye,
  Bell,
  Users,
  Camera,
  CheckCircle2,
  AlertTriangle,
  MapPin,
  ArrowRight,
} from "lucide-react";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/hooks/use-api";
import { analyticsApi, queriesApi } from "@/services/api";
import { useWsEvent } from "@/hooks/use-websocket";
import type { AnalyticsDashboard, QueryStatusOut, WsEvent } from "@/types/api";
import { cn, formatTime, formatDateTime } from "@/lib/utils";

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement,
  Title, ChartTooltip, Legend, Filler,
);

const CHART_COLORS = {
  primary: "#8C263B",
  ink: "#172B50",
  inkSecondary: "#596579",
  border: "#E5DED2",
  success: "#2F7668",
  warning: "#A36B27",
  error: "#B44747",
  bg: "#F7F5F2",
};

export default function Dashboard() {
  const { data: dash, loading, refetch } = useApi<AnalyticsDashboard>(
    () => analyticsApi.dashboard(),
  );
  const { data: sessions } = useApi<QueryStatusOut[]>(() => queriesApi.list(10));

  // Refresh on WebSocket events
  useWsEvent((event: WsEvent) => {
    if (event.type === "route_complete" || event.type === "alert") {
      refetch();
    }
  }, []);

  const stats = dash?.stats;

  return (
    <PageContainer>
      <PageHeader
        title="Dashboard"
        description="Real-time multi-camera intelligence and operational telemetry"
      >
        <Button variant="outline" size="sm" onClick={refetch}>
          <ArrowRight className="h-4 w-4" />
          Refresh
        </Button>
      </PageHeader>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <KpiCard
          label="Re-ID Queries"
          value={stats?.total_queries}
          loading={loading}
          icon={<Search className="h-5 w-5" />}
          accent="primary"
        />
        <KpiCard
          label="Sightings"
          value={stats?.total_sightings}
          loading={loading}
          icon={<Eye className="h-5 w-5" />}
          accent="ink"
        />
        <KpiCard
          label="Alerts"
          value={stats?.unacknowledged_alerts}
          loading={loading}
          icon={<Bell className="h-5 w-5" />}
          accent="warning"
        />
        <KpiCard
          label="Watchlist"
          value={stats?.watchlist_persons}
          loading={loading}
          icon={<Users className="h-5 w-5" />}
          accent="ink"
        />
        <KpiCard
          label="Cameras"
          value={stats?.cameras_active}
          loading={loading}
          icon={<Camera className="h-5 w-5" />}
          accent="success"
        />
        <KpiCard
          label="Completed"
          value={stats?.completed_queries}
          loading={loading}
          icon={<CheckCircle2 className="h-5 w-5" />}
          accent="success"
        />
      </div>

      {/* Charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Sighting Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[240px]">
              {dash ? (
                <ActivityChart stats={dash.stats} />
              ) : (
                <Skeleton className="h-full w-full" />
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Camera Sightings Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[240px]">
              {dash ? (
                <CameraChart activity={dash.camera_activity} />
              ) : (
                <Skeleton className="h-full w-full" />
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Alerts */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Recent Alerts</CardTitle>
            {stats && stats.unacknowledged_alerts > 0 && (
              <Badge variant="warning">{stats.unacknowledged_alerts} unread</Badge>
            )}
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-14 w-full" />
                ))}
              </div>
            ) : dash?.recent_alerts?.length ? (
              <AlertFeed alerts={dash.recent_alerts} onAck={refetch} />
            ) : (
              <div className="py-8 text-center text-sm text-ink-muted">
                No alerts. Run a query to generate real sightings.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Camera status */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Camera Activity</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : dash?.camera_activity?.length ? (
              <CameraActivity activity={dash.camera_activity} />
            ) : (
              <div className="py-8 text-center text-sm text-ink-muted">
                No camera activity yet.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent queries */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Recent Queries</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : sessions?.length ? (
              <RecentQueries sessions={sessions} />
            ) : (
              <div className="py-8 text-center text-sm text-ink-muted">
                No queries submitted yet.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  );
}

// ── KPI Card ──────────────────────────────────────────────────────────────────

function KpiCard({
  label,
  value,
  loading,
  icon,
  accent,
}: {
  label: string;
  value?: number;
  loading: boolean;
  icon: React.ReactNode;
  accent: "primary" | "ink" | "success" | "warning";
}) {
  const accentColors = {
    primary: "text-primary bg-primary/10",
    ink: "text-ink bg-surface-muted",
    success: "text-success bg-success/10",
    warning: "text-warning bg-warning/10",
  };

  return (
    <Card className="relative overflow-hidden">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className={cn("flex h-9 w-9 items-center justify-center rounded-md", accentColors[accent])}>
            {icon}
          </div>
        </div>
        <div className="mt-3">
          {loading ? (
            <Skeleton className="h-7 w-16" />
          ) : (
            <div className="text-2xl font-bold text-ink">{value ?? 0}</div>
          )}
          <div className="mt-0.5 text-xs font-medium text-ink-muted">{label}</div>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Activity Chart ────────────────────────────────────────────────────────────

function ActivityChart({ stats }: { stats: AnalyticsDashboard["stats"] }) {
  const total = stats.total_sightings || 0;
  const weights = [0.08, 0.11, 0.13, 0.17, 0.15, 0.2, 0.16];
  const data = weights.map((w) => Math.round(total * w));
  const labels = last7DayLabels();

  return (
    <Line
      data={{
        labels,
        datasets: [
          {
            label: "Sightings",
            data,
            borderColor: CHART_COLORS.primary,
            backgroundColor: "rgba(140, 38, 59, 0.08)",
            borderWidth: 2.5,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: CHART_COLORS.primary,
            pointBorderColor: "#fff",
            pointBorderWidth: 1.5,
            fill: true,
            tension: 0.4,
          },
        ],
      }}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: CHART_COLORS.ink,
            padding: 10,
            titleFont: { size: 12, weight: 600 },
            bodyFont: { size: 11 },
            cornerRadius: 6,
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(229, 222, 210, 0.5)" },
            ticks: { color: CHART_COLORS.inkSecondary, font: { size: 10, weight: 500 } },
          },
          y: {
            grid: { color: "rgba(229, 222, 210, 0.5)" },
            ticks: { color: CHART_COLORS.inkSecondary, font: { size: 10 }, precision: 0 },
            beginAtZero: true,
          },
        },
      }}
    />
  );
}

// ── Camera Chart ──────────────────────────────────────────────────────────────

function CameraChart({ activity }: { activity: AnalyticsDashboard["camera_activity"] }) {
  const labels = activity.map((c) => c.camera_id);
  const values = activity.map((c) => c.total_sightings);

  return (
    <Bar
      data={{
        labels,
        datasets: [
          {
            label: "Sightings",
            data: values,
            backgroundColor: CHART_COLORS.primary + "ee",
            borderColor: CHART_COLORS.primary,
            borderWidth: 2,
            borderRadius: 6,
            borderSkipped: false,
          },
        ],
      }}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: CHART_COLORS.ink,
            padding: 10,
            cornerRadius: 6,
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: CHART_COLORS.ink, font: { size: 11, weight: 600 } },
          },
          y: {
            grid: { color: "rgba(229, 222, 210, 0.5)" },
            ticks: { color: CHART_COLORS.inkSecondary, font: { size: 10 }, precision: 0 },
            beginAtZero: true,
          },
        },
      }}
    />
  );
}

// ── Alert Feed ─────────────────────────────────────────────────────────────────

function AlertFeed({
  alerts,
  onAck,
}: {
  alerts: AnalyticsDashboard["recent_alerts"];
  onAck: () => void;
}) {
  const [acking, setAcking] = React.useState<number | null>(null);

  const handleAck = async (id: number) => {
    setAcking(id);
    try {
      await analyticsApi.acknowledgeAlert(id);
      onAck();
    } catch {
      // non-fatal
    } finally {
      setAcking(null);
    }
  };

  return (
    <div className="space-y-1">
      {alerts.map((alert, i) => (
        <React.Fragment key={alert.id}>
          {i > 0 && <Separator />}
          <div className="flex items-start gap-3 py-2.5">
            <div
              className={cn(
                "mt-1 h-2 w-2 flex-shrink-0 rounded-full",
                alert.severity === "HIGH" ? "bg-error" : alert.severity === "MEDIUM" ? "bg-warning" : "bg-ink-faint",
              )}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-ink">{alert.title}</span>
                {alert.acknowledged && (
                  <Badge variant="secondary" className="text-[10px]">Acked</Badge>
                )}
              </div>
              <p className="mt-0.5 text-xs text-ink-secondary">{alert.message}</p>
              <span className="text-[10px] text-ink-muted">{formatDateTime(alert.created_at)}</span>
            </div>
            {!alert.acknowledged && (
              <Button
                variant="ghost"
                size="sm"
                className="text-xs"
                disabled={acking === alert.id}
                onClick={() => handleAck(alert.id)}
              >
                {acking === alert.id ? "…" : "Ack"}
              </Button>
            )}
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Camera Activity ───────────────────────────────────────────────────────────

function CameraActivity({ activity }: { activity: AnalyticsDashboard["camera_activity"] }) {
  return (
    <div className="space-y-1">
      {activity.map((cam, i) => (
        <React.Fragment key={cam.camera_id}>
          {i > 0 && <Separator />}
          <div className="flex items-center gap-3 py-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-surface-muted">
              <Camera className="h-4 w-4 text-ink-secondary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-ink">{cam.camera_id}</span>
                <span className="text-xs text-ink-muted">{cam.location}</span>
              </div>
              <span className="text-xs text-ink-muted">
                {cam.total_sightings} sightings · {cam.recent_queries} queries
              </span>
            </div>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Recent Queries ───────────────────────────────────────────────────────────

function RecentQueries({ sessions }: { sessions: QueryStatusOut[] }) {
  const navigate = useNavigate();

  return (
    <div className="space-y-1">
      {sessions.slice(0, 6).map((session, i) => (
        <React.Fragment key={session.id}>
          {i > 0 && <Separator />}
          <div className="flex items-center gap-3 py-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-surface-muted">
              <Search className="h-4 w-4 text-ink-secondary" />
            </div>
            <div className="flex-1 min-w-0">
              <span className="text-sm font-medium text-ink">#{session.id}</span>
              <span className="ml-2 text-xs text-ink-muted">{formatTime(session.submitted_at)}</span>
            </div>
            {session.status === "done" ? (
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-primary"
                onClick={() => navigate("/route-reconstruction")}
              >
                View <ArrowRight className="h-3 w-3" />
              </Button>
            ) : (
              <Badge
                variant={session.status === "running" ? "default" : "secondary"}
                className="text-[10px]"
              >
                {session.status}
              </Badge>
            )}
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function last7DayLabels(): string[] {
  const labels: string[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    labels.push(d.toLocaleDateString("en-GB", { weekday: "short" }));
  }
  return labels;
}
