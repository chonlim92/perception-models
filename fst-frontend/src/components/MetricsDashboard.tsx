// [IMPLEMENTED BY CLAUDE - was missing]
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { MetricsSummary } from '../types';

interface Props {
  nodeId: string;
  metrics: MetricsSummary | undefined;
}

export function MetricsDashboard({ nodeId: _nodeId, metrics }: Props) {
  if (!metrics || !metrics.metrics || Object.keys(metrics.metrics).length === 0) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-sm font-semibold text-gray-500 uppercase mb-2">
          Metrics Dashboard
        </h3>
        <p className="text-sm text-gray-400">
          No evaluation data yet. Attach recordings and run evaluations.
        </p>
      </div>
    );
  }

  const chartData = Object.entries(metrics.metrics).map(([name, stats]) => ({
    name,
    mean: Number(stats.mean.toFixed(3)),
    min: Number(stats.min.toFixed(3)),
    max: Number(stats.max.toFixed(3)),
    std: Number(stats.std.toFixed(3)),
  }));

  const kpiThresholds = metrics.kpi_configs?.reduce(
    (acc, kpi) => ({ ...acc, [kpi.metric_name]: kpi.threshold }),
    {} as Record<string, number>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-900">Metrics Dashboard</h3>
        <span className="text-xs text-gray-500">
          {metrics.total_recordings} recordings
        </span>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(metrics.metrics).map(([name, stats]) => {
          const threshold = kpiThresholds?.[name];
          const isPassing = threshold ? stats.mean >= threshold : true;
          return (
            <div
              key={name}
              className={`p-3 rounded-lg border ${
                isPassing ? 'border-gray-200 bg-white' : 'border-red-200 bg-red-50'
              }`}
            >
              <div className="text-xs text-gray-500 uppercase truncate">{name}</div>
              <div className="text-lg font-bold text-gray-900">
                {stats.mean.toFixed(3)}
              </div>
              <div className="text-[10px] text-gray-400">
                &plusmn;{stats.std.toFixed(3)} | [{stats.min.toFixed(2)}, {stats.max.toFixed(2)}]
              </div>
              {threshold && (
                <div className="text-[10px] mt-1">
                  <span className={isPassing ? 'text-green-600' : 'text-red-600'}>
                    Threshold: {threshold}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Bar Chart */}
      {chartData.length > 0 && (
        <div className="h-48 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                formatter={(value: number) => value.toFixed(4)}
              />
              <Bar dataKey="mean" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              {Object.entries(kpiThresholds || {}).map(([name, threshold]) => (
                <ReferenceLine
                  key={name}
                  y={threshold}
                  stroke="#ef4444"
                  strokeDasharray="3 3"
                  label={{ value: `KPI`, fontSize: 9, fill: '#ef4444' }}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Failing Metrics Alert */}
      {metrics.failing_metrics && metrics.failing_metrics.length > 0 && (
        <div className="bg-red-50 border border-red-200 p-3 rounded-lg">
          <div className="text-sm font-medium text-red-800">KPI Failures</div>
          <ul className="mt-1 space-y-1">
            {metrics.failing_metrics.map((metric) => (
              <li key={metric} className="text-xs text-red-700">
                &bull; {metric} below threshold
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
