// [IMPLEMENTED BY CLAUDE - was missing]
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { useTreeStore } from '../store/useTreeStore';
import { MetricsDashboard } from './MetricsDashboard';
import { SuggestionsPanel } from './SuggestionsPanel';
import { RecordingsPanel } from './RecordingsPanel';
import type { TreeNode } from '../types';

interface Props {
  nodeId: string;
}

function findNode(tree: TreeNode | null, id: string): TreeNode | null {
  if (!tree) return null;
  if (tree.id === id) return tree;
  for (const child of tree.children) {
    const found = findNode(child, id);
    if (found) return found;
  }
  return null;
}

export function NodeDetailPanel({ nodeId }: Props) {
  const { tree, selectNode } = useTreeStore();
  const queryClient = useQueryClient();
  const node = findNode(tree, nodeId);

  const { data: metrics } = useQuery({
    queryKey: ['metrics', nodeId],
    queryFn: () => api.getNodeMetrics(nodeId),
  });

  const { data: suggestions } = useQuery({
    queryKey: ['suggestions', nodeId],
    queryFn: () => api.getSuggestions(nodeId),
  });

  const analyzeMutation = useMutation({
    mutationFn: () => api.triggerAnalysis(nodeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['suggestions', nodeId] });
    },
  });

  if (!node) {
    return (
      <div className="p-4 text-gray-500">Node not found</div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      {/* Node Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-bold text-gray-900">{node.name}</h2>
          <p className="text-sm text-gray-500 font-mono">{node.id}</p>
          <p className="text-sm text-gray-600 mt-1">{node.description}</p>
        </div>
        <button
          onClick={() => selectNode(null)}
          className="text-gray-400 hover:text-gray-600"
        >
          &times;
        </button>
      </div>

      {/* Layer Badge */}
      <div className="flex gap-2">
        <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded">
          Layer {node.layer}
        </span>
        {metrics?.kpi_status && (
          <span
            className={`px-2 py-0.5 text-xs rounded ${
              metrics.kpi_status === 'pass'
                ? 'bg-green-100 text-green-800'
                : metrics.kpi_status === 'warn'
                ? 'bg-yellow-100 text-yellow-800'
                : 'bg-red-100 text-red-800'
            }`}
          >
            KPI: {metrics.kpi_status.toUpperCase()}
          </span>
        )}
      </div>

      {/* Detection Method */}
      {node.detection_method && (
        <div className="bg-gray-50 p-3 rounded-lg">
          <h3 className="text-xs font-semibold text-gray-500 uppercase mb-1">
            Detection Method
          </h3>
          <p className="text-sm text-gray-700">{node.detection_method}</p>
        </div>
      )}

      {/* Metrics Dashboard */}
      <MetricsDashboard nodeId={nodeId} metrics={metrics} />

      {/* Recordings */}
      <RecordingsPanel nodeId={nodeId} />

      {/* Root Cause Analysis */}
      <div className="border-t pt-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-900">
            Root Cause Analysis
          </h3>
          <button
            onClick={() => analyzeMutation.mutate()}
            disabled={analyzeMutation.isPending}
            className="px-3 py-1 bg-purple-600 text-white text-xs rounded hover:bg-purple-700 disabled:opacity-50"
          >
            {analyzeMutation.isPending ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
        {suggestions && suggestions.length > 0 && (
          <SuggestionsPanel suggestions={suggestions} nodeId={nodeId} />
        )}
      </div>
    </div>
  );
}
