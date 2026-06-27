// [IMPLEMENTED BY CLAUDE - was missing]
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { Suggestion } from '../types';

interface Props {
  suggestions: Suggestion[];
  nodeId: string;
}

export function SuggestionsPanel({ suggestions, nodeId }: Props) {
  const queryClient = useQueryClient();

  const approveMutation = useMutation({
    mutationFn: (id: string) =>
      api.approveSuggestion(id, { reviewed_by: 'developer' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['suggestions', nodeId] });
      queryClient.invalidateQueries({ queryKey: ['tree'] });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: (id: string) =>
      api.rejectSuggestion(id, { reviewed_by: 'developer' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['suggestions', nodeId] });
    },
  });

  const pendingSuggestions = suggestions.filter((s) => s.status === 'pending');
  const resolvedSuggestions = suggestions.filter((s) => s.status !== 'pending');

  if (suggestions.length === 0) {
    return (
      <p className="text-sm text-gray-400">No suggestions yet. Run analysis first.</p>
    );
  }

  return (
    <div className="space-y-3">
      {/* Pending Suggestions */}
      {pendingSuggestions.map((suggestion) => (
        <div
          key={suggestion.id}
          className={`border rounded-lg p-3 ${
            suggestion.impact_estimate === 'high'
              ? 'border-red-200 bg-red-50'
              : suggestion.impact_estimate === 'medium'
              ? 'border-yellow-200 bg-yellow-50'
              : 'border-gray-200 bg-gray-50'
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span
                  className={`px-1.5 py-0.5 text-[10px] font-medium rounded uppercase ${
                    suggestion.suggestion_type === 'split'
                      ? 'bg-purple-100 text-purple-700'
                      : suggestion.suggestion_type === 'investigate'
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-gray-100 text-gray-700'
                  }`}
                >
                  {suggestion.suggestion_type}
                </span>
                <span
                  className={`px-1.5 py-0.5 text-[10px] rounded ${
                    suggestion.impact_estimate === 'high'
                      ? 'bg-red-200 text-red-800'
                      : suggestion.impact_estimate === 'medium'
                      ? 'bg-yellow-200 text-yellow-800'
                      : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  {suggestion.impact_estimate} impact
                </span>
              </div>
              <h4 className="text-sm font-medium text-gray-900 mt-1.5">
                {suggestion.title}
              </h4>
              <p className="text-xs text-gray-600 mt-1">{suggestion.description}</p>
              <div className="text-[10px] text-gray-400 mt-1">
                Confidence: {(suggestion.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => approveMutation.mutate(suggestion.id)}
              disabled={approveMutation.isPending}
              className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 disabled:opacity-50"
            >
              Approve & Apply
            </button>
            <button
              onClick={() => rejectMutation.mutate(suggestion.id)}
              disabled={rejectMutation.isPending}
              className="px-3 py-1 bg-gray-200 text-gray-700 text-xs rounded hover:bg-gray-300 disabled:opacity-50"
            >
              Reject
            </button>
          </div>
        </div>
      ))}

      {/* Resolved Suggestions (collapsed) */}
      {resolvedSuggestions.length > 0 && (
        <details className="text-xs text-gray-500">
          <summary className="cursor-pointer hover:text-gray-700">
            {resolvedSuggestions.length} resolved suggestion(s)
          </summary>
          <div className="mt-2 space-y-1">
            {resolvedSuggestions.map((s) => (
              <div key={s.id} className="flex items-center gap-2 p-1">
                <span
                  className={`w-2 h-2 rounded-full ${
                    s.status === 'approved' ? 'bg-green-500' : 'bg-gray-400'
                  }`}
                />
                <span className="truncate">{s.title}</span>
                <span className="text-gray-400">({s.status})</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
