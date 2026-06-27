// [IMPLEMENTED BY CLAUDE - was missing]
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';

interface Props {
  nodeId: string;
}

export function RecordingsPanel({ nodeId }: Props) {
  const { data: recordings, isLoading } = useQuery({
    queryKey: ['recordings', nodeId],
    queryFn: () => api.getNodeRecordings(nodeId),
  });

  return (
    <div className="border-t pt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900">
          Recordings/Measurements
        </h3>
        <span className="text-xs text-gray-500">
          {recordings?.length || 0} attached
        </span>
      </div>

      {isLoading && <p className="text-xs text-gray-400">Loading...</p>}

      {recordings && recordings.length > 0 ? (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {recordings.map((rec: any) => (
            <div
              key={rec.id}
              className="flex items-center justify-between p-2 bg-gray-50 rounded text-xs"
            >
              <div className="flex-1 min-w-0">
                <div className="font-mono text-gray-700 truncate">{rec.id}</div>
                <div className="text-gray-400 truncate">{rec.path}</div>
                {rec.duration && (
                  <span className="text-gray-500">{rec.duration.toFixed(1)}s</span>
                )}
              </div>
              {rec.location && (
                <span className="px-1.5 py-0.5 bg-blue-50 text-blue-600 rounded text-[10px] ml-2">
                  {rec.location}
                </span>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-400">
          No recordings attached. Use the API or bulk import to attach recordings.
        </p>
      )}
    </div>
  );
}
