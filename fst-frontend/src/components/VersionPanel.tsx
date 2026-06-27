// [IMPLEMENTED BY CLAUDE - was missing]
import { useTreeStore } from '../store/useTreeStore';

export function VersionPanel() {
  const { versions, currentVersionId } = useTreeStore();

  if (!versions || versions.length === 0) return null;

  return (
    <div className="bg-white border-t border-gray-200 px-6 py-2">
      <div className="flex items-center gap-4 overflow-x-auto">
        <span className="text-xs font-semibold text-gray-500 uppercase whitespace-nowrap">
          Versions:
        </span>
        {versions.slice(0, 10).map((v) => (
          <button
            key={v.id}
            className={`px-2 py-1 text-xs rounded whitespace-nowrap transition-colors ${
              v.id === currentVersionId
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {v.version}
            {v.is_current && ' (current)'}
          </button>
        ))}
        {versions.length > 10 && (
          <span className="text-xs text-gray-400">
            +{versions.length - 10} more
          </span>
        )}
      </div>
    </div>
  );
}
