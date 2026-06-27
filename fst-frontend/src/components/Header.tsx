// [IMPLEMENTED BY CLAUDE - was missing]
import { useTreeStore } from '../store/useTreeStore';

export function Header() {
  const { versions, currentVersionId } = useTreeStore();
  const currentVersion = versions.find((v) => v.id === currentVersionId);

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-bold text-gray-900">
          FST - Functional Scenario Tree
        </h1>
        {currentVersion && (
          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-md font-mono">
            {currentVersion.version}
          </span>
        )}
      </div>
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-500">
          PEGASUS/ASAM 6-Layer Taxonomy
        </span>
        <button className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors">
          New Version
        </button>
      </div>
    </header>
  );
}
