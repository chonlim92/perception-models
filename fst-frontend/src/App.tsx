// [IMPLEMENTED BY CLAUDE - was missing]
import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTreeStore } from './store/useTreeStore';
import { api } from './api/client';
import { TreeVisualization } from './components/TreeVisualization';
import { NodeDetailPanel } from './components/NodeDetailPanel';
import { VersionPanel } from './components/VersionPanel';
import { Header } from './components/Header';

export default function App() {
  const { setTree, setVersions, setCurrentVersionId, selectedNodeId } = useTreeStore();

  const { data: treeData, isLoading } = useQuery({
    queryKey: ['tree'],
    queryFn: api.getTree,
  });

  const { data: versions } = useQuery({
    queryKey: ['versions'],
    queryFn: api.getTreeVersions,
  });

  useEffect(() => {
    if (treeData?.tree_data?.root) {
      setTree(treeData.tree_data.root);
      setCurrentVersionId(treeData.id);
    }
  }, [treeData, setTree, setCurrentVersionId]);

  useEffect(() => {
    if (versions) {
      setVersions(versions);
    }
  }, [versions, setVersions]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-lg text-gray-600">Loading Functional Scenario Tree...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 relative">
          <TreeVisualization />
        </div>
        {selectedNodeId && (
          <div className="w-[480px] border-l border-gray-200 overflow-y-auto bg-white">
            <NodeDetailPanel nodeId={selectedNodeId} />
          </div>
        )}
      </div>
      <VersionPanel />
    </div>
  );
}
