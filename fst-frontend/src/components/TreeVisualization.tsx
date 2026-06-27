// [IMPLEMENTED BY CLAUDE - was missing]
import { useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  NodeTypes,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useTreeStore } from '../store/useTreeStore';
import type { TreeNode } from '../types';
import { ScenarioNode } from './ScenarioNode';

const nodeTypes: NodeTypes = {
  scenario: ScenarioNode,
};

const LAYER_COLORS: Record<number, string> = {
  0: '#6b7280',
  1: '#3b82f6',
  2: '#8b5cf6',
  3: '#f59e0b',
  4: '#ef4444',
  5: '#10b981',
  6: '#06b6d4',
};

function treeToFlow(
  node: TreeNode,
  x: number,
  y: number,
  nodes: Node[],
  edges: Edge[],
  depth: number = 0
): number {
  const nodeWidth = 180;
  const nodeHeight = 50;
  const horizontalSpacing = 200;
  const verticalSpacing = 80;

  nodes.push({
    id: node.id,
    type: 'scenario',
    position: { x, y },
    data: {
      label: node.name,
      layer: node.layer,
      description: node.description,
      childCount: node.children.length,
      color: LAYER_COLORS[node.layer] || '#6b7280',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  });

  if (node.parent_id) {
    edges.push({
      id: `${node.parent_id}-${node.id}`,
      source: node.parent_id,
      target: node.id,
      type: 'smoothstep',
      style: { stroke: '#94a3b8', strokeWidth: 1.5 },
    });
  }

  let childX = x - ((node.children.length - 1) * horizontalSpacing) / 2;
  const childY = y + verticalSpacing + nodeHeight;

  for (const child of node.children) {
    const width = treeToFlow(child, childX, childY, nodes, edges, depth + 1);
    childX += Math.max(horizontalSpacing, width);
  }

  return Math.max(nodeWidth, node.children.length * horizontalSpacing);
}

export function TreeVisualization() {
  const { tree, selectNode } = useTreeStore();

  const { initialNodes, initialEdges } = useMemo(() => {
    if (!tree) return { initialNodes: [], initialEdges: [] };
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    treeToFlow(tree, 600, 50, nodes, edges);
    return { initialNodes: nodes, initialEdges: edges };
  }, [tree]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  if (!tree) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No tree data loaded
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={onNodeClick}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.1}
      maxZoom={2}
    >
      <Background color="#e2e8f0" gap={20} />
      <Controls />
      <MiniMap
        nodeColor={(node) => node.data?.color || '#6b7280'}
        maskColor="rgba(0,0,0,0.1)"
      />
    </ReactFlow>
  );
}
