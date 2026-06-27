// [IMPLEMENTED BY CLAUDE - was missing]
import { memo } from 'react';
import { Handle, Position } from 'reactflow';

interface ScenarioNodeData {
  label: string;
  layer: number;
  description: string;
  childCount: number;
  color: string;
  kpiStatus?: 'pass' | 'warn' | 'fail';
}

function ScenarioNodeComponent({ data }: { data: ScenarioNodeData }) {
  const statusColors = {
    pass: 'border-kpi-pass bg-green-50',
    warn: 'border-kpi-warn bg-yellow-50',
    fail: 'border-kpi-fail bg-red-50',
  };

  const borderClass = data.kpiStatus
    ? statusColors[data.kpiStatus]
    : 'border-gray-300 bg-white';

  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />
      <div
        className={`px-3 py-2 rounded-lg border-2 shadow-sm min-w-[140px] cursor-pointer
          hover:shadow-md transition-shadow ${borderClass}`}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: data.color }}
          />
          <span className="text-xs font-medium text-gray-900 truncate">
            {data.label}
          </span>
        </div>
        {data.childCount > 0 && (
          <div className="text-[10px] text-gray-500 mt-0.5 ml-4">
            {data.childCount} children
          </div>
        )}
        {data.kpiStatus && (
          <div className="mt-1 flex items-center gap-1">
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                data.kpiStatus === 'pass'
                  ? 'bg-kpi-pass'
                  : data.kpiStatus === 'warn'
                  ? 'bg-kpi-warn'
                  : 'bg-kpi-fail'
              }`}
            />
            <span className="text-[10px] text-gray-600 uppercase">
              {data.kpiStatus}
            </span>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </>
  );
}

export const ScenarioNode = memo(ScenarioNodeComponent);
