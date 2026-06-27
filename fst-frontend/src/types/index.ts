// [IMPLEMENTED BY CLAUDE - was missing]

export interface TreeNode {
  id: string;
  name: string;
  layer: number;
  description: string;
  detection_method: string;
  parent_id: string | null;
  children: TreeNode[];
}

export interface TreeVersion {
  id: string;
  version: string;
  created_at: string;
  created_by: string;
  change_description: string;
  parent_version_id: string | null;
  is_current?: boolean;
  tree_data?: { root: TreeNode };
}

export interface Recording {
  id: string;
  path: string;
  timestamp?: string;
  duration?: number;
  location?: string;
  attributes?: Record<string, unknown>;
}

export interface MetricsSummary {
  metrics: Record<string, MetricStats>;
  total_recordings: number;
  kpi_status: 'pass' | 'warn' | 'fail';
  failing_metrics: string[];
  kpi_configs: KPIConfig[];
}

export interface MetricStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  count: number;
}

export interface KPIConfig {
  metric_name: string;
  threshold: number;
  direction: 'above' | 'below';
  warning_margin: number;
}

export interface EvaluationScript {
  id: string;
  node_id: string;
  name: string;
  script_content: string;
  created_at: string;
  last_run?: string;
}

export interface Pattern {
  attribute: string;
  attribute_key: string;
  attribute_value: string;
  prevalence_in_failures: number;
  prevalence_overall: number;
  lift: number;
  confidence: number;
  failing_count_with_attr: number;
  sample_recordings: string[];
}

export interface Suggestion {
  id: string;
  node_id: string;
  suggestion_type: 'split' | 'reassign' | 'adjust_threshold' | 'investigate';
  title: string;
  description: string;
  evidence: Record<string, unknown>;
  proposed_changes: Record<string, unknown>;
  confidence: number;
  impact_estimate: 'high' | 'medium' | 'low';
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  reviewed_by?: string;
  reviewed_at?: string;
  notes?: string;
}

export interface AnalysisResult {
  node_id: string;
  status: string;
  failing_count: number;
  total_count: number;
  failure_rate: number;
  patterns: Pattern[];
  suggestions: Suggestion[];
  analysis_timestamp: string;
}
