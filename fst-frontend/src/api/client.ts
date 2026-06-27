// [IMPLEMENTED BY CLAUDE - was missing]

const BASE_URL = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Tree
  getTree: () => request<any>('/tree'),
  getTreeVersions: () => request<any[]>('/tree/versions'),
  getTreeVersion: (id: string) => request<any>(`/tree/versions/${id}`),
  createVersion: (data: { version?: string; change_description: string; created_by: string }) =>
    request<any>('/tree/versions', { method: 'POST', body: JSON.stringify(data) }),
  updateNode: (nodeId: string, data: { name?: string; description?: string }) =>
    request<any>(`/tree/nodes/${nodeId}`, { method: 'PUT', body: JSON.stringify(data) }),
  addChild: (parentId: string, data: { id: string; name: string; layer: number; description?: string }) =>
    request<any>(`/tree/nodes/${parentId}/children`, { method: 'POST', body: JSON.stringify(data) }),
  deleteNode: (nodeId: string) =>
    request<any>(`/tree/nodes/${nodeId}`, { method: 'DELETE' }),
  splitNode: (nodeId: string, data: { split_criteria: string; branch_names: string[]; auto_reassign: boolean }) =>
    request<any>(`/tree/nodes/${nodeId}/split`, { method: 'POST', body: JSON.stringify(data) }),

  // Recordings
  getNodeRecordings: (nodeId: string) => request<any[]>(`/nodes/${nodeId}/recordings`),
  attachRecording: (nodeId: string, data: { recording_id: string; path?: string }) =>
    request<any>(`/nodes/${nodeId}/recordings`, { method: 'POST', body: JSON.stringify(data) }),
  detachRecording: (nodeId: string, recordingId: string) =>
    request<any>(`/nodes/${nodeId}/recordings/${recordingId}`, { method: 'DELETE' }),
  bulkImport: (data: { recordings: any[]; node_id?: string }) =>
    request<any>('/recordings/bulk-import', { method: 'POST', body: JSON.stringify(data) }),

  // Metrics
  getNodeMetrics: (nodeId: string) => request<any>(`/nodes/${nodeId}/metrics`),
  runEvaluation: (nodeId: string, data?: { script_id?: string; recording_ids?: string[] }) =>
    request<any>(`/nodes/${nodeId}/evaluate`, { method: 'POST', body: JSON.stringify(data || {}) }),
  getKPI: (nodeId: string) => request<any>(`/nodes/${nodeId}/kpi`),
  setKPI: (nodeId: string, data: { metric_name: string; threshold: number; direction: string; warning_margin?: number }) =>
    request<any>(`/nodes/${nodeId}/kpi`, { method: 'PUT', body: JSON.stringify(data) }),

  // Scripts
  getScripts: (nodeId: string) => request<any[]>(`/nodes/${nodeId}/scripts`),
  addScript: (nodeId: string, data: { name: string; script_content: string }) =>
    request<any>(`/nodes/${nodeId}/scripts`, { method: 'POST', body: JSON.stringify(data) }),

  // Analysis
  triggerAnalysis: (nodeId: string) =>
    request<any>(`/nodes/${nodeId}/analyze`, { method: 'POST' }),
  getSuggestions: (nodeId: string, status?: string) =>
    request<any[]>(`/nodes/${nodeId}/suggestions${status ? `?status=${status}` : ''}`),
  approveSuggestion: (suggestionId: string, data: { reviewed_by: string; notes?: string }) =>
    request<any>(`/suggestions/${suggestionId}/approve`, { method: 'POST', body: JSON.stringify(data) }),
  rejectSuggestion: (suggestionId: string, data: { reviewed_by: string; notes?: string }) =>
    request<any>(`/suggestions/${suggestionId}/reject`, { method: 'POST', body: JSON.stringify(data) }),
};
