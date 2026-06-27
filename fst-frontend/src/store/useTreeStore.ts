// [IMPLEMENTED BY CLAUDE - was missing]

import { create } from 'zustand';
import type { TreeNode, TreeVersion, Suggestion } from '../types';

interface TreeState {
  tree: TreeNode | null;
  versions: TreeVersion[];
  currentVersionId: string | null;
  selectedNodeId: string | null;
  suggestions: Suggestion[];
  loading: boolean;
  error: string | null;

  setTree: (tree: TreeNode) => void;
  setVersions: (versions: TreeVersion[]) => void;
  setCurrentVersionId: (id: string) => void;
  selectNode: (nodeId: string | null) => void;
  setSuggestions: (suggestions: Suggestion[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useTreeStore = create<TreeState>((set) => ({
  tree: null,
  versions: [],
  currentVersionId: null,
  selectedNodeId: null,
  suggestions: [],
  loading: false,
  error: null,

  setTree: (tree) => set({ tree }),
  setVersions: (versions) => set({ versions }),
  setCurrentVersionId: (id) => set({ currentVersionId: id }),
  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),
  setSuggestions: (suggestions) => set({ suggestions }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));
