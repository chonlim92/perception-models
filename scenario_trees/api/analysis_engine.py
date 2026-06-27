# [IMPLEMENTED BY CLAUDE - was missing]
"""
Root Cause Analysis Engine for FST nodes.

When a node fails its KPI threshold, this engine:
1. Identifies failing recordings
2. Mines common patterns/attributes in failures
3. Computes statistical lift to find discriminating factors
4. Generates actionable suggestions (split node, reassign, adjust threshold)
"""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .database import FSTDatabase


class RootCauseAnalyzer:
    """Analyzes failing nodes to identify patterns and suggest corrective actions."""

    def __init__(self, db: FSTDatabase) -> None:
        self.db = db

    def analyze_node(self, node_id: str) -> Dict[str, Any]:
        """
        Run full root cause analysis on a node that is failing its KPIs.

        Returns analysis results including patterns and generated suggestions.
        """
        kpis = self.db.get_kpis(node_id)
        if not kpis:
            return {
                "node_id": node_id,
                "status": "no_kpis_configured",
                "patterns": [],
                "suggestions": [],
            }

        results = self.db.get_evaluation_results(node_id)
        if not results:
            return {
                "node_id": node_id,
                "status": "no_evaluation_data",
                "patterns": [],
                "suggestions": [],
            }

        failing_recordings, passing_recordings = self._classify_recordings(
            results, kpis
        )

        if not failing_recordings:
            return {
                "node_id": node_id,
                "status": "all_passing",
                "failing_count": 0,
                "total_count": len(failing_recordings) + len(passing_recordings),
                "patterns": [],
                "suggestions": [],
            }

        patterns = self._mine_patterns(
            node_id, failing_recordings, passing_recordings
        )

        suggestions = self._generate_suggestions(node_id, patterns, failing_recordings)

        for suggestion in suggestions:
            self.db.add_suggestion(
                node_id=node_id,
                suggestion_type=suggestion["suggestion_type"],
                title=suggestion["title"],
                description=suggestion["description"],
                evidence=suggestion["evidence"],
                proposed_changes=suggestion["proposed_changes"],
                confidence=suggestion["confidence"],
                impact_estimate=suggestion["impact_estimate"],
            )

        total = len(failing_recordings) + len(passing_recordings)
        return {
            "node_id": node_id,
            "status": "analysis_complete",
            "failing_count": len(failing_recordings),
            "total_count": total,
            "failure_rate": len(failing_recordings) / total if total > 0 else 0,
            "patterns": patterns,
            "suggestions": suggestions,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _classify_recordings(
        self,
        results: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """Classify recordings as failing or passing based on KPI thresholds."""
        recording_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for r in results:
            recording_metrics[r["recording_id"]][r["metric_name"]].append(
                r["metric_value"]
            )

        failing = set()
        passing = set()

        for rec_id, metrics in recording_metrics.items():
            is_failing = False
            for kpi in kpis:
                metric_name = kpi["metric_name"]
                if metric_name not in metrics:
                    continue
                avg_value = sum(metrics[metric_name]) / len(metrics[metric_name])
                threshold = kpi["threshold"]
                direction = kpi["direction"]

                if direction == "above" and avg_value < threshold:
                    is_failing = True
                    break
                elif direction == "below" and avg_value > threshold:
                    is_failing = True
                    break

            if is_failing:
                failing.add(rec_id)
            else:
                passing.add(rec_id)

        return list(failing), list(passing)

    def _mine_patterns(
        self,
        node_id: str,
        failing_ids: List[str],
        passing_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Mine common attributes in failing recordings vs passing ones."""
        failing_attrs = self._collect_attributes(failing_ids)
        passing_attrs = self._collect_attributes(passing_ids)
        all_ids = failing_ids + passing_ids

        patterns = []
        all_attribute_keys = set()
        for attrs in failing_attrs.values():
            all_attribute_keys.update(attrs.keys())
        for attrs in passing_attrs.values():
            all_attribute_keys.update(attrs.keys())

        for attr_key in all_attribute_keys:
            attr_values_in_failures = Counter()
            attr_values_in_passing = Counter()

            for rec_id in failing_ids:
                attrs = failing_attrs.get(rec_id, {})
                if attr_key in attrs:
                    val = attrs[attr_key]
                    if isinstance(val, list):
                        for v in val:
                            attr_values_in_failures[str(v)] += 1
                    else:
                        attr_values_in_failures[str(val)] += 1

            for rec_id in passing_ids:
                attrs = passing_attrs.get(rec_id, {})
                if attr_key in attrs:
                    val = attrs[attr_key]
                    if isinstance(val, list):
                        for v in val:
                            attr_values_in_passing[str(v)] += 1
                    else:
                        attr_values_in_passing[str(val)] += 1

            for value, fail_count in attr_values_in_failures.items():
                prevalence_failures = fail_count / len(failing_ids) if failing_ids else 0
                pass_count = attr_values_in_passing.get(value, 0)
                prevalence_overall = (fail_count + pass_count) / len(all_ids) if all_ids else 0

                if prevalence_overall == 0:
                    continue

                lift = prevalence_failures / prevalence_overall if prevalence_overall > 0 else 0

                if lift > 1.5 and prevalence_failures > 0.3:
                    sample_recs = [
                        rec_id for rec_id in failing_ids
                        if attr_key in failing_attrs.get(rec_id, {})
                        and str(failing_attrs[rec_id].get(attr_key, "")) == value
                    ][:5]

                    confidence = min(1.0, (lift - 1.0) * prevalence_failures)

                    patterns.append({
                        "attribute": f"{attr_key}={value}",
                        "attribute_key": attr_key,
                        "attribute_value": value,
                        "prevalence_in_failures": round(prevalence_failures, 3),
                        "prevalence_overall": round(prevalence_overall, 3),
                        "lift": round(lift, 3),
                        "confidence": round(confidence, 3),
                        "failing_count_with_attr": fail_count,
                        "sample_recordings": sample_recs,
                    })

        patterns.sort(key=lambda p: p["lift"], reverse=True)
        return patterns[:10]  # Top 10 patterns

    def _collect_attributes(self, recording_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect attributes for a set of recordings."""
        attrs = {}
        for rec_id in recording_ids:
            rec = self.db.get_recording(rec_id)
            if rec and rec.get("attributes"):
                attrs[rec_id] = rec["attributes"]
            else:
                attrs[rec_id] = {}
        return attrs

    def _generate_suggestions(
        self,
        node_id: str,
        patterns: List[Dict[str, Any]],
        failing_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate actionable suggestions based on discovered patterns."""
        suggestions = []

        for pattern in patterns[:3]:
            if pattern["lift"] >= 2.0 and pattern["prevalence_in_failures"] >= 0.5:
                attr_key = pattern["attribute_key"]
                attr_value = pattern["attribute_value"]

                suggestion = {
                    "suggestion_type": "split",
                    "title": f"Split node by '{attr_key}': with/without '{attr_value}'",
                    "description": (
                        f"Analysis found that {pattern['prevalence_in_failures']*100:.0f}% "
                        f"of failing recordings have {attr_key}={attr_value} "
                        f"(vs {pattern['prevalence_overall']*100:.0f}% overall, "
                        f"lift={pattern['lift']:.1f}x). "
                        f"Splitting this node into sub-branches for recordings "
                        f"with and without '{attr_value}' would isolate the failure mode "
                        f"and allow targeted investigation."
                    ),
                    "evidence": {
                        "pattern": pattern,
                        "failing_sample": pattern["sample_recordings"],
                    },
                    "proposed_changes": {
                        "action": "split_node",
                        "parent_node_id": node_id,
                        "new_branches": [
                            {
                                "name": f"with_{attr_value}",
                                "filter": {attr_key: attr_value},
                            },
                            {
                                "name": f"without_{attr_value}",
                                "filter": {attr_key: {"$ne": attr_value}},
                            },
                        ],
                        "reassign_recordings": True,
                    },
                    "confidence": pattern["confidence"],
                    "impact_estimate": "high" if pattern["lift"] >= 3.0 else "medium",
                }
                suggestions.append(suggestion)

            elif pattern["lift"] >= 1.5:
                suggestions.append({
                    "suggestion_type": "investigate",
                    "title": f"Investigate correlation: '{pattern['attribute']}'",
                    "description": (
                        f"Recordings with {pattern['attribute']} are "
                        f"{pattern['lift']:.1f}x more likely to fail. "
                        f"This warrants investigation but may not justify a split."
                    ),
                    "evidence": {"pattern": pattern},
                    "proposed_changes": {
                        "action": "flag_for_review",
                        "attribute": pattern["attribute"],
                    },
                    "confidence": pattern["confidence"] * 0.7,
                    "impact_estimate": "low",
                })

        if not patterns and failing_ids:
            suggestions.append({
                "suggestion_type": "adjust_threshold",
                "title": "Consider adjusting KPI threshold",
                "description": (
                    "No clear discriminating pattern found in failing recordings. "
                    "The failures may be uniformly distributed, suggesting the "
                    "KPI threshold may be too aggressive for this scenario category."
                ),
                "evidence": {
                    "failing_count": len(failing_ids),
                    "no_patterns_found": True,
                },
                "proposed_changes": {
                    "action": "adjust_threshold",
                    "recommendation": "Review and potentially relax threshold",
                },
                "confidence": 0.4,
                "impact_estimate": "medium",
            })

        return suggestions

    def apply_split_suggestion(
        self, suggestion_id: str, node_id: str, tree_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply an approved split suggestion to the tree.

        Creates new child nodes and reassigns recordings based on the filter criteria.
        Returns the updated tree data.
        """
        suggestions = self.db.get_suggestions(node_id=node_id)
        suggestion = next(
            (s for s in suggestions if s["id"] == suggestion_id), None
        )
        if not suggestion:
            return tree_data

        proposed = suggestion["proposed_changes"]
        if proposed.get("action") != "split_node":
            return tree_data

        new_branches = proposed.get("new_branches", [])
        recordings = self.db.get_node_recordings(node_id)

        for branch in new_branches:
            branch_name = branch["name"]
            branch_filter = branch["filter"]
            new_node_id = f"{node_id}.{branch_name}"

            matching_recordings = self._filter_recordings(
                recordings, branch_filter
            )

            for rec_id in matching_recordings:
                self.db.attach_recording(new_node_id, rec_id)

        return tree_data

    def _filter_recordings(
        self, recording_ids: List[str], filter_criteria: Dict[str, Any]
    ) -> List[str]:
        """Filter recordings by attribute criteria."""
        matching = []
        for rec_id in recording_ids:
            rec = self.db.get_recording(rec_id)
            if not rec:
                continue
            attrs = rec.get("attributes", {})
            matches = True
            for key, value in filter_criteria.items():
                if isinstance(value, dict) and "$ne" in value:
                    if attrs.get(key) == value["$ne"]:
                        matches = False
                        break
                else:
                    rec_val = attrs.get(key)
                    if isinstance(rec_val, list):
                        if value not in rec_val:
                            matches = False
                            break
                    elif rec_val != value:
                        matches = False
                        break
            if matches:
                matching.append(rec_id)
        return matching
