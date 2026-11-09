"""Unified 6-stage weight-space compiler across 6 model classes, Actor x {Transfer, Transform, Save} x Horizon DAG decomposer, and leak-free holdout recursive RL trial engine."""

from __future__ import annotations

import base64
import hashlib
import importlib
import itertools
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .boil_down_and_swarm import AdaptiveBoilDownEngine
from .fabric_substrate import DISPATCH_AFFINITY_MODES, LocalAdaptiveFabricStore
from .grid_and_env import AdaptiveNodeGrid, EffortAndBudgetGate


MODEL_CLASSES: Tuple[str, ...] = (
    "embedding",
    "search_retrieval",
    "decision_router",
    "lora_xs_cartridge",
    "memory_as_lora",
    "steered_llm",
)

ACTOR_MODES: Tuple[str, ...] = (
    "person",
    "machine",
    "swarm",
)

TRIAD_OPS: Tuple[str, ...] = (
    "transfer",
    "transform",
    "save",
)

BENCHMARK_ARCHETYPES: Tuple[str, ...] = (
    "general_horizon",
    "bitemporal_tabular_oof",
    "symbolic_grid_synthesis",
    "combinatorial_packing",
    "repository_patch_repair",
)


def encode_safetensors_bytes(
    tensors: Mapping[str, Tuple[Sequence[int], Sequence[float]]],
    metadata: Optional[Mapping[str, str]] = None,
    f32_width: int = 1,
) -> bytes:
    """Serialize float32 tensors into spec-compliant .safetensors binary layout."""
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    chunks: List[bytes] = []
    offset = 0
    for name in sorted(tensors.keys()):
        shape, flat_vals = tensors[name]
        clean_vals = [float(v) for v in flat_vals]
        packed = struct.pack(f"<{len(clean_vals)}f", *clean_vals)
        byte_len = len(clean_vals) * int(f32_width)
        assert len(packed) == byte_len
        header[str(name)] = {
            "dtype": "F32",
            "shape": [int(dim) for dim in shape],
            "data_offsets": [offset, offset + len(packed)],
        }
        chunks.append(packed)
        offset += len(packed)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)


def decode_safetensors_bytes(
    raw_bytes: bytes,
    header_len_bytes: int,
    f32_width: int,
) -> Dict[str, Any]:
    """Parse spec-compliant .safetensors binary payload into float32 arrays and metadata."""
    if len(raw_bytes) < header_len_bytes:
        raise ValueError("safetensors_payload_too_short")
    (hlen,) = struct.unpack("<Q", raw_bytes[:header_len_bytes])
    body_start = header_len_bytes + int(hlen)
    if len(raw_bytes) < body_start:
        raise ValueError("safetensors_truncated_header")
    header = json.loads(raw_bytes[header_len_bytes:body_start].decode("utf-8"))
    data_buf = raw_bytes[body_start:]
    tensors: Dict[str, Dict[str, Any]] = {}
    meta = dict(header.get("__metadata__") or {})
    for key, info in header.items():
        if key == "__metadata__":
            continue
        start_off, end_off = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        seg = data_buf[start_off:end_off]
        count = max(0, len(seg) // max(1, int(f32_width)))
        values = list(struct.unpack(f"<{count}f", seg[: count * int(f32_width)]))
        tensors[str(key)] = {
            "dtype": str(info.get("dtype") or "F32"),
            "shape": [int(s) for s in info.get("shape", [])],
            "values": values,
        }
    return {"metadata": meta, "tensors": tensors}


class SixStageWeightPipelineEngine:
    """Executes the unified 6-stage weight-space compilation pipeline across 6 model classes."""

    def __init__(
        self,
        grid: AdaptiveNodeGrid,
        store: LocalAdaptiveFabricStore,
        boil_engine: AdaptiveBoilDownEngine,
        budget_gate: EffortAndBudgetGate,
    ) -> None:
        self.grid = grid
        self.store = store
        self.boil_engine = boil_engine
        self.budget_gate = budget_gate
        self.promoted_units: Dict[str, Dict[str, Any]] = {}
        self.unit_history: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _unit_key(namespace: str, model_class: str) -> str:
        return f"{namespace}::{model_class}"

    @staticmethod
    def _orthonormal_rows(vectors: Sequence[Sequence[float]], eps: float, tol: float) -> List[List[float]]:
        """Modified Gram-Schmidt orthonormal basis over row vectors."""
        basis: List[List[float]] = []
        for vec in vectors:
            work = [float(v) for v in vec]
            dim = len(work)
            for q_row in basis:
                proj = sum(work[i] * q_row[i] for i in range(dim))
                for i in range(dim):
                    work[i] -= proj * q_row[i]
            norm = math.sqrt(sum(v * v for v in work))
            if norm > tol:
                inv = 1.0 / max(eps, norm)
                basis.append([v * inv for v in work])
        return basis

    def _compute_anchor_orthonormal_basis(
        self,
        anchor_rows: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], str]:
        eps = float(self.grid.runtime_constants["eps_floor"])
        tol = float(self.grid.runtime_constants["nullspace_rank_tolerance"])
        if not anchor_rows:
            return [], "empty_anchor_basis"
        try:
            np = importlib.import_module("numpy")
            a_mat = np.asarray(anchor_rows, dtype=np.float64)
            _, sing_vals, vt = np.linalg.svd(a_mat, full_matrices=False)
            rank = int(np.sum(sing_vals > tol))
            q_rows = [[float(v) for v in vt[r_idx]] for r_idx in range(rank)]
            return q_rows, "numpy_svd_nullspace"
        except Exception:
            return self._orthonormal_rows(anchor_rows, eps=eps, tol=tol), "stdlib_gram_schmidt_nullspace"

    @staticmethod
    def _project_rows_into_nullspace(
        delta_rows: Sequence[Sequence[float]],
        q_basis: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Project each row vector v -> v - sum_q (v . q) q so (v_proj . a) = 0 for all anchors a."""
        projected: List[List[float]] = []
        for row in delta_rows:
            work = [float(v) for v in row]
            dim = len(work)
            for q_vec in q_basis:
                dot = sum(work[i] * q_vec[i] for i in range(dim))
                for i in range(dim):
                    work[i] -= dot * q_vec[i]
            projected.append(work)
        return projected

    def _induce_symbolic_skeleton(
        self,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        buckets: int,
        mask_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stage 1: Induce class-conditional invariant half-spaces and analytical ReLU skeleton W_sym."""
        rc = self.grid.runtime_constants
        min_sup = int(rc["symbolic_invariant_min_support"])
        margin_th = float(rc["symbolic_invariant_margin_threshold"])
        gain_scale = float(rc["symbolic_skeleton_gain_scale"])
        eps = float(rc["eps_floor"])

        num_classes = len(labels)
        label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
        class_sums = [[0.0] * buckets for _ in range(num_classes)]
        class_counts = [0] * num_classes
        x_cache: List[List[float]] = []

        for item in rows:
            xv = self.boil_engine._featurize_text(str(item["input_text"]), buckets, mask_seed=mask_seed)
            x_cache.append(xv)
            c_idx = label_to_idx.get(str(item["target_label"]), 0)
            class_counts[c_idx] += 1
            for b_i in range(buckets):
                class_sums[c_idx][b_i] += xv[b_i]

        class_means = [
            [
                class_sums[c_i][b_i] / float(max(1, class_counts[c_i]))
                for b_i in range(buckets)
            ]
            for c_i in range(num_classes)
        ]

        w_sym = [[0.0] * buckets for _ in range(num_classes)]
        invariant_rules: List[Dict[str, Any]] = []
        for c_i in range(num_classes):
            if class_counts[c_i] < min_sup:
                continue
            other_denom = float(max(1, num_classes - 1))
            for b_i in range(buckets):
                other_mean = sum(
                    class_means[o_i][b_i] for o_i in range(num_classes) if o_i != c_i
                ) / other_denom
                margin = class_means[c_i][b_i] - other_mean
                if abs(margin) >= margin_th:
                    w_sym[c_i][b_i] = margin * gain_scale
                    invariant_rules.append(
                        {
                            "class_label": labels[c_i],
                            "bucket_index": b_i,
                            "signed_margin": round(margin, int(rc["ratio_round_digits"])),
                        }
                    )
            row_norm = math.sqrt(sum(v * v for v in w_sym[c_i]))
            if row_norm > eps:
                inv_n = 1.0 / row_norm
                w_sym[c_i] = [v * inv_n for v in w_sym[c_i]]

        violations = 0
        checked_rules = 0
        for r_idx, item in enumerate(rows):
            c_true = label_to_idx.get(str(item["target_label"]), 0)
            sym_row = w_sym[c_true]
            if any(abs(v) > eps for v in sym_row):
                checked_rules += 1
                own_act = max(0.0, sum(sym_row[b_i] * x_cache[r_idx][b_i] for b_i in range(buckets)))
                if own_act <= 0.0:
                    violations += 1

        violation_rate = float(violations) / float(max(1, checked_rules)) if checked_rules > 0 else 0.0
        return {
            "w_sym": w_sym,
            "invariant_count": len(invariant_rules),
            "invariants": invariant_rules[: int(rc["hybrid_search_default_limit"])],
            "gradient_epochs": 0,
            "symbolic_rule_violation_rate": round(violation_rate, int(rc["ratio_round_digits"])),
            "x_cache": x_cache,
        }

    def _prune_channels_and_fit_ridge(
        self,
        namespace: str,
        model_class: str,
        rows: Sequence[Mapping[str, Any]],
        x_cache: Sequence[Sequence[float]],
        labels: Sequence[str],
        buckets: int,
        initial_hidden: int,
        retained_hidden: int,
        w_sym: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        """Stage 2: Wanda-style historical-trace channel pruning + closed-form ridge recalibration."""
        rc = self.grid.runtime_constants
        reg = float(rc["ridge_regularization"])
        eps = float(rc["eps_floor"])
        pct = float(rc["percent_scale"])
        round_d = int(rc["ratio_round_digits"])

        raw_hidden = (
            int(rc["embedding_raw_dim_default"])
            if model_class == "embedding"
            else max(1, initial_hidden)
        )
        keep_hidden = max(1, min(raw_hidden, retained_hidden))
        full_proj = self.boil_engine._build_polytope_projection(
            f"{namespace}:{model_class}",
            buckets,
            raw_hidden,
        )

        # Score each hidden channel via historical-trace activation RMS * weight L1 norm
        channel_scores: List[Tuple[float, int]] = []
        n_samples = max(1, len(x_cache))
        for h_idx in range(raw_hidden):
            row_w = full_proj[h_idx]
            w_norm = sum(abs(v) for v in row_w) / float(max(1, buckets))
            act_sq = 0.0
            for s_idx in range(len(x_cache)):
                dot = sum(row_w[b_i] * float(x_cache[s_idx][b_i]) for b_i in range(buckets))
                relu_v = dot if dot > 0.0 else 0.0
                act_sq += relu_v * relu_v
            act_rms = math.sqrt(act_sq / float(n_samples) + eps)
            channel_scores.append((w_norm * act_rms, h_idx))

        channel_scores.sort(key=lambda item: item[0], reverse=True)
        kept_indices = sorted(idx for _, idx in channel_scores[:keep_hidden])
        pruned_proj = [full_proj[h_idx] for h_idx in kept_indices]

        num_classes = len(labels)
        label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
        dim = 1 + keep_hidden
        gram = [[0.0] * dim for _ in range(dim)]
        rhs = [[0.0] * num_classes for _ in range(dim)]
        z_rows: List[List[float]] = []
        residual_targets: List[List[float]] = []

        for r_idx, item in enumerate(rows):
            xv = x_cache[r_idx]
            zv: List[float] = [1.0]
            for row_w in pruned_proj:
                dot = sum(row_w[b_i] * float(xv[b_i]) for b_i in range(buckets))
                zv.append(dot if dot > 0.0 else 0.0)
            z_rows.append(zv)

            c_true = label_to_idx.get(str(item["target_label"]), 0)
            y_res: List[float] = []
            for c_i in range(num_classes):
                target_v = 1.0 if c_i == c_true else 0.0
                sym_v = max(0.0, sum(float(w_sym[c_i][b_i]) * float(xv[b_i]) for b_i in range(buckets)))
                y_res.append(target_v - sym_v)
            residual_targets.append(y_res)

            for i in range(dim):
                zi = zv[i]
                for j in range(dim):
                    gram[i][j] += zi * zv[j]
                for c_i in range(num_classes):
                    rhs[i][c_i] += zi * y_res[c_i]

        for i in range(dim):
            gram[i][i] += reg
        ridge_w = self.boil_engine._solve_linear_system(gram, rhs, eps)

        mse_sum = 0.0
        for r_idx in range(len(rows)):
            zv = z_rows[r_idx]
            yv = residual_targets[r_idx]
            for c_i in range(num_classes):
                pred_c = sum(zv[i] * ridge_w[i][c_i] for i in range(dim))
                diff = pred_c - yv[c_i]
                mse_sum += diff * diff
        mse = mse_sum / float(max(1, len(rows) * num_classes))
        ram_reduction_pct = round((1.0 - (float(keep_hidden) / float(max(1, raw_hidden)))) * pct, round_d)

        return {
            "initial_channels": raw_hidden,
            "retained_channels": keep_hidden,
            "kept_channel_indices": kept_indices,
            "ram_reduction_pct": ram_reduction_pct,
            "recalibrated_ridge_mse": round(mse, round_d),
            "pruned_proj": pruned_proj,
            "ridge_w": ridge_w,
            "residual_targets": residual_targets,
        }

    def _fit_subspace_cartridge(
        self,
        namespace: str,
        model_class: str,
        x_cache: Sequence[Sequence[float]],
        pruned_proj: Sequence[Sequence[float]],
        residual_targets: Sequence[Sequence[float]],
        subspace_rank: int,
        num_classes: int,
        seed_tag: int = 0,
        null_edit: bool = False,
        loaded_r_mat: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Stage 3: Singular-subspace LoRA-XS (r x r float32, 256B at r=8) solve."""
        rc = self.grid.runtime_constants
        reg = float(rc["ridge_regularization"])
        eps = float(rc["eps_floor"])
        f32_bytes = int(rc["float32_byte_width"])

        buckets = len(x_cache[0]) if x_cache else 1
        keep_hidden = len(pruned_proj)
        feat_dim = 1 + buckets + keep_hidden
        v_mat, u_mat = self.boil_engine._build_frozen_subspace_bases(
            namespace=f"{namespace}:{model_class}",
            feat_dim=feat_dim,
            subspace_rank=subspace_rank,
            num_classes=num_classes,
            seed_tag=seed_tag,
        )

        if null_edit:
            r_mat = [[0.0] * subspace_rank for _ in range(subspace_rank)]
        elif loaded_r_mat is not None:
            r_mat = [[float(v) for v in row] for row in loaded_r_mat]
        else:
            gram = [[0.0] * subspace_rank for _ in range(subspace_rank)]
            rhs = [[0.0] * subspace_rank for _ in range(subspace_rank)]
            for r_idx, xv in enumerate(x_cache):
                phi = self.boil_engine._polytope_lift(xv, pruned_proj)
                sv = [
                    sum(phi[d_i] * v_mat[d_i][k_i] for d_i in range(feat_dim))
                    for k_i in range(subspace_rank)
                ]
                yv = residual_targets[r_idx]
                # Project target residual onto U_r^T
                yu = [
                    sum(yv[c_i] * u_mat[k_i][c_i] for c_i in range(num_classes))
                    for k_i in range(subspace_rank)
                ]
                for i in range(subspace_rank):
                    si = sv[i]
                    for j in range(subspace_rank):
                        gram[i][j] += si * sv[j]
                        rhs[i][j] += si * yu[j]
            for i in range(subspace_rank):
                gram[i][i] += reg
            r_mat = self.boil_engine._solve_linear_system(gram, rhs, eps)

        flat_r = [float(r_mat[i][j]) for i in range(subspace_rank) for j in range(subspace_rank)]
        packed_bin = struct.pack(f"<{subspace_rank * subspace_rank}f", *flat_r)
        unpacked_flat = list(struct.unpack(f"<{subspace_rank * subspace_rank}f", packed_bin))
        r_quant = [
            unpacked_flat[i * subspace_rank : (i + 1) * subspace_rank]
            for i in range(subspace_rank)
        ]
        return {
            "subspace_rank": subspace_rank,
            "cartridge_bytes": subspace_rank * subspace_rank * f32_bytes,
            "cartridge_b64": base64.b64encode(packed_bin).decode("ascii"),
            "packed_bin": packed_bin,
            "feat_dim": feat_dim,
            "v_mat": v_mat,
            "u_mat": u_mat,
            "r_mat": r_quant,
        }

    def _compile_memory_as_lora_slots(
        self,
        namespace: str,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        buckets: int,
        max_slots: int,
        episodic_memories: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 4: Memory-as-LoRA orthogonal episodic trace-to-weight slot compilation."""
        rc = self.grid.runtime_constants
        eps = float(rc["eps_floor"])
        tol = float(rc["nullspace_rank_tolerance"])
        toks_per_turn = int(rc["memory_tokens_per_turn_estimate"])
        turns_per_slot = int(rc["memory_turns_per_slot_estimate"])
        round_d = int(rc["ratio_round_digits"])

        source_items: List[Dict[str, str]] = []
        if episodic_memories:
            for idx, mem in enumerate(episodic_memories):
                source_items.append(
                    {
                        "slot_id": str(mem.get("slot_id") or f"slot_{idx + 1}"),
                        "fact_text": str(mem.get("fact_text") or mem.get("input_text") or ""),
                        "target_label": str(mem.get("target_label") or labels[0]),
                    }
                )
        else:
            envelopes = self.store.query_trace_envelopes(namespace=namespace, limit=max_slots)
            for idx, env in enumerate(envelopes):
                if str(env.get("input_text") or "").strip() and str(env.get("target_label") or "").strip():
                    source_items.append(
                        {
                            "slot_id": f"env_{env['envelope_id']}",
                            "fact_text": str(env["input_text"]),
                            "target_label": str(env["target_label"]),
                        }
                    )
            if not source_items:
                for idx, r_item in enumerate(rows[:max_slots]):
                    source_items.append(
                        {
                            "slot_id": f"trace_slot_{idx + 1}",
                            "fact_text": str(r_item["input_text"]),
                            "target_label": str(r_item["target_label"]),
                        }
                    )

        capped_items = source_items[: max(1, max_slots)]
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        orthogonal_keys: List[List[float]] = []
        compiled_slots: List[Dict[str, Any]] = []

        for item in capped_items:
            raw_vec = self.boil_engine._featurize_text(item["fact_text"], buckets)
            work = list(raw_vec)
            for prev_k in orthogonal_keys:
                proj = sum(work[b_i] * prev_k[b_i] for b_i in range(buckets))
                for b_i in range(buckets):
                    work[b_i] -= proj * prev_k[b_i]
            norm = math.sqrt(sum(v * v for v in work))
            if norm <= tol:
                continue
            inv_n = 1.0 / max(eps, norm)
            unit_k = [v * inv_n for v in work]
            orthogonal_keys.append(unit_k)
            c_idx = label_to_idx.get(item["target_label"], 0)
            compiled_slots.append(
                {
                    "slot_id": item["slot_id"],
                    "target_label": labels[c_idx],
                    "class_index": c_idx,
                    "key_vec": unit_k,
                    "raw_vec": raw_vec,
                }
            )

        max_cross_dot = 0.0
        for i in range(len(orthogonal_keys)):
            for j in range(i + 1, len(orthogonal_keys)):
                dot = abs(sum(orthogonal_keys[i][b_i] * orthogonal_keys[j][b_i] for b_i in range(buckets)))
                if dot > max_cross_dot:
                    max_cross_dot = dot

        eliminated_tokens = len(compiled_slots) * toks_per_turn * turns_per_slot
        return {
            "slot_count": len(compiled_slots),
            "max_slots": max_slots,
            "slot_orthogonality_max_inner_product": round(max_cross_dot, int(rc["attenuation_round_digits"])),
            "context_tokens_eliminated": eliminated_tokens,
            "slots": compiled_slots,
        }

    def _project_residual_onto_anchor_nullspace(
        self,
        buckets: int,
        num_classes: int,
        w_sym: Sequence[Sequence[float]],
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
        anchor_texts: Optional[Sequence[str]] = None,
        null_edit: bool = False,
    ) -> Dict[str, Any]:
        """Stage 5: SVD / Gram-Schmidt Null(W_sym) projection guaranteeing 0.0 drift on anchors."""
        rc = self.grid.runtime_constants
        reg = float(rc["ridge_regularization"])
        eps = float(rc["eps_floor"])
        tol = float(rc["nullspace_rank_tolerance"])
        drift_digits = int(rc["attenuation_round_digits"])

        anchor_pool: List[List[float]] = []
        for sym_row in w_sym:
            if any(abs(v) > tol for v in sym_row):
                anchor_pool.append([float(v) for v in sym_row])
        if anchor_vectors:
            for av in anchor_vectors:
                padded = [float(av[i]) if i < len(av) else 0.0 for i in range(buckets)]
                anchor_pool.append(padded)
        if anchor_texts:
            for atxt in anchor_texts:
                anchor_pool.append(self.boil_engine._featurize_text(str(atxt), buckets))

        q_basis, solver_backend = self._compute_anchor_orthonormal_basis(anchor_pool)

        label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
        if null_edit:
            delta_null = [[0.0] * buckets for _ in range(num_classes)]
        else:
            # Fit direct linear residual in Null(W_sym) over input feature space
            x_proj_rows: List[List[float]] = []
            y_targets: List[List[float]] = []
            for item in rows:
                xv = self.boil_engine._featurize_text(str(item["input_text"]), buckets)
                xv_null = self._project_rows_into_nullspace([xv], q_basis)[0]
                x_proj_rows.append(xv_null)
                y_row = [0.0] * num_classes
                c_idx = label_to_idx.get(str(item["target_label"]), 0)
                y_row[c_idx] = 1.0
                y_targets.append(y_row)

            gram = [[0.0] * buckets for _ in range(buckets)]
            rhs = [[0.0] * num_classes for _ in range(buckets)]
            for r_idx in range(len(x_proj_rows)):
                xp = x_proj_rows[r_idx]
                yt = y_targets[r_idx]
                for i in range(buckets):
                    xpi = xp[i]
                    if xpi == 0.0:
                        continue
                    for j in range(buckets):
                        gram[i][j] += xpi * xp[j]
                    for c_i in range(num_classes):
                        rhs[i][c_i] += xpi * yt[c_i]
            for i in range(buckets):
                gram[i][i] += reg
            solved_cols = self.boil_engine._solve_linear_system(gram, rhs, eps)
            raw_delta_rows = [
                [solved_cols[b_i][c_i] for b_i in range(buckets)]
                for c_i in range(num_classes)
            ]
            delta_null = self._project_rows_into_nullspace(raw_delta_rows, q_basis)

        max_drift = 0.0
        for a_vec in anchor_pool:
            for c_i in range(num_classes):
                base_out = sum(float(w_sym[c_i][b_i]) * a_vec[b_i] for b_i in range(buckets))
                edited_out = sum(
                    (float(w_sym[c_i][b_i]) + delta_null[c_i][b_i]) * a_vec[b_i]
                    for b_i in range(buckets)
                )
                drift = abs(edited_out - base_out)
                if drift > max_drift:
                    max_drift = drift

        return {
            "nullspace_solver": solver_backend,
            "anchor_constraint_count": len(anchor_pool),
            "anchor_rank": len(q_basis),
            "nullspace_free_dims": max(0, buckets - len(q_basis)),
            "anchor_max_drift": round(max_drift, drift_digits),
            "raw_anchor_max_drift": max_drift,
            "delta_null": delta_null,
        }

    def _score_unit_inputs(
        self,
        xv: Sequence[float],
        w_sym: Sequence[Sequence[float]],
        pruned_proj: Sequence[Sequence[float]],
        v_mat: Sequence[Sequence[float]],
        u_mat: Sequence[Sequence[float]],
        r_mat: Sequence[Sequence[float]],
        delta_null: Sequence[Sequence[float]],
        slots: Sequence[Mapping[str, Any]],
        contrastive_bias: Sequence[float],
        num_classes: int,
    ) -> List[float]:
        buckets = len(xv)
        feat_dim = len(v_mat)
        rank_r = len(r_mat)
        phi = self.boil_engine._polytope_lift(xv, pruned_proj)
        sv = [sum(phi[d_i] * float(v_mat[d_i][r_i]) for d_i in range(feat_dim)) for r_i in range(rank_r)]
        qv = [sum(sv[r_i] * float(r_mat[r_i][r_j]) for r_i in range(rank_r)) for r_j in range(rank_r)]
        lora_logits = [sum(qv[r_j] * float(u_mat[r_j][c_i]) for r_j in range(rank_r)) for c_i in range(num_classes)]

        slot_bonus = [0.0] * num_classes
        for sl in slots:
            raw_v = sl["raw_vec"]
            sim = sum(float(xv[b_i]) * float(raw_v[b_i]) for b_i in range(buckets))
            if sim > 0.0:
                slot_bonus[int(sl["class_index"])] += sim * float(self.grid.runtime_constants["steered_contrastive_scale"])

        scores: List[float] = []
        for c_i in range(num_classes):
            sym_act = max(0.0, sum(float(w_sym[c_i][b_i]) * float(xv[b_i]) for b_i in range(buckets)))
            null_act = sum(float(delta_null[c_i][b_i]) * float(xv[b_i]) for b_i in range(buckets))
            bias_c = float(contrastive_bias[c_i]) if c_i < len(contrastive_bias) else 0.0
            scores.append(sym_act + lora_logits[c_i] + null_act + slot_bonus[c_i] + bias_c)
        return scores

    def compile_weight_unit(
        self,
        namespace: str,
        model_class: str = "lora_xs_cartridge",
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        holdout_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        episodic_memories: Optional[Sequence[Mapping[str, Any]]] = None,
        anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
        anchor_texts: Optional[Sequence[str]] = None,
        maker_lineage: str = "maker_primary",
        checker_lineage: str = "checker_independent",
        subspace_rank: Optional[int] = None,
        safetensors_path: Optional[str] = None,
        null_edit: bool = False,
        enforce_thinking_off_gate: Optional[bool] = None,
        export_dir: Optional[Path] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all 6 stages across any of the 6 model classes with lineage & thinking=off gating."""
        t0 = time.perf_counter()
        rc = self.grid.runtime_constants
        cell = self.grid.resolve_cell(regime)
        m_class = str(model_class or "lora_xs_cartridge").strip().lower()
        if m_class not in MODEL_CLASSES:
            return {
                "ok": False,
                "error": "unsupported_model_class",
                "supported_model_classes": list(MODEL_CLASSES),
            }

        clean_maker = str(maker_lineage or "").strip()
        clean_checker = str(checker_lineage or "").strip()
        if not clean_maker or not clean_checker or clean_maker == clean_checker:
            return {
                "ok": False,
                "promoted": False,
                "error": "shared_maker_checker_lineage_rejected",
                "maker_lineage": clean_maker,
                "checker_lineage": clean_checker,
            }

        loaded_r: Optional[List[List[float]]] = None
        if safetensors_path:
            st_file = Path(str(safetensors_path))
            if not st_file.exists():
                return {"ok": False, "error": "safetensors_file_not_found", "safetensors_path": str(st_file)}
            parsed_st = decode_safetensors_bytes(
                st_file.read_bytes(),
                header_len_bytes=int(rc["safetensors_header_len_bytes"]),
                f32_width=int(rc["float32_byte_width"]),
            )
            if "r_matrix" in parsed_st["tensors"]:
                t_info = parsed_st["tensors"]["r_matrix"]
                r_dim = int(t_info["shape"][0])
                vals = t_info["values"]
                loaded_r = [vals[i * r_dim : (i + 1) * r_dim] for i in range(r_dim)]

        rows = list(samples) if samples is not None else self.store.list_execution_traces(namespace)
        min_support = int(cell["boil_down_min_trace_support"])
        if len(rows) < min_support:
            return {
                "ok": False,
                "error": "insufficient_trace_support",
                "recorded_traces": len(rows),
                "required_traces": min_support,
            }

        labels = sorted({str(r["target_label"]) for r in rows})
        num_classes = len(labels)
        if num_classes <= 1:
            return {"ok": False, "error": "need_at_least_two_distinct_labels", "labels": labels}

        buckets = int(cell["micro_feature_buckets"])
        init_hidden = int(cell["micro_hidden_units"])
        keep_hidden = (
            int(cell["projected_subspace_dim"])
            if m_class == "embedding"
            else int(cell["pruned_hidden_units"])
        )
        rank_r = (
            len(loaded_r)
            if loaded_r is not None
            else max(1, int(subspace_rank) if subspace_rank is not None else int(cell["lora_xs_subspace_rank"]))
        )
        max_slots = int(cell["memory_lora_max_slots"])

        def _fit_all_stages(
            train_rows: Sequence[Mapping[str, Any]],
            seed_val: int,
            mask_seed: Optional[int],
        ) -> Dict[str, Any]:
            s1 = self._induce_symbolic_skeleton(train_rows, labels, buckets, mask_seed=mask_seed)
            s2 = self._prune_channels_and_fit_ridge(
                namespace=namespace,
                model_class=m_class,
                rows=train_rows,
                x_cache=s1["x_cache"],
                labels=labels,
                buckets=buckets,
                initial_hidden=init_hidden,
                retained_hidden=keep_hidden,
                w_sym=s1["w_sym"],
            )
            s3 = self._fit_subspace_cartridge(
                namespace=namespace,
                model_class=m_class,
                x_cache=s1["x_cache"],
                pruned_proj=s2["pruned_proj"],
                residual_targets=s2["residual_targets"],
                subspace_rank=rank_r,
                num_classes=num_classes,
                seed_tag=seed_val,
                null_edit=null_edit,
                loaded_r_mat=loaded_r,
            )
            s4 = self._compile_memory_as_lora_slots(
                namespace=namespace,
                rows=train_rows,
                labels=labels,
                buckets=buckets,
                max_slots=max_slots,
                episodic_memories=episodic_memories,
            )
            s5 = self._project_residual_onto_anchor_nullspace(
                buckets=buckets,
                num_classes=num_classes,
                w_sym=s1["w_sym"],
                rows=train_rows,
                labels=labels,
                anchor_vectors=anchor_vectors,
                anchor_texts=anchor_texts,
                null_edit=null_edit,
            )
            # Contrastive class prior folding for steered_llm
            c_counts = [0] * num_classes
            l_map = {lab: idx for idx, lab in enumerate(labels)}
            for item in train_rows:
                c_counts[l_map.get(str(item["target_label"]), 0)] += 1
            c_bias = [
                (float(c_counts[c_i]) / float(max(1, len(train_rows))) - (1.0 / float(num_classes)))
                * float(rc["steered_contrastive_scale"])
                if m_class == "steered_llm" and not null_edit
                else 0.0
                for c_i in range(num_classes)
            ]
            return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "c_bias": c_bias}

        def _eval_split(
            train_s: Sequence[Mapping[str, Any]],
            eval_s: Sequence[Mapping[str, Any]],
            seed_val: int,
            mask_seed: Optional[int],
        ) -> float:
            fitted = _fit_all_stages(train_s, seed_val=seed_val, mask_seed=mask_seed)
            hits = 0
            for item in eval_s:
                xv = self.boil_engine._featurize_text(str(item["input_text"]), buckets, mask_seed=mask_seed)
                sc = self._score_unit_inputs(
                    xv=xv,
                    w_sym=fitted["s1"]["w_sym"],
                    pruned_proj=fitted["s2"]["pruned_proj"],
                    v_mat=fitted["s3"]["v_mat"],
                    u_mat=fitted["s3"]["u_mat"],
                    r_mat=fitted["s3"]["r_mat"],
                    delta_null=fitted["s5"]["delta_null"],
                    slots=fitted["s4"]["slots"],
                    contrastive_bias=fitted["c_bias"],
                    num_classes=num_classes,
                )
                p_idx = max(range(num_classes), key=lambda c: sc[c])
                if labels[p_idx] == str(item["target_label"]):
                    hits += 1
            return float(hits) / float(max(1, len(eval_s)))

        # Stage 6: Matched thinking=off control gate + incumbent regression check + canary step
        gate = self.boil_engine._evaluate_thinking_off_confound_gate(
            rows=rows,
            labels=labels,
            candidate_eval_fn=_eval_split,
            holdout_samples=holdout_samples,
            null_edit=null_edit,
            regime=regime,
            enforce_gate=enforce_thinking_off_gate,
        )

        full_fit = _fit_all_stages(rows, seed_val=0, mask_seed=None)
        s1, s2, s3, s4, s5 = (
            full_fit["s1"],
            full_fit["s2"],
            full_fit["s3"],
            full_fit["s4"],
            full_fit["s5"],
        )

        ukey = self._unit_key(namespace, m_class)
        incumbent = self.promoted_units.get(ukey)
        cand_holdout = float(gate["candidate_holdout_accuracy"])

        if not gate["passed"]:
            return {
                "ok": False,
                "promoted": False,
                "rolled_back": incumbent is not None,
                "error": "blocked_by_thinking_off_confound_gate",
                "namespace": namespace,
                "model_class": m_class,
                "pipeline_stages_completed": int(rc["pipeline_stage_count"]),
                "cartridge_bytes": s3["cartridge_bytes"],
                "anchor_max_drift": s5["anchor_max_drift"],
                "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
                "confound_gate": gate,
            }

        if incumbent is not None and cand_holdout + float(rc["eps_floor"]) < float(incumbent["holdout_accuracy"]):
            self.store.canary_policy_step(
                policy_id=f"weight_unit:{ukey}",
                mode="record",
                baseline_arm="incumbent_weight_unit",
                canary_arm="candidate_weight_unit",
                used_canary=True,
                reward=cand_holdout,
            )
            return {
                "ok": False,
                "promoted": False,
                "rolled_back": True,
                "error": "canary_holdout_regression_auto_rolled_back",
                "namespace": namespace,
                "model_class": m_class,
                "incumbent_holdout_accuracy": incumbent["holdout_accuracy"],
                "candidate_holdout_accuracy": cand_holdout,
                "active_unit_generation": incumbent["unit_generation"],
            }

        # Record canary promotion telemetry
        canary_res = self.store.canary_policy_step(
            policy_id=f"weight_unit:{ukey}",
            mode="record",
            baseline_arm="direct_thinking_off",
            canary_arm=f"{m_class}_v040",
            used_canary=True,
            reward=cand_holdout,
        )

        # Compute train accuracy on full trace corpus
        train_hits = 0
        for r_idx, item in enumerate(rows):
            sc = self._score_unit_inputs(
                xv=s1["x_cache"][r_idx],
                w_sym=s1["w_sym"],
                pruned_proj=s2["pruned_proj"],
                v_mat=s3["v_mat"],
                u_mat=s3["u_mat"],
                r_mat=s3["r_mat"],
                delta_null=s5["delta_null"],
                slots=s4["slots"],
                contrastive_bias=full_fit["c_bias"],
                num_classes=num_classes,
            )
            best_c = max(range(num_classes), key=lambda c: sc[c])
            if labels[best_c] == str(item["target_label"]):
                train_hits += 1
        train_acc = round(float(train_hits) / float(len(rows)), int(rc["ratio_round_digits"]))

        # Export spec-compliant .safetensors + .bin cartridges if export_dir provided
        st_out_path: Optional[str] = None
        bin_out_path: Optional[str] = None
        flat_r = [float(s3["r_mat"][i][j]) for i in range(rank_r) for j in range(rank_r)]
        flat_sym = [float(s1["w_sym"][c_i][b_i]) for c_i in range(num_classes) for b_i in range(buckets)]
        flat_null = [float(s5["delta_null"][c_i][b_i]) for c_i in range(num_classes) for b_i in range(buckets)]
        st_bytes = encode_safetensors_bytes(
            tensors={
                "r_matrix": ([rank_r, rank_r], flat_r),
                "w_sym": ([num_classes, buckets], flat_sym),
                "delta_null": ([num_classes, buckets], flat_null),
            },
            metadata={
                "namespace": namespace,
                "model_class": m_class,
                "maker_lineage": clean_maker,
                "checker_lineage": clean_checker,
            },
            f32_width=int(rc["float32_byte_width"]),
        )
        if export_dir is not None:
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
                st_file = export_dir / f"weight_unit_{namespace}_{m_class}.safetensors"
                st_file.write_bytes(st_bytes)
                st_out_path = str(st_file)
                bin_file = export_dir / f"cartridge_{namespace}_{m_class}.bin"
                bin_file.write_bytes(s3["packed_bin"])
                bin_out_path = str(bin_file)
            except Exception:
                st_out_path = None
                bin_out_path = None

        prev_gen = int(incumbent["unit_generation"]) if incumbent is not None else 0
        if incumbent is not None:
            self.unit_history.setdefault(ukey, []).append(incumbent)

        posting_prune_pct = round(
            float(rc["search_posting_prune_quantile_permille"])
            / float(rc["permille_scale"])
            * float(rc["percent_scale"]),
            int(rc["ratio_round_digits"]),
        )
        elapsed_ms = round(
            (time.perf_counter() - t0) * float(rc["ms_per_second"]),
            int(rc["ngram_span"]),
        )

        unit_spec: Dict[str, Any] = {
            "namespace": namespace,
            "model_class": m_class,
            "unit_generation": prev_gen + 1,
            "maker_lineage": clean_maker,
            "checker_lineage": clean_checker,
            "buckets": buckets,
            "labels": labels,
            "subspace_rank": rank_r,
            "cartridge_bytes": s3["cartridge_bytes"],
            "cartridge_b64": s3["cartridge_b64"],
            "safetensors_path": st_out_path,
            "cartridge_bin_path": bin_out_path,
            "w_sym": s1["w_sym"],
            "pruned_proj": s2["pruned_proj"],
            "v_mat": s3["v_mat"],
            "u_mat": s3["u_mat"],
            "r_mat": s3["r_mat"],
            "slots": s4["slots"],
            "delta_null": s5["delta_null"],
            "contrastive_bias": full_fit["c_bias"],
            "ram_reduction_pct": s2["ram_reduction_pct"],
            "posting_reduction_pct": posting_prune_pct if m_class == "search_retrieval" else 0.0,
            "context_tokens_eliminated": s4["context_tokens_eliminated"],
            "anchor_max_drift": s5["anchor_max_drift"],
            "train_accuracy": train_acc,
            "holdout_accuracy": cand_holdout,
            "compile_ms": elapsed_ms,
        }
        self.promoted_units[ukey] = unit_spec
        self.promoted_units[namespace] = unit_spec

        # Also mirror into boil_engine so /v1/chat/completions can serve 0-token inference immediately
        self.boil_engine.lora_xs_cartridges[namespace] = {
            "backend": "lora_xs_subspace_cartridge",
            "namespace": namespace,
            "subspace_rank": rank_r,
            "cartridge_bytes": s3["cartridge_bytes"],
            "cartridge_b64": s3["cartridge_b64"],
            "cartridge_path": bin_out_path,
            "buckets": buckets,
            "hidden": len(s2["pruned_proj"]),
            "feat_dim": s3["feat_dim"],
            "labels": labels,
            "proj_w": s2["pruned_proj"],
            "v_mat": s3["v_mat"],
            "u_mat": s3["u_mat"],
            "r_mat": s3["r_mat"],
            "train_accuracy": train_acc,
            "compile_ms": elapsed_ms,
        }
        self.boil_engine.compiled_routers[namespace] = self.boil_engine.lora_xs_cartridges[namespace]

        return {
            "ok": True,
            "promoted": True,
            "rolled_back": False,
            "namespace": namespace,
            "model_class": m_class,
            "unit_generation": prev_gen + 1,
            "maker_lineage": clean_maker,
            "checker_lineage": clean_checker,
            "labels": labels,
            "pipeline_stages_completed": int(rc["pipeline_stage_count"]),
            "stages": {
                "stage_1_symbolic_skeleton": {
                    "invariant_count": s1["invariant_count"],
                    "gradient_epochs": s1["gradient_epochs"],
                    "symbolic_rule_violation_rate": s1["symbolic_rule_violation_rate"],
                },
                "stage_2_activation_channel_pruning": {
                    "initial_channels": s2["initial_channels"],
                    "retained_channels": s2["retained_channels"],
                    "ram_reduction_pct": s2["ram_reduction_pct"],
                    "recalibrated_ridge_mse": s2["recalibrated_ridge_mse"],
                },
                "stage_3_lora_xs_subspace": {
                    "subspace_rank": rank_r,
                    "cartridge_bytes": s3["cartridge_bytes"],
                    "cartridge_b64": s3["cartridge_b64"],
                    "safetensors_path": st_out_path,
                },
                "stage_4_memory_as_lora_slots": {
                    "slot_count": s4["slot_count"],
                    "slot_orthogonality_max_inner_product": s4["slot_orthogonality_max_inner_product"],
                    "context_tokens_eliminated": s4["context_tokens_eliminated"],
                },
                "stage_5_svd_anchor_nullspace": {
                    "nullspace_solver": s5["nullspace_solver"],
                    "anchor_constraint_count": s5["anchor_constraint_count"],
                    "anchor_rank": s5["anchor_rank"],
                    "nullspace_free_dims": s5["nullspace_free_dims"],
                    "anchor_max_drift": s5["anchor_max_drift"],
                },
                "stage_6_matched_gate_and_canary": {
                    "gate_verdict": gate["verdict"],
                    "direct_thinking_off_accuracy": gate["direct_thinking_off_accuracy"],
                    "candidate_holdout_accuracy": cand_holdout,
                    "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
                    "canary_event": canary_res["event"],
                },
            },
            "train_accuracy": train_acc,
            "holdout_accuracy": cand_holdout,
            "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
            "anchor_max_drift": s5["anchor_max_drift"],
            "ram_reduction_pct": s2["ram_reduction_pct"],
            "cartridge_bytes": s3["cartridge_bytes"],
            "safetensors_path": st_out_path,
            "compile_ms": elapsed_ms,
            "llm_tokens_per_inference": 0,
        }

    def predict_weight_unit(
        self,
        namespace: str,
        input_text: str,
        model_class: Optional[str] = None,
        candidate_docs: Optional[Sequence[Mapping[str, Any]]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run 0-LLM-token inference on any promoted 6-stage weight unit."""
        t0 = time.perf_counter()
        rc = self.grid.runtime_constants
        ukey = self._unit_key(namespace, str(model_class)) if model_class else namespace
        spec = self.promoted_units.get(ukey) or self.promoted_units.get(namespace)
        if spec is None:
            return {"ok": False, "error": "no_promoted_weight_unit_for_namespace"}

        m_class = str(spec["model_class"])
        buckets = int(spec["buckets"])
        labels: List[str] = list(spec["labels"])
        num_classes = len(labels)
        xv = self.boil_engine._featurize_text(input_text, buckets)
        scores = self._score_unit_inputs(
            xv=xv,
            w_sym=spec["w_sym"],
            pruned_proj=spec["pruned_proj"],
            v_mat=spec["v_mat"],
            u_mat=spec["u_mat"],
            r_mat=spec["r_mat"],
            delta_null=spec["delta_null"],
            slots=spec["slots"],
            contrastive_bias=spec["contrastive_bias"],
            num_classes=num_classes,
        )
        best_c = max(range(num_classes), key=lambda c: scores[c])
        pred_label = labels[best_c]
        eps = float(rc["eps_floor"])
        round_d = int(rc["ratio_round_digits"])

        extra: Dict[str, Any] = {}
        if m_class == "embedding":
            raw_emb = [
                sum(float(row_w[b_i]) * xv[b_i] for b_i in range(buckets))
                for row_w in spec["pruned_proj"]
            ]
            emb_norm = math.sqrt(sum(v * v for v in raw_emb) + eps)
            extra["embedding_dim"] = len(raw_emb)
            extra["whitened_embedding"] = [round(v / emb_norm, round_d) for v in raw_emb]
            extra["ram_reduction_pct"] = spec["ram_reduction_pct"]
        elif m_class == "search_retrieval":
            w_lex = float(rc["rrf_lexical_weight_default"])
            w_bilin = float(rc["rrf_bilinear_weight_default"])
            q_tokens = {tok for tok in input_text.lower().split() if tok}
            docs = list(candidate_docs or [])
            ranked_docs: List[Dict[str, Any]] = []
            for d_idx, doc in enumerate(docs):
                d_txt = str(doc.get("text") or "")
                d_tokens = {tok for tok in d_txt.lower().split() if tok}
                lex_ov = float(len(q_tokens & d_tokens)) / float(max(1, len(q_tokens)))
                dv = self.boil_engine._featurize_text(d_txt, buckets)
                bilin = max(0.0, sum(xv[b_i] * dv[b_i] for b_i in range(buckets)))
                fused = w_lex * lex_ov + w_bilin * bilin
                ranked_docs.append(
                    {
                        "doc_id": str(doc.get("doc_id") or f"doc_{d_idx + 1}"),
                        "fused_score": round(fused, round_d),
                    }
                )
            ranked_docs.sort(key=lambda item: item["fused_score"], reverse=True)
            extra["ranked_docs"] = ranked_docs
            extra["posting_reduction_pct"] = spec["posting_reduction_pct"]
        elif m_class == "decision_router":
            _, cal = self.budget_gate.calibrate_llm_request(
                {"messages": [{"role": "user", "content": input_text}]},
                regime=regime,
            )
            cell = self.grid.resolve_cell(regime)
            extra["recommended_effort"] = cal["selected_effort"]
            extra["recommended_bundle_k"] = int(cell["bundle_cardinality_k"])
            extra["recommended_dispatch_mode"] = str(cell["dispatch_affinity_mode"])
        elif m_class == "memory_as_lora":
            best_slot = None
            best_sim = -1.0
            for sl in spec["slots"]:
                sim = sum(xv[b_i] * float(sl["raw_vec"][b_i]) for b_i in range(buckets))
                if sim > best_sim:
                    best_sim = sim
                    best_slot = sl["slot_id"]
            extra["matched_slot_id"] = best_slot
            extra["context_tokens_saved"] = spec["context_tokens_eliminated"]
        elif m_class == "steered_llm":
            extra["steering_logits"] = [round(s, round_d) for s in scores]
            extra["ram_reduction_pct"] = spec["ram_reduction_pct"]

        elapsed_ms = (time.perf_counter() - t0) * float(rc["ms_per_second"])
        return {
            "ok": True,
            "namespace": namespace,
            "model_class": m_class,
            "unit_generation": spec["unit_generation"],
            "predicted_label": pred_label,
            "cartridge_bytes": spec["cartridge_bytes"],
            "anchor_max_drift": spec["anchor_max_drift"],
            "latency_ms": round(elapsed_ms, int(rc["ngram_span"])),
            "llm_tokens_spent": 0,
            **extra,
        }

    def rollback_weight_unit(self, namespace: str, model_class: str = "lora_xs_cartridge") -> Dict[str, Any]:
        """Instant <1 ms auto-rollback to the prior promoted weight unit snapshot."""
        ukey = self._unit_key(namespace, model_class)
        hist = self.unit_history.get(ukey) or []
        if not hist:
            return {"ok": False, "error": "no_prior_unit_snapshot_to_restore", "namespace": namespace}
        restored = hist.pop()
        self.promoted_units[ukey] = restored
        self.promoted_units[namespace] = restored
        return {
            "ok": True,
            "rolled_back": True,
            "namespace": namespace,
            "model_class": model_class,
            "restored_generation": restored["unit_generation"],
            "restored_holdout_accuracy": restored["holdout_accuracy"],
        }


class HorizonTriadDecomposer:
    """Universal Actor x {Transfer, Transform, Save} x Horizon check-gated atomic DAG compiler."""

    def __init__(self, grid: AdaptiveNodeGrid, store: LocalAdaptiveFabricStore) -> None:
        self.grid = grid
        self.store = store

    def enumerate_triad_catalog(self) -> Dict[str, Any]:
        combos = list(itertools.product(TRIAD_OPS, repeat=int(self.grid.runtime_constants["ngram_span"])))
        entries = [
            {
                "triad_index": idx + 1,
                "triad_code": "-".join(step[0].upper() for step in c),
                "steps": list(c),
                "ends_with_bitemporal_save": c[-1] == "save",
            }
            for idx, c in enumerate(combos)
        ]
        return {
            "ok": True,
            "base_ops": list(TRIAD_OPS),
            "actor_modes": list(ACTOR_MODES),
            "total_three_step_triads": len(entries),
            "triads": entries,
        }

    def decompose_horizon(
        self,
        workflow_id: str,
        goal: str,
        actor_mode: str = "swarm",
        triad_sequence: Optional[Sequence[str]] = None,
        horizon_steps: Optional[int] = None,
        checkpoint_cadence_steps: Optional[int] = None,
        benchmark_archetype: str = "general_horizon",
        namespace: str = "horizon_dag",
        maker_lineage: str = "maker_pool_a",
        checker_lineage: str = "checker_pool_b",
        enqueue_immediately: bool = True,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        cell = self.grid.resolve_cell(regime)

        actor = str(actor_mode or "swarm").strip().lower()
        if actor not in ACTOR_MODES:
            return {"ok": False, "error": "unsupported_actor_mode", "supported_actors": list(ACTOR_MODES)}

        archetype = str(benchmark_archetype or "general_horizon").strip().lower()
        if archetype not in BENCHMARK_ARCHETYPES:
            archetype = "general_horizon"

        max_h = int(cell["horizon_max_steps"])
        steps_h = max(1, min(max_h, int(horizon_steps) if horizon_steps is not None else int(rc["macro_dag_phase_count"])))
        cadence = max(
            1,
            min(
                steps_h,
                int(checkpoint_cadence_steps)
                if checkpoint_cadence_steps is not None
                else int(cell["checkpoint_cadence_steps"]),
            ),
        )

        clean_seq = [
            str(op).strip().lower()
            for op in (triad_sequence or TRIAD_OPS)
            if str(op).strip().lower() in TRIAD_OPS
        ]
        if not clean_seq:
            clean_seq = list(TRIAD_OPS)

        full_milli = float(rc["full_confidence_milli"])
        if actor == "person":
            base_milli = int(rc["person_step_confidence_milli"])
        elif actor == "machine":
            base_milli = int(rc["machine_step_confidence_milli"])
        else:
            base_milli = int(rc["swarm_step_confidence_milli"])
        step_p = float(base_milli) / full_milli

        disjoint_lineage = bool(
            str(maker_lineage).strip()
            and str(checker_lineage).strip()
            and str(maker_lineage).strip() != str(checker_lineage).strip()
        )
        round_d = int(rc["ratio_round_digits"])

        now = time.time()
        nodes: List[Dict[str, Any]] = []
        raw_surv = 1.0
        gated_surv = 1.0
        checkpoint_count = 0

        for idx in range(steps_h):
            step_num = idx + 1
            triad_op = clean_seq[idx % len(clean_seq)]
            is_save_boundary = (triad_op == "save") or (step_num % cadence == 0)
            checked_reset = is_save_boundary and disjoint_lineage

            raw_surv *= step_p
            if checked_reset:
                gated_surv = 1.0
                checkpoint_count += 1
                eff_milli = int(rc["checked_save_reset_milli"])
            else:
                gated_surv *= step_p
                eff_milli = base_milli

            tid = f"{workflow_id}:h{step_num}:{triad_op}"
            deps = [nodes[-1]["task_id"]] if nodes else []
            pfx_sig = f"pfx_{archetype}_{triad_op}"
            schema_tag = f"{archetype}:{triad_op}"

            node_info = {
                "step_index": step_num,
                "task_id": tid,
                "actor_mode": actor,
                "triad_op": triad_op,
                "benchmark_archetype": archetype,
                "depends_on": deps,
                "bitemporal_checkpoint": checked_reset,
                "valid_from": round(now + float(idx), int(rc["pair_divisor"])),
                "unguarded_survival": round(raw_surv, round_d),
                "check_gated_survival": round(gated_surv, round_d),
                "prefix_signature": pfx_sig,
                "tool_schema_tag": schema_tag,
            }
            nodes.append(node_info)

            if enqueue_immediately:
                self.store.enqueue_task(
                    task_id=tid,
                    namespace=namespace,
                    operation=f"{actor}_{triad_op}",
                    payload={
                        "goal": goal,
                        "step_index": step_num,
                        "actor_mode": actor,
                        "triad_op": triad_op,
                        "benchmark_archetype": archetype,
                        "bitemporal_checkpoint": checked_reset,
                        "maker_lineage": maker_lineage,
                        "checker_lineage": checker_lineage,
                    },
                    depends_on=deps,
                    prefix_signature=pfx_sig,
                    capability_tag=f"{actor}:{triad_op}",
                    tool_schema_tag=schema_tag,
                    origin_id=workflow_id,
                    dag_depth=idx,
                    dag_width=1,
                    granularity_tier="atomic",
                    step_confidence_milli=eff_milli,
                    regime=regime,
                )

        return {
            "ok": True,
            "workflow_id": workflow_id,
            "namespace": namespace,
            "actor_mode": actor,
            "benchmark_archetype": archetype,
            "horizon_steps": steps_h,
            "checkpoint_cadence_steps": cadence,
            "disjoint_maker_checker_lineage": disjoint_lineage,
            "bitemporal_checkpoints": checkpoint_count,
            "final_unguarded_survival": round(raw_surv, round_d),
            "final_check_gated_survival": round(gated_surv, round_d),
            "survival_gain": round(gated_surv - raw_surv, round_d),
            "enqueued": enqueue_immediately,
            "recommended_dispatch_modes": list(DISPATCH_AFFINITY_MODES),
            "nodes": nodes,
        }


class RecursiveHoldoutTrialEngine:
    """Leak-free holdout UCB + policy-gradient exploration over NodeGrid cells and weight cartridges."""

    def __init__(self, grid: AdaptiveNodeGrid, store: LocalAdaptiveFabricStore) -> None:
        self.grid = grid
        self.store = store
        self._init_trial_tables()

    def _init_trial_tables(self) -> None:
        with self.store._lock:
            conn = self.store._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS recursive_trials (
                        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        campaign_id TEXT NOT NULL,
                        arm_key TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        cartridge_id TEXT NOT NULL,
                        train_reward REAL NOT NULL,
                        holdout_reward REAL NOT NULL,
                        thinking_off_holdout_reward REAL NOT NULL,
                        generalization_gap REAL NOT NULL,
                        cost REAL NOT NULL,
                        maker_id TEXT NOT NULL,
                        checker_id TEXT NOT NULL,
                        maker_lineage TEXT NOT NULL,
                        checker_lineage TEXT NOT NULL,
                        promoted INTEGER NOT NULL,
                        verdict TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS recursive_arm_preferences (
                        campaign_id TEXT NOT NULL,
                        arm_key TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        cartridge_id TEXT NOT NULL,
                        preference_logit REAL NOT NULL DEFAULT 0.0,
                        trial_count INTEGER NOT NULL DEFAULT 0,
                        best_holdout REAL NOT NULL DEFAULT 0.0,
                        PRIMARY KEY (campaign_id, arm_key)
                    );
                    """
                )
            finally:
                self.store._close_if_disk(conn)

    @staticmethod
    def _arm_key(params: Mapping[str, Any], cartridge_id: str) -> str:
        p_str = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
        return f"{cartridge_id}::{p_str}"

    def propose_trial(
        self,
        campaign_id: str,
        cartridge_ids: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        ucb_c = float(rc["ucb_exploration_weight"])
        round_d = int(rc["ratio_round_digits"])
        k = max(1, min(int(rc["max_batch_proposals"]), int(batch_size or rc["default_batch_proposals"])))
        carts = [str(c) for c in (cartridge_ids or ["cartridge_default"])]

        grid_prop = self.store.propose_optimization_batch(
            campaign_id=campaign_id,
            strategy="ucb_surrogate",
            batch_size=k,
        )
        candidate_cells = (
            grid_prop["proposals"]
            if grid_prop.get("ok") and grid_prop.get("proposals")
            else [self.grid.resolve_cell(regime)]
        )

        with self.store._lock:
            conn = self.store._connect()
            try:
                pref_rows = {
                    str(r["arm_key"]): {
                        "logit": float(r["preference_logit"]),
                        "n": int(r["trial_count"]),
                        "best_holdout": float(r["best_holdout"]),
                    }
                    for r in conn.execute(
                        "SELECT * FROM recursive_arm_preferences WHERE campaign_id = ?",
                        (campaign_id,),
                    ).fetchall()
                }
            finally:
                self.store._close_if_disk(conn)

        total_trials = 1 + sum(item["n"] for item in pref_rows.values())
        scored_arms: List[Dict[str, Any]] = []
        raw_logits: List[float] = []
        for params in candidate_cells:
            for cid in carts:
                akey = self._arm_key(params, cid)
                st = pref_rows.get(akey, {"logit": 0.0, "n": 0, "best_holdout": 0.0})
                raw_logits.append(st["logit"])
                ucb_bonus = (
                    float(rc["percent_scale"])
                    if st["n"] == 0
                    else ucb_c * math.sqrt(math.log(float(total_trials)) / float(st["n"]))
                )
                scored_arms.append(
                    {
                        "arm_key": akey,
                        "params": params,
                        "cartridge_id": cid,
                        "preference_logit": round(st["logit"], round_d),
                        "trial_count": st["n"],
                        "combined_acquisition_score": round(st["best_holdout"] + st["logit"] + ucb_bonus, round_d),
                    }
                )

        max_l = max(raw_logits) if raw_logits else 0.0
        exp_vals = [math.exp(l_v - max_l) for l_v in raw_logits]
        denom = sum(exp_vals) if exp_vals else 1.0
        for idx, arm in enumerate(scored_arms):
            arm["policy_probability"] = round(exp_vals[idx] / denom, round_d)

        scored_arms.sort(key=lambda item: item["combined_acquisition_score"], reverse=True)
        return {
            "ok": True,
            "campaign_id": campaign_id,
            "proposals": scored_arms[:k],
        }

    def record_trial(
        self,
        campaign_id: str,
        params: Mapping[str, Any],
        cartridge_id: str,
        train_reward: float,
        holdout_reward: float,
        thinking_off_holdout_reward: float,
        cost: float = 0.0,
        maker_id: str = "maker_1",
        checker_id: str = "checker_1",
        maker_lineage: str = "lineage_a",
        checker_lineage: str = "lineage_b",
        auto_expand: bool = True,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        cell = self.grid.resolve_cell(regime)
        max_gap = float(rc["max_train_holdout_gap"])
        min_imp = float(rc["min_holdout_improvement"])
        lr = float(cell["policy_gradient_step_milli"]) / float(rc["full_confidence_milli"])
        round_d = int(rc["ratio_round_digits"])

        t_rew = float(train_reward)
        h_rew = float(holdout_reward)
        off_rew = float(thinking_off_holdout_reward)
        gen_gap = round(t_rew - h_rew, round_d)
        akey = self._arm_key(params, cartridge_id)
        now = time.time()

        disjoint_ids = bool(str(maker_id).strip() and str(checker_id).strip() and str(maker_id) != str(checker_id))
        disjoint_lin = bool(
            str(maker_lineage).strip()
            and str(checker_lineage).strip()
            and str(maker_lineage) != str(checker_lineage)
        )

        with self.store._lock:
            conn = self.store._connect()
            try:
                inc_row = conn.execute(
                    "SELECT COALESCE(MAX(holdout_reward), 0.0) AS m FROM recursive_trials WHERE campaign_id = ? AND promoted = 1",
                    (campaign_id,),
                ).fetchone()
                incumbent_best = float(inc_row["m"]) if inc_row is not None else 0.0

                if not (disjoint_ids and disjoint_lin):
                    promoted = False
                    verdict = "rejected_shared_lineage"
                elif gen_gap > max_gap:
                    promoted = False
                    verdict = "holdout_overfit_rejected"
                elif h_rew < (off_rew + min_imp):
                    promoted = False
                    verdict = "blocked_by_thinking_off_holdout_gate"
                elif h_rew < incumbent_best:
                    promoted = False
                    verdict = "holdout_regression_auto_rolled_back"
                else:
                    promoted = True
                    verdict = "promoted_on_sealed_holdout"

                baseline_ref = max(off_rew, incumbent_best)
                advantage = (h_rew - baseline_ref) if promoted else -abs(max(min_imp, gen_gap))
                pref_row = conn.execute(
                    "SELECT * FROM recursive_arm_preferences WHERE campaign_id = ? AND arm_key = ?",
                    (campaign_id, akey),
                ).fetchone()
                prev_logit = float(pref_row["preference_logit"]) if pref_row else 0.0
                prev_n = int(pref_row["trial_count"]) if pref_row else 0
                prev_best = float(pref_row["best_holdout"]) if pref_row else 0.0
                new_logit = prev_logit + lr * advantage
                new_best = max(prev_best, h_rew) if promoted else prev_best

                conn.execute(
                    """
                    INSERT INTO recursive_trials (
                        campaign_id, arm_key, params_json, cartridge_id,
                        train_reward, holdout_reward, thinking_off_holdout_reward,
                        generalization_gap, cost, maker_id, checker_id,
                        maker_lineage, checker_lineage, promoted, verdict, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        akey,
                        json.dumps(dict(params), sort_keys=True),
                        cartridge_id,
                        t_rew,
                        h_rew,
                        off_rew,
                        gen_gap,
                        float(cost),
                        str(maker_id),
                        str(checker_id),
                        str(maker_lineage),
                        str(checker_lineage),
                        1 if promoted else 0,
                        verdict,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO recursive_arm_preferences (
                        campaign_id, arm_key, params_json, cartridge_id,
                        preference_logit, trial_count, best_holdout
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        akey,
                        json.dumps(dict(params), sort_keys=True),
                        cartridge_id,
                        new_logit,
                        prev_n + 1,
                        new_best,
                    ),
                )
                conn.commit()
            finally:
                self.store._close_if_disk(conn)

        grid_update: Optional[Dict[str, Any]] = None
        if promoted:
            # Record both maker and independent checker outcomes so NodeGrid lifecycle advances cleanly
            self.store.record_optimization_outcome(
                campaign_id=campaign_id,
                params=params,
                reward=h_rew,
                cost=cost,
                agent_id=maker_id,
                auto_expand=auto_expand,
            )
            grid_update = self.store.record_optimization_outcome(
                campaign_id=campaign_id,
                params=params,
                reward=h_rew,
                cost=cost,
                agent_id=checker_id,
                auto_expand=auto_expand,
            )

        frontier = max(incumbent_best, h_rew) if promoted else incumbent_best
        return {
            "ok": promoted,
            "promoted": promoted,
            "auto_rolled_back": not promoted,
            "verdict": verdict,
            "campaign_id": campaign_id,
            "cartridge_id": cartridge_id,
            "train_reward": round(t_rew, round_d),
            "holdout_reward": round(h_rew, round_d),
            "thinking_off_holdout_reward": round(off_rew, round_d),
            "generalization_gap": gen_gap,
            "sealed_holdout_frontier": round(frontier, round_d),
            "updated_preference_logit": round(new_logit, round_d),
            "grid_update": grid_update,
        }

    def summarize_campaign(self, campaign_id: str) -> Dict[str, Any]:
        round_d = int(self.grid.runtime_constants["ratio_round_digits"])
        with self.store._lock:
            conn = self.store._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM recursive_trials WHERE campaign_id = ? ORDER BY trial_id ASC",
                    (campaign_id,),
                ).fetchall()
            finally:
                self.store._close_if_disk(conn)

        frontier = 0.0
        frontier_curve: List[float] = []
        promoted_count = 0
        rejected_overfit = 0
        trials_out: List[Dict[str, Any]] = []
        for r in rows:
            prom = int(r["promoted"]) == 1
            h_rew = float(r["holdout_reward"])
            if prom and h_rew > frontier:
                frontier = h_rew
            if prom:
                promoted_count += 1
            if str(r["verdict"]) == "holdout_overfit_rejected":
                rejected_overfit += 1
            frontier_curve.append(round(frontier, round_d))
            trials_out.append(
                {
                    "trial_id": int(r["trial_id"]),
                    "cartridge_id": str(r["cartridge_id"]),
                    "train_reward": round(float(r["train_reward"]), round_d),
                    "holdout_reward": round(h_rew, round_d),
                    "generalization_gap": round(float(r["generalization_gap"]), round_d),
                    "promoted": prom,
                    "verdict": str(r["verdict"]),
                }
            )
        return {
            "ok": True,
            "campaign_id": campaign_id,
            "total_trials": len(trials_out),
            "promoted_trials": promoted_count,
            "rejected_overfit_trials": rejected_overfit,
            "sealed_holdout_frontier": round(frontier, round_d),
            "holdout_frontier_curve": frontier_curve,
            "trials": trials_out,
        }
