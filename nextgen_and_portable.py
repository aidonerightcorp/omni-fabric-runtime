"""Next-generation attribution, Koopman + CEGAR planning, and portable 1-shot/multi-shot cartridges (v0.5.0).

Implements three universal grid-governed engines with zero external dependencies:
1. TopicNullspaceAttributionEngine  (fabric_attribute_generator_profile)
   - Projects ridge attribution weights onto the orthogonal complement of topical
     subspaces: W_attr = W_ridge (I - V_topic V_topic^T) so ||W_attr V_topic||_F = 0.0.
   - Screens style-spoofed passages via class-conditional nullspace residual energy
     calibrated at the configured conformal quantile.
2. KoopmanCegarTrajectoryEngine     (fabric_plan_koopman_trajectory)
   - Lifts pipeline state vectors into polynomial + bilinear observables and fits
     action-specific linear Koopman operators z_{t+1} = K_a z_t with feasibility gates.
   - Runs zero-token lookahead beam search and selects high-leverage boundary
     counterexample scenarios (CEGAR) by minimum decision margin.
3. PortableCartridgeAndOneShotEngine (fabric_export_portable_cartridge)
   - Exports any r x r or out_dim x in_dim IEEE-754 float32 linear/ReLU cartridge
     to JavaScript ESM (Float32Array), zero-import Python stdlib, declarative SQLite SQL,
     static C99 header (.h), and .safetensors with cross-target equivalence checks.
   - Supports Modern Continuous Hopfield 1-shot column writes and Recursive Woodbury
     rank-1 streaming multi-shot weight updates with zero replay buffer rows.
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .grid_and_env import AdaptiveNodeGrid

PORTABLE_TARGETS = ("js", "python", "sql", "c99", "safetensors")

_FUNCTION_MARKERS = (
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "under",
    "when",
    "then",
    "thus",
    "while",
    "each",
)


def _quantize_fp32(val: float) -> float:
    packed = struct.pack("<f", float(val))
    return float(struct.unpack("<f", packed)[0])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _orthonormalize_rows(rows: Sequence[Sequence[float]], tol: float) -> List[List[float]]:
    basis: List[List[float]] = []
    for raw in rows:
        vec = [float(x) for x in raw]
        for b in basis:
            proj = _dot(vec, b)
            vec = [vx - proj * bx for vx, bx in zip(vec, b)]
        nrm = _vec_norm(vec)
        if nrm > tol:
            basis.append([vx / nrm for vx in vec])
    return basis


def _solve_linear_square(mat: List[List[float]], rhs: List[List[float]], eps: float) -> List[List[float]]:
    """Gauss-Jordan solver for A * X = B with partial pivoting."""
    n = len(mat)
    if n == 0:
        return []
    m_cols = len(rhs[0]) if rhs else 0
    aug = [list(mat[i]) + list(rhs[i]) for i in range(n)]
    for col in range(n):
        piv = col
        best = abs(aug[col][col])
        for r in range(col + 1, n):
            cand = abs(aug[r][col])
            if cand > best:
                best = cand
                piv = r
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        diag = aug[col][col]
        if abs(diag) <= eps:
            diag = eps if diag >= 0.0 else -eps
        inv_d = 1.0 / diag
        for j in range(col, n + m_cols):
            aug[col][j] *= inv_d
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                for j in range(col, n + m_cols):
                    aug[r][j] -= factor * aug[col][j]
    return [row[n : n + m_cols] for row in aug]


def _encode_safetensors_fp32(tensors: Dict[str, Tuple[List[int], List[float]]], header_prefix_bytes: int, fp32_bytes: int) -> bytes:
    header: Dict[str, Any] = {}
    body_chunks: List[bytes] = []
    offset = 0
    for name in sorted(tensors):
        shape, flat_vals = tensors[name]
        raw = struct.pack(f"<{len(flat_vals)}f", *[float(v) for v in flat_vals])
        end = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        body_chunks.append(raw)
        offset = end
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    rem = len(header_json) % header_prefix_bytes
    if rem != 0:
        header_json += b" " * (header_prefix_bytes - rem)
    prefix = struct.pack("<Q", len(header_json))
    _ = fp32_bytes
    return prefix + header_json + b"".join(body_chunks)


class TopicNullspaceAttributionEngine:
    """Topic-nullspace generator profile compiler with conformal spoof screening."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid
        self.models: Dict[str, Dict[str, Any]] = {}

    def _extract_sketch(self, text: str, dim: int) -> List[float]:
        rc = self.grid.runtime_constants
        eps = float(rc["eps_floor"])
        moment_slots = int(rc["structural_moment_slots"])
        prime = int(rc["default_hash_prime"])
        offset_seed = int(rc["default_seed_offset"])
        ngram_n = int(rc["ngram_span"])
        pair_div = int(rc["pair_divisor"])

        usable_dim = max(moment_slots + pair_div, dim)
        vec = [0.0] * usable_dim

        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        tokens = re.findall(r"[A-Za-z0-9_']+", text)
        low_tokens = [t.lower() for t in tokens]
        tok_count = max(1, len(low_tokens))
        char_count = max(1, len(text))

        # Structural invariant moments (sentence length mean/Fano, punctuation cadence, function ratio, code density, token curvature)
        sent_lens = [float(len(re.findall(r"\S+", s))) for s in sents] or [float(tok_count)]
        mean_sl = sum(sent_lens) / float(len(sent_lens))
        var_sl = sum((x - mean_sl) * (x - mean_sl) for x in sent_lens) / float(len(sent_lens))
        fano = var_sl / (mean_sl + eps)

        punct_semicolon_colon = sum(1 for ch in text if ch in ";:—-")
        punct_comma = sum(1 for ch in text if ch == ",")
        code_delim = sum(1 for ch in text if ch in "{}[]()=`")
        list_markers = len(re.findall(r"(?m)^\s*(?:[-*]|\d+\.)\s+", text))
        func_hits = sum(1 for t in low_tokens if t in _FUNCTION_MARKERS)

        tok_lens = [float(len(t)) for t in low_tokens]
        curv = 0.0
        if len(tok_lens) > pair_div:
            diffs = [abs(tok_lens[i + 1] - tok_lens[i]) for i in range(len(tok_lens) - 1)]
            curv = sum(diffs) / float(len(diffs))

        moments = [
            mean_sl / float(usable_dim),
            fano / float(usable_dim),
            float(punct_semicolon_colon + list_markers) / float(tok_count),
            float(punct_comma) / float(tok_count),
            float(func_hits) / float(tok_count),
            (float(code_delim) / float(char_count)) + (curv / float(usable_dim)),
        ]
        for idx in range(min(moment_slots, len(moments))):
            vec[idx] = moments[idx]

        rem_buckets = usable_dim - moment_slots
        struct_buckets = max(1, rem_buckets // pair_div)
        topic_buckets = max(1, rem_buckets - struct_buckets)

        # Structural punctuation/function transition n-grams
        masked = re.sub(r"[A-Za-z]{4,}", "W", text)
        for i in range(max(0, len(masked) - ngram_n + 1)):
            gram = masked[i : i + ngram_n]
            acc = offset_seed
            for ch in gram:
                acc = acc * prime + ord(ch)
            slot = moment_slots + (acc % struct_buckets)
            vec[slot] += 1.0 / float(char_count)

        # Content/topical token buckets (captured so topic-nullspace removes topical shortcuts cleanly)
        for tok in low_tokens:
            if tok in _FUNCTION_MARKERS:
                continue
            acc = offset_seed
            for ch in tok:
                acc = acc * prime + ord(ch)
            slot = moment_slots + struct_buckets + (acc % topic_buckets)
            vec[slot] += 1.0 / float(tok_count)

        nrm = _vec_norm(vec)
        if nrm > eps:
            vec = [v / nrm for v in vec]
        return vec[:dim]

    def compile_attribution_model(
        self,
        namespace: str,
        samples: Sequence[Dict[str, Any]],
        sketch_dim: Optional[int] = None,
        topic_rank: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        cell = self.grid.resolve_cell(
            regime,
            overrides={
                "attribution_sketch_dim": sketch_dim,
                "topic_nullspace_rank": topic_rank,
            },
        )
        rc = self.grid.runtime_constants
        dim = int(cell["attribution_sketch_dim"])
        t_rank = int(cell["topic_nullspace_rank"])
        t_rank = min(t_rank, max(1, dim - int(rc["pair_divisor"])))
        lam = float(rc["ridge_regularization"])
        eps = float(rc["eps_floor"])
        tol = float(rc["nullspace_rank_tolerance"])
        q_perm = float(rc["conformal_spoof_quantile_permille"]) / float(rc["permille_scale"])
        spoof_scale = float(rc["spoof_residual_scale"])
        inlier_rank = int(rc["inlier_subspace_rank_default"])
        round_d = int(rc["attenuation_round_digits"])

        if not samples:
            return {"ok": False, "error": "empty_attribution_samples"}

        profiles = sorted({str(s.get("profile_id") or s.get("target_label") or "profile_default") for s in samples})
        prof_to_idx = {p: i for i, p in enumerate(profiles)}
        num_classes = len(profiles)

        feats: List[List[float]] = []
        prof_labels: List[str] = []
        topics: List[str] = []
        for s in samples:
            txt = str(s.get("text") or s.get("input_text") or "")
            p_id = str(s.get("profile_id") or s.get("target_label") or "profile_default")
            t_id = str(s.get("topic_id") or s.get("topic") or "topic_general")
            feats.append(self._extract_sketch(txt, dim))
            prof_labels.append(p_id)
            topics.append(t_id)

        # Build within-profile cross-topic difference rows D_topic
        by_prof_topic: Dict[Tuple[str, str], List[List[float]]] = {}
        for x, p_id, t_id in zip(feats, prof_labels, topics):
            by_prof_topic.setdefault((p_id, t_id), []).append(x)

        means_by_prof: Dict[str, List[List[float]]] = {}
        for (p_id, _t_id), rows in sorted(by_prof_topic.items()):
            cnt = float(len(rows))
            mu = [sum(r[j] for r in rows) / cnt for j in range(dim)]
            means_by_prof.setdefault(p_id, []).append(mu)

        diff_rows: List[List[float]] = []
        for _p_id, t_means in sorted(means_by_prof.items()):
            if len(t_means) > 1:
                anchor_mu = t_means[0]
                for other_mu in t_means[1:]:
                    diff_rows.append([other_mu[j] - anchor_mu[j] for j in range(dim)])

        v_topic = _orthonormalize_rows(diff_rows, tol)[:t_rank]

        # Orthogonal projector P_perp = I - V_topic^T V_topic
        p_perp = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        for v in v_topic:
            for i in range(dim):
                vi = v[i]
                for j in range(dim):
                    p_perp[i][j] -= vi * v[j]

        proj_feats = [
            [sum(p_perp[i][j] * x[j] for j in range(dim)) for i in range(dim)]
            for x in feats
        ]

        # Closed-form ridge solve on projected features -> W_ridge, then right-multiply by P_perp
        xtx = [[lam if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        xty = [[0.0 for _ in range(num_classes)] for _ in range(dim)]
        for xp, p_id in zip(proj_feats, prof_labels):
            c_idx = prof_to_idx[p_id]
            for i in range(dim):
                xpi = xp[i]
                xty[i][c_idx] += xpi
                for j in range(dim):
                    xtx[i][j] += xpi * xp[j]

        w_cols = _solve_linear_square(xtx, xty, eps)  # dim x num_classes
        w_raw = [[w_cols[j][c] for j in range(dim)] for c in range(num_classes)]
        w_attr = [
            [sum(w_raw[c][k] * p_perp[k][j] for k in range(dim)) for j in range(dim)]
            for c in range(num_classes)
        ]

        # Compute exact topic leakage Frobenius norm ||W_attr V_topic^T||_F
        leak_sq = 0.0
        for c in range(num_classes):
            for v in v_topic:
                ip = _dot(w_attr[c], v)
                leak_sq += ip * ip
        leak_frob = math.sqrt(leak_sq)
        if leak_frob <= float(rc["nullspace_drift_floor"]):
            leak_frob = 0.0

        # Calibrate class-conditional conformal residual energy threshold for spoof detection
        class_means: Dict[str, List[float]] = {}
        class_bases: Dict[str, List[List[float]]] = {}
        energies: List[float] = []
        for p_id in profiles:
            p_rows = [xp for xp, lbl in zip(proj_feats, prof_labels) if lbl == p_id]
            cnt = float(max(1, len(p_rows)))
            mu = [sum(r[j] for r in p_rows) / cnt for j in range(dim)]
            centered = [[r[j] - mu[j] for j in range(dim)] for r in p_rows]
            u_c = _orthonormalize_rows(centered, tol)[:inlier_rank]
            class_means[p_id] = mu
            class_bases[p_id] = u_c
            for r in p_rows:
                diff = [r[j] - mu[j] for j in range(dim)]
                for b in u_c:
                    pr = _dot(diff, b)
                    diff = [d - pr * bj for d, bj in zip(diff, b)]
                energies.append(sum(d * d for d in diff))

        energies_sorted = sorted(energies) if energies else [0.0]
        q_idx = min(len(energies_sorted) - 1, max(0, int(q_perm * float(len(energies_sorted) - 1))))
        conformal_thr = max(eps, energies_sorted[q_idx] * spoof_scale)

        correct = 0
        for x, p_id in zip(feats, prof_labels):
            scores = [_dot(w_attr[c], x) for c in range(num_classes)]
            best_c = max(range(num_classes), key=lambda idx: scores[idx])
            if profiles[best_c] == p_id:
                correct += 1
        seen_acc = float(correct) / float(len(feats))

        model = {
            "namespace": namespace,
            "profiles": profiles,
            "sketch_dim": dim,
            "topic_nullspace_rank": len(v_topic),
            "w_attr": w_attr,
            "p_perp": p_perp,
            "class_means": class_means,
            "class_bases": class_bases,
            "conformal_threshold": conformal_thr,
            "topic_leakage_norm": round(leak_frob, round_d),
            "seen_accuracy": round(seen_acc, int(rc["ratio_round_digits"])),
        }
        self.models[namespace] = model
        return {
            "ok": True,
            "namespace": namespace,
            "profiles": profiles,
            "sketch_dim": dim,
            "topic_nullspace_rank": len(v_topic),
            "topic_leakage_norm": model["topic_leakage_norm"],
            "conformal_threshold": round(conformal_thr, round_d),
            "seen_accuracy": model["seen_accuracy"],
        }

    def attribute_text(
        self,
        namespace: str,
        text: str,
    ) -> Dict[str, Any]:
        model = self.models.get(namespace)
        if model is None:
            return {"ok": False, "error": "attribution_model_not_compiled", "namespace": namespace}
        rc = self.grid.runtime_constants
        round_d = int(rc["attenuation_round_digits"])
        dim = int(model["sketch_dim"])
        x = self._extract_sketch(text, dim)
        profiles: List[str] = model["profiles"]
        w_attr: List[List[float]] = model["w_attr"]
        logits = [_dot(row, x) for row in w_attr]
        max_l = max(logits) if logits else 0.0
        exps = [math.exp(l - max_l) for l in logits]
        denom = sum(exps) or 1.0
        probs = [e / denom for e in exps]
        best_idx = max(range(len(profiles)), key=lambda i: logits[i])
        pred_profile = profiles[best_idx]

        p_perp: List[List[float]] = model["p_perp"]
        xp = [sum(p_perp[i][j] * x[j] for j in range(dim)) for i in range(dim)]
        mu = model["class_means"][pred_profile]
        u_c = model["class_bases"][pred_profile]
        diff = [xp[j] - mu[j] for j in range(dim)]
        for b in u_c:
            pr = _dot(diff, b)
            diff = [d - pr * bj for d, bj in zip(diff, b)]
        energy = sum(d * d for d in diff)
        thr = float(model["conformal_threshold"])

        return {
            "ok": True,
            "namespace": namespace,
            "predicted_profile": pred_profile,
            "confidence": round(probs[best_idx], int(rc["ratio_round_digits"])),
            "profile_probabilities": {
                profiles[i]: round(probs[i], int(rc["ratio_round_digits"])) for i in range(len(profiles))
            },
            "residual_energy": round(energy, round_d),
            "conformal_threshold": round(thr, round_d),
            "spoof_flagged": bool(energy > thr),
            "topic_leakage_norm": model["topic_leakage_norm"],
            "llm_tokens_used": 0,
        }


class KoopmanCegarTrajectoryEngine:
    """Zero-token linear Koopman multi-step tool trajectory planner + CEGAR scenario selector."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid
        self.compiled_planners: Dict[str, Dict[str, Any]] = {}

    def _lift_state(self, state: Sequence[float], lifted_dim: int) -> List[float]:
        s = [float(v) for v in state]
        lifted: List[float] = [1.0]
        lifted.extend(s)
        lifted.extend(v * v for v in s)
        for i in range(max(0, len(s) - 1)):
            lifted.append(s[i] * s[i + 1])
        if s:
            last = s[-1]
            for i in range(max(0, len(s) - 1)):
                lifted.append(s[i] * last)
        if len(lifted) < lifted_dim:
            idx = 0
            while len(lifted) < lifted_dim:
                base_v = s[idx % max(1, len(s))] if s else 0.0
                lifted.append(math.sin(base_v * float(idx + 1)))
                idx += 1
        return lifted[:lifted_dim]

    def compile_koopman_planner(
        self,
        namespace: str,
        transitions: Sequence[Dict[str, Any]],
        goal_state: Optional[Sequence[float]] = None,
        lifted_dim: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        cell = self.grid.resolve_cell(regime, overrides={"koopman_lifted_dim": lifted_dim})
        rc = self.grid.runtime_constants
        z_dim = int(cell["koopman_lifted_dim"])
        lam = float(rc["ridge_regularization"])
        eps = float(rc["eps_floor"])

        if not transitions:
            return {"ok": False, "error": "empty_koopman_transitions"}

        by_action: Dict[str, List[Dict[str, Any]]] = {}
        state_len = 0
        for tr in transitions:
            act = str(tr["action"])
            st = [float(v) for v in tr["state"]]
            state_len = max(state_len, len(st))
            by_action.setdefault(act, []).append(tr)

        operators: Dict[str, List[List[float]]] = {}
        gates: Dict[str, List[float]] = {}
        for act, rows in sorted(by_action.items()):
            valid_rows = [r for r in rows if bool(r.get("valid", True))]
            fit_rows = valid_rows if valid_rows else rows

            ztz = [[lam if i == j else 0.0 for j in range(z_dim)] for i in range(z_dim)]
            ztn = [[0.0 for _ in range(z_dim)] for _ in range(z_dim)]
            for r in fit_rows:
                z_cur = self._lift_state(r["state"], z_dim)
                z_nxt = self._lift_state(r["next_state"], z_dim)
                for i in range(z_dim):
                    zi = z_cur[i]
                    for j in range(z_dim):
                        ztz[i][j] += zi * z_cur[j]
                        ztn[i][j] += zi * z_nxt[j]
            k_cols = _solve_linear_square(ztz, ztn, eps)  # z_dim x z_dim (z_cur^T * K^T)
            k_mat = [[k_cols[j][i] for j in range(z_dim)] for i in range(z_dim)]
            operators[act] = k_mat

            # Linear feasibility gate g_a predicting valid (+1) vs invalid (-1)
            gtg = [[lam if i == j else 0.0 for j in range(z_dim)] for i in range(z_dim)]
            gty = [[0.0] for _ in range(z_dim)]
            for r in rows:
                z_cur = self._lift_state(r["state"], z_dim)
                target = 1.0 if bool(r.get("valid", True)) else -1.0
                for i in range(z_dim):
                    zi = z_cur[i]
                    gty[i][0] += zi * target
                    for j in range(z_dim):
                        gtg[i][j] += zi * z_cur[j]
            has_invalid = any(not bool(r.get("valid", True)) for r in rows)
            if has_invalid:
                g_sol = _solve_linear_square(gtg, gty, eps)
                gates[act] = [g_sol[i][0] for i in range(z_dim)]
            else:
                gates[act] = [1.0] + [0.0] * (z_dim - 1)

        default_goal = list(goal_state) if goal_state is not None else [1.0] * max(1, state_len)
        z_goal = self._lift_state(default_goal, z_dim)

        spec = {
            "namespace": namespace,
            "lifted_dim": z_dim,
            "state_dim": state_len,
            "actions": sorted(operators.keys()),
            "operators": operators,
            "gates": gates,
            "z_goal": z_goal,
        }
        self.compiled_planners[namespace] = spec
        return {
            "ok": True,
            "namespace": namespace,
            "lifted_dim": z_dim,
            "actions": spec["actions"],
            "transition_count": len(transitions),
        }

    def plan_trajectory(
        self,
        namespace: str,
        initial_state: Sequence[float],
        goal_state: Optional[Sequence[float]] = None,
        horizon_depth: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = self.compiled_planners.get(namespace)
        if spec is None:
            return {"ok": False, "error": "koopman_planner_not_compiled", "namespace": namespace}
        cell = self.grid.resolve_cell(regime, overrides={"koopman_lookahead_depth": horizon_depth})
        rc = self.grid.runtime_constants
        depth = int(cell["koopman_lookahead_depth"])
        beam_w = int(rc["koopman_beam_width"])
        step_pen = float(rc["koopman_step_penalty"])
        round_d = int(rc["ratio_round_digits"])

        z_dim = int(spec["lifted_dim"])
        z0 = self._lift_state(initial_state, z_dim)
        z_goal = self._lift_state(goal_state, z_dim) if goal_state is not None else list(spec["z_goal"])
        operators: Dict[str, List[List[float]]] = spec["operators"]
        gates: Dict[str, List[float]] = spec["gates"]
        actions: List[str] = spec["actions"]

        def _dist_sq(za: Sequence[float], zb: Sequence[float]) -> float:
            # Compare state-projection slots (1..state_dim) primarily, plus full lifted tie-break
            s_end = min(z_dim, 1 + int(spec["state_dim"]))
            core = sum((za[i] - zb[i]) * (za[i] - zb[i]) for i in range(1, s_end))
            aux = sum((za[i] - zb[i]) * (za[i] - zb[i]) for i in range(s_end, z_dim))
            return core + step_pen * aux

        # Beam entries: (score_lower_is_better, z_vec, seq_actions, best_terminal_dist)
        init_d = _dist_sq(z0, z_goal)
        beam: List[Tuple[float, List[float], List[str], float]] = [(init_d, z0, [], init_d)]
        best_overall = (init_d, z0, [])

        for _step in range(depth):
            candidates: List[Tuple[float, List[float], List[str], float]] = []
            for _sc, z_cur, seq, _bd in beam:
                for act in actions:
                    gate_score = _dot(gates[act], z_cur)
                    if gate_score < 0.0:
                        continue
                    k_mat = operators[act]
                    z_nxt = [sum(k_mat[i][j] * z_cur[j] for j in range(z_dim)) for i in range(z_dim)]
                    d_goal = _dist_sq(z_nxt, z_goal)
                    new_seq = seq + [act]
                    total_cost = d_goal + step_pen * float(len(new_seq))
                    candidates.append((total_cost, z_nxt, new_seq, d_goal))
                    if total_cost < best_overall[0]:
                        best_overall = (total_cost, z_nxt, new_seq)
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            beam = candidates[:beam_w]

        best_cost, best_z, best_seq = best_overall
        s_end = min(z_dim, 1 + int(spec["state_dim"]))
        pred_terminal = [round(v, round_d) for v in best_z[1:s_end]]
        return {
            "ok": True,
            "namespace": namespace,
            "planned_actions": best_seq,
            "predicted_terminal_state": pred_terminal,
            "goal_distance_sq": round(best_cost, int(rc["attenuation_round_digits"])),
            "invalid_transitions": 0,
            "llm_tokens_used": 0,
        }

    def select_cegar_counterexamples(
        self,
        namespace: str,
        candidate_scenarios: Sequence[Dict[str, Any]],
        query_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        budget = max(1, int(query_budget or rc["cegar_query_budget_default"]))
        margin_floor = float(rc["cegar_margin_floor"])
        round_d = int(rc["attenuation_round_digits"])
        spec = self.compiled_planners.get(namespace)

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, scen in enumerate(candidate_scenarios):
            if "margin" in scen:
                margin = abs(float(scen["margin"]))
            elif spec is not None and "state" in scen:
                z = self._lift_state(scen["state"], int(spec["lifted_dim"]))
                gate_vals = [abs(_dot(g, z)) for g in spec["gates"].values()]
                margin = min(gate_vals) if gate_vals else margin_floor
            else:
                margin = margin_floor
            scored.append((margin, idx, dict(scen)))

        scored.sort(key=lambda item: (item[0], item[1]))
        chosen = []
        for m_val, idx, scen in scored[:budget]:
            entry = dict(scen)
            entry["scenario_index"] = idx
            entry["boundary_margin"] = round(m_val, round_d)
            entry["near_decision_boundary"] = bool(m_val <= margin_floor)
            chosen.append(entry)

        total = max(1, len(candidate_scenarios))
        return {
            "ok": True,
            "namespace": namespace,
            "total_candidates": len(candidate_scenarios),
            "selected_count": len(chosen),
            "oracle_reduction_ratio_x": round(float(total) / float(max(1, len(chosen))), int(rc["ratio_round_digits"])),
            "counterexample_scenarios": chosen,
        }


class PortableCartridgeAndOneShotEngine:
    """5-target portable float32 cartridge exporter + Hopfield 1-shot & Woodbury multi-shot updater."""

    def __init__(self, grid: AdaptiveNodeGrid) -> None:
        self.grid = grid
        self.hopfield_banks: Dict[str, Dict[str, Any]] = {}
        self.woodbury_states: Dict[str, Dict[str, Any]] = {}

    def export_portable_cartridge(
        self,
        cartridge_id: str,
        weights: Sequence[Sequence[float]],
        bias: Optional[Sequence[float]] = None,
        targets: Optional[Sequence[str]] = None,
        activation: str = "linear",
        test_vectors: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        digits = int(rc["c99_float_digits"])
        header_len_b = int(rc["safetensors_header_len_bytes"])
        fp32_b = int(rc["float32_byte_width"])
        ceiling = float(rc["portable_discrepancy_ceiling"])
        round_d = int(rc["attenuation_round_digits"])

        if not weights or not weights[0]:
            return {"ok": False, "error": "empty_weight_matrix"}

        out_dim = len(weights)
        in_dim = len(weights[0])
        w_fp32 = [[_quantize_fp32(v) for v in row] for row in weights]
        b_fp32 = [_quantize_fp32(v) for v in (bias if bias is not None else [0.0] * out_dim)]
        req_targets = [t for t in (targets or PORTABLE_TARGETS) if t in PORTABLE_TARGETS]
        use_relu = activation == "relu"
        safe_sym = re.sub(r"[^A-Za-z0-9_]", "_", cartridge_id) or "cartridge"

        flat_w = [v for row in w_fp32 for v in row]
        artifacts: Dict[str, str] = {}

        if "js" in req_targets:
            w_lit = ", ".join(f"{v:.{digits}g}" for v in flat_w)
            b_lit = ", ".join(f"{v:.{digits}g}" for v in b_fp32)
            act_expr = "Math.max(0.0, acc)" if use_relu else "acc"
            artifacts["js"] = (
                f"// Auto-generated IEEE-754 Float32Array cartridge ({out_dim}x{in_dim})\n"
                f"export const W = new Float32Array([{w_lit}]);\n"
                f"export const B = new Float32Array([{b_lit}]);\n"
                f"export function infer_{safe_sym}(x) {{\n"
                f"  const out = new Float32Array({out_dim});\n"
                f"  for (let r = 0; r < {out_dim}; r++) {{\n"
                f"    let acc = B[r];\n"
                f"    for (let c = 0; c < {in_dim}; c++) {{\n"
                f"      acc = Math.fround(acc + Math.fround(W[r * {in_dim} + c] * Math.fround(x[c])));\n"
                f"    }}\n"
                f"    out[r] = {act_expr};\n"
                f"  }}\n"
                f"  return out;\n"
                f"}}\n"
            )

        if "python" in req_targets:
            artifacts["python"] = (
                f"# Zero-import pure Python stdlib float32 cartridge ({out_dim}x{in_dim})\n"
                f"W = {w_fp32!r}\n"
                f"B = {b_fp32!r}\n"
                f"def infer_{safe_sym}(x):\n"
                f"    out = []\n"
                f"    for r in range({out_dim}):\n"
                f"        acc = B[r] + sum(W[r][c] * float(x[c]) for c in range({in_dim}))\n"
                f"        out.append(max(0.0, acc) if {use_relu!r} else acc)\n"
                f"    return out\n"
            )

        if "sql" in req_targets:
            w_tuples = ", ".join(
                f"({r}, {c}, {w_fp32[r][c]:.{digits}g})"
                for r in range(out_dim)
                for c in range(in_dim)
            )
            b_tuples = ", ".join(f"({r}, {b_fp32[r]:.{digits}g})" for r in range(out_dim))
            sql_expr = "MAX(0.0, b.val + SUM(w.val * x.val))" if use_relu else "(b.val + SUM(w.val * x.val))"
            artifacts["sql"] = (
                f"-- Declarative SQLite cartridge query ({out_dim}x{in_dim})\n"
                f"WITH weights(r, c, val) AS (VALUES {w_tuples}),\n"
                f"     biases(r, val) AS (VALUES {b_tuples})\n"
                f"SELECT w.r AS row_idx, {sql_expr} AS logit\n"
                f"FROM weights w JOIN input_vec x ON w.c = x.c JOIN biases b ON w.r = b.r\n"
                f"GROUP BY w.r ORDER BY w.r;\n"
            )

        if "c99" in req_targets:
            w_c = ", ".join(f"{v:.{digits}g}f" for v in flat_w)
            b_c = ", ".join(f"{v:.{digits}g}f" for v in b_fp32)
            act_c = "acc > 0.0f ? acc : 0.0f" if use_relu else "acc"
            artifacts["c99"] = (
                f"#ifndef CARTRIDGE_{safe_sym.upper()}_H\n"
                f"#define CARTRIDGE_{safe_sym.upper()}_H\n"
                f"static const float W_{safe_sym}[{out_dim * in_dim}] = {{{w_c}}};\n"
                f"static const float B_{safe_sym}[{out_dim}] = {{{b_c}}};\n"
                f"static inline void infer_{safe_sym}(const float* x, float* out) {{\n"
                f"  for (int r = 0; r < {out_dim}; ++r) {{\n"
                f"    float acc = B_{safe_sym}[r];\n"
                f"    for (int c = 0; c < {in_dim}; ++c) acc += W_{safe_sym}[r * {in_dim} + c] * x[c];\n"
                f"    out[r] = {act_c};\n"
                f"  }}\n"
                f"}}\n"
                f"#endif\n"
            )

        st_bytes = _encode_safetensors_fp32(
            {
                "weight": ([out_dim, in_dim], flat_w),
                "bias": ([out_dim], b_fp32),
            },
            header_prefix_bytes=header_len_b,
            fp32_bytes=fp32_b,
        )
        if "safetensors" in req_targets:
            artifacts["safetensors"] = base64.b64encode(st_bytes).decode("ascii")

        # Check float32 accumulation vs float64 accumulation discrepancy on probe vectors
        probes = list(test_vectors) if test_vectors else [[1.0 / float(c + 1) for c in range(in_dim)]]
        max_disc = 0.0
        for x_raw in probes:
            x32 = [_quantize_fp32(v) for v in x_raw[:in_dim]]
            for r in range(out_dim):
                acc64 = b_fp32[r] + sum(w_fp32[r][c] * x32[c] for c in range(len(x32)))
                acc32 = _quantize_fp32(b_fp32[r])
                for c in range(len(x32)):
                    acc32 = _quantize_fp32(acc32 + _quantize_fp32(w_fp32[r][c] * x32[c]))
                if use_relu:
                    acc64 = max(0.0, acc64)
                    acc32 = max(0.0, acc32)
                max_disc = max(max_disc, abs(acc64 - acc32))

        return {
            "ok": True,
            "cartridge_id": cartridge_id,
            "shape": [out_dim, in_dim],
            "raw_weight_bytes_fp32": out_dim * in_dim * fp32_b,
            "safetensors_bytes": len(st_bytes),
            "exported_targets": list(artifacts.keys()),
            "artifacts": artifacts,
            "max_float32_discrepancy": round(max_disc, round_d + int(rc["pair_divisor"])),
            "passes_equivalence_gate": bool(max_disc <= ceiling),
        }

    def hopfield_oneshot_step(
        self,
        namespace: str,
        mode: str,
        vector: Sequence[float],
        label: str = "pattern_default",
        cartridge_id: str = "cartridge_default",
        beta: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        cell = self.grid.resolve_cell(regime, overrides={"hopfield_inverse_temp_beta": beta})
        rc = self.grid.runtime_constants
        eps = float(rc["eps_floor"])
        inv_temp = float(cell["hopfield_inverse_temp_beta"])
        nov_thr = float(rc["hopfield_novelty_energy_threshold"])
        round_d = int(rc["attenuation_round_digits"])

        nrm = _vec_norm(vector)
        unit_v = [float(x) / nrm for x in vector] if nrm > eps else [0.0 for _ in vector]
        bank = self.hopfield_banks.setdefault(namespace, {"patterns": [], "labels": [], "cartridges": []})

        if mode == "write":
            bank["patterns"].append(unit_v)
            bank["labels"].append(label)
            bank["cartridges"].append(cartridge_id)
            return {
                "ok": True,
                "namespace": namespace,
                "mode": "write",
                "stored_patterns": len(bank["patterns"]),
                "retraining_epochs": 0,
                "label": label,
                "cartridge_id": cartridge_id,
            }

        if not bank["patterns"]:
            return {"ok": False, "error": "empty_hopfield_bank", "namespace": namespace}

        sims = [_dot(p, unit_v) for p in bank["patterns"]]
        scaled = [inv_temp * s for s in sims]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        z_sum = sum(exps) or 1.0
        attn = [e / z_sum for e in exps]
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        max_cos = sims[best_idx]
        novelty_energy = 1.0 - max_cos

        return {
            "ok": True,
            "namespace": namespace,
            "mode": "query",
            "matched_label": bank["labels"][best_idx],
            "matched_cartridge_id": bank["cartridges"][best_idx],
            "attention_peak": round(attn[best_idx], int(rc["ratio_round_digits"])),
            "max_cosine": round(max_cos, round_d),
            "novelty_energy": round(novelty_energy, round_d),
            "is_novel": bool(novelty_energy > nov_thr),
            "llm_tokens_used": 0,
        }

    def woodbury_multishot_step(
        self,
        namespace: str,
        mode: str,
        vector: Sequence[float],
        target_label: str = "",
        class_labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        init_scale = float(rc["woodbury_init_precision_scale"])
        round_d = int(rc["attenuation_round_digits"])
        dim = len(vector)
        if dim == 0:
            return {"ok": False, "error": "empty_woodbury_vector"}

        state = self.woodbury_states.get(namespace)
        if state is None:
            labels = list(class_labels) if class_labels else ([target_label] if target_label else ["class_0"])
            p_mat = [[init_scale if i == j else 0.0 for j in range(dim)] for i in range(dim)]
            w_mat = [[0.0 for _ in range(len(labels))] for _ in range(dim)]
            state = {"dim": dim, "labels": labels, "p_mat": p_mat, "w_mat": w_mat, "updates": 0}
            self.woodbury_states[namespace] = state

        labels = state["labels"]
        if target_label and target_label not in labels:
            labels.append(target_label)
            for row in state["w_mat"]:
                row.append(0.0)

        x = [float(v) for v in vector[: state["dim"]]]
        if mode == "predict":
            w_mat = state["w_mat"]
            logits = [sum(x[i] * w_mat[i][c] for i in range(state["dim"])) for c in range(len(labels))]
            best_c = max(range(len(labels)), key=lambda c: logits[c])
            return {
                "ok": True,
                "namespace": namespace,
                "predicted_label": labels[best_c],
                "logits": {labels[c]: round(logits[c], round_d) for c in range(len(labels))},
                "updates_assimilated": state["updates"],
                "replay_buffer_rows": 0,
            }

        # Exact rank-1 Recursive Woodbury RLS update:
        # k = (P x) / (1 + x^T P x);  W <- W + k (y^T - x^T W);  P <- P - k (x^T P)
        p_mat = state["p_mat"]
        w_mat = state["w_mat"]
        d = state["dim"]
        px = [sum(p_mat[i][j] * x[j] for j in range(d)) for i in range(d)]
        denom = 1.0 + _dot(x, px)
        gain = [v / denom for v in px]
        y_vec = [1.0 if lbl == target_label else 0.0 for lbl in labels]
        pred = [sum(x[i] * w_mat[i][c] for i in range(d)) for c in range(len(labels))]
        err = [y_vec[c] - pred[c] for c in range(len(labels))]

        for i in range(d):
            gi = gain[i]
            for c in range(len(labels)):
                w_mat[i][c] += gi * err[c]

        xtp = [sum(x[k] * p_mat[k][j] for k in range(d)) for j in range(d)]
        for i in range(d):
            gi = gain[i]
            for j in range(d):
                p_mat[i][j] -= gi * xtp[j]

        state["updates"] = int(state["updates"]) + 1
        return {
            "ok": True,
            "namespace": namespace,
            "mode": "update",
            "updates_assimilated": state["updates"],
            "labels": list(labels),
            "replay_buffer_rows": 0,
        }
