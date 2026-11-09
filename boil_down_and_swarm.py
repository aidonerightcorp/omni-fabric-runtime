"""Progressive neural boil-down engine with LoRA-XS 256B subspace cartridges, matched thinking=off confound gate, OpenAI-v1 loopback swarm proxy, and selective upstream Fabric bridge."""

from __future__ import annotations

import base64
import hashlib
import http.client
import importlib
import ipaddress
import json
import math
import socket
import ssl
import struct
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .fabric_substrate import LocalAdaptiveFabricStore
from .grid_and_env import AdaptiveNodeGrid, EffortAndBudgetGate

_ALWAYS_BLOCKED_NETWORKS: Tuple[Any, ...] = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "100.64.0.0/10",
        "169.254.0.0/16",
        "192.0.0.0/24",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "fe80::/10",
        "ff00::/8",
        "fd00:ec2::254/128",
    )
)
_PRIVATE_OR_LOOPBACK_NETWORKS: Tuple[Any, ...] = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::1/128",
        "fc00::/7",
    )
)
_BLOCKED_METADATA_HOSTS: Tuple[str, ...] = (
    "metadata.google.internal",
    "metadata",
    "instance-data",
)
_LOOPBACK_LITERAL_IPS: Tuple[Any, ...] = (
    ipaddress.ip_address("127.0.0.1"),
    ipaddress.ip_address("::1"),
)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection pinned to a pre-resolved IP address so DNS rebinding is impossible."""

    def __init__(self, pinned_ip: str, port: int, timeout: float) -> None:
        super().__init__(pinned_ip, port=port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection pinned to a pre-resolved IP while preserving original TLS SNI hostname."""

    def __init__(self, pinned_ip: str, sni_host: str, port: int, timeout: float, tls_ctx: ssl.SSLContext) -> None:
        super().__init__(pinned_ip, port=port, timeout=timeout, context=tls_ctx)
        self._pinned_ip = pinned_ip
        self._sni_host = sni_host
        self._tls_ctx = tls_ctx

    def connect(self) -> None:
        raw_sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._tls_ctx.wrap_socket(raw_sock, server_hostname=self._sni_host)


def _vet_and_resolve_url(
    base_url: str,
    mode: str,
    allow_local: bool,
    grid: AdaptiveNodeGrid,
) -> Dict[str, Any]:
    cleaned = base_url.strip()
    if not cleaned:
        return {"ok": False, "error": "empty_endpoint_url"}
    if "?" in cleaned or "#" in cleaned:
        return {"ok": False, "error": "url_query_or_fragment_forbidden"}

    parsed = urllib.parse.urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return {"ok": False, "error": "invalid_url_scheme"}
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return {"ok": False, "error": "url_userinfo_forbidden"}
    if parsed.query or parsed.fragment:
        return {"ok": False, "error": "url_query_or_fragment_forbidden"}

    orig_host = (parsed.hostname or "").strip().lower()
    if not orig_host:
        return {"ok": False, "error": "missing_hostname"}
    if orig_host in _BLOCKED_METADATA_HOSTS or orig_host.endswith(".metadata.google.internal"):
        return {"ok": False, "error": "blocked_metadata_host"}

    try:
        raw_port = parsed.port
    except ValueError:
        return {"ok": False, "error": "invalid_url_port"}

    http_def = int(grid.runtime_constants["http_default_port"])
    https_def = int(grid.runtime_constants["https_default_port"])
    max_port = int(grid.runtime_constants["max_tcp_port"])
    default_port = https_def if scheme == "https" else http_def
    port = int(raw_port) if raw_port is not None else default_port
    if port < 1 or port > max_port:
        return {"ok": False, "error": "invalid_url_port"}

    if mode == "local_runner":
        if not allow_local:
            return {"ok": False, "error": "local_runner_disabled_by_config"}
        if orig_host not in ("127.0.0.1", "::1", "localhost"):
            return {"ok": False, "error": "local_runner_must_be_loopback"}

    try:
        infos = socket.getaddrinfo(orig_host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return {"ok": False, "error": f"dns_resolution_failed: {exc}"}
    if not infos:
        return {"ok": False, "error": "dns_resolution_empty"}

    pinned_ip = ""
    for _fam, _stype, _proto, _canon, sockaddr in infos:
        raw_addr = str(sockaddr[0])
        if "%" in raw_addr:
            return {"ok": False, "error": "ipv6_zone_scope_forbidden"}
        try:
            ip_raw = ipaddress.ip_address(raw_addr)
        except ValueError:
            return {"ok": False, "error": "invalid_resolved_ip"}
        ip_obj = ip_raw.ipv4_mapped if getattr(ip_raw, "ipv4_mapped", None) is not None else ip_raw

        if (
            ip_obj.is_multicast
            or ip_obj.is_unspecified
            or ip_obj.is_link_local
            or (ip_obj.is_reserved and not ip_obj.is_loopback)
            or any(ip_obj in net for net in _ALWAYS_BLOCKED_NETWORKS)
        ):
            return {"ok": False, "error": f"blocked_special_or_metadata_ip:{ip_obj}"}

        if mode == "local_runner":
            if ip_obj not in _LOOPBACK_LITERAL_IPS:
                return {"ok": False, "error": f"local_runner_non_loopback_ip:{ip_obj}"}
        else:
            is_priv_or_loop = (
                ip_obj.is_loopback
                or ip_obj.is_private
                or any(ip_obj in net for net in _PRIVATE_OR_LOOPBACK_NETWORKS)
            )
            if is_priv_or_loop and not allow_local:
                return {"ok": False, "error": f"private_or_loopback_upstream_refused:{ip_obj}"}

        if not pinned_ip:
            pinned_ip = str(ip_obj)

    host_fmt = f"[{orig_host}]" if ":" in orig_host else orig_host
    host_header = host_fmt if port == default_port else f"{host_fmt}:{port}"
    return {
        "ok": True,
        "scheme": scheme,
        "orig_host": orig_host,
        "pinned_ip": pinned_ip,
        "port": port,
        "base_path": parsed.path.rstrip("/"),
        "host_header": host_header,
    }


def _send_vetted_http(
    base_url: str,
    route: str,
    *,
    method: str = "GET",
    body_bytes: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout_s: float = 1.0,
    mode: str = "upstream",
    allow_local: bool = False,
    grid: AdaptiveNodeGrid,
) -> Dict[str, Any]:
    vet = _vet_and_resolve_url(base_url, mode=mode, allow_local=allow_local, grid=grid)
    if not vet.get("ok"):
        return vet

    clean_route = route if route.startswith("/") else f"/{route}"
    full_path = f"{vet['base_path']}{clean_route}" or "/"
    req_headers: Dict[str, str] = dict(headers or {})
    req_headers["Host"] = str(vet["host_header"])
    req_headers["Connection"] = "close"
    if body_bytes is not None and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(body_bytes))

    conn: http.client.HTTPConnection
    if vet["scheme"] == "https":
        tls_ctx = ssl.create_default_context()
        conn = _PinnedHTTPSConnection(
            pinned_ip=str(vet["pinned_ip"]),
            sni_host=str(vet["orig_host"]),
            port=int(vet["port"]),
            timeout=timeout_s,
            tls_ctx=tls_ctx,
        )
    else:
        conn = _PinnedHTTPConnection(
            pinned_ip=str(vet["pinned_ip"]),
            port=int(vet["port"]),
            timeout=timeout_s,
        )

    ok_min = int(grid.runtime_constants["http_ok_min"])
    ok_max = int(grid.runtime_constants["http_ok_max"])
    try:
        conn.request(method, full_path, body=body_bytes, headers=req_headers)
        resp = conn.getresponse()
        status = int(resp.status)
        raw_text = resp.read().decode("utf-8", errors="replace")
        if ok_min <= status <= ok_max:
            parsed_json = json.loads(raw_text) if raw_text.strip() else {}
            return {"ok": True, "status_code": status, "data": parsed_json}
        return {"ok": False, "status_code": status, "error": raw_text or f"http_status_{status}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass


class AdaptiveBoilDownEngine:
    """Compiles traces into 0-token micro-routers and 256B LoRA-XS cartridges gated by matched thinking=off controls."""

    def __init__(self, grid: AdaptiveNodeGrid, store: LocalAdaptiveFabricStore) -> None:
        self.grid = grid
        self.store = store
        self.compiled_routers: Dict[str, Dict[str, Any]] = {}
        self.lora_xs_cartridges: Dict[str, Dict[str, Any]] = {}

    def _featurize_text(
        self,
        text: str,
        num_buckets: int,
        mask_seed: Optional[int] = None,
    ) -> List[float]:
        vec = [0.0] * num_buckets
        cleaned = f"^{text.strip().lower()}$"
        span = int(self.grid.runtime_constants["ngram_span"])
        pair_div = int(self.grid.runtime_constants["pair_divisor"])
        mask_mod = int(self.grid.runtime_constants["confound_mask_modulus"])
        for idx in range(max(1, len(cleaned) - span + 1)):
            gram = cleaned[idx : idx + span]
            digest = hashlib.blake2s(gram.encode("utf-8")).digest()
            bucket = digest[0] % num_buckets
            if mask_seed is not None and ((bucket + int(mask_seed)) % mask_mod == 0):
                continue
            sign = 1.0 if (digest[1] % pair_div == 0) else -1.0
            vec[bucket] += sign
        norm_sq = sum(v * v for v in vec)
        eps = float(self.grid.runtime_constants["eps_floor"])
        inv_norm = 1.0 / math.sqrt(norm_sq + eps)
        return [v * inv_norm for v in vec]

    def _build_polytope_projection(self, namespace: str, buckets: int, hidden: int) -> List[List[float]]:
        pair_div = int(self.grid.runtime_constants["pair_divisor"])
        scale = 1.0 / math.sqrt(float(max(1, buckets)))
        proj_w: List[List[float]] = []
        for h_idx in range(hidden):
            row_w: List[float] = []
            for b_idx in range(buckets):
                d = hashlib.blake2s(f"relu:{namespace}:{h_idx}:{b_idx}".encode("utf-8")).digest()
                s = 1.0 if (d[0] % pair_div == 0) else -1.0
                row_w.append(s * scale)
            proj_w.append(row_w)
        return proj_w

    @staticmethod
    def _polytope_lift(
        x_vec: Sequence[float],
        proj_w: Sequence[Sequence[float]],
    ) -> List[float]:
        buckets = len(x_vec)
        lifted: List[float] = [1.0, *[float(v) for v in x_vec]]
        for row_w in proj_w:
            dot = sum(float(row_w[b]) * float(x_vec[b]) for b in range(buckets))
            lifted.append(dot if dot > 0.0 else 0.0)
        return lifted

    def _build_frozen_subspace_bases(
        self,
        namespace: str,
        feat_dim: int,
        subspace_rank: int,
        num_classes: int,
        seed_tag: int = 0,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Construct deterministic orthonormal frozen (V_r, U_r) bases for LoRA-XS: f(x) = (phi(x) @ V_r) @ R @ U_r."""
        eps = float(self.grid.runtime_constants["eps_floor"])
        pair_div = int(self.grid.runtime_constants["pair_divisor"])

        # V_r: feat_dim x subspace_rank with orthonormal columns via modified Gram-Schmidt
        v_cols: List[List[float]] = []
        for col_idx in range(subspace_rank):
            raw_col: List[float] = []
            for d_idx in range(feat_dim):
                dig = hashlib.blake2s(
                    f"loraxs_v:{namespace}:{seed_tag}:{col_idx}:{d_idx}".encode("utf-8")
                ).digest()
                val = (float(dig[0]) - float(dig[1])) / float(self.grid.runtime_constants["percent_scale"])
                if val == 0.0:
                    val = 1.0 if (d_idx % pair_div == 0) else -1.0
                raw_col.append(val)
            for prev in v_cols:
                proj = sum(raw_col[i] * prev[i] for i in range(feat_dim))
                for i in range(feat_dim):
                    raw_col[i] -= proj * prev[i]
            norm = math.sqrt(sum(c * c for c in raw_col) + eps)
            v_cols.append([c / norm for c in raw_col])
        v_mat = [[v_cols[col_idx][d_idx] for col_idx in range(subspace_rank)] for d_idx in range(feat_dim)]

        # U_r: subspace_rank x num_classes with orthonormal columns when num_classes <= subspace_rank
        u_cols: List[List[float]] = []
        for c_idx in range(num_classes):
            raw_u: List[float] = []
            for r_idx in range(subspace_rank):
                dig = hashlib.blake2s(
                    f"loraxs_u:{namespace}:{seed_tag}:{c_idx}:{r_idx}".encode("utf-8")
                ).digest()
                val = (float(dig[0]) - float(dig[1])) / float(self.grid.runtime_constants["percent_scale"])
                if val == 0.0:
                    val = 1.0 if ((r_idx + c_idx) % pair_div == 0) else -1.0
                raw_u.append(val)
            if c_idx < subspace_rank:
                for prev_u in u_cols:
                    proj = sum(raw_u[i] * prev_u[i] for i in range(subspace_rank))
                    for i in range(subspace_rank):
                        raw_u[i] -= proj * prev_u[i]
            norm_u = math.sqrt(sum(v * v for v in raw_u) + eps)
            u_cols.append([v / norm_u for v in raw_u])
        u_mat = [[u_cols[c_idx][r_idx] for c_idx in range(num_classes)] for r_idx in range(subspace_rank)]
        return v_mat, u_mat

    @staticmethod
    def _solve_linear_system(a_mat: List[List[float]], b_mat: List[List[float]], eps: float) -> List[List[float]]:
        n = len(a_mat)
        cols = len(b_mat[0]) if b_mat else 0
        aug = [list(a_mat[i]) + list(b_mat[i]) for i in range(n)]
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
            if abs(diag) < eps:
                diag = eps
            inv_d = 1.0 / diag
            for j in range(col, n + cols):
                aug[col][j] *= inv_d
            for r in range(n):
                if r != col:
                    factor = aug[r][col]
                    if factor != 0.0:
                        for j in range(col, n + cols):
                            aug[r][j] -= factor * aug[col][j]
        return [row[n : n + cols] for row in aug]

    def _fit_stdlib_ridge_on_rows(
        self,
        namespace: str,
        train_rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        buckets: int,
        hidden: int,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        reg = float(self.grid.runtime_constants["ridge_regularization"])
        eps = float(self.grid.runtime_constants["eps_floor"])
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        num_classes = len(labels)
        proj_w = self._build_polytope_projection(namespace, buckets, hidden)

        h_feats: List[List[float]] = []
        targets: List[List[float]] = []
        for item in train_rows:
            x_vec = self._featurize_text(str(item["input_text"]), buckets)
            z_vec: List[float] = [1.0]
            for h_idx in range(hidden):
                dot = sum(proj_w[h_idx][b] * x_vec[b] for b in range(buckets))
                z_vec.append(dot if dot > 0.0 else 0.0)
            h_feats.append(z_vec)
            y_vec = [0.0] * num_classes
            t_lab = str(item["target_label"])
            if t_lab in label_to_idx:
                y_vec[label_to_idx[t_lab]] = 1.0
            targets.append(y_vec)

        dim = hidden + 1
        gram = [[0.0] * dim for _ in range(dim)]
        rhs = [[0.0] * num_classes for _ in range(dim)]
        for n_idx in range(len(h_feats)):
            zf = h_feats[n_idx]
            yv = targets[n_idx]
            for i in range(dim):
                zi = zf[i]
                for j in range(dim):
                    gram[i][j] += zi * zf[j]
                for c in range(num_classes):
                    rhs[i][c] += zi * yv[c]
        for i in range(dim):
            gram[i][i] += reg
        readout_w = self._solve_linear_system(gram, rhs, eps)
        return proj_w, readout_w

    def _fit_lora_xs_on_rows(
        self,
        namespace: str,
        train_rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        buckets: int,
        hidden: int,
        subspace_rank: int,
        seed_tag: int = 0,
        null_edit: bool = False,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
        reg = float(self.grid.runtime_constants["ridge_regularization"])
        eps = float(self.grid.runtime_constants["eps_floor"])
        num_classes = len(labels)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        proj_w = self._build_polytope_projection(namespace, buckets, hidden)
        feat_dim = 1 + buckets + hidden
        v_mat, u_mat = self._build_frozen_subspace_bases(
            namespace=namespace,
            feat_dim=feat_dim,
            subspace_rank=subspace_rank,
            num_classes=num_classes,
            seed_tag=seed_tag,
        )
        if null_edit:
            r_zero = [[0.0] * subspace_rank for _ in range(subspace_rank)]
            return proj_w, v_mat, u_mat, r_zero

        z_sub_list: List[List[float]] = []
        y_u_list: List[List[float]] = []
        for item in train_rows:
            x_vec = self._featurize_text(str(item["input_text"]), buckets)
            phi = self._polytope_lift(x_vec, proj_w)
            s_vec = [
                sum(phi[d_idx] * v_mat[d_idx][r_idx] for d_idx in range(feat_dim))
                for r_idx in range(subspace_rank)
            ]
            z_sub_list.append(s_vec)
            t_lab = str(item["target_label"])
            c_target = label_to_idx.get(t_lab, 0)
            # y_row @ U_r^T picks column c_target of U_r^T, i.e., [u_mat[r_idx][c_target] for r_idx in range(r)]
            y_u_list.append([u_mat[r_idx][c_target] for r_idx in range(subspace_rank)])

        gram = [[0.0] * subspace_rank for _ in range(subspace_rank)]
        rhs = [[0.0] * subspace_rank for _ in range(subspace_rank)]
        for n_idx in range(len(z_sub_list)):
            sv = z_sub_list[n_idx]
            yu = y_u_list[n_idx]
            for i in range(subspace_rank):
                si = sv[i]
                for j in range(subspace_rank):
                    gram[i][j] += si * sv[j]
                    rhs[i][j] += si * yu[j]
        for i in range(subspace_rank):
            gram[i][i] += reg

        r_mat = self._solve_linear_system(gram, rhs, eps)
        return proj_w, v_mat, u_mat, r_mat

    def _evaluate_thinking_off_confound_gate(
        self,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        candidate_eval_fn: Any,
        holdout_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        null_edit: bool = False,
        regime: Optional[str] = None,
        enforce_gate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Evaluate candidate weight operator against BOTH verbose_baseline and direct_thinking_off_baseline across seeds."""
        rc = self.grid.runtime_constants
        cell = self.grid.resolve_cell(regime)
        seed_count = int(rc["confound_eval_seed_count"])
        seed_base = int(rc["default_seed_offset"])
        seed_stride = int(rc["confound_seed_stride"])
        min_split_n = int(rc["confound_min_holdout_split_samples"])
        min_gain = float(rc["confound_min_gain_over_thinking_off"])
        round_d = int(rc["ratio_round_digits"])

        require_gate = (
            bool(cell["require_thinking_off_baseline_gate"])
            if enforce_gate is None
            else bool(enforce_gate)
        )
        seeds = [seed_base + s_idx * seed_stride for s_idx in range(seed_count)]
        num_classes = max(1, len(labels))
        default_off_label = str(labels[0])

        verbose_scores: List[float] = []
        off_scores: List[float] = []
        cand_scores: List[float] = []
        per_seed: List[Dict[str, Any]] = []

        for s_idx, seed_val in enumerate(seeds):
            if holdout_samples is not None and len(holdout_samples) > 0:
                train_split = list(rows)
                eval_split = list(holdout_samples)
                mask_seed_arg: Optional[int] = None
            elif len(rows) >= min_split_n:
                by_label: Dict[str, List[int]] = {}
                for r_i, item in enumerate(rows):
                    by_label.setdefault(str(item["target_label"]), []).append(r_i)
                holdout_idx: set[int] = set()
                for grp in by_label.values():
                    if len(grp) > 1:
                        holdout_idx.add(grp[s_idx % len(grp)])
                if not holdout_idx:
                    holdout_idx.add(s_idx % len(rows))
                train_split = [rows[i] for i in range(len(rows)) if i not in holdout_idx]
                eval_split = [rows[i] for i in range(len(rows)) if i in holdout_idx]
                mask_seed_arg = None
            else:
                train_split = list(rows)
                eval_split = list(rows)
                mask_seed_arg = seed_val

            v_hits = 0.0
            off_hits = 0.0
            for item in eval_split:
                t_lab = str(item["target_label"])
                if "verbose_correct" in item:
                    v_hits += 1.0 if bool(item["verbose_correct"]) else 0.0
                elif "verbose_pred" in item:
                    v_hits += 1.0 if str(item["verbose_pred"]) == t_lab else 0.0
                else:
                    v_hits += 1.0 / float(num_classes)

                if "thinking_off_correct" in item:
                    off_hits += 1.0 if bool(item["thinking_off_correct"]) else 0.0
                elif "thinking_off_pred" in item:
                    off_hits += 1.0 if str(item["thinking_off_pred"]) == t_lab else 0.0
                else:
                    off_hits += 1.0 if default_off_label == t_lab else 0.0

            denom = float(max(1, len(eval_split)))
            v_acc = v_hits / denom
            off_acc = off_hits / denom
            if null_edit:
                c_acc = off_acc
            else:
                c_acc = float(candidate_eval_fn(train_split, eval_split, seed_val, mask_seed_arg))

            verbose_scores.append(v_acc)
            off_scores.append(off_acc)
            cand_scores.append(c_acc)
            per_seed.append(
                {
                    "seed_id": seed_val,
                    "verbose_accuracy": round(v_acc, round_d),
                    "thinking_off_accuracy": round(off_acc, round_d),
                    "candidate_accuracy": round(c_acc, round_d),
                }
            )

        mean_v = round(sum(verbose_scores) / float(len(verbose_scores)), round_d)
        mean_off = round(sum(off_scores) / float(len(off_scores)), round_d)
        mean_cand = round(sum(cand_scores) / float(len(cand_scores)), round_d)
        gain_over_verbose = round(mean_cand - mean_v, round_d)
        gain_over_off = round(mean_cand - mean_off, round_d)

        passed = (not require_gate) or (gain_over_off >= min_gain and mean_cand > mean_v)
        verdict = (
            "promoted_exceeds_matched_thinking_off_control"
            if (passed and require_gate)
            else ("promoted_gate_disabled" if passed else "rejected_thinking_off_confound")
        )
        return {
            "passed": passed,
            "verdict": verdict,
            "require_thinking_off_baseline_gate": require_gate,
            "min_required_gain_over_thinking_off": min_gain,
            "evaluated_seeds": seeds,
            "per_seed": per_seed,
            "verbose_baseline_accuracy": mean_v,
            "direct_thinking_off_accuracy": mean_off,
            "candidate_holdout_accuracy": mean_cand,
            "apparent_gain_over_verbose": gain_over_verbose,
            "matched_control_gain_over_thinking_off": gain_over_off,
        }

    def compile_stdlib_micro_router(
        self,
        namespace: str,
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        holdout_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        null_edit: bool = False,
        enforce_thinking_off_gate: Optional[bool] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cell = self.grid.resolve_cell(regime)
        min_support = int(cell["boil_down_min_trace_support"])
        buckets = int(cell["micro_feature_buckets"])
        hidden = int(cell["pruned_hidden_units"])

        rows = list(samples) if samples is not None else self.store.list_execution_traces(namespace)
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

        def _eval_split(
            train_s: Sequence[Mapping[str, Any]],
            eval_s: Sequence[Mapping[str, Any]],
            _seed: int,
            mask_seed: Optional[int],
        ) -> float:
            pw, rw = self._fit_stdlib_ridge_on_rows(namespace, train_s, labels, buckets, hidden)
            dim_local = hidden + 1
            hits = 0
            for item in eval_s:
                xv = self._featurize_text(str(item["input_text"]), buckets, mask_seed=mask_seed)
                zv: List[float] = [1.0]
                for h_i in range(hidden):
                    dot = sum(pw[h_i][b] * xv[b] for b in range(buckets))
                    zv.append(dot if dot > 0.0 else 0.0)
                sc = [sum(zv[i] * rw[i][c] for i in range(dim_local)) for c in range(num_classes)]
                p_idx = max(range(num_classes), key=lambda c: sc[c])
                if labels[p_idx] == str(item["target_label"]):
                    hits += 1
            return float(hits) / float(max(1, len(eval_s)))

        gate = self._evaluate_thinking_off_confound_gate(
            rows=rows,
            labels=labels,
            candidate_eval_fn=_eval_split,
            holdout_samples=holdout_samples,
            null_edit=null_edit,
            regime=regime,
            enforce_gate=enforce_thinking_off_gate,
        )
        if not gate["passed"]:
            return {
                "ok": False,
                "promoted": False,
                "error": "blocked_by_thinking_off_confound_gate",
                "gate_verdict": gate["verdict"],
                "backend": "stdlib_relu_ridge",
                "namespace": namespace,
                "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
                "apparent_gain_over_verbose": gate["apparent_gain_over_verbose"],
                "confound_gate": gate,
            }

        proj_w, readout_w = self._fit_stdlib_ridge_on_rows(namespace, rows, labels, buckets, hidden)
        dim = hidden + 1
        correct = 0
        for item in rows:
            x_vec = self._featurize_text(str(item["input_text"]), buckets)
            z_vec: List[float] = [1.0]
            for h_idx in range(hidden):
                dot = sum(proj_w[h_idx][b] * x_vec[b] for b in range(buckets))
                z_vec.append(dot if dot > 0.0 else 0.0)
            scores = [sum(z_vec[i] * readout_w[i][c] for i in range(dim)) for c in range(num_classes)]
            pred_idx = max(range(num_classes), key=lambda c: scores[c])
            if labels[pred_idx] == str(item["target_label"]):
                correct += 1

        acc = float(correct) / float(len(rows))
        elapsed_ms = (time.perf_counter() - t0) * float(self.grid.runtime_constants["ms_per_second"])
        router_spec = {
            "backend": "stdlib_relu_ridge",
            "namespace": namespace,
            "buckets": buckets,
            "hidden": hidden,
            "labels": labels,
            "proj_w": proj_w,
            "readout_w": readout_w,
            "train_accuracy": acc,
            "compile_ms": elapsed_ms,
        }
        self.compiled_routers[namespace] = router_spec
        return {
            "ok": True,
            "promoted": True,
            "gate_verdict": gate["verdict"],
            "backend": "stdlib_relu_ridge",
            "namespace": namespace,
            "labels": labels,
            "sample_count": len(rows),
            "train_accuracy": acc,
            "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
            "confound_gate": gate,
            "compile_ms": round(elapsed_ms, int(self.grid.runtime_constants["ngram_span"])),
            "llm_tokens_per_inference": 0,
        }

    def compile_lora_xs_cartridge(
        self,
        namespace: str,
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        holdout_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        subspace_rank: Optional[int] = None,
        null_edit: bool = False,
        enforce_thinking_off_gate: Optional[bool] = None,
        export_dir: Optional[Path] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize an r x r float32 LoRA-XS singular subspace cartridge (256 bytes at r=8) gated by thinking=off control."""
        t0 = time.perf_counter()
        cell = self.grid.resolve_cell(regime)
        min_support = int(cell["boil_down_min_trace_support"])
        buckets = int(cell["micro_feature_buckets"])
        hidden = int(cell["pruned_hidden_units"])
        rank_r = max(1, int(subspace_rank) if subspace_rank is not None else int(cell["lora_xs_subspace_rank"]))
        f32_bytes = int(self.grid.runtime_constants["float32_byte_width"])

        rows = list(samples) if samples is not None else self.store.list_execution_traces(namespace)
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

        feat_dim = 1 + buckets + hidden

        def _eval_lora_split(
            train_s: Sequence[Mapping[str, Any]],
            eval_s: Sequence[Mapping[str, Any]],
            seed_val: int,
            mask_seed: Optional[int],
        ) -> float:
            pw, vm, um, rm = self._fit_lora_xs_on_rows(
                namespace=namespace,
                train_rows=train_s,
                labels=labels,
                buckets=buckets,
                hidden=hidden,
                subspace_rank=rank_r,
                seed_tag=seed_val,
                null_edit=null_edit,
            )
            hits = 0
            for item in eval_s:
                xv = self._featurize_text(str(item["input_text"]), buckets, mask_seed=mask_seed)
                phi = self._polytope_lift(xv, pw)
                sv = [sum(phi[d_i] * vm[d_i][r_i] for d_i in range(feat_dim)) for r_i in range(rank_r)]
                qv = [sum(sv[r_i] * rm[r_i][r_j] for r_i in range(rank_r)) for r_j in range(rank_r)]
                logits = [sum(qv[r_j] * um[r_j][c_i] for r_j in range(rank_r)) for c_i in range(num_classes)]
                p_idx = max(range(num_classes), key=lambda c: logits[c])
                if labels[p_idx] == str(item["target_label"]):
                    hits += 1
            return float(hits) / float(max(1, len(eval_s)))

        gate = self._evaluate_thinking_off_confound_gate(
            rows=rows,
            labels=labels,
            candidate_eval_fn=_eval_lora_split,
            holdout_samples=holdout_samples,
            null_edit=null_edit,
            regime=regime,
            enforce_gate=enforce_thinking_off_gate,
        )

        proj_w, v_mat, u_mat, r_mat = self._fit_lora_xs_on_rows(
            namespace=namespace,
            train_rows=rows,
            labels=labels,
            buckets=buckets,
            hidden=hidden,
            subspace_rank=rank_r,
            seed_tag=0,
            null_edit=null_edit,
        )
        flat_r = [float(r_mat[i][j]) for i in range(rank_r) for j in range(rank_r)]
        packed_bytes = struct.pack(f"<{rank_r * rank_r}f", *flat_r)
        expected_bytes = rank_r * rank_r * f32_bytes
        assert len(packed_bytes) == expected_bytes
        cartridge_b64 = base64.b64encode(packed_bytes).decode("ascii")

        if not gate["passed"]:
            return {
                "ok": False,
                "promoted": False,
                "error": "blocked_by_thinking_off_confound_gate",
                "gate_verdict": gate["verdict"],
                "backend": "lora_xs_subspace_cartridge",
                "namespace": namespace,
                "subspace_rank": rank_r,
                "cartridge_bytes": len(packed_bytes),
                "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
                "apparent_gain_over_verbose": gate["apparent_gain_over_verbose"],
                "confound_gate": gate,
            }

        # Evaluate train accuracy using quantized float32 unpacked R matrix for exact parity
        unpacked_flat = list(struct.unpack(f"<{rank_r * rank_r}f", packed_bytes))
        r_quant = [
            unpacked_flat[i * rank_r : (i + 1) * rank_r]
            for i in range(rank_r)
        ]
        correct = 0
        for item in rows:
            xv = self._featurize_text(str(item["input_text"]), buckets)
            phi = self._polytope_lift(xv, proj_w)
            sv = [sum(phi[d_i] * v_mat[d_i][r_i] for d_i in range(feat_dim)) for r_i in range(rank_r)]
            qv = [sum(sv[r_i] * r_quant[r_i][r_j] for r_i in range(rank_r)) for r_j in range(rank_r)]
            logits = [sum(qv[r_j] * u_mat[r_j][c_i] for r_j in range(rank_r)) for c_i in range(num_classes)]
            pred_c = max(range(num_classes), key=lambda c: logits[c])
            if labels[pred_c] == str(item["target_label"]):
                correct += 1
        train_acc = float(correct) / float(len(rows))
        elapsed_ms = (time.perf_counter() - t0) * float(self.grid.runtime_constants["ms_per_second"])

        bin_path: Optional[str] = None
        if export_dir is not None:
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
                out_bin = export_dir / f"lora_xs_{namespace}.bin"
                out_bin.write_bytes(packed_bytes)
                bin_path = str(out_bin)
            except Exception:
                bin_path = None

        cartridge_spec = {
            "backend": "lora_xs_subspace_cartridge",
            "namespace": namespace,
            "subspace_rank": rank_r,
            "cartridge_bytes": len(packed_bytes),
            "cartridge_b64": cartridge_b64,
            "cartridge_path": bin_path,
            "buckets": buckets,
            "hidden": hidden,
            "feat_dim": feat_dim,
            "labels": labels,
            "proj_w": proj_w,
            "v_mat": v_mat,
            "u_mat": u_mat,
            "r_mat": r_quant,
            "train_accuracy": train_acc,
            "compile_ms": elapsed_ms,
        }
        self.lora_xs_cartridges[namespace] = cartridge_spec
        self.compiled_routers[namespace] = cartridge_spec
        return {
            "ok": True,
            "promoted": True,
            "gate_verdict": gate["verdict"],
            "backend": "lora_xs_subspace_cartridge",
            "namespace": namespace,
            "labels": labels,
            "subspace_rank": rank_r,
            "cartridge_bytes": len(packed_bytes),
            "cartridge_b64": cartridge_b64,
            "cartridge_path": bin_path,
            "sample_count": len(rows),
            "train_accuracy": train_acc,
            "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
            "apparent_gain_over_verbose": gate["apparent_gain_over_verbose"],
            "confound_gate": gate,
            "compile_ms": round(elapsed_ms, int(self.grid.runtime_constants["ngram_span"])),
            "llm_tokens_per_inference": 0,
        }

    def hot_swap_lora_xs_cartridge(
        self,
        namespace: str,
        cartridge_b64: str,
        labels: Sequence[str],
        subspace_rank: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hot-swap a serialized r x r float32 LoRA-XS cartridge (e.g., 256 bytes) onto frozen deterministic bases."""
        cell = self.grid.resolve_cell(regime)
        buckets = int(cell["micro_feature_buckets"])
        hidden = int(cell["pruned_hidden_units"])
        rank_r = max(1, int(subspace_rank) if subspace_rank is not None else int(cell["lora_xs_subspace_rank"]))
        f32_bytes = int(self.grid.runtime_constants["float32_byte_width"])
        expected_bytes = rank_r * rank_r * f32_bytes

        raw_bytes = base64.b64decode(cartridge_b64.encode("ascii"))
        if len(raw_bytes) != expected_bytes:
            return {
                "ok": False,
                "error": "invalid_cartridge_byte_length",
                "expected_bytes": expected_bytes,
                "received_bytes": len(raw_bytes),
            }
        clean_labels = [str(lab) for lab in labels]
        if len(clean_labels) <= 1:
            return {"ok": False, "error": "need_at_least_two_distinct_labels"}

        unpacked_flat = list(struct.unpack(f"<{rank_r * rank_r}f", raw_bytes))
        r_quant = [unpacked_flat[i * rank_r : (i + 1) * rank_r] for i in range(rank_r)]
        proj_w = self._build_polytope_projection(namespace, buckets, hidden)
        feat_dim = 1 + buckets + hidden
        v_mat, u_mat = self._build_frozen_subspace_bases(
            namespace=namespace,
            feat_dim=feat_dim,
            subspace_rank=rank_r,
            num_classes=len(clean_labels),
            seed_tag=0,
        )
        spec = {
            "backend": "lora_xs_subspace_cartridge",
            "namespace": namespace,
            "subspace_rank": rank_r,
            "cartridge_bytes": len(raw_bytes),
            "cartridge_b64": cartridge_b64,
            "buckets": buckets,
            "hidden": hidden,
            "feat_dim": feat_dim,
            "labels": clean_labels,
            "proj_w": proj_w,
            "v_mat": v_mat,
            "u_mat": u_mat,
            "r_mat": r_quant,
            "train_accuracy": 1.0,
            "compile_ms": 0.0,
        }
        self.lora_xs_cartridges[namespace] = spec
        self.compiled_routers[namespace] = spec
        return {
            "ok": True,
            "hot_swapped": True,
            "backend": "lora_xs_subspace_cartridge",
            "namespace": namespace,
            "subspace_rank": rank_r,
            "cartridge_bytes": len(raw_bytes),
            "labels": clean_labels,
        }

    def predict_micro_router(self, namespace: str, input_text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        spec = self.compiled_routers.get(namespace)
        if spec is None:
            return {"ok": False, "error": "no_compiled_router_for_namespace"}

        backend = str(spec["backend"])
        buckets = int(spec["buckets"])
        labels: List[str] = list(spec["labels"])
        x_vec = self._featurize_text(input_text, buckets)

        if backend == "lora_xs_subspace_cartridge":
            rank_r = int(spec["subspace_rank"])
            feat_dim = int(spec["feat_dim"])
            proj_w = spec["proj_w"]
            v_mat = spec["v_mat"]
            u_mat = spec["u_mat"]
            r_mat = spec["r_mat"]
            phi = self._polytope_lift(x_vec, proj_w)
            sv = [sum(phi[d_i] * v_mat[d_i][r_i] for d_i in range(feat_dim)) for r_i in range(rank_r)]
            qv = [sum(sv[r_i] * r_mat[r_i][r_j] for r_i in range(rank_r)) for r_j in range(rank_r)]
            scores = [sum(qv[r_j] * u_mat[r_j][c_i] for r_j in range(rank_r)) for c_i in range(len(labels))]
            best_c = max(range(len(labels)), key=lambda c: scores[c])
            latency_ms = (time.perf_counter() - t0) * float(self.grid.runtime_constants["ms_per_second"])
            return {
                "ok": True,
                "namespace": namespace,
                "backend": backend,
                "cartridge_bytes": int(spec["cartridge_bytes"]),
                "subspace_rank": rank_r,
                "predicted_label": labels[best_c],
                "latency_ms": latency_ms,
                "llm_tokens_spent": 0,
            }

        hidden = int(spec["hidden"])
        proj_w = spec["proj_w"]
        readout_w = spec["readout_w"]
        z_vec: List[float] = [1.0]
        for h_idx in range(hidden):
            dot = sum(proj_w[h_idx][b] * x_vec[b] for b in range(buckets))
            z_vec.append(dot if dot > 0.0 else 0.0)

        dim = hidden + 1
        scores = [sum(z_vec[i] * readout_w[i][c] for i in range(dim)) for c in range(len(labels))]
        best_c = max(range(len(labels)), key=lambda c: scores[c])
        latency_ms = (time.perf_counter() - t0) * float(self.grid.runtime_constants["ms_per_second"])
        return {
            "ok": True,
            "namespace": namespace,
            "backend": backend,
            "predicted_label": labels[best_c],
            "latency_ms": latency_ms,
            "llm_tokens_spent": 0,
        }

    # ── Optional Neural Tier (NumPy / PyTorch / Safetensors) ──────────────

    def run_neural_surgery_or_pruning(
        self,
        operation: str,
        namespace: str = "default",
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        holdout_samples: Optional[Sequence[Mapping[str, Any]]] = None,
        anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
        edit_vectors: Optional[Sequence[Sequence[float]]] = None,
        edit_targets: Optional[Sequence[Sequence[float]]] = None,
        null_edit: bool = False,
        enforce_thinking_off_gate: Optional[bool] = None,
        export_dir: Optional[Path] = None,
        masked_modules: Optional[Tuple[str, ...]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        if masked_modules and ("numpy" in masked_modules or "torch" in masked_modules):
            return self.compile_stdlib_micro_router(
                namespace=namespace,
                samples=samples,
                holdout_samples=holdout_samples,
                null_edit=null_edit,
                enforce_thinking_off_gate=enforce_thinking_off_gate,
                regime=regime,
            )

        try:
            np = importlib.import_module("numpy")
            torch = importlib.import_module("torch")
            torch.set_num_threads(1)
        except Exception:
            return self.compile_stdlib_micro_router(
                namespace=namespace,
                samples=samples,
                holdout_samples=holdout_samples,
                null_edit=null_edit,
                enforce_thinking_off_gate=enforce_thinking_off_gate,
                regime=regime,
            )

        cell = self.grid.resolve_cell(regime)
        reg = float(self.grid.runtime_constants["ridge_regularization"])
        tol = float(self.grid.runtime_constants["nullspace_rank_tolerance"])
        pct_scale = float(self.grid.runtime_constants["percent_scale"])
        min_gain = float(self.grid.runtime_constants["confound_min_gain_over_thinking_off"])
        round_d = int(self.grid.runtime_constants["ratio_round_digits"])
        require_gate = (
            bool(cell["require_thinking_off_baseline_gate"])
            if enforce_thinking_off_gate is None
            else bool(enforce_thinking_off_gate)
        )

        if operation == "nullspace_surgery":
            if not anchor_vectors or not edit_vectors or not edit_targets:
                return {"ok": False, "error": "missing_anchor_or_edit_vectors"}
            a_mat = np.asarray(anchor_vectors, dtype=np.float64)
            e_mat = np.asarray(edit_vectors, dtype=np.float64)
            y_mat = np.asarray(edit_targets, dtype=np.float64)
            feat_d = a_mat.shape[1]
            out_d = y_mat.shape[1]
            w_base = np.zeros((out_d, feat_d), dtype=np.float64)

            _, sing_vals, vt = np.linalg.svd(a_mat, full_matrices=True)
            rank = int(np.sum(sing_vals > tol))
            v_null = vt[rank:].T
            p_null = v_null @ v_null.T

            if null_edit:
                w_edited = w_base.copy()
            else:
                e_proj = e_mat @ p_null
                gram = e_proj.T @ e_proj + reg * np.eye(feat_d, dtype=np.float64)
                delta_t = p_null @ np.linalg.solve(gram, e_proj.T @ (y_mat - (e_mat @ w_base.T)))
                w_edited = w_base + delta_t.T

            anchor_drift = float(np.max(np.abs((a_mat @ w_edited.T) - (a_mat @ w_base.T))))
            edit_true = np.argmax(y_mat, axis=1)
            base_preds = np.argmax(e_mat @ w_base.T, axis=1)
            edit_preds = np.argmax(e_mat @ w_edited.T, axis=1)
            off_acc = round(float(np.mean(base_preds == edit_true)), round_d)
            edit_acc = round(float(np.mean(edit_preds == edit_true)), round_d)
            gain_off = round(edit_acc - off_acc, round_d)

            if require_gate and gain_off < min_gain:
                return {
                    "ok": False,
                    "promoted": False,
                    "error": "blocked_by_thinking_off_confound_gate",
                    "gate_verdict": "rejected_thinking_off_confound",
                    "backend": "numpy_svd_nullspace",
                    "operation": operation,
                    "anchor_max_logit_drift": anchor_drift,
                    "direct_thinking_off_accuracy": off_acc,
                    "unseen_edit_accuracy": edit_acc,
                    "matched_control_gain_over_thinking_off": gain_off,
                }
            return {
                "ok": True,
                "promoted": True,
                "gate_verdict": "promoted_exceeds_matched_thinking_off_control" if require_gate else "promoted_gate_disabled",
                "backend": "numpy_svd_nullspace",
                "operation": operation,
                "anchor_max_logit_drift": anchor_drift,
                "direct_thinking_off_accuracy": off_acc,
                "unseen_edit_accuracy": edit_acc,
                "matched_control_gain_over_thinking_off": gain_off,
                "sgd_epochs": 0,
            }

        # Default neural operation: synthesize multi-head torch micro-net + Wanda channel pruning
        rows = list(samples) if samples is not None else self.store.list_execution_traces(namespace)
        min_support = int(cell["boil_down_min_trace_support"])
        if len(rows) < min_support:
            return {"ok": False, "error": "insufficient_trace_support", "recorded_traces": len(rows)}

        buckets = int(cell["micro_feature_buckets"])
        full_h = int(cell["micro_hidden_units"])
        keep_h = int(cell["pruned_hidden_units"])
        steps = int(self.grid.runtime_constants["neural_train_steps"])
        lr = float(self.grid.runtime_constants["neural_learning_rate"])

        labels = sorted({str(r["target_label"]) for r in rows})
        lmap = {lab: idx for idx, lab in enumerate(labels)}

        def _fit_and_score_wanda(
            train_s: Sequence[Mapping[str, Any]],
            eval_s: Sequence[Mapping[str, Any]],
            seed_val: int,
            mask_seed: Optional[int],
        ) -> float:
            x_tr = np.asarray([self._featurize_text(str(r["input_text"]), buckets) for r in train_s], dtype=np.float32)
            y_tr = np.asarray([lmap[str(r["target_label"])] for r in train_s], dtype=np.int64)
            torch.manual_seed(int(seed_val))
            fc1 = torch.nn.Linear(buckets, full_h)
            fc2 = torch.nn.Linear(full_h, len(labels))
            opt_s = torch.optim.Adam(list(fc1.parameters()) + list(fc2.parameters()), lr=lr)
            ce = torch.nn.CrossEntropyLoss()
            xt = torch.from_numpy(x_tr)
            yt = torch.from_numpy(y_tr)
            for _ in range(steps):
                opt_s.zero_grad()
                loss_s = ce(fc2(torch.relu(fc1(xt))), yt)
                loss_s.backward()
                opt_s.step()
            with torch.no_grad():
                act_s = torch.relu(fc1(xt)).numpy()
                w1_s = fc1.weight.detach().numpy()
                w2_s = fc2.weight.detach().numpy()
                imp_s = np.mean(np.abs(act_s), axis=0) * np.linalg.norm(w2_s, axis=0)
                k_idx = np.argsort(-imp_s)[:keep_h]
                hp_s = np.maximum(0.0, x_tr.astype(np.float64) @ w1_s[k_idx].T)
                zd_s = np.concatenate([np.ones((hp_s.shape[0], 1), dtype=np.float64), hp_s], axis=1)
                wr_s = np.linalg.solve(
                    zd_s.T @ zd_s + reg * np.eye(keep_h + 1, dtype=np.float64),
                    zd_s.T @ np.eye(len(labels), dtype=np.float64)[y_tr],
                )
                x_ev = np.asarray(
                    [self._featurize_text(str(r["input_text"]), buckets, mask_seed=mask_seed) for r in eval_s],
                    dtype=np.float64,
                )
                y_ev = np.asarray([lmap[str(r["target_label"])] for r in eval_s], dtype=np.int64)
                hp_ev = np.maximum(0.0, x_ev @ w1_s[k_idx].T)
                zd_ev = np.concatenate([np.ones((hp_ev.shape[0], 1), dtype=np.float64), hp_ev], axis=1)
                pr_ev = np.argmax(zd_ev @ wr_s, axis=1)
                return float(np.mean(pr_ev == y_ev))

        gate = self._evaluate_thinking_off_confound_gate(
            rows=rows,
            labels=labels,
            candidate_eval_fn=_fit_and_score_wanda,
            holdout_samples=holdout_samples,
            null_edit=null_edit,
            regime=regime,
            enforce_gate=enforce_thinking_off_gate,
        )
        if not gate["passed"]:
            return {
                "ok": False,
                "promoted": False,
                "error": "blocked_by_thinking_off_confound_gate",
                "gate_verdict": gate["verdict"],
                "backend": "torch_wanda_pruned_micro_net",
                "namespace": namespace,
                "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
                "confound_gate": gate,
            }

        x_np = np.asarray([self._featurize_text(str(r["input_text"]), buckets) for r in rows], dtype=np.float32)
        y_np = np.asarray([lmap[str(r["target_label"])] for r in rows], dtype=np.int64)

        torch.manual_seed(int(self.grid.runtime_constants["default_seed_offset"]))
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        fc_in = torch.nn.Linear(buckets, full_h)
        fc_out = torch.nn.Linear(full_h, len(labels))
        opt = torch.optim.Adam(list(fc_in.parameters()) + list(fc_out.parameters()), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(steps):
            opt.zero_grad()
            act = torch.relu(fc_in(x_t))
            logits = fc_out(act)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            act_full = torch.relu(fc_in(x_t)).numpy()
            w1_np = fc_in.weight.detach().numpy()
            b1_np = fc_in.bias.detach().numpy()
            w2_np = fc_out.weight.detach().numpy()

            act_norm = np.mean(np.abs(act_full), axis=0)
            col_norm = np.linalg.norm(w2_np, axis=0)
            importance = act_norm * col_norm
            keep_idx = np.argsort(-importance)[:keep_h]

            h_pruned = np.maximum(0.0, x_np.astype(np.float64) @ w1_np[keep_idx].T)
            ones = np.ones((h_pruned.shape[0], 1), dtype=np.float64)
            z_design = np.concatenate([ones, h_pruned], axis=1)
            eye = np.eye(keep_h + 1, dtype=np.float64)
            y_oh = np.eye(len(labels), dtype=np.float64)[y_np]
            w_ridge = np.linalg.solve(z_design.T @ z_design + reg * eye, z_design.T @ y_oh)

            preds = np.argmax(z_design @ w_ridge, axis=1)
            pruned_acc = float(np.mean(preds == y_np))

        dense_params = buckets * full_h + full_h + full_h * len(labels) + len(labels)
        pruned_params = buckets * keep_h + keep_h + (keep_h + 1) * len(labels)
        ram_reduction_pct = (1.0 - (float(pruned_params) / float(max(1, dense_params)))) * pct_scale

        proj_kept = [[float(v) for v in w1_np[idx]] for idx in keep_idx]
        self.compiled_routers[namespace] = {
            "backend": "torch_wanda_pruned_micro_net",
            "namespace": namespace,
            "buckets": buckets,
            "hidden": keep_h,
            "labels": labels,
            "proj_w": proj_kept,
            "readout_w": [[float(v) for v in row] for row in w_ridge],
            "train_accuracy": pruned_acc,
            "compile_ms": 0.0,
        }

        safetensors_path: Optional[str] = None
        if export_dir is not None:
            try:
                st_mod = importlib.import_module("safetensors.torch")
                export_dir.mkdir(parents=True, exist_ok=True)
                out_file = export_dir / f"micro_net_{namespace}.safetensors"
                st_mod.save_file(
                    {
                        "trunk_weight": torch.from_numpy(w1_np[keep_idx].copy()),
                        "trunk_bias": torch.from_numpy(b1_np[keep_idx].copy()),
                        "readout_ridge": torch.from_numpy(w_ridge.astype(np.float32)),
                    },
                    str(out_file),
                )
                safetensors_path = str(out_file)
            except Exception:
                safetensors_path = None

        return {
            "ok": True,
            "promoted": True,
            "gate_verdict": gate["verdict"],
            "backend": "torch_wanda_pruned_micro_net",
            "namespace": namespace,
            "dense_hidden_units": full_h,
            "pruned_hidden_units": keep_h,
            "dense_params": dense_params,
            "pruned_params": pruned_params,
            "ram_reduction_pct": round(ram_reduction_pct, int(self.grid.runtime_constants["pair_divisor"])),
            "pruned_accuracy": pruned_acc,
            "matched_control_gain_over_thinking_off": gate["matched_control_gain_over_thinking_off"],
            "confound_gate": gate,
            "safetensors_path": safetensors_path,
        }


class LoopbackSwarmProxyManager:
    """On-demand loopback HTTP server (/v1/models, /v1/chat/completions, /swarm/*) for Hermes, GrokBot, and OpenAI-v1 swarms."""

    def __init__(
        self,
        grid: AdaptiveNodeGrid,
        store: LocalAdaptiveFabricStore,
        boil_engine: AdaptiveBoilDownEngine,
        budget_gate: EffortAndBudgetGate,
        local_runner_url: str = "http://127.0.0.1:11434",
        allow_local_runner: bool = False,
    ) -> None:
        self.grid = grid
        self.store = store
        self.boil_engine = boil_engine
        self.budget_gate = budget_gate
        self.local_runner_url = local_runner_url.rstrip("/")
        self.allow_local_runner = bool(allow_local_runner)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.requests_served: int = 0
        self.zero_token_hits: int = 0

    def _build_models_catalog(self) -> Dict[str, Any]:
        now_i = int(time.time())
        models: List[Dict[str, Any]] = [
            {
                "id": "omni-fabric-auto",
                "object": "model",
                "created": now_i,
                "owned_by": "omni-fabric-runtime",
                "zero_llm_tokens": False,
            }
        ]
        for ns, spec in sorted(self.boil_engine.compiled_routers.items()):
            backend = str(spec["backend"])
            model_id = f"loraxs:{ns}" if backend == "lora_xs_subspace_cartridge" else f"boildown:{ns}"
            entry: Dict[str, Any] = {
                "id": model_id,
                "object": "model",
                "created": now_i,
                "owned_by": "omni-fabric-runtime",
                "namespace": ns,
                "backend": backend,
                "zero_llm_tokens": True,
            }
            if backend == "lora_xs_subspace_cartridge":
                entry["cartridge_bytes"] = int(spec["cartridge_bytes"])
                entry["subspace_rank"] = int(spec["subspace_rank"])
            models.append(entry)
        return {"object": "list", "data": models}

    def _resolve_namespace_from_request(self, data: Mapping[str, Any]) -> str:
        explicit_ns = str(data.get("namespace") or "").strip()
        if explicit_ns:
            return explicit_ns
        raw_model = str(data.get("model") or "").strip()
        for prefix in ("loraxs:", "boildown:"):
            if raw_model.startswith(prefix):
                return raw_model[len(prefix) :]
        if raw_model in self.boil_engine.compiled_routers:
            return raw_model
        return "swarm"

    def start(self, host: str = "127.0.0.1", port: int = 0) -> Dict[str, Any]:
        clean_host = host.strip().lower()
        if clean_host not in ("127.0.0.1", "localhost", "::1"):
            return {"ok": False, "running": False, "error": "loopback_bind_host_required"}
        if self._server is not None:
            addr = self._server.server_address
            return {
                "ok": True,
                "running": True,
                "listen_url": f"http://{addr[0]}:{addr[1]}",
                "openai_base_url": f"http://{addr[0]}:{addr[1]}/v1",
                "requests_served": self.requests_served,
                "zero_token_hits": self.zero_token_hits,
            }

        manager = self

        class _SwarmHandler(BaseHTTPRequestHandler):
            def log_message(self, _fmt: str, *_args: Any) -> None:
                return

            def _send_json(self, code: int, body: Mapping[str, Any]) -> None:
                raw = json.dumps(dict(body), sort_keys=True).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _guard_loopback_request(self) -> bool:
                forbid_code = int(manager.grid.runtime_constants["http_status_forbidden"])
                host_hdr = (self.headers.get("Host") or "").strip().lower()
                valid_host = host_hdr in ("127.0.0.1", "localhost", "[::1]") or host_hdr.startswith(
                    ("127.0.0.1:", "localhost:", "[::1]:")
                )
                if not valid_host:
                    self._send_json(forbid_code, {"ok": False, "error": "forbidden_host_header"})
                    return False

                origin_hdr = (self.headers.get("Origin") or "").strip()
                if origin_hdr:
                    parsed_origin = urllib.parse.urlsplit(origin_hdr)
                    origin_scheme = parsed_origin.scheme.lower()
                    origin_host = (parsed_origin.hostname or "").strip().lower()
                    if origin_scheme not in ("http", "https") or origin_host not in ("127.0.0.1", "localhost", "::1"):
                        self._send_json(forbid_code, {"ok": False, "error": "forbidden_origin_header"})
                        return False
                return True

            def do_GET(self) -> None:
                if not self._guard_loopback_request():
                    return
                ok_code = int(manager.grid.runtime_constants["http_status_ok"])
                parsed_path = urllib.parse.urlparse(self.path)
                route = parsed_path.path

                if route.startswith("/health"):
                    self._send_json(
                        ok_code,
                        {
                            "ok": True,
                            "requests_served": manager.requests_served,
                            "zero_token_hits": manager.zero_token_hits,
                            "compiled_routers": sorted(manager.boil_engine.compiled_routers.keys()),
                            "lora_xs_cartridges": sorted(manager.boil_engine.lora_xs_cartridges.keys()),
                            "substrate": manager.store.summarize_status(),
                        },
                    )
                    return

                if route in ("/v1/models", "/models"):
                    manager.requests_served += 1
                    self._send_json(ok_code, manager._build_models_catalog())
                    return

                if route == "/swarm/traces":
                    qs = urllib.parse.parse_qs(parsed_path.query)
                    ns = qs["namespace"][0] if "namespace" in qs and qs["namespace"] else None
                    rmode = qs["reasoning_mode"][0] if "reasoning_mode" in qs and qs["reasoning_mode"] else None
                    items = manager.store.query_trace_envelopes(namespace=ns, reasoning_mode=rmode)
                    self._send_json(ok_code, {"ok": True, "count": len(items), "envelopes": items})
                    return

                self._send_json(
                    int(manager.grid.runtime_constants["http_status_bad_request"]),
                    {"ok": False, "error": "unknown_route"},
                )

            def do_POST(self) -> None:
                if not self._guard_loopback_request():
                    return
                manager.requests_served += 1
                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                data = json.loads(raw_body or "{}")
                ok_code = int(manager.grid.runtime_constants["http_status_ok"])

                if self.path == "/swarm/enqueue":
                    res = manager.store.enqueue_task(
                        task_id=str(data["task_id"]),
                        namespace=str(data.get("namespace") or "swarm"),
                        operation=str(data.get("operation") or "execute"),
                        payload=dict(data.get("payload") or {}),
                        contract_tag=str(data.get("contract_tag") or "v1"),
                        depends_on=data.get("depends_on"),
                        prefix_signature=data.get("prefix_signature"),
                        capability_tag=data.get("capability_tag"),
                        tool_schema_tag=data.get("tool_schema_tag"),
                        origin_id=data.get("origin_id"),
                        step_confidence_milli=data.get("step_confidence_milli"),
                    )
                    self._send_json(ok_code, res)
                    return

                if self.path == "/swarm/claim":
                    res = manager.store.claim_task(
                        worker_id=str(data["worker_id"]),
                        namespace=data.get("namespace"),
                        task_id=data.get("task_id"),
                    )
                    code = int(res.get("status_code") or ok_code)
                    self._send_json(code, res)
                    return

                if self.path in ("/swarm/claim-bundle", "/swarm/claim_bundle"):
                    res = manager.store.claim_bundle(
                        worker_id=str(data["worker_id"]),
                        bundle_size=data.get("bundle_size"),
                        dispatch_mode=data.get("dispatch_mode"),
                        namespace=data.get("namespace"),
                        preferred_prefix_signature=data.get("preferred_prefix_signature"),
                        preferred_capability_tag=data.get("preferred_capability_tag"),
                        preferred_tool_schema_tag=data.get("preferred_tool_schema_tag"),
                    )
                    self._send_json(ok_code, res)
                    return

                if self.path == "/swarm/checkpoint":
                    res = manager.store.checkpoint_task(
                        task_id=str(data["task_id"]),
                        worker_id=str(data["worker_id"]),
                        lease_epoch=int(data["lease_epoch"]),
                        fence_token=str(data["fence_token"]),
                        checkpoint=dict(data.get("checkpoint") or {}),
                        completed=bool(data.get("completed", False)),
                    )
                    code = int(res.get("status_code") or ok_code)
                    self._send_json(code, res)
                    return

                if self.path == "/swarm/trace":
                    if str(data.get("action") or "") == "query":
                        items = manager.store.query_trace_envelopes(
                            namespace=data.get("namespace"),
                            reasoning_mode=data.get("reasoning_mode"),
                            prefix_signature=data.get("prefix_signature"),
                            task_id=data.get("task_id"),
                            seed_id=data.get("seed_id"),
                            limit=data.get("limit"),
                        )
                        self._send_json(ok_code, {"ok": True, "count": len(items), "envelopes": items})
                        return
                    res = manager.store.record_trace_envelope(
                        namespace=str(data.get("namespace") or "swarm"),
                        task_id=str(data.get("task_id") or ""),
                        object_id=str(data.get("object_id") or ""),
                        worker_id=str(data.get("worker_id") or ""),
                        input_text=str(data.get("input_text") or ""),
                        target_label=str(data.get("target_label") or ""),
                        prefix_signature=str(data.get("prefix_signature") or ""),
                        reasoning_mode=str(data.get("reasoning_mode") or "adaptive"),
                        thinking_tokens=int(data.get("thinking_tokens") or 0),
                        answer_tokens=int(data.get("answer_tokens") or 0),
                        tokens_used=data.get("tokens_used"),
                        latency_ms=float(data.get("latency_ms") or 0.0),
                        activation_norm_digest=str(data.get("activation_norm_digest") or ""),
                        attenuation_confidence_milli=data.get("attenuation_confidence_milli"),
                        seed_id=int(data.get("seed_id") or 0),
                    )
                    self._send_json(ok_code, res)
                    return

                if self.path == "/swarm/submit":
                    has_enriched_trace = any(
                        k in data
                        for k in (
                            "reasoning_mode",
                            "thinking_tokens",
                            "answer_tokens",
                            "activation_norm_digest",
                            "seed_id",
                            "complete_task",
                        )
                    )
                    if has_enriched_trace:
                        res = manager.store.submit_result(
                            task_id=str(data["task_id"]),
                            maker_id=str(data["maker_id"]),
                            checker_id=data.get("checker_id"),
                            namespace=str(data.get("namespace") or "swarm"),
                            title=str(data.get("title") or data["task_id"]),
                            summary_text=str(data.get("summary_text") or ""),
                            payload=dict(data.get("payload") or {}),
                            lease_epoch=data.get("lease_epoch"),
                            fence_token=data.get("fence_token"),
                            complete_task=bool(data.get("complete_task", False)),
                            input_text=str(data.get("input_text") or ""),
                            target_label=str(data.get("target_label") or ""),
                            prefix_signature=str(data.get("prefix_signature") or ""),
                            reasoning_mode=str(data.get("reasoning_mode") or "adaptive"),
                            thinking_tokens=int(data.get("thinking_tokens") or 0),
                            answer_tokens=int(data.get("answer_tokens") or 0),
                            latency_ms=float(data.get("latency_ms") or 0.0),
                            activation_norm_digest=str(data.get("activation_norm_digest") or ""),
                            attenuation_confidence_milli=data.get("attenuation_confidence_milli"),
                            seed_id=int(data.get("seed_id") or 0),
                        )
                    else:
                        res = manager.store.stage_or_promote_object(
                            task_id=str(data["task_id"]),
                            maker_id=str(data["maker_id"]),
                            checker_id=data.get("checker_id"),
                            namespace=str(data.get("namespace") or "swarm"),
                            title=str(data.get("title") or data["task_id"]),
                            summary_text=str(data.get("summary_text") or ""),
                            payload=dict(data.get("payload") or {}),
                            attenuation_confidence_milli=data.get("attenuation_confidence_milli"),
                        )
                    code = int(res.get("status_code") or ok_code)
                    self._send_json(code, res)
                    return

                if self.path == "/swarm/boil_down":
                    ns = str(data.get("namespace") or "swarm")
                    mode = str(data.get("mode") or "stdlib_relu_ridge")
                    if mode == "hot_swap_lora_xs":
                        res = manager.boil_engine.hot_swap_lora_xs_cartridge(
                            namespace=ns,
                            cartridge_b64=str(data["cartridge_b64"]),
                            labels=list(data.get("labels") or []),
                            subspace_rank=data.get("subspace_rank"),
                        )
                    elif mode == "lora_xs":
                        res = manager.boil_engine.compile_lora_xs_cartridge(
                            namespace=ns,
                            samples=data.get("samples"),
                            holdout_samples=data.get("holdout_samples"),
                            subspace_rank=data.get("subspace_rank"),
                            null_edit=bool(data.get("null_edit", False)),
                        )
                    else:
                        res = manager.boil_engine.compile_stdlib_micro_router(
                            namespace=ns,
                            samples=data.get("samples"),
                            holdout_samples=data.get("holdout_samples"),
                            null_edit=bool(data.get("null_edit", False)),
                        )
                    self._send_json(ok_code, res)
                    return

                if self.path in ("/v1/chat/completions", "/chat/completions"):
                    ns = manager._resolve_namespace_from_request(data)
                    if data.get("hot_swap_cartridge_b64") and data.get("labels"):
                        manager.boil_engine.hot_swap_lora_xs_cartridge(
                            namespace=ns,
                            cartridge_b64=str(data["hot_swap_cartridge_b64"]),
                            labels=list(data["labels"]),
                            subspace_rank=data.get("subspace_rank"),
                        )
                    msgs = data.get("messages") or []
                    prompt_txt = ""
                    if isinstance(msgs, list) and msgs:
                        prompt_txt = str(msgs[-1].get("content") or "")

                    # Rung 1: Compiled 0-token micro-router or 256B LoRA-XS cartridge check!
                    pred = manager.boil_engine.predict_micro_router(ns, prompt_txt)
                    if pred.get("ok"):
                        manager.zero_token_hits += 1
                        backend_name = str(pred["backend"])
                        rung_name = (
                            "rung_1_lora_xs_cartridge"
                            if backend_name == "lora_xs_subspace_cartridge"
                            else "rung_1_compiled_micro_router"
                        )
                        self._send_json(
                            ok_code,
                            {
                                "id": f"boildown-{manager.requests_served}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": str(data.get("model") or f"boildown:{ns}"),
                                "rung": rung_name,
                                "backend": backend_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "finish_reason": "stop",
                                        "message": {
                                            "role": "assistant",
                                            "content": json.dumps(
                                                {
                                                    "route": pred["predicted_label"],
                                                    "boil_down": True,
                                                    "backend": backend_name,
                                                }
                                            ),
                                        },
                                    }
                                ],
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            },
                        )
                        return

                    # Rung 2: Optional local model runner forward if model requested & runner explicitly enabled
                    if data.get("use_local_runner") and manager.local_runner_url and manager.allow_local_runner:
                        runner_payload = {
                            "model": str(data.get("model") or ""),
                            "messages": msgs,
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": 0.0,
                                "num_ctx": int(manager.grid.runtime_constants["default_runner_ctx_tokens"]),
                            },
                        }
                        runner_res = _send_vetted_http(
                            manager.local_runner_url,
                            "/api/chat",
                            method="POST",
                            body_bytes=json.dumps(runner_payload).encode("utf-8"),
                            headers={"Content-Type": "application/json", "Accept": "application/json"},
                            timeout_s=float(manager.grid.runtime_constants["upstream_http_timeout_sec"]),
                            mode="local_runner",
                            allow_local=manager.allow_local_runner,
                            grid=manager.grid,
                        )
                        if runner_res.get("ok") and isinstance(runner_res.get("data"), Mapping):
                            out_json = runner_res["data"]
                            content_str = str((out_json.get("message") or {}).get("content") or "{}")
                            eval_toks = int(out_json.get("eval_count") or 0) + int(out_json.get("prompt_eval_count") or 0)
                            if data.get("trace_label"):
                                manager.store.record_execution_trace(ns, prompt_txt, str(data["trace_label"]), tokens_used=eval_toks)
                            self._send_json(
                                ok_code,
                                {
                                    "id": f"runner-{manager.requests_served}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": str(data.get("model") or "local-runner"),
                                    "rung": "rung_2_constrained_local_runner",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": content_str},
                                        }
                                    ],
                                    "usage": {"prompt_tokens": 0, "completion_tokens": eval_toks, "total_tokens": eval_toks},
                                },
                            )
                            return

                    # Record trace during Wave-1 seeding
                    label = str(data.get("trace_label") or "general_execute")
                    sim_tokens = int(
                        data.get("simulated_wave1_tokens")
                        or (
                            manager.grid.runtime_constants["structured_output_token_floor"]
                            if "structured_output_token_floor" in manager.grid.runtime_constants
                            else max(1, len(prompt_txt))
                        )
                    )
                    manager.store.record_execution_trace(ns, prompt_txt, label, tokens_used=sim_tokens)
                    self._send_json(
                        ok_code,
                        {
                            "id": f"wave1-{manager.requests_served}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": str(data.get("model") or "omni-fabric-auto"),
                            "rung": "rung_3_trace_harvest",
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": json.dumps({"route": label, "boil_down": False})},
                                }
                            ],
                            "usage": {"prompt_tokens": 0, "completion_tokens": sim_tokens, "total_tokens": sim_tokens},
                        },
                    )
                    return

                self._send_json(
                    int(manager.grid.runtime_constants["http_status_bad_request"]),
                    {"ok": False, "error": "unknown_route"},
                )

        self._server = ThreadingHTTPServer((host, int(port)), _SwarmHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        addr = self._server.server_address
        return {
            "ok": True,
            "running": True,
            "listen_url": f"http://{addr[0]}:{addr[1]}",
            "openai_base_url": f"http://{addr[0]}:{addr[1]}/v1",
        }

    def stop(self) -> Dict[str, Any]:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
        return {
            "ok": True,
            "running": False,
            "requests_served": self.requests_served,
            "zero_token_hits": self.zero_token_hits,
        }

    def status(self) -> Dict[str, Any]:
        if self._server is None:
            return {
                "ok": True,
                "running": False,
                "requests_served": self.requests_served,
                "zero_token_hits": self.zero_token_hits,
            }
        addr = self._server.server_address
        return {
            "ok": True,
            "running": True,
            "listen_url": f"http://{addr[0]}:{addr[1]}",
            "openai_base_url": f"http://{addr[0]}:{addr[1]}/v1",
            "requests_served": self.requests_served,
            "zero_token_hits": self.zero_token_hits,
        }


class FederatedFabricBridge:
    """Selective sync & thin-client bridge to owner-configured upstream Fabric endpoint."""

    def __init__(
        self,
        grid: AdaptiveNodeGrid,
        store: LocalAdaptiveFabricStore,
        data_dir: Optional[Path],
        allow_local_upstream: bool = False,
    ) -> None:
        self.grid = grid
        self.store = store
        self.data_dir = data_dir
        self.allow_local_upstream = bool(allow_local_upstream)

    def _token_path(self, base_url: str) -> Optional[Path]:
        if self.data_dir is None:
            return None
        origin = urllib.parse.urlparse(base_url).netloc.replace(":", "_") or "default"
        return self.data_dir / f".fabric_worker_token_{origin}.json"

    def _load_worker_token(self, base_url: str) -> Optional[str]:
        tpath = self._token_path(base_url)
        if tpath is None or not tpath.exists():
            return None
        try:
            obj = json.loads(tpath.read_text(encoding="utf-8"))
            tok = obj.get("worker_token")
            return str(tok) if tok else None
        except Exception:
            return None

    def _save_worker_token(self, base_url: str, w_tok: str) -> None:
        tpath = self._token_path(base_url)
        if tpath is None:
            return
        try:
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.write_text(json.dumps({"worker_token": w_tok}), encoding="utf-8")
            tpath.chmod(int(self.grid.runtime_constants["octal_owner_rw_mode"]))
        except Exception:
            return

    def _http_json(
        self,
        base_url: str,
        route: str,
        method: str = "GET",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_base = base_url.strip().rstrip("/")
        if not clean_base:
            return {"ok": False, "error": "upstream_endpoint_not_configured"}
        headers: Dict[str, str] = {"Accept": "application/json"}
        tok = self._load_worker_token(clean_base)
        if tok:
            headers["X-Worker-Token"] = tok
        body_bytes = None
        if payload is not None:
            body_bytes = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"

        timeout_s = float(self.grid.runtime_constants["upstream_http_timeout_sec"])
        res = _send_vetted_http(
            clean_base,
            route,
            method=method,
            body_bytes=body_bytes,
            headers=headers,
            timeout_s=timeout_s,
            mode="upstream",
            allow_local=self.allow_local_upstream,
            grid=self.grid,
        )
        if res.get("ok"):
            parsed = res.get("data")
            if isinstance(parsed, dict) and parsed.get("worker_token"):
                self._save_worker_token(clean_base, str(parsed["worker_token"]))
        return res

    def check_upstream_pulse(self, base_url: str) -> Dict[str, Any]:
        return self._http_json(base_url, "/q/pulse", method="GET")

    def pull_unseen_frontier(self, base_url: str, worker_id: str, query: str = "") -> Dict[str, Any]:
        qs = urllib.parse.urlencode({"worker_id": worker_id, "unseen_only": "1", "q": query})
        res = self._http_json(base_url, f"/q/next?{qs}", method="GET")
        if not res.get("ok"):
            return res
        data = res.get("data") or {}
        items = data.get("tasks") or data.get("items") or ([] if not isinstance(data, list) else data)
        imported = 0
        if isinstance(items, list):
            for idx, item in enumerate(items):
                if isinstance(item, Mapping):
                    tid = str(item.get("id") or item.get("task_id") or f"remote_{idx + 1}")
                    self.store.enqueue_task(
                        task_id=f"sync:{tid}",
                        namespace=str(item.get("namespace") or item.get("domain") or "federated"),
                        operation="remote_pulled",
                        payload=dict(item),
                        visibility="sync_ready",
                    )
                    imported += 1
        return {"ok": True, "imported_tasks": imported, "upstream_response": data}

    def push_sync_ready_objects(self, base_url: str) -> Dict[str, Any]:
        ready = self.store.list_sync_ready_objects()
        pushed = 0
        responses: List[Dict[str, Any]] = []
        for obj in ready:
            res = self._http_json(
                base_url,
                "/q/submit",
                method="POST",
                payload={
                    "id": obj["object_id"],
                    "title": obj["title"],
                    "summary": obj["summary_text"],
                    "maker_id": obj["maker_id"],
                    "checker_id": obj["checker_id"],
                    "artifact": obj["payload"],
                },
            )
            responses.append(res)
            if res.get("ok"):
                pushed += 1
        return {"ok": True, "sync_ready_candidates": len(ready), "pushed_count": pushed, "responses": responses}

    def remote_mcp_list_or_call(
        self,
        base_url: str,
        tool_name: Optional[str] = None,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not tool_name:
            rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        else:
            rpc = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": dict(arguments or {})},
            }
        return self._http_json(base_url, "/mcp", method="POST", payload=rpc)
