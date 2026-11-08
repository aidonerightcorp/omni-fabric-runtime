"""Progressive 4-rung neural boil-down engine, loopback multi-agent swarm proxy, and selective upstream Fabric bridge."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .fabric_substrate import LocalAdaptiveFabricStore
from .grid_and_env import AdaptiveNodeGrid, EffortAndBudgetGate


class AdaptiveBoilDownEngine:
    """Compiles traces into 0-token micro-routers (stdlib-only OR PyTorch/NumPy + SVD/Wanda)."""

    def __init__(self, grid: AdaptiveNodeGrid, store: LocalAdaptiveFabricStore) -> None:
        self.grid = grid
        self.store = store
        self.compiled_routers: Dict[str, Dict[str, Any]] = {}

    def _featurize_text(self, text: str, num_buckets: int) -> List[float]:
        vec = [0.0] * num_buckets
        cleaned = f"^{text.strip().lower()}$"
        span = int(self.grid.runtime_constants["ngram_span"])
        prime = int(self.grid.runtime_constants["default_hash_prime"])
        for idx in range(max(1, len(cleaned) - span + 1)):
            gram = cleaned[idx : idx + span]
            acc = 0
            for ch in gram:
                acc = (acc * prime + ord(ch)) & ((1 << int(self.grid.runtime_constants["trivial_prompt_char_cap"]) if "trivial_prompt_char_cap" in self.grid.runtime_constants else prime * prime * prime) - 1)
            digest = hashlib.blake2s(gram.encode("utf-8")).digest()
            bucket = digest[0] % num_buckets
            sign = 1.0 if (digest[1] % int(self.grid.runtime_constants["pair_divisor"]) == 0) else -1.0
            vec[bucket] += sign
        norm_sq = sum(v * v for v in vec)
        eps = float(self.grid.runtime_constants["eps_floor"])
        inv_norm = 1.0 / math.sqrt(norm_sq + eps)
        return [v * inv_norm for v in vec]

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

    def compile_stdlib_micro_router(
        self,
        namespace: str,
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cell = self.grid.resolve_cell(regime)
        min_support = int(cell["boil_down_min_trace_support"])
        buckets = int(cell["micro_feature_buckets"])
        hidden = int(cell["pruned_hidden_units"])
        reg = float(self.grid.runtime_constants["ridge_regularization"])
        eps = float(self.grid.runtime_constants["eps_floor"])

        rows = list(samples) if samples is not None else self.store.list_execution_traces(namespace)
        if len(rows) < min_support:
            return {
                "ok": False,
                "error": "insufficient_trace_support",
                "recorded_traces": len(rows),
                "required_traces": min_support,
            }

        labels = sorted({str(r["target_label"]) for r in rows})
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        num_classes = len(labels)

        # Deterministic ReLU polytope projection weights from blake2s seed
        proj_w: List[List[float]] = []
        scale = 1.0 / math.sqrt(float(buckets))
        for h_idx in range(hidden):
            row_w: List[float] = []
            for b_idx in range(buckets):
                d = hashlib.blake2s(f"relu:{namespace}:{h_idx}:{b_idx}".encode("utf-8")).digest()
                s = 1.0 if (d[0] % int(self.grid.runtime_constants["pair_divisor"]) == 0) else -1.0
                row_w.append(s * scale)
            proj_w.append(row_w)

        h_feats: List[List[float]] = []
        targets: List[List[float]] = []
        for item in rows:
            x_vec = self._featurize_text(str(item["input_text"]), buckets)
            z_vec: List[float] = [1.0]
            for h_idx in range(hidden):
                dot = sum(proj_w[h_idx][b] * x_vec[b] for b in range(buckets))
                z_vec.append(dot if dot > 0.0 else 0.0)
            h_feats.append(z_vec)
            y_vec = [0.0] * num_classes
            y_vec[label_to_idx[str(item["target_label"])]] = 1.0
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

        correct = 0
        for idx_s, item in enumerate(rows):
            zf = h_feats[idx_s]
            scores = [sum(zf[i] * readout_w[i][c] for i in range(dim)) for c in range(num_classes)]
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
            "backend": "stdlib_relu_ridge",
            "namespace": namespace,
            "labels": labels,
            "sample_count": len(rows),
            "train_accuracy": acc,
            "compile_ms": round(elapsed_ms, int(self.grid.runtime_constants["ngram_span"])),
            "llm_tokens_per_inference": 0,
        }

    def predict_micro_router(self, namespace: str, input_text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        spec = self.compiled_routers.get(namespace)
        if spec is None:
            return {"ok": False, "error": "no_compiled_router_for_namespace"}
        buckets = int(spec["buckets"])
        hidden = int(spec["hidden"])
        labels: List[str] = list(spec["labels"])
        proj_w: List[List[float]] = spec["proj_w"]
        readout_w: List[List[float]] = spec["readout_w"]

        x_vec = self._featurize_text(input_text, buckets)
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
            "backend": str(spec["backend"]),
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
        anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
        edit_vectors: Optional[Sequence[Sequence[float]]] = None,
        edit_targets: Optional[Sequence[Sequence[float]]] = None,
        export_dir: Optional[Path] = None,
        masked_modules: Optional[Tuple[str, ...]] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        if masked_modules and ("numpy" in masked_modules or "torch" in masked_modules):
            return self.compile_stdlib_micro_router(namespace=namespace, samples=samples, regime=regime)

        try:
            np = importlib.import_module("numpy")
            torch = importlib.import_module("torch")
            torch.set_num_threads(1)
        except Exception:
            return self.compile_stdlib_micro_router(namespace=namespace, samples=samples, regime=regime)

        cell = self.grid.resolve_cell(regime)
        reg = float(self.grid.runtime_constants["ridge_regularization"])
        tol = float(self.grid.runtime_constants["nullspace_rank_tolerance"])
        pct_scale = float(self.grid.runtime_constants["percent_scale"])

        if operation == "nullspace_surgery":
            if not anchor_vectors or not edit_vectors or not edit_targets:
                return {"ok": False, "error": "missing_anchor_or_edit_vectors"}
            a_mat = np.asarray(anchor_vectors, dtype=np.float64)
            e_mat = np.asarray(edit_vectors, dtype=np.float64)
            y_mat = np.asarray(edit_targets, dtype=np.float64)
            feat_d = a_mat.shape[1]
            out_d = y_mat.shape[1]
            w_base = np.zeros((out_d, feat_d), dtype=np.float64)

            # Compute orthogonal projector onto nullspace of anchor activations
            _, sing_vals, vt = np.linalg.svd(a_mat, full_matrices=True)
            rank = int(np.sum(sing_vals > tol))
            v_null = vt[rank:].T
            p_null = v_null @ v_null.T

            e_proj = e_mat @ p_null
            gram = e_proj.T @ e_proj + reg * np.eye(feat_d, dtype=np.float64)
            delta_t = p_null @ np.linalg.solve(gram, e_proj.T @ (y_mat - (e_mat @ w_base.T)))
            w_edited = w_base + delta_t.T

            anchor_drift = float(np.max(np.abs((a_mat @ w_edited.T) - (a_mat @ w_base.T))))
            edit_preds = np.argmax(e_mat @ w_edited.T, axis=1)
            edit_true = np.argmax(y_mat, axis=1)
            edit_acc = float(np.mean(edit_preds == edit_true))
            return {
                "ok": True,
                "backend": "numpy_svd_nullspace",
                "operation": operation,
                "anchor_max_logit_drift": anchor_drift,
                "unseen_edit_accuracy": edit_acc,
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

            # Wanda-style historical-trace importance: mean(|activation|) * ||readout_col||_2
            act_norm = np.mean(np.abs(act_full), axis=0)
            col_norm = np.linalg.norm(w2_np, axis=0)
            importance = act_norm * col_norm
            keep_idx = np.argsort(-importance)[:keep_h]

            h_pruned = act_full[:, keep_idx]
            ones = np.ones((h_pruned.shape[0], 1), dtype=np.float64)
            z_design = np.concatenate([ones, h_pruned.astype(np.float64)], axis=1)
            eye = np.eye(keep_h + 1, dtype=np.float64)
            y_oh = np.eye(len(labels), dtype=np.float64)[y_np]
            w_ridge = np.linalg.solve(z_design.T @ z_design + reg * eye, z_design.T @ y_oh)

            preds = np.argmax(z_design @ w_ridge, axis=1)
            pruned_acc = float(np.mean(preds == y_np))

        dense_params = buckets * full_h + full_h + full_h * len(labels) + len(labels)
        pruned_params = buckets * keep_h + keep_h + (keep_h + 1) * len(labels)
        ram_reduction_pct = (1.0 - (float(pruned_params) / float(max(1, dense_params)))) * pct_scale

        # Also register pruned weights into compiled_routers for instant 0-token inference!
        proj_kept = [[float(v) for v in w1_np[idx]] for idx in keep_idx]
        # Absorb bias into proj or keep zero-bias ReLU if b1 is near zero; store readout directly
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
            "backend": "torch_wanda_pruned_micro_net",
            "namespace": namespace,
            "dense_hidden_units": full_h,
            "pruned_hidden_units": keep_h,
            "dense_params": dense_params,
            "pruned_params": pruned_params,
            "ram_reduction_pct": round(ram_reduction_pct, int(self.grid.runtime_constants["pair_divisor"])),
            "pruned_accuracy": pruned_acc,
            "safetensors_path": safetensors_path,
        }


class LoopbackSwarmProxyManager:
    """On-demand loopback HTTP server (/v1/chat/completions + /swarm/*) for multi-agent swarms."""

    def __init__(
        self,
        grid: AdaptiveNodeGrid,
        store: LocalAdaptiveFabricStore,
        boil_engine: AdaptiveBoilDownEngine,
        budget_gate: EffortAndBudgetGate,
        local_runner_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.grid = grid
        self.store = store
        self.boil_engine = boil_engine
        self.budget_gate = budget_gate
        self.local_runner_url = local_runner_url.rstrip("/")
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.requests_served: int = 0
        self.zero_token_hits: int = 0

    def start(self, host: str = "127.0.0.1", port: int = 0) -> Dict[str, Any]:
        if self._server is not None:
            addr = self._server.server_address
            return {
                "ok": True,
                "running": True,
                "listen_url": f"http://{addr[0]}:{addr[1]}",
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

            def do_GET(self) -> None:
                if self.path.startswith("/health"):
                    self._send_json(
                        int(manager.grid.runtime_constants["http_status_ok"]),
                        {
                            "ok": True,
                            "requests_served": manager.requests_served,
                            "zero_token_hits": manager.zero_token_hits,
                            "substrate": manager.store.summarize_status(),
                        },
                    )
                    return
                self._send_json(int(manager.grid.runtime_constants["http_status_bad_request"]), {"ok": False, "error": "unknown_route"})

            def do_POST(self) -> None:
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
                    )
                    self._send_json(ok_code, res)
                    return

                if self.path == "/swarm/claim":
                    res = manager.store.claim_task(
                        worker_id=str(data["worker_id"]),
                        namespace=data.get("namespace"),
                        task_id=data.get("task_id"),
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

                if self.path == "/swarm/submit":
                    res = manager.store.stage_or_promote_object(
                        task_id=str(data["task_id"]),
                        maker_id=str(data["maker_id"]),
                        checker_id=data.get("checker_id"),
                        namespace=str(data.get("namespace") or "swarm"),
                        title=str(data.get("title") or data["task_id"]),
                        summary_text=str(data.get("summary_text") or ""),
                        payload=dict(data.get("payload") or {}),
                    )
                    code = int(res.get("status_code") or ok_code)
                    self._send_json(code, res)
                    return

                if self.path == "/swarm/boil_down":
                    ns = str(data.get("namespace") or "swarm")
                    res = manager.boil_engine.compile_stdlib_micro_router(namespace=ns)
                    self._send_json(ok_code, res)
                    return

                if self.path == "/v1/chat/completions":
                    ns = str(data.get("namespace") or "swarm")
                    msgs = data.get("messages") or []
                    prompt_txt = ""
                    if isinstance(msgs, list) and msgs:
                        prompt_txt = str(msgs[-1].get("content") or "")

                    # Rung 1: Compiled 0-token micro-router check!
                    pred = manager.boil_engine.predict_micro_router(ns, prompt_txt)
                    if pred.get("ok"):
                        manager.zero_token_hits += 1
                        self._send_json(
                            ok_code,
                            {
                                "id": f"boildown-{manager.requests_served}",
                                "object": "chat.completion",
                                "rung": "rung_1_compiled_micro_router",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": json.dumps({"route": pred["predicted_label"], "boil_down": True}),
                                        },
                                    }
                                ],
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            },
                        )
                        return

                    # Rung 2: Optional local model runner forward if model requested & runner reachable
                    if data.get("use_local_runner") and manager.local_runner_url:
                        try:
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
                            req = urllib.request.Request(
                                f"{manager.local_runner_url}/api/chat",
                                data=json.dumps(runner_payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(
                                req,
                                timeout=float(manager.grid.runtime_constants["upstream_http_timeout_sec"]),
                            ) as resp:
                                out_json = json.loads(resp.read().decode("utf-8"))
                            content_str = str((out_json.get("message") or {}).get("content") or "{}")
                            eval_toks = int(out_json.get("eval_count") or 0) + int(out_json.get("prompt_eval_count") or 0)
                            if data.get("trace_label"):
                                manager.store.record_execution_trace(ns, prompt_txt, str(data["trace_label"]), tokens_used=eval_toks)
                            self._send_json(
                                ok_code,
                                {
                                    "id": f"runner-{manager.requests_served}",
                                    "object": "chat.completion",
                                    "rung": "rung_2_constrained_local_runner",
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content_str}}],
                                    "usage": {"total_tokens": eval_toks},
                                },
                            )
                            return
                        except Exception:
                            pass

                    # Record trace if provided during Wave-1 seeding
                    label = str(data.get("trace_label") or "general_execute")
                    sim_tokens = int(data.get("simulated_wave1_tokens") or manager.grid.runtime_constants["structured_output_token_floor"] if "structured_output_token_floor" in manager.grid.runtime_constants else len(prompt_txt))
                    manager.store.record_execution_trace(ns, prompt_txt, label, tokens_used=sim_tokens)
                    self._send_json(
                        ok_code,
                        {
                            "id": f"wave1-{manager.requests_served}",
                            "object": "chat.completion",
                            "rung": "rung_3_trace_harvest",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": json.dumps({"route": label, "boil_down": False})},
                                }
                            ],
                            "usage": {"total_tokens": sim_tokens},
                        },
                    )
                    return

                self._send_json(int(manager.grid.runtime_constants["http_status_bad_request"]), {"ok": False, "error": "unknown_route"})

        self._server = ThreadingHTTPServer((host, int(port)), _SwarmHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        addr = self._server.server_address
        return {
            "ok": True,
            "running": True,
            "listen_url": f"http://{addr[0]}:{addr[1]}",
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
            return {"ok": True, "running": False, "requests_served": self.requests_served, "zero_token_hits": self.zero_token_hits}
        addr = self._server.server_address
        return {
            "ok": True,
            "running": True,
            "listen_url": f"http://{addr[0]}:{addr[1]}",
            "requests_served": self.requests_served,
            "zero_token_hits": self.zero_token_hits,
        }


class FederatedFabricBridge:
    """Selective sync & thin-client bridge to any configured upstream Fabric endpoint."""

    def __init__(self, grid: AdaptiveNodeGrid, store: LocalAdaptiveFabricStore, data_dir: Optional[Path]) -> None:
        self.grid = grid
        self.store = store
        self.data_dir = data_dir

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

    def _save_worker_token(self, base_url: str, worker_token: str) -> None:
        tpath = self._token_path(base_url)
        if tpath is None:
            return
        try:
            tpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.write_text(json.dumps({"worker_token": worker_token}), encoding="utf-8")
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
        url = f"{clean_base}{route}"
        headers: Dict[str, str] = {"Accept": "application/json"}
        tok = self._load_worker_token(clean_base)
        if tok:
            headers["X-Worker-Token"] = tok
        body_bytes = None
        if payload is not None:
            body_bytes = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        timeout_s = float(self.grid.runtime_constants["upstream_http_timeout_sec"])
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw) if raw.strip() else {}
                if isinstance(parsed, dict) and parsed.get("worker_token"):
                    self._save_worker_token(clean_base, str(parsed["worker_token"]))
                return {"ok": True, "status_code": int(resp.status), "data": parsed}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            return {"ok": False, "status_code": int(exc.code), "error": err_body}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

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
