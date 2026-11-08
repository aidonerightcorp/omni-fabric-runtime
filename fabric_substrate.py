"""SQLite WAL durable substrate: fenced CAS queue, affinity bundle dispatch, enriched trace envelopes, bitemporal maker!=checker store, FTS5 RRF search, attenuation-gated Macro-DAG compiler, and NodeGrid + RL canary optimizer."""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .grid_and_env import AdaptiveNodeGrid


ACTION_PHASES: Tuple[str, ...] = (
    "acquire",
    "analyze",
    "transform",
    "coordinate",
    "check",
    "publish",
)

DISPATCH_AFFINITY_MODES: Tuple[str, ...] = (
    "single_isolated",
    "back_to_back_sequential",
    "simultaneous_batched_pack",
    "speculative_pipeline",
)

REASONING_MODES: Tuple[str, ...] = (
    "off",
    "minimal",
    "adaptive",
    "deep",
)


class LocalAdaptiveFabricStore:
    """Unified local SQLite WAL/memory substrate governed by AdaptiveNodeGrid."""

    def __init__(
        self,
        grid: AdaptiveNodeGrid,
        db_path: Optional[Path] = None,
        ephemeral_memory: bool = False,
    ) -> None:
        self.grid = grid
        self.ephemeral_memory = ephemeral_memory or (db_path is None)
        self.db_path = db_path
        self._lock = threading.RLock()
        self._secret = self._load_or_create_secret(db_path)
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._fts5_enabled = False
        self._init_schema()

    def _load_or_create_secret(self, db_path: Optional[Path]) -> bytes:
        nbytes = int(self.grid.runtime_constants["fence_digest_char_len"])
        if db_path is None or self.ephemeral_memory:
            return secrets.token_bytes(nbytes)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            key_file = db_path.parent / ".omni_fabric_fence.key"
            if key_file.exists():
                return key_file.read_bytes()
            raw = secrets.token_bytes(nbytes)
            key_file.write_bytes(raw)
            key_file.chmod(int(self.grid.runtime_constants["octal_owner_rw_mode"]))
            return raw
        except Exception:
            return secrets.token_bytes(nbytes)

    def _connect(self) -> sqlite3.Connection:
        if self.ephemeral_memory or self.db_path is None:
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        timeout_s = (
            float(self.grid.runtime_constants["sqlite_busy_timeout_ms"])
            / float(self.grid.runtime_constants["ms_per_second"])
        )
        conn = sqlite3.connect(str(self.db_path), timeout=timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(self.grid.runtime_constants['sqlite_busy_timeout_ms'])}")
        return conn

    def _close_if_disk(self, conn: sqlite3.Connection) -> None:
        if not self.ephemeral_memory and self.db_path is not None:
            conn.close()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_ddl: str) -> None:
        existing = {
            str(r["name"])
            for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_ddl}")

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS task_leases (
                        task_id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        task_key TEXT NOT NULL,
                        contract_tag TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        depends_on_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'queued',
                        lease_owner TEXT,
                        lease_epoch INTEGER NOT NULL DEFAULT 0,
                        fence_token TEXT,
                        lease_expires_at REAL NOT NULL DEFAULT 0.0,
                        checkpoint_json TEXT NOT NULL DEFAULT '{}',
                        visibility TEXT NOT NULL DEFAULT 'local_only',
                        prefix_signature TEXT NOT NULL DEFAULT '',
                        capability_tag TEXT NOT NULL DEFAULT 'general',
                        tool_schema_tag TEXT NOT NULL DEFAULT 'default_schema',
                        origin_id TEXT NOT NULL DEFAULT '',
                        dag_depth INTEGER NOT NULL DEFAULT 0,
                        dag_width INTEGER NOT NULL DEFAULT 1,
                        granularity_tier TEXT NOT NULL DEFAULT 'atomic',
                        step_confidence_milli INTEGER NOT NULL DEFAULT 1000,
                        cumulative_attenuation_survival REAL NOT NULL DEFAULT 1.0,
                        attenuation_halted INTEGER NOT NULL DEFAULT 0,
                        checked_reset INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS task_objects (
                        object_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        task_key TEXT NOT NULL,
                        contract_tag TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        maker_id TEXT NOT NULL,
                        checker_id TEXT,
                        visibility TEXT NOT NULL DEFAULT 'local_only',
                        title TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        valid_from REAL NOT NULL,
                        valid_to REAL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS seen_novelty_keys (
                        agent_scope TEXT NOT NULL,
                        object_id TEXT NOT NULL,
                        seen_at REAL NOT NULL,
                        PRIMARY KEY (agent_scope, object_id)
                    );
                    CREATE TABLE IF NOT EXISTS grid_campaigns (
                        campaign_id TEXT PRIMARY KEY,
                        generation INTEGER NOT NULL,
                        axes_json TEXT NOT NULL,
                        constraints_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS grid_ledger (
                        campaign_id TEXT NOT NULL,
                        cell_key TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        lifecycle_state TEXT NOT NULL,
                        evaluations_json TEXT NOT NULL,
                        best_reward REAL NOT NULL DEFAULT 0.0,
                        mean_cost REAL NOT NULL DEFAULT 0.0,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (campaign_id, cell_key)
                    );
                    CREATE TABLE IF NOT EXISTS rl_canary_state (
                        policy_id TEXT PRIMARY KEY,
                        baseline_arm TEXT NOT NULL,
                        canary_arm TEXT NOT NULL,
                        baseline_sum REAL NOT NULL DEFAULT 0.0,
                        baseline_n INTEGER NOT NULL DEFAULT 0,
                        canary_sum REAL NOT NULL DEFAULT 0.0,
                        canary_n INTEGER NOT NULL DEFAULT 0,
                        promotions INTEGER NOT NULL DEFAULT 0,
                        rollbacks INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS execution_traces (
                        trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        target_label TEXT NOT NULL,
                        tokens_used INTEGER NOT NULL DEFAULT 0,
                        latency_ms REAL NOT NULL DEFAULT 0.0,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS trace_envelopes (
                        envelope_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL DEFAULT '',
                        object_id TEXT NOT NULL DEFAULT '',
                        namespace TEXT NOT NULL,
                        worker_id TEXT NOT NULL DEFAULT '',
                        input_text TEXT NOT NULL DEFAULT '',
                        target_label TEXT NOT NULL DEFAULT '',
                        prefix_signature TEXT NOT NULL DEFAULT '',
                        reasoning_mode TEXT NOT NULL DEFAULT 'adaptive',
                        thinking_tokens INTEGER NOT NULL DEFAULT 0,
                        answer_tokens INTEGER NOT NULL DEFAULT 0,
                        tokens_used INTEGER NOT NULL DEFAULT 0,
                        latency_ms REAL NOT NULL DEFAULT 0.0,
                        activation_norm_digest TEXT NOT NULL DEFAULT '',
                        attenuation_confidence_milli INTEGER NOT NULL DEFAULT 1000,
                        seed_id INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS subagent_workers (
                        worker_id TEXT PRIMARY KEY,
                        role TEXT NOT NULL,
                        status TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """
                )
                for col_name, col_ddl in (
                    ("prefix_signature", "TEXT NOT NULL DEFAULT ''"),
                    ("capability_tag", "TEXT NOT NULL DEFAULT 'general'"),
                    ("tool_schema_tag", "TEXT NOT NULL DEFAULT 'default_schema'"),
                    ("origin_id", "TEXT NOT NULL DEFAULT ''"),
                    ("dag_depth", "INTEGER NOT NULL DEFAULT 0"),
                    ("dag_width", "INTEGER NOT NULL DEFAULT 1"),
                    ("granularity_tier", "TEXT NOT NULL DEFAULT 'atomic'"),
                    ("step_confidence_milli", "INTEGER NOT NULL DEFAULT 1000"),
                    ("cumulative_attenuation_survival", "REAL NOT NULL DEFAULT 1.0"),
                    ("attenuation_halted", "INTEGER NOT NULL DEFAULT 0"),
                    ("checked_reset", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    self._ensure_column(conn, "task_leases", col_name, col_ddl)
                try:
                    conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS task_objects_fts USING fts5(object_id UNINDEXED, title, summary_text, namespace)"
                    )
                    self._fts5_enabled = True
                except Exception:
                    self._fts5_enabled = False
            finally:
                self._close_if_disk(conn)

    def _compute_task_key(
        self,
        namespace: str,
        operation: str,
        contract_tag: str,
        payload: Mapping[str, Any],
    ) -> str:
        canonical = json.dumps(
            {"ns": namespace, "op": operation, "ct": contract_tag, "p": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        char_len = int(self.grid.runtime_constants["task_key_char_len"])
        return hashlib.blake2s(canonical).hexdigest()[:char_len]

    def _compute_prefix_signature(
        self,
        namespace: str,
        capability_tag: str,
        prefix_hint: str = "",
    ) -> str:
        char_len = int(self.grid.runtime_constants["task_key_char_len"])
        raw = f"{namespace}:{capability_tag}:{prefix_hint}".encode("utf-8")
        return f"pfx_{hashlib.blake2s(raw).hexdigest()[:char_len]}"

    def _compute_fence_token(self, task_id: str, worker_id: str, epoch: int) -> str:
        msg = f"{task_id}:{worker_id}:{epoch}".encode("utf-8")
        char_len = int(self.grid.runtime_constants["fence_digest_char_len"])
        return hmac.new(self._secret, msg, hashlib.blake2s).hexdigest()[:char_len]

    @staticmethod
    def _normalize_reasoning_mode(raw_mode: Optional[str]) -> str:
        cleaned = str(raw_mode or "adaptive").strip().lower()
        if cleaned in ("none", "direct", "disabled"):
            return "off"
        if cleaned in REASONING_MODES:
            return cleaned
        return "adaptive"

    def _recompute_downstream_attenuation(
        self,
        conn: sqlite3.Connection,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        cell = self.grid.resolve_cell(regime)
        full_milli = float(self.grid.runtime_constants["full_confidence_milli"])
        cutoff = float(cell["attenuation_circuit_breaker_milli"]) / full_milli
        round_digits = int(self.grid.runtime_constants["attenuation_round_digits"])

        rows = conn.execute(
            "SELECT task_id, depends_on_json, step_confidence_milli, cumulative_attenuation_survival, attenuation_halted, checked_reset FROM task_leases ORDER BY dag_depth ASC, created_at ASC"
        ).fetchall()
        if not rows:
            return {"halted_tasks": 0, "active_tasks": 0, "cutoff": cutoff}

        state_map: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            tid = str(r["task_id"])
            state_map[tid] = {
                "deps": json.loads(str(r["depends_on_json"] or "[]")),
                "step_milli": int(r["step_confidence_milli"]),
                "survival": float(r["cumulative_attenuation_survival"]),
                "halted": int(r["attenuation_halted"]),
                "checked": int(r["checked_reset"]),
            }

        for _ in range(max(1, len(rows))):
            changed = False
            for tid, item in state_map.items():
                if item["checked"] == 1:
                    new_surv = 1.0
                    new_halt = 0
                else:
                    parent_survs: List[float] = []
                    for dep_id in item["deps"]:
                        dep_str = str(dep_id)
                        if dep_str in state_map:
                            p_item = state_map[dep_str]
                            parent_survs.append(1.0 if p_item["checked"] == 1 else float(p_item["survival"]))
                    incoming = min(parent_survs) if parent_survs else 1.0
                    step_factor = max(0.0, min(1.0, float(item["step_milli"]) / full_milli))
                    new_surv = round(incoming * step_factor, round_digits)
                    new_halt = 1 if new_surv < cutoff else 0

                if abs(new_surv - float(item["survival"])) > float(self.grid.runtime_constants["eps_floor"]) or new_halt != int(item["halted"]):
                    item["survival"] = new_surv
                    item["halted"] = new_halt
                    changed = True
            if not changed:
                break

        now = time.time()
        halted_count = 0
        for tid, item in state_map.items():
            if int(item["halted"]) == 1:
                halted_count += 1
            conn.execute(
                """
                UPDATE task_leases
                SET cumulative_attenuation_survival = ?,
                    attenuation_halted = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (float(item["survival"]), int(item["halted"]), now, tid),
            )
        return {
            "halted_tasks": halted_count,
            "active_tasks": len(state_map) - halted_count,
            "cutoff": cutoff,
        }

    def _build_resume_packet(
        self,
        task_id: str,
        namespace: str,
        operation: str,
        epoch: int,
        checkpoint: Mapping[str, Any],
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        cell = self.grid.resolve_cell(regime)
        cap = int(cell["resume_packet_byte_cap"])
        packet: Dict[str, Any] = {
            "task_id": task_id,
            "ns": namespace,
            "op": operation,
            "epoch": epoch,
            "ckpt": dict(checkpoint),
        }
        encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= cap:
            return packet
        compact_note = json.dumps(checkpoint, sort_keys=True)[
            : max(1, cap // int(self.grid.runtime_constants["pair_divisor"]))
        ]
        return {
            "task_id": task_id,
            "ns": namespace,
            "op": operation,
            "epoch": epoch,
            "ckpt_head": compact_note,
        }

    # ── 1. Monotonic CAS Fenced Task Queue + Affinity Bundle Claiming ─────

    def enqueue_task(
        self,
        task_id: str,
        namespace: str,
        operation: str,
        payload: Mapping[str, Any],
        contract_tag: str = "v1",
        depends_on: Optional[Sequence[str]] = None,
        visibility: str = "local_only",
        prefix_signature: Optional[str] = None,
        capability_tag: Optional[str] = None,
        tool_schema_tag: Optional[str] = None,
        origin_id: Optional[str] = None,
        dag_depth: Optional[int] = None,
        dag_width: Optional[int] = None,
        granularity_tier: Optional[str] = None,
        step_confidence_milli: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        deps = [str(d) for d in (depends_on or []) if str(d).strip()]
        tkey = self._compute_task_key(namespace, operation, contract_tag, payload)
        cap_tag = str(capability_tag or payload.get("capability_tag") or operation or "general")
        schema_tag = str(tool_schema_tag or payload.get("tool_schema_tag") or f"{contract_tag}:{operation}")
        pfx_sig = str(
            prefix_signature
            or payload.get("prefix_signature")
            or self._compute_prefix_signature(namespace, cap_tag, str(payload.get("prefix_hint") or ""))
        )
        orig_id = str(origin_id or payload.get("origin_id") or task_id)
        gran_tier = str(granularity_tier or payload.get("granularity_tier") or "atomic")
        full_milli = int(self.grid.runtime_constants["full_confidence_milli"])
        step_milli = (
            max(0, min(full_milli, int(step_confidence_milli)))
            if step_confidence_milli is not None
            else full_milli
        )

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT status, lease_epoch, task_key, prefix_signature, capability_tag, tool_schema_tag, cumulative_attenuation_survival, attenuation_halted FROM task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if existing is not None:
                    conn.execute("COMMIT")
                    return {
                        "ok": True,
                        "idempotent": True,
                        "task_id": task_id,
                        "status": str(existing["status"]),
                        "lease_epoch": int(existing["lease_epoch"]),
                        "task_key": str(existing["task_key"]),
                        "prefix_signature": str(existing["prefix_signature"]),
                        "capability_tag": str(existing["capability_tag"]),
                        "tool_schema_tag": str(existing["tool_schema_tag"]),
                        "cumulative_attenuation_survival": float(existing["cumulative_attenuation_survival"]),
                        "attenuation_halted": bool(int(existing["attenuation_halted"])),
                    }

                resolved_depth = int(dag_depth) if dag_depth is not None else 0
                if dag_depth is None and deps:
                    parent_depths: List[int] = []
                    for dep_id in deps:
                        prow = conn.execute(
                            "SELECT dag_depth FROM task_leases WHERE task_id = ?",
                            (dep_id,),
                        ).fetchone()
                        if prow is not None:
                            parent_depths.append(int(prow["dag_depth"]))
                    if parent_depths:
                        resolved_depth = max(parent_depths) + 1
                resolved_width = max(1, int(dag_width)) if dag_width is not None else 1

                conn.execute(
                    """
                    INSERT INTO task_leases (
                        task_id, namespace, operation, task_key, contract_tag,
                        payload_json, depends_on_json, status, lease_epoch,
                        lease_expires_at, checkpoint_json, visibility,
                        prefix_signature, capability_tag, tool_schema_tag,
                        origin_id, dag_depth, dag_width, granularity_tier,
                        step_confidence_milli, cumulative_attenuation_survival,
                        attenuation_halted, checked_reset, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0.0, '{}', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 0, 0, ?, ?)
                    """,
                    (
                        task_id,
                        namespace,
                        operation,
                        tkey,
                        contract_tag,
                        json.dumps(dict(payload), sort_keys=True),
                        json.dumps(deps),
                        visibility,
                        pfx_sig,
                        cap_tag,
                        schema_tag,
                        orig_id,
                        resolved_depth,
                        resolved_width,
                        gran_tier,
                        step_milli,
                        now,
                        now,
                    ),
                )
                self._recompute_downstream_attenuation(conn, regime=regime)
                inserted = conn.execute(
                    "SELECT cumulative_attenuation_survival, attenuation_halted FROM task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                surv = float(inserted["cumulative_attenuation_survival"]) if inserted else 1.0
                halted = bool(int(inserted["attenuation_halted"])) if inserted else False
                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "idempotent": False,
                    "task_id": task_id,
                    "status": "queued",
                    "lease_epoch": 0,
                    "task_key": tkey,
                    "depends_on": deps,
                    "prefix_signature": pfx_sig,
                    "capability_tag": cap_tag,
                    "tool_schema_tag": schema_tag,
                    "origin_id": orig_id,
                    "dag_depth": resolved_depth,
                    "dag_width": resolved_width,
                    "granularity_tier": gran_tier,
                    "cumulative_attenuation_survival": surv,
                    "attenuation_halted": halted,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def _lease_row_to_claim_dict(
        self,
        row: sqlite3.Row,
        worker_id: str,
        next_epoch: int,
        fence: str,
        exp_at: float,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        tid = str(row["task_id"])
        ckpt = json.loads(str(row["checkpoint_json"] or "{}"))
        resume = self._build_resume_packet(
            tid,
            str(row["namespace"]),
            str(row["operation"]),
            next_epoch,
            ckpt,
            regime=regime,
        )
        return {
            "ok": True,
            "task_id": tid,
            "namespace": str(row["namespace"]),
            "operation": str(row["operation"]),
            "task_key": str(row["task_key"]),
            "contract_tag": str(row["contract_tag"]),
            "prefix_signature": str(row["prefix_signature"]),
            "capability_tag": str(row["capability_tag"]),
            "tool_schema_tag": str(row["tool_schema_tag"]),
            "origin_id": str(row["origin_id"]),
            "dag_depth": int(row["dag_depth"]),
            "granularity_tier": str(row["granularity_tier"]),
            "cumulative_attenuation_survival": float(row["cumulative_attenuation_survival"]),
            "worker_id": worker_id,
            "lease_epoch": next_epoch,
            "fence_token": fence,
            "lease_expires_at": exp_at,
            "payload": json.loads(str(row["payload_json"])),
            "resume_packet": resume,
        }

    def claim_task(
        self,
        worker_id: str,
        namespace: Optional[str] = None,
        task_id: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        cell = self.grid.resolve_cell(regime)
        ttl = float(cell["lease_ttl_seconds"])
        max_epochs = int(cell["max_retry_epochs"])

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                if task_id:
                    halted_row = conn.execute(
                        "SELECT cumulative_attenuation_survival, attenuation_halted FROM task_leases WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if halted_row is not None and int(halted_row["attenuation_halted"]) == 1:
                        conn.execute("COMMIT")
                        return {
                            "ok": False,
                            "status_code": int(self.grid.runtime_constants["http_status_conflict"]),
                            "error": "attenuation_circuit_breaker_halted",
                            "task_id": task_id,
                            "cumulative_attenuation_survival": float(halted_row["cumulative_attenuation_survival"]),
                        }

                completed_rows = conn.execute(
                    "SELECT task_id FROM task_leases WHERE status = 'completed'"
                ).fetchall()
                done_ids = {str(r["task_id"]) for r in completed_rows}

                query = """
                    SELECT * FROM task_leases
                    WHERE (status = 'queued' OR (status = 'leased' AND lease_expires_at <= ?))
                      AND lease_epoch < ?
                      AND attenuation_halted = 0
                """
                params: List[Any] = [now, max_epochs]
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if namespace:
                    query += " AND namespace = ?"
                    params.append(namespace)
                query += " ORDER BY created_at ASC"

                candidates = conn.execute(query, tuple(params)).fetchall()
                chosen = None
                for row in candidates:
                    deps = json.loads(str(row["depends_on_json"] or "[]"))
                    if all(str(d) in done_ids for d in deps):
                        chosen = row
                        break

                if chosen is None:
                    conn.execute("COMMIT")
                    return {"ok": False, "error": "no_ready_task"}

                tid = str(chosen["task_id"])
                prev_epoch = int(chosen["lease_epoch"])
                next_epoch = prev_epoch + 1
                fence = self._compute_fence_token(tid, worker_id, next_epoch)
                exp_at = now + ttl

                cur = conn.execute(
                    """
                    UPDATE task_leases
                    SET status = 'leased',
                        lease_owner = ?,
                        lease_epoch = ?,
                        fence_token = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE task_id = ? AND lease_epoch = ?
                    """,
                    (worker_id, next_epoch, fence, exp_at, now, tid, prev_epoch),
                )
                if cur.rowcount != 1:
                    conn.execute("ROLLBACK")
                    return {"ok": False, "error": "cas_contention_retry"}

                conn.execute("COMMIT")
                return self._lease_row_to_claim_dict(
                    chosen,
                    worker_id=worker_id,
                    next_epoch=next_epoch,
                    fence=fence,
                    exp_at=exp_at,
                    regime=regime,
                )
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def _select_affinity_bundle_rows(
        self,
        ready_rows: Sequence[sqlite3.Row],
        target_k: int,
        dispatch_mode: str,
        preferred_prefix_signature: Optional[str] = None,
        preferred_capability_tag: Optional[str] = None,
        preferred_tool_schema_tag: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        if not ready_rows or target_k <= 0:
            return []
        if dispatch_mode == "single_isolated":
            return [ready_rows[0]]

        if dispatch_mode == "back_to_back_sequential":
            prefix_counts: Dict[str, int] = {}
            cap_counts: Dict[str, int] = {}
            earliest_prefix: Dict[str, float] = {}
            for r in ready_rows:
                pfx = str(r["prefix_signature"])
                cap = str(r["capability_tag"])
                cts = float(r["created_at"])
                prefix_counts[pfx] = prefix_counts.get(pfx, 0) + 1
                cap_counts[cap] = cap_counts.get(cap, 0) + 1
                if pfx not in earliest_prefix or cts < earliest_prefix[pfx]:
                    earliest_prefix[pfx] = cts

            def _seq_key(r: sqlite3.Row) -> Tuple[Any, ...]:
                pfx = str(r["prefix_signature"])
                cap = str(r["capability_tag"])
                pref_pfx_hit = 0 if (preferred_prefix_signature and pfx == preferred_prefix_signature) else 1
                pref_cap_hit = 0 if (preferred_capability_tag and cap == preferred_capability_tag) else 1
                return (
                    pref_pfx_hit,
                    pref_cap_hit,
                    -prefix_counts.get(pfx, 0),
                    -cap_counts.get(cap, 0),
                    earliest_prefix.get(pfx, float(r["created_at"])),
                    pfx,
                    cap,
                    float(r["created_at"]),
                )

            ordered = sorted(ready_rows, key=_seq_key)
            return list(ordered[:target_k])

        if dispatch_mode == "simultaneous_batched_pack":
            schema_counts: Dict[str, int] = {}
            earliest_schema: Dict[str, float] = {}
            for r in ready_rows:
                sch = str(r["tool_schema_tag"])
                cts = float(r["created_at"])
                schema_counts[sch] = schema_counts.get(sch, 0) + 1
                if sch not in earliest_schema or cts < earliest_schema[sch]:
                    earliest_schema[sch] = cts

            def _pack_key(r: sqlite3.Row) -> Tuple[Any, ...]:
                sch = str(r["tool_schema_tag"])
                pref_sch_hit = 0 if (preferred_tool_schema_tag and sch == preferred_tool_schema_tag) else 1
                return (
                    pref_sch_hit,
                    -schema_counts.get(sch, 0),
                    earliest_schema.get(sch, float(r["created_at"])),
                    sch,
                    str(r["prefix_signature"]),
                    float(r["created_at"]),
                )

            ordered = sorted(ready_rows, key=_pack_key)
            return list(ordered[:target_k])

        # speculative_pipeline
        origin_counts: Dict[str, int] = {}
        for r in ready_rows:
            orig = str(r["origin_id"])
            origin_counts[orig] = origin_counts.get(orig, 0) + 1

        def _pipe_key(r: sqlite3.Row) -> Tuple[Any, ...]:
            orig = str(r["origin_id"])
            return (
                -origin_counts.get(orig, 0),
                orig,
                int(r["dag_depth"]),
                str(r["prefix_signature"]),
                float(r["created_at"]),
            )

        ordered = sorted(ready_rows, key=_pipe_key)
        return list(ordered[:target_k])

    def _estimate_kv_reuse_and_schedule(
        self,
        claimed: Sequence[Mapping[str, Any]],
        dispatch_mode: str,
    ) -> Tuple[float, Dict[str, Any]]:
        rc = self.grid.runtime_constants
        round_d = int(rc["ratio_round_digits"])
        exact_r = float(rc["kv_prefix_exact_match_ratio"])
        cap_r = float(rc["kv_capability_match_ratio"])
        pack_r = float(rc["kv_schema_pack_match_ratio"])
        pipe_r = float(rc["kv_speculative_stage_ratio"])
        base_r = float(rc["kv_disjoint_baseline_ratio"])

        steps: List[Dict[str, Any]] = []
        transition_scores: List[float] = []

        for idx, item in enumerate(claimed):
            pfx = str(item["prefix_signature"])
            cap = str(item["capability_tag"])
            sch = str(item["tool_schema_tag"])
            if idx == 0:
                if dispatch_mode == "back_to_back_sequential":
                    directive = "cold_load_prefix_and_cartridge"
                elif dispatch_mode == "simultaneous_batched_pack":
                    directive = "pack_anchor_schema_slot"
                elif dispatch_mode == "speculative_pipeline":
                    directive = "pipeline_lead_stage"
                else:
                    directive = "isolated_single_lease"
            else:
                ref = claimed[idx - 1] if dispatch_mode != "simultaneous_batched_pack" else claimed[0]
                ref_pfx = str(ref["prefix_signature"])
                ref_cap = str(ref["capability_tag"])
                ref_sch = str(ref["tool_schema_tag"])
                if pfx and pfx == ref_pfx:
                    transition_scores.append(exact_r)
                    directive = (
                        "reuse_warm_kv_prefix_and_cartridge"
                        if dispatch_mode == "back_to_back_sequential"
                        else "packed_shared_prefix_slot"
                    )
                elif dispatch_mode == "simultaneous_batched_pack" and sch and sch == ref_sch:
                    transition_scores.append(pack_r)
                    directive = "pack_compatible_tool_schema_slot"
                elif dispatch_mode == "speculative_pipeline" and (cap == ref_cap or str(item["origin_id"]) == str(ref["origin_id"])):
                    transition_scores.append(pipe_r)
                    directive = "speculative_prefetch_pipeline_stage"
                elif cap and cap == ref_cap:
                    transition_scores.append(cap_r)
                    directive = "reuse_capability_cartridge"
                elif sch and sch == ref_sch:
                    transition_scores.append(pack_r)
                    directive = "reuse_tool_schema_binding"
                else:
                    transition_scores.append(base_r)
                    directive = "switch_prefix_context"

            steps.append(
                {
                    "slot_index": idx,
                    "task_id": str(item["task_id"]),
                    "execution_directive": directive,
                    "prefix_signature": pfx,
                    "capability_tag": cap,
                    "tool_schema_tag": sch,
                }
            )

        ratio = (
            round(sum(transition_scores) / float(len(transition_scores)), round_d)
            if transition_scores
            else 0.0
        )
        pattern_map = {
            "single_isolated": "isolated_single_task_turn",
            "back_to_back_sequential": "serial_warm_kv_prefix_and_lora_xs_cartridge_reuse",
            "simultaneous_batched_pack": "single_turn_packed_compatible_tool_schema_batch",
            "speculative_pipeline": "pipelined_stage_overlap_with_independent_check",
        }
        schedule = {
            "mode": dispatch_mode,
            "pattern": pattern_map.get(dispatch_mode, "serial_warm_kv_prefix_and_lora_xs_cartridge_reuse"),
            "distinct_prefixes": len({str(c["prefix_signature"]) for c in claimed}),
            "distinct_tool_schemas": len({str(c["tool_schema_tag"]) for c in claimed}),
            "steps": steps,
        }
        return ratio, schedule

    def claim_bundle(
        self,
        worker_id: str,
        bundle_size: Optional[int] = None,
        dispatch_mode: Optional[str] = None,
        namespace: Optional[str] = None,
        preferred_prefix_signature: Optional[str] = None,
        preferred_capability_tag: Optional[str] = None,
        preferred_tool_schema_tag: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        cell = self.grid.resolve_cell(regime)
        ttl = float(cell["lease_ttl_seconds"])
        max_epochs = int(cell["max_retry_epochs"])
        min_k = int(self.grid.runtime_constants["bundle_min_k"])
        max_k = int(self.grid.runtime_constants["bundle_max_k"])

        mode = str(dispatch_mode or cell["dispatch_affinity_mode"]).strip()
        if mode not in DISPATCH_AFFINITY_MODES:
            mode = str(cell["dispatch_affinity_mode"])

        raw_k = int(bundle_size) if bundle_size is not None else int(cell["bundle_cardinality_k"])
        target_k = 1 if mode == "single_isolated" else max(min_k, min(max_k, raw_k))

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                completed_rows = conn.execute(
                    "SELECT task_id FROM task_leases WHERE status = 'completed'"
                ).fetchall()
                done_ids = {str(r["task_id"]) for r in completed_rows}

                query = """
                    SELECT * FROM task_leases
                    WHERE (status = 'queued' OR (status = 'leased' AND lease_expires_at <= ?))
                      AND lease_epoch < ?
                      AND attenuation_halted = 0
                """
                params: List[Any] = [now, max_epochs]
                if namespace:
                    query += " AND namespace = ?"
                    params.append(namespace)
                query += " ORDER BY created_at ASC"

                candidates = conn.execute(query, tuple(params)).fetchall()
                ready_rows: List[sqlite3.Row] = []
                for row in candidates:
                    deps = json.loads(str(row["depends_on_json"] or "[]"))
                    if all(str(d) in done_ids for d in deps):
                        ready_rows.append(row)

                chosen_rows = self._select_affinity_bundle_rows(
                    ready_rows=ready_rows,
                    target_k=target_k,
                    dispatch_mode=mode,
                    preferred_prefix_signature=preferred_prefix_signature,
                    preferred_capability_tag=preferred_capability_tag,
                    preferred_tool_schema_tag=preferred_tool_schema_tag,
                )
                if not chosen_rows:
                    conn.execute("COMMIT")
                    return {
                        "ok": False,
                        "error": "no_ready_tasks_for_bundle",
                        "dispatch_mode": mode,
                        "requested_bundle_size": target_k,
                        "claimed_count": 0,
                        "tasks": [],
                    }

                exp_at = now + ttl
                claimed_items: List[Dict[str, Any]] = []
                for row in chosen_rows:
                    tid = str(row["task_id"])
                    prev_epoch = int(row["lease_epoch"])
                    next_epoch = prev_epoch + 1
                    fence = self._compute_fence_token(tid, worker_id, next_epoch)
                    cur = conn.execute(
                        """
                        UPDATE task_leases
                        SET status = 'leased',
                            lease_owner = ?,
                            lease_epoch = ?,
                            fence_token = ?,
                            lease_expires_at = ?,
                            updated_at = ?
                        WHERE task_id = ? AND lease_epoch = ?
                        """,
                        (worker_id, next_epoch, fence, exp_at, now, tid, prev_epoch),
                    )
                    if cur.rowcount != 1:
                        conn.execute("ROLLBACK")
                        return {"ok": False, "error": "cas_contention_retry"}
                    claimed_items.append(
                        self._lease_row_to_claim_dict(
                            row,
                            worker_id=worker_id,
                            next_epoch=next_epoch,
                            fence=fence,
                            exp_at=exp_at,
                            regime=regime,
                        )
                    )

                conn.execute("COMMIT")
                reuse_ratio, schedule = self._estimate_kv_reuse_and_schedule(claimed_items, mode)
                char_len = int(self.grid.runtime_constants["task_key_char_len"])
                bundle_digest = hashlib.blake2s(
                    f"{worker_id}:{now}:{','.join(c['task_id'] for c in claimed_items)}".encode("utf-8")
                ).hexdigest()[:char_len]
                return {
                    "ok": True,
                    "bundle_id": f"bdl_{bundle_digest}",
                    "worker_id": worker_id,
                    "dispatch_mode": mode,
                    "requested_bundle_size": target_k,
                    "claimed_count": len(claimed_items),
                    "estimated_kv_prefix_reuse_ratio": reuse_ratio,
                    "recommended_execution_schedule": schedule,
                    "tasks": claimed_items,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def checkpoint_task(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        fence_token: str,
        checkpoint: Mapping[str, Any],
        completed: bool = False,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        cell = self.grid.resolve_cell(regime)
        ttl = float(cell["lease_ttl_seconds"])
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return {"ok": False, "error": "unknown_task_id"}

                active_epoch = int(row["lease_epoch"])
                active_owner = str(row["lease_owner"] or "")
                expected_fence = str(row["fence_token"] or "")
                if (
                    active_epoch != int(lease_epoch)
                    or active_owner != worker_id
                    or not hmac.compare_digest(expected_fence, str(fence_token))
                ):
                    conn.execute("COMMIT")
                    return {
                        "ok": False,
                        "status_code": int(self.grid.runtime_constants["http_status_conflict"]),
                        "error": "stale_lease_epoch_fenced",
                        "active_lease_epoch": active_epoch,
                        "attempted_lease_epoch": int(lease_epoch),
                    }

                new_status = "completed" if completed else "leased"
                new_exp = 0.0 if completed else (now + ttl)
                conn.execute(
                    """
                    UPDATE task_leases
                    SET status = ?,
                        checkpoint_json = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE task_id = ? AND lease_epoch = ?
                    """,
                    (
                        new_status,
                        json.dumps(dict(checkpoint), sort_keys=True),
                        new_exp,
                        now,
                        task_id,
                        active_epoch,
                    ),
                )
                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "task_id": task_id,
                    "status": new_status,
                    "lease_epoch": active_epoch,
                    "resume_packet": self._build_resume_packet(
                        task_id,
                        str(row["namespace"]),
                        str(row["operation"]),
                        active_epoch,
                        checkpoint,
                        regime=regime,
                    ),
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def reclaim_stale_leases(self, worker_id_filter: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                if worker_id_filter:
                    cur = conn.execute(
                        """
                        UPDATE task_leases
                        SET lease_expires_at = 0.0, updated_at = ?
                        WHERE status = 'leased' AND lease_owner = ?
                        """,
                        (now, worker_id_filter),
                    )
                else:
                    cur = conn.execute(
                        """
                        UPDATE task_leases
                        SET status = 'queued', updated_at = ?
                        WHERE status = 'leased' AND lease_expires_at <= ?
                        """,
                        (now, now),
                    )
                count = int(cur.rowcount)
                conn.execute("COMMIT")
                return {"ok": True, "reclaimed_count": count}
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    # ── 2. Bitemporal Maker != Checker Store, Enriched Submit & FTS5 RRF ──

    def stage_or_promote_object(
        self,
        task_id: str,
        maker_id: str,
        title: str,
        summary_text: str,
        payload: Mapping[str, Any],
        checker_id: Optional[str] = None,
        namespace: str = "default",
        task_key: Optional[str] = None,
        contract_tag: str = "v1",
        visibility: str = "local_only",
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        attenuation_confidence_milli: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        v_from = float(valid_from) if valid_from is not None else now
        if checker_id is not None and str(checker_id).strip() == str(maker_id).strip():
            return {
                "ok": False,
                "status_code": int(self.grid.runtime_constants["http_status_conflict"]),
                "error": "self_approval_blocked_maker_equals_checker",
                "maker_id": maker_id,
                "checker_id": checker_id,
            }

        resolved_key = task_key or self._compute_task_key(namespace, "artifact", contract_tag, payload)
        char_len = int(self.grid.runtime_constants["task_key_char_len"])
        obj_digest = hashlib.blake2s(
            f"{task_id}:{resolved_key}:{contract_tag}:{maker_id}".encode("utf-8")
        ).hexdigest()[:char_len]
        object_id = f"obj_{namespace}_{obj_digest}"
        has_independent_checker = bool(checker_id and str(checker_id).strip())
        stage = "promoted" if has_independent_checker else "staged"
        full_milli = int(self.grid.runtime_constants["full_confidence_milli"])

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO task_objects (
                        object_id, task_id, namespace, task_key, contract_tag,
                        stage, maker_id, checker_id, visibility, title,
                        summary_text, payload_json, valid_from, valid_to, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        object_id,
                        task_id,
                        namespace,
                        resolved_key,
                        contract_tag,
                        stage,
                        maker_id,
                        checker_id,
                        visibility,
                        title,
                        summary_text,
                        json.dumps(dict(payload), sort_keys=True),
                        v_from,
                        valid_to,
                        now,
                    ),
                )
                if self._fts5_enabled:
                    conn.execute("DELETE FROM task_objects_fts WHERE object_id = ?", (object_id,))
                    conn.execute(
                        "INSERT INTO task_objects_fts (object_id, title, summary_text, namespace) VALUES (?, ?, ?, ?)",
                        (object_id, title, summary_text, namespace),
                    )

                attenuation_reset = False
                attenuation_summary: Optional[Dict[str, Any]] = None
                trow = conn.execute(
                    "SELECT task_id FROM task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if trow is not None:
                    if has_independent_checker:
                        conn.execute(
                            """
                            UPDATE task_leases
                            SET checked_reset = 1,
                                step_confidence_milli = ?,
                                cumulative_attenuation_survival = 1.0,
                                attenuation_halted = 0,
                                updated_at = ?
                            WHERE task_id = ?
                            """,
                            (full_milli, now, task_id),
                        )
                        attenuation_reset = True
                        attenuation_summary = self._recompute_downstream_attenuation(conn, regime=regime)
                    elif attenuation_confidence_milli is not None:
                        clamped_conf = max(0, min(full_milli, int(attenuation_confidence_milli)))
                        conn.execute(
                            """
                            UPDATE task_leases
                            SET checked_reset = 0,
                                step_confidence_milli = ?,
                                updated_at = ?
                            WHERE task_id = ?
                            """,
                            (clamped_conf, now, task_id),
                        )
                        attenuation_summary = self._recompute_downstream_attenuation(conn, regime=regime)

                conn.execute("COMMIT")
                out: Dict[str, Any] = {
                    "ok": True,
                    "object_id": object_id,
                    "stage": stage,
                    "task_key": resolved_key,
                    "contract_tag": contract_tag,
                    "maker_id": maker_id,
                    "checker_id": checker_id,
                    "visibility": visibility,
                    "attenuation_reset_to_unity": attenuation_reset,
                }
                if attenuation_summary is not None:
                    out["attenuation_state"] = attenuation_summary
                return out
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def submit_result(
        self,
        task_id: str,
        maker_id: str,
        title: str,
        summary_text: str = "",
        payload: Optional[Mapping[str, Any]] = None,
        checker_id: Optional[str] = None,
        namespace: str = "default",
        contract_tag: str = "v1",
        visibility: str = "local_only",
        lease_epoch: Optional[int] = None,
        fence_token: Optional[str] = None,
        complete_task: bool = False,
        input_text: str = "",
        target_label: str = "",
        prefix_signature: str = "",
        reasoning_mode: str = "adaptive",
        thinking_tokens: int = 0,
        answer_tokens: int = 0,
        latency_ms: float = 0.0,
        activation_norm_digest: str = "",
        attenuation_confidence_milli: Optional[int] = None,
        seed_id: int = 0,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        body_payload = dict(payload or {})
        if complete_task and lease_epoch is not None and fence_token is not None:
            ckpt_res = self.checkpoint_task(
                task_id=task_id,
                worker_id=maker_id,
                lease_epoch=int(lease_epoch),
                fence_token=str(fence_token),
                checkpoint={"submitted_by": maker_id, "title": title},
                completed=True,
                regime=regime,
            )
            if not ckpt_res.get("ok"):
                return ckpt_res

        full_milli = int(self.grid.runtime_constants["full_confidence_milli"])
        eff_conf = (
            full_milli
            if (checker_id and str(checker_id).strip() and str(checker_id).strip() != str(maker_id).strip())
            else (
                max(0, min(full_milli, int(attenuation_confidence_milli)))
                if attenuation_confidence_milli is not None
                else full_milli
            )
        )
        obj_res = self.stage_or_promote_object(
            task_id=task_id,
            maker_id=maker_id,
            checker_id=checker_id,
            namespace=namespace,
            contract_tag=contract_tag,
            visibility=visibility,
            title=title,
            summary_text=summary_text,
            payload=body_payload,
            attenuation_confidence_milli=attenuation_confidence_milli,
            regime=regime,
        )
        if not obj_res.get("ok"):
            return obj_res

        env_res = self.record_trace_envelope(
            namespace=namespace,
            task_id=task_id,
            object_id=str(obj_res["object_id"]),
            worker_id=maker_id,
            input_text=input_text,
            target_label=target_label,
            prefix_signature=prefix_signature,
            reasoning_mode=reasoning_mode,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            latency_ms=latency_ms,
            activation_norm_digest=activation_norm_digest,
            attenuation_confidence_milli=eff_conf,
            seed_id=seed_id,
        )
        return {
            **obj_res,
            "trace_envelope": env_res,
        }

    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "task_id": str(row["task_id"]),
                    "namespace": str(row["namespace"]),
                    "operation": str(row["operation"]),
                    "status": str(row["status"]),
                    "lease_epoch": int(row["lease_epoch"]),
                    "prefix_signature": str(row["prefix_signature"]),
                    "capability_tag": str(row["capability_tag"]),
                    "tool_schema_tag": str(row["tool_schema_tag"]),
                    "origin_id": str(row["origin_id"]),
                    "dag_depth": int(row["dag_depth"]),
                    "dag_width": int(row["dag_width"]),
                    "granularity_tier": str(row["granularity_tier"]),
                    "step_confidence_milli": int(row["step_confidence_milli"]),
                    "cumulative_attenuation_survival": float(row["cumulative_attenuation_survival"]),
                    "attenuation_halted": bool(int(row["attenuation_halted"])),
                    "checked_reset": bool(int(row["checked_reset"])),
                }
            finally:
                self._close_if_disk(conn)

    def lookup_reusable_object(
        self,
        task_key: str,
        contract_tag: str = "v1",
        as_of: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        ts = float(as_of) if as_of is not None else time.time()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM task_objects
                    WHERE task_key = ?
                      AND contract_tag = ?
                      AND stage = 'promoted'
                      AND valid_from <= ?
                      AND (valid_to IS NULL OR valid_to > ?)
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (task_key, contract_tag, ts, ts),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "object_id": str(row["object_id"]),
                    "task_id": str(row["task_id"]),
                    "namespace": str(row["namespace"]),
                    "task_key": str(row["task_key"]),
                    "contract_tag": str(row["contract_tag"]),
                    "maker_id": str(row["maker_id"]),
                    "checker_id": str(row["checker_id"]),
                    "title": str(row["title"]),
                    "summary_text": str(row["summary_text"]),
                    "payload": json.loads(str(row["payload_json"])),
                    "llm_tokens_spent": 0,
                }
            finally:
                self._close_if_disk(conn)

    def hybrid_search_objects(
        self,
        query: str,
        namespace: Optional[str] = None,
        promoted_only: bool = True,
        unseen_only_for: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        rrf_k = float(self.grid.runtime_constants["rrf_rank_constant"])
        def_lim = int(self.grid.runtime_constants["hybrid_search_default_limit"])
        max_lim = int(self.grid.runtime_constants["hybrid_search_max_limit"])
        top_n = max(1, min(max_lim, int(limit or def_lim)))
        tokens = [tok.lower() for tok in query.replace("-", " ").split() if tok.strip()]

        with self._lock:
            conn = self._connect()
            try:
                sql = "SELECT * FROM task_objects WHERE 1=1"
                args: List[Any] = []
                if promoted_only:
                    sql += " AND stage = 'promoted'"
                if namespace:
                    sql += " AND namespace = ?"
                    args.append(namespace)
                sql += " ORDER BY created_at DESC"
                all_rows = conn.execute(sql, tuple(args)).fetchall()

                seen_set: set[str] = set()
                if unseen_only_for:
                    srows = conn.execute(
                        "SELECT object_id FROM seen_novelty_keys WHERE agent_scope = ?",
                        (unseen_only_for,),
                    ).fetchall()
                    seen_set = {str(r["object_id"]) for r in srows}

                fts_ranks: Dict[str, int] = {}
                if self._fts5_enabled and tokens:
                    fts_q = " OR ".join(f'"{t}"' for t in tokens)
                    try:
                        frows = conn.execute(
                            "SELECT object_id FROM task_objects_fts WHERE task_objects_fts MATCH ? ORDER BY bm25(task_objects_fts)",
                            (fts_q,),
                        ).fetchall()
                        for idx, fr in enumerate(frows):
                            fts_ranks[str(fr["object_id"])] = idx + 1
                    except Exception:
                        fts_ranks = {}

                scored: List[Tuple[float, sqlite3.Row]] = []
                for rec_idx, row in enumerate(all_rows):
                    oid = str(row["object_id"])
                    if unseen_only_for and oid in seen_set:
                        continue
                    recency_rank = rec_idx + 1
                    text_blob = f"{row['title']} {row['summary_text']} {row['namespace']}".lower()
                    lex_hits = sum(1 for t in tokens if t in text_blob)
                    if tokens and lex_hits == 0 and oid not in fts_ranks:
                        continue
                    bm25_rank = fts_ranks.get(oid, max_lim + recency_rank)
                    score = (1.0 / (rrf_k + float(bm25_rank))) + (1.0 / (rrf_k + float(recency_rank)))
                    scored.append((score, row))

                scored.sort(key=lambda item: item[0], reverse=True)
                chosen = scored[:top_n]
                now = time.time()
                results: List[Dict[str, Any]] = []
                for score, row in chosen:
                    oid = str(row["object_id"])
                    if unseen_only_for:
                        conn.execute(
                            "INSERT OR IGNORE INTO seen_novelty_keys (agent_scope, object_id, seen_at) VALUES (?, ?, ?)",
                            (unseen_only_for, oid, now),
                        )
                    results.append(
                        {
                            "object_id": oid,
                            "title": str(row["title"]),
                            "namespace": str(row["namespace"]),
                            "stage": str(row["stage"]),
                            "rrf_score": round(score, int(self.grid.runtime_constants["macro_dag_phase_count"])),
                            "summary_text": str(row["summary_text"]),
                            "visibility": str(row["visibility"]),
                        }
                    )
                if unseen_only_for:
                    conn.commit()
                return {"ok": True, "count": len(results), "results": results}
            finally:
                self._close_if_disk(conn)

    # ── 3. Attenuation-Tracked Macro-DAG & Action-Lattice Compiler ────────

    def compile_macro_dag(
        self,
        workflow_id: str,
        goal: str,
        target_task_count: int,
        namespace: str = "macro_workflow",
        origin_id: Optional[str] = None,
        step_confidence_milli: Optional[int] = None,
        regime: Optional[str] = None,
        enqueue_immediately: bool = True,
    ) -> Dict[str, Any]:
        rc = self.grid.runtime_constants
        min_t = int(rc["macro_dag_min_tasks"])
        max_t = int(rc["macro_dag_max_tasks"])
        total = max(min_t, min(max_t, int(target_task_count)))
        phase_n = len(ACTION_PHASES)

        coarse_max = int(rc["coarse_granularity_max_tasks"])
        balanced_max = int(rc["balanced_granularity_max_tasks"])
        if total <= coarse_max:
            granularity_tier = "coarse_macro"
        elif total <= balanced_max:
            granularity_tier = "balanced_meso"
        else:
            granularity_tier = "fine_micro"

        resolved_origin = str(origin_id or workflow_id)
        full_milli = int(rc["full_confidence_milli"])
        def_step_milli = int(rc["default_step_confidence_milli"])
        hop_milli = (
            max(0, min(full_milli, int(step_confidence_milli)))
            if step_confidence_milli is not None
            else def_step_milli
        )
        cell = self.grid.resolve_cell(regime)
        cutoff_milli = int(cell["attenuation_circuit_breaker_milli"])
        cutoff = float(cutoff_milli) / float(full_milli)
        round_d = int(rc["attenuation_round_digits"])

        phase_counts: List[int] = [0] * phase_n
        for idx in range(total):
            p_idx = (idx * phase_n) // total
            phase_counts[p_idx] += 1
        dag_width = max(1, max(phase_counts))
        populated_phases = [c for c in phase_counts if c > 0]
        dag_depth = len(populated_phases)

        phase_buckets: List[List[str]] = [[] for _ in range(phase_n)]
        node_survivals: Dict[str, float] = {}
        compiled_nodes: List[Dict[str, Any]] = []
        halted_ids: List[str] = []

        for idx in range(total):
            phase_idx = (idx * phase_n) // total
            phase_name = ACTION_PHASES[phase_idx]
            tid = f"{workflow_id}:p{phase_idx + 1}:{phase_name}:{idx + 1}"
            deps: List[str] = []
            prev_nonempty_idx = -1
            for scan_p in range(phase_idx - 1, -1, -1):
                if phase_buckets[scan_p]:
                    prev_nonempty_idx = scan_p
                    break
            if prev_nonempty_idx >= 0:
                prev_list = phase_buckets[prev_nonempty_idx]
                parent = prev_list[idx % len(prev_list)]
                deps.append(parent)

            depth_val = 0 if not deps else (phase_idx)
            node_step_milli = full_milli if not deps else hop_milli
            parent_surv = min(node_survivals[d] for d in deps) if deps else 1.0
            cum_surv = (
                1.0
                if not deps
                else round(parent_surv * (float(node_step_milli) / float(full_milli)), round_d)
            )
            is_halted = cum_surv < cutoff
            if is_halted:
                halted_ids.append(tid)
            node_survivals[tid] = cum_surv

            cap_tag = f"phase_{phase_name}"
            schema_tag = f"schema_{phase_name}"
            pfx_sig = self._compute_prefix_signature(namespace, cap_tag, resolved_origin)

            node = {
                "task_id": tid,
                "origin_id": resolved_origin,
                "phase": phase_name,
                "dag_depth": depth_val,
                "dag_width": dag_width,
                "granularity_tier": granularity_tier,
                "prefix_signature": pfx_sig,
                "capability_tag": cap_tag,
                "tool_schema_tag": schema_tag,
                "depends_on": deps,
                "cumulative_attenuation_survival": cum_surv,
                "attenuation_halted": is_halted,
                "payload": {
                    "goal": goal,
                    "origin_id": resolved_origin,
                    "phase": phase_name,
                    "step_index": idx + 1,
                    "total_steps": total,
                    "granularity_tier": granularity_tier,
                },
            }
            phase_buckets[phase_idx].append(tid)
            compiled_nodes.append(node)
            if enqueue_immediately:
                self.enqueue_task(
                    task_id=tid,
                    namespace=namespace,
                    operation=f"workflow_{phase_name}",
                    payload=node["payload"],
                    depends_on=deps,
                    prefix_signature=pfx_sig,
                    capability_tag=cap_tag,
                    tool_schema_tag=schema_tag,
                    origin_id=resolved_origin,
                    dag_depth=depth_val,
                    dag_width=dag_width,
                    granularity_tier=granularity_tier,
                    step_confidence_milli=node_step_milli,
                    regime=regime,
                )

        first_nonempty = next((b for b in phase_buckets if b), [])
        last_nonempty = next((b for b in reversed(phase_buckets) if b), [])
        min_surv = min(node_survivals.values()) if node_survivals else 1.0
        return {
            "ok": True,
            "workflow_id": workflow_id,
            "origin_id": resolved_origin,
            "total_tasks": total,
            "dag_depth": dag_depth,
            "dag_width": dag_width,
            "granularity_tier": granularity_tier,
            "step_confidence_milli": hop_milli,
            "attenuation_circuit_breaker_milli": cutoff_milli,
            "min_cumulative_attenuation_survival": min_surv,
            "halted_task_count": len(halted_ids),
            "halted_task_ids": halted_ids,
            "phases": list(ACTION_PHASES),
            "enqueued": enqueue_immediately,
            "root_tasks": first_nonempty,
            "terminal_tasks": last_nonempty,
            "sample_nodes": compiled_nodes[:min_t],
        }

    def compile_workflow_dag(
        self,
        workflow_id: str,
        goal: str,
        target_task_count: int,
        namespace: str = "macro_workflow",
        enqueue_immediately: bool = True,
        origin_id: Optional[str] = None,
        step_confidence_milli: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.compile_macro_dag(
            workflow_id=workflow_id,
            goal=goal,
            target_task_count=target_task_count,
            namespace=namespace,
            origin_id=origin_id,
            step_confidence_milli=step_confidence_milli,
            regime=regime,
            enqueue_immediately=enqueue_immediately,
        )

    # ── 4. Parameterized NodeGrid + GridLedger + Contextual RL Canary ─────

    @staticmethod
    def _enumerate_cells(
        axes: Sequence[Mapping[str, Any]],
        constraints: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        domains: List[List[Any]] = []
        names: List[str] = []
        for ax in axes:
            names.append(str(ax["name"]))
            kind = str(ax["kind"])
            if kind == "integer_range":
                lo = int(ax["low"])
                hi = int(ax["high"])
                st = max(1, int(ax.get("step", 1)))
                domains.append(list(range(lo, hi + 1, st)))
            elif kind == "choice":
                domains.append(list(ax["choices"]))
            elif kind == "boolean":
                domains.append([False, True])
            else:
                domains.append([ax.get("default")])

        valid: List[Dict[str, Any]] = []
        for combo in itertools.product(*domains):
            cell = dict(zip(names, combo))
            ok = True
            for rule in constraints:
                lv = cell.get(str(rule["left"]))
                rv = cell.get(str(rule["right"]))
                op = str(rule["op"])
                if lv is not None and rv is not None:
                    if op == "le" and lv > rv:
                        ok = False
                    elif op == "lt" and lv >= rv:
                        ok = False
            if ok:
                valid.append(cell)
        return valid

    def define_optimization_space(
        self,
        campaign_id: str,
        axes: Sequence[Mapping[str, Any]],
        constraints: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        ax_list = [dict(a) for a in axes]
        c_list = [dict(c) for c in (constraints or [])]
        cells = self._enumerate_cells(ax_list, c_list)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO grid_campaigns (campaign_id, generation, axes_json, constraints_json, created_at)
                    VALUES (?, 1, ?, ?, ?)
                    """,
                    (campaign_id, json.dumps(ax_list), json.dumps(c_list), now),
                )
                for cell in cells:
                    ckey = json.dumps(cell, sort_keys=True, separators=(",", ":"))
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO grid_ledger (
                            campaign_id, cell_key, params_json, lifecycle_state,
                            evaluations_json, best_reward, mean_cost, updated_at
                        ) VALUES (?, ?, ?, 'applicable', '[]', 0.0, 0.0, ?)
                        """,
                        (campaign_id, ckey, json.dumps(cell, sort_keys=True), now),
                    )
                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "campaign_id": campaign_id,
                    "generation": 1,
                    "applicable_cells": len(cells),
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def propose_optimization_batch(
        self,
        campaign_id: str,
        strategy: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        strat = str(strategy or self.grid.resolve_cell()["exploration_strategy"])
        def_b = int(self.grid.runtime_constants["default_batch_proposals"])
        max_b = int(self.grid.runtime_constants["max_batch_proposals"])
        k = max(1, min(max_b, int(batch_size or def_b)))
        ucb_c = float(self.grid.runtime_constants["ucb_exploration_weight"])

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM grid_ledger WHERE campaign_id = ? ORDER BY cell_key ASC",
                    (campaign_id,),
                ).fetchall()
                if not rows:
                    return {"ok": False, "error": "unknown_campaign_id"}

                total_evals = 1
                parsed_rows: List[Dict[str, Any]] = []
                for r in rows:
                    evals = json.loads(str(r["evaluations_json"]))
                    total_evals += len(evals)
                    parsed_rows.append(
                        {
                            "cell_key": str(r["cell_key"]),
                            "params": json.loads(str(r["params_json"])),
                            "state": str(r["lifecycle_state"]),
                            "n": len(evals),
                            "best_reward": float(r["best_reward"]),
                            "mean_cost": float(r["mean_cost"]),
                        }
                    )

                if strat == "latin_maximin":
                    stride = max(1, len(parsed_rows) // k)
                    picked = [parsed_rows[(i * stride) % len(parsed_rows)] for i in range(min(k, len(parsed_rows)))]
                elif strat == "successive_halving":
                    parsed_rows.sort(
                        key=lambda item: (item["n"] == 0, item["best_reward"] - item["mean_cost"]),
                        reverse=True,
                    )
                    picked = parsed_rows[:k]
                elif strat == "ucb_surrogate":
                    def _ucb(item: Dict[str, Any]) -> float:
                        if item["n"] == 0:
                            return float(self.grid.runtime_constants["percent_scale"])
                        return item["best_reward"] + ucb_c * math.sqrt(math.log(float(total_evals)) / float(item["n"]))

                    parsed_rows.sort(key=_ucb, reverse=True)
                    picked = parsed_rows[:k]
                else:
                    parsed_rows.sort(key=lambda item: (item["n"], item["cell_key"]))
                    picked = parsed_rows[:k]

                now = time.time()
                proposals: List[Dict[str, Any]] = []
                for item in picked:
                    if item["state"] == "applicable":
                        conn.execute(
                            "UPDATE grid_ledger SET lifecycle_state = 'proposed', updated_at = ? WHERE campaign_id = ? AND cell_key = ?",
                            (now, campaign_id, item["cell_key"]),
                        )
                    proposals.append(item["params"])
                conn.commit()
                return {
                    "ok": True,
                    "campaign_id": campaign_id,
                    "strategy": strat,
                    "proposals": proposals,
                }
            finally:
                self._close_if_disk(conn)

    def record_optimization_outcome(
        self,
        campaign_id: str,
        params: Mapping[str, Any],
        reward: float,
        cost: float,
        agent_id: str,
        auto_expand: bool = True,
    ) -> Dict[str, Any]:
        now = time.time()
        quorum = int(self.grid.resolve_cell()["maker_checker_quorum"])
        ckey = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                camp = conn.execute(
                    "SELECT * FROM grid_campaigns WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()
                if camp is None:
                    conn.execute("COMMIT")
                    return {"ok": False, "error": "unknown_campaign_id"}

                row = conn.execute(
                    "SELECT * FROM grid_ledger WHERE campaign_id = ? AND cell_key = ?",
                    (campaign_id, ckey),
                ).fetchone()
                evals: List[Dict[str, Any]] = json.loads(str(row["evaluations_json"])) if row else []
                evals.append({"agent_id": agent_id, "reward": float(reward), "cost": float(cost), "ts": now})
                distinct_agents = {str(e["agent_id"]) for e in evals}
                best_rew = max(float(e["reward"]) for e in evals)
                mean_c = sum(float(e["cost"]) for e in evals) / float(len(evals))
                promoted = len(distinct_agents) >= quorum
                state = "promoted" if promoted else "evaluated"

                conn.execute(
                    """
                    INSERT OR REPLACE INTO grid_ledger (
                        campaign_id, cell_key, params_json, lifecycle_state,
                        evaluations_json, best_reward, mean_cost, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        ckey,
                        json.dumps(dict(params), sort_keys=True),
                        state,
                        json.dumps(evals),
                        best_rew,
                        mean_c,
                        now,
                    ),
                )

                gen = int(camp["generation"])
                expanded: List[str] = []
                if promoted and auto_expand:
                    axes = json.loads(str(camp["axes_json"]))
                    constraints = json.loads(str(camp["constraints_json"]))
                    step_mult = int(self.grid.runtime_constants["boundary_expand_steps"])
                    changed = False
                    for ax in axes:
                        if str(ax.get("kind")) == "integer_range":
                            aname = str(ax["name"])
                            step = max(1, int(ax.get("step", 1)))
                            if params.get(aname) == int(ax["high"]):
                                ax["high"] = int(ax["high"]) + step * step_mult
                                expanded.append(f"{aname}:high->{ax['high']}")
                                changed = True
                    if changed:
                        gen += 1
                        conn.execute(
                            "UPDATE grid_campaigns SET generation = ?, axes_json = ? WHERE campaign_id = ?",
                            (gen, json.dumps(axes), campaign_id),
                        )
                        for new_cell in self._enumerate_cells(axes, constraints):
                            nk = json.dumps(new_cell, sort_keys=True, separators=(",", ":"))
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO grid_ledger (
                                    campaign_id, cell_key, params_json, lifecycle_state,
                                    evaluations_json, best_reward, mean_cost, updated_at
                                ) VALUES (?, ?, ?, 'applicable', '[]', 0.0, 0.0, ?)
                                """,
                                (campaign_id, nk, json.dumps(new_cell, sort_keys=True), now),
                            )

                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "campaign_id": campaign_id,
                    "lifecycle_state": state,
                    "distinct_evaluators": len(distinct_agents),
                    "maker_checker_quorum_met": promoted,
                    "generation": gen,
                    "expanded_axes": expanded,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    def canary_policy_step(
        self,
        policy_id: str,
        mode: str = "select",
        baseline_arm: str = "reuse_then_check",
        canary_arm: str = "boil_down_micro_net",
        used_canary: bool = False,
        reward: float = 0.0,
        sample_index: int = 0,
    ) -> Dict[str, Any]:
        now = time.time()
        cell = self.grid.resolve_cell()
        permille = int(cell["canary_traffic_permille"])
        scale = int(self.grid.runtime_constants["permille_scale"])
        min_samples = int(self.grid.runtime_constants["canary_min_samples"])
        rb_margin = float(self.grid.runtime_constants["canary_rollback_margin"])
        pr_margin = float(self.grid.runtime_constants["canary_promote_margin"])

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM rl_canary_state WHERE policy_id = ?",
                    (policy_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO rl_canary_state (policy_id, baseline_arm, canary_arm, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (policy_id, baseline_arm, canary_arm, now),
                    )
                    row = conn.execute(
                        "SELECT * FROM rl_canary_state WHERE policy_id = ?",
                        (policy_id,),
                    ).fetchone()

                assert row is not None
                b_arm = str(row["baseline_arm"])
                c_arm = str(row["canary_arm"])
                b_sum = float(row["baseline_sum"])
                b_n = int(row["baseline_n"])
                c_sum = float(row["canary_sum"])
                c_n = int(row["canary_n"])
                promos = int(row["promotions"])
                rbs = int(row["rollbacks"])

                if mode == "select":
                    conn.execute("COMMIT")
                    slot = (abs(int(sample_index)) * int(self.grid.runtime_constants["default_hash_prime"])) % scale
                    route_canary = slot < permille
                    return {
                        "ok": True,
                        "policy_id": policy_id,
                        "selected_arm": c_arm if route_canary else b_arm,
                        "used_canary": route_canary,
                        "canary_permille": permille,
                    }

                if used_canary:
                    c_sum += float(reward)
                    c_n += 1
                else:
                    b_sum += float(reward)
                    b_n += 1

                event = "steady"
                if b_n > 0 and c_n >= min_samples:
                    b_mean = b_sum / float(b_n)
                    c_mean = c_sum / float(c_n)
                    if c_mean < (b_mean - rb_margin):
                        rbs += 1
                        c_sum = 0.0
                        c_n = 0
                        event = "canary_auto_rolled_back"
                    elif c_mean > (b_mean + pr_margin):
                        promos += 1
                        b_arm = c_arm
                        b_sum = c_sum
                        b_n = c_n
                        c_sum = 0.0
                        c_n = 0
                        event = "canary_promoted_to_default"

                conn.execute(
                    """
                    UPDATE rl_canary_state
                    SET baseline_arm = ?, canary_arm = ?, baseline_sum = ?, baseline_n = ?,
                        canary_sum = ?, canary_n = ?, promotions = ?, rollbacks = ?, updated_at = ?
                    WHERE policy_id = ?
                    """,
                    (b_arm, c_arm, b_sum, b_n, c_sum, c_n, promos, rbs, now, policy_id),
                )
                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "policy_id": policy_id,
                    "event": event,
                    "active_default_arm": b_arm,
                    "promotions": promos,
                    "rollbacks": rbs,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                self._close_if_disk(conn)

    # ── 5. Enriched Trace Envelopes, Execution Traces & Subagent Registry ─

    def record_trace_envelope(
        self,
        namespace: str,
        task_id: str = "",
        object_id: str = "",
        worker_id: str = "",
        input_text: str = "",
        target_label: str = "",
        prefix_signature: str = "",
        reasoning_mode: str = "adaptive",
        thinking_tokens: int = 0,
        answer_tokens: int = 0,
        tokens_used: Optional[int] = None,
        latency_ms: float = 0.0,
        activation_norm_digest: str = "",
        attenuation_confidence_milli: Optional[int] = None,
        seed_id: int = 0,
        mirror_execution_trace: bool = True,
    ) -> Dict[str, Any]:
        now = time.time()
        norm_mode = self._normalize_reasoning_mode(reasoning_mode)
        t_toks = max(0, int(thinking_tokens))
        a_toks = max(0, int(answer_tokens))
        total_toks = max(0, int(tokens_used)) if tokens_used is not None else (t_toks + a_toks)
        full_milli = int(self.grid.runtime_constants["full_confidence_milli"])
        conf_milli = (
            max(0, min(full_milli, int(attenuation_confidence_milli)))
            if attenuation_confidence_milli is not None
            else full_milli
        )
        char_len = int(self.grid.runtime_constants["task_key_char_len"])
        pfx_sig = str(prefix_signature).strip() or self._compute_prefix_signature(namespace, target_label or "trace")
        act_digest = str(activation_norm_digest).strip()
        if not act_digest:
            act_digest = f"act_{hashlib.blake2s(f'{namespace}:{norm_mode}:{input_text}'.encode('utf-8')).hexdigest()[:char_len]}"

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO trace_envelopes (
                        task_id, object_id, namespace, worker_id, input_text, target_label,
                        prefix_signature, reasoning_mode, thinking_tokens, answer_tokens,
                        tokens_used, latency_ms, activation_norm_digest,
                        attenuation_confidence_milli, seed_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(task_id),
                        str(object_id),
                        namespace,
                        str(worker_id),
                        str(input_text),
                        str(target_label),
                        pfx_sig,
                        norm_mode,
                        t_toks,
                        a_toks,
                        total_toks,
                        float(latency_ms),
                        act_digest,
                        conf_milli,
                        int(seed_id),
                        now,
                    ),
                )
                envelope_id = int(cur.lastrowid or 0)
                if mirror_execution_trace and input_text.strip() and target_label.strip():
                    conn.execute(
                        """
                        INSERT INTO execution_traces (namespace, input_text, target_label, tokens_used, latency_ms, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (namespace, str(input_text), str(target_label), total_toks, float(latency_ms), now),
                    )
                conn.commit()
                return {
                    "ok": True,
                    "envelope_id": envelope_id,
                    "task_id": str(task_id),
                    "object_id": str(object_id),
                    "namespace": namespace,
                    "prefix_signature": pfx_sig,
                    "reasoning_mode": norm_mode,
                    "thinking_tokens": t_toks,
                    "answer_tokens": a_toks,
                    "tokens_used": total_toks,
                    "activation_norm_digest": act_digest,
                    "attenuation_confidence_milli": conf_milli,
                    "seed_id": int(seed_id),
                }
            finally:
                self._close_if_disk(conn)

    def query_trace_envelopes(
        self,
        namespace: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        prefix_signature: Optional[str] = None,
        task_id: Optional[str] = None,
        seed_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        def_lim = int(self.grid.runtime_constants["trace_query_default_limit"])
        max_lim = int(self.grid.runtime_constants["trace_query_max_limit"])
        top_n = max(1, min(max_lim, int(limit or def_lim)))

        with self._lock:
            conn = self._connect()
            try:
                sql = "SELECT * FROM trace_envelopes WHERE 1=1"
                args: List[Any] = []
                if namespace:
                    sql += " AND namespace = ?"
                    args.append(namespace)
                if reasoning_mode:
                    sql += " AND reasoning_mode = ?"
                    args.append(self._normalize_reasoning_mode(reasoning_mode))
                if prefix_signature:
                    sql += " AND prefix_signature = ?"
                    args.append(prefix_signature)
                if task_id:
                    sql += " AND task_id = ?"
                    args.append(task_id)
                if seed_id is not None:
                    sql += " AND seed_id = ?"
                    args.append(int(seed_id))
                sql += " ORDER BY envelope_id ASC LIMIT ?"
                args.append(top_n)
                rows = conn.execute(sql, tuple(args)).fetchall()
                return [
                    {
                        "envelope_id": int(r["envelope_id"]),
                        "task_id": str(r["task_id"]),
                        "object_id": str(r["object_id"]),
                        "namespace": str(r["namespace"]),
                        "worker_id": str(r["worker_id"]),
                        "input_text": str(r["input_text"]),
                        "target_label": str(r["target_label"]),
                        "prefix_signature": str(r["prefix_signature"]),
                        "reasoning_mode": str(r["reasoning_mode"]),
                        "thinking_tokens": int(r["thinking_tokens"]),
                        "answer_tokens": int(r["answer_tokens"]),
                        "tokens_used": int(r["tokens_used"]),
                        "latency_ms": float(r["latency_ms"]),
                        "activation_norm_digest": str(r["activation_norm_digest"]),
                        "attenuation_confidence_milli": int(r["attenuation_confidence_milli"]),
                        "seed_id": int(r["seed_id"]),
                    }
                    for r in rows
                ]
            finally:
                self._close_if_disk(conn)

    def record_execution_trace(
        self,
        namespace: str,
        input_text: str,
        target_label: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO execution_traces (namespace, input_text, target_label, tokens_used, latency_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (namespace, input_text, target_label, int(tokens_used), float(latency_ms), time.time()),
                )
                conn.commit()
            finally:
                self._close_if_disk(conn)

    def list_execution_traces(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if namespace:
                    rows = conn.execute(
                        "SELECT * FROM execution_traces WHERE namespace = ? ORDER BY trace_id ASC",
                        (namespace,),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM execution_traces ORDER BY trace_id ASC").fetchall()
                return [
                    {
                        "namespace": str(r["namespace"]),
                        "input_text": str(r["input_text"]),
                        "target_label": str(r["target_label"]),
                        "tokens_used": int(r["tokens_used"]),
                        "latency_ms": float(r["latency_ms"]),
                    }
                    for r in rows
                ]
            finally:
                self._close_if_disk(conn)

    def update_subagent_worker(self, worker_id: str, role: str, status: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO subagent_workers (worker_id, role, status, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (worker_id, role, status, time.time()),
                )
                conn.commit()
            finally:
                self._close_if_disk(conn)

    def list_sync_ready_objects(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM task_objects WHERE stage = 'promoted' AND visibility = 'sync_ready' ORDER BY created_at ASC"
                ).fetchall()
                return [
                    {
                        "object_id": str(r["object_id"]),
                        "task_id": str(r["task_id"]),
                        "namespace": str(r["namespace"]),
                        "title": str(r["title"]),
                        "summary_text": str(r["summary_text"]),
                        "maker_id": str(r["maker_id"]),
                        "checker_id": str(r["checker_id"]),
                        "payload": json.loads(str(r["payload_json"])),
                    }
                    for r in rows
                ]
            finally:
                self._close_if_disk(conn)

    def summarize_status(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                status_counts: Dict[str, int] = {}
                for r in conn.execute("SELECT status, COUNT(*) AS c FROM task_leases GROUP BY status").fetchall():
                    status_counts[str(r["status"])] = int(r["c"])
                halted = int(
                    conn.execute("SELECT COUNT(*) FROM task_leases WHERE attenuation_halted = 1").fetchone()[0]
                )
                promoted = int(
                    conn.execute("SELECT COUNT(*) FROM task_objects WHERE stage = 'promoted'").fetchone()[0]
                )
                staged = int(
                    conn.execute("SELECT COUNT(*) FROM task_objects WHERE stage = 'staged'").fetchone()[0]
                )
                traces = int(conn.execute("SELECT COUNT(*) FROM execution_traces").fetchone()[0])
                envelopes = int(conn.execute("SELECT COUNT(*) FROM trace_envelopes").fetchone()[0])
                subagents = int(conn.execute("SELECT COUNT(*) FROM subagent_workers").fetchone()[0])
                campaigns = int(conn.execute("SELECT COUNT(*) FROM grid_campaigns").fetchone()[0])
                return {
                    "grid_id": self.grid.grid_id,
                    "ephemeral_memory": self.ephemeral_memory,
                    "fts5_enabled": self._fts5_enabled,
                    "tasks_by_status": status_counts,
                    "halted_tasks": halted,
                    "promoted_objects": promoted,
                    "staged_objects": staged,
                    "execution_traces": traces,
                    "trace_envelopes": envelopes,
                    "subagent_workers": subagents,
                    "grid_campaigns": campaigns,
                }
            finally:
                self._close_if_disk(conn)
