# omni-fabric-runtime

Environment-adaptive runtime plugin for Hermes Agent (`>=0.20.0`, Manifest v2) that unites durable task fencing, distinct `maker != checker` object reuse, hybrid `FTS5` search, Macro-DAG workflow compilation, parameterized `NodeGrid` + contextual RL canary optimization, progressive 4-rung neural boil-down, a loopback multi-agent swarm proxy, and optional federated Fabric sync in one zero-dependency package.

## Automatic Host Capability Detection & 5 Setup Profiles

At session start (or via `fabric_probe_and_configure` / `/fabric-mode`), the plugin inspects what can run locally inside the current environment and selects one of five operational setups:

| Profile | Local Requirements | Active Local & Federated Capabilities |
|---|---|---|
| `stdlib_airgap` | Python 3 stdlib + SQLite WAL/FTS5 (`0` pip packages, `0` network) | Monotonic `CAS lease_epoch` queue, `maker != checker` object cache, `FTS5` BM25 + RRF search, `3..250` Macro-DAG compiler, `NodeGrid` + RL canary optimizer, and pure-Python `0`-epoch ReLU/ridge micro-router compiler |
| `local_neural` | `+ numpy` & `torch` (`+ safetensors` optional), `0` network | Everything in `stdlib_airgap` plus multi-head `.safetensors` micro-net synthesis, `0`-epoch SVD nullspace weight surgery (`0.0` anchor drift), and Wanda trace channel pruning |
| `local_runner_swarm` | `+ loopback open-model runner` (`127.0.0.1:11434`) | Everything above plus on-demand loopback `/v1/chat/completions` + `/swarm/*` distilling proxy for subagents and external harnesses with right-sized context & JSON schema constraints |
| `hybrid_fabric_sync` | Local SQLite WAL cell `+ upstream_endpoint` | Runs private `local_only` work, subagent swarms, and boil-down locally while syncing `unseen_only` frontier tasks, `sync_ready` promoted objects, and `/mcp` calls with a remote Fabric endpoint |
| `thin_remote_client` | Ephemeral / read-only container `+ upstream_endpoint` | Zero disk footprint (`:memory:` buffer); bridges queue, object, optimization, and MCP operations directly to `upstream_endpoint` |

## Public Extension Surfaces

- **8 Tools:** `fabric_probe_and_configure`, `fabric_task_queue`, `fabric_object_store`, `fabric_compile_workflow_dag`, `fabric_grid_and_rl_optimize`, `fabric_boil_down_and_edit`, `fabric_swarm_proxy_control`, `fabric_federated_sync`
- **10 Lifecycle, Subagent & Kanban Hooks:** `on_session_start`, `pre_llm_call`, `pre_verify`, `post_api_request`, `post_tool_call`, `subagent_start`, `subagent_stop`, `kanban_task_claimed`, `kanban_task_completed`, `on_kanban_worker_stale_claim`
- **1 Request Middleware:** `llm_request` (`<0.2 ms` reasoning-effort calibration + structured JSON `max_tokens` floor)
- **2 Slash Commands:** `/fabric-status`, `/fabric-mode`

## Privacy & Network Disclosure

- Default `upstream_endpoint` is `""` (local-only): `0` external network requests, `0` telemetry, and `local_only` default visibility for staged Task Objects.
- Remote synchronization (`fabric_federated_sync`) runs only when an operator explicitly configures `upstream_endpoint`.
