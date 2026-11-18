# Task Plan

## Goal
Continue the Tenstorrent Blackhole rmsnorm + silu profiling sweep that was
inherited from Claude session `50ee0691-5d11-4762-841d-2b37787c7e2d` and
opencode session `ses_22664831affesQbs7rsPtwMPBi`. Resume from
`planning/HANDOFF.md` step 1 once the EPCC remote's 2 MB hugepage
reservation is restored.

## Migration Source
- Prior Claude session id: `50ee0691-5d11-4762-841d-2b37787c7e2d` (transcript
  dumped at `planning/claude_dump/session_50ee0691-5d11-4762-841d-2b37787c7e2d/session.jsonl`).
- Prior opencode session id: `ses_22664831affesQbs7rsPtwMPBi` (the session
  this task is being migrated *out of*; final state was: planning files seeded,
  hugepages still 0).

## Acceptance Criteria
- Hugepages on `nextgenio-amd02` are > 0 (`grep HugePages_Total /proc/meminfo`).
- Stale all-`-1.0` rmsnorm/silu CSVs on the remote are removed.
- `run_blackhole_rmsnorm_silu.sh mini 10 30` finishes and produces non-empty
  rmsnorm/silu CSVs.
- `MIN_ROWS=50 bash scripts/pull_and_train_blackhole.sh` completes and writes
  `data/trained_models/{rmsnorm,silu}_tt_bh_p150b_xgb.pth` locally.
- `examples/trace_blackhole.py` reports
  `loaded models: ['gemm', 'attn', 'rmsnorm', 'silu']`.
- New CSVs committed atomically (no AI-author markers) on
  `yanwei/custom_platform_support`.

## Phases
1. Recover prior session + skill artifacts. Status: complete (already on disk
   under `planning/claude_dump/`).
2. Re-authenticate `ts-ssh` and confirm hugepages on
   `nextgenio-amd02`. Status: **blocked — new failure mode.** `ts-auth` ran
   non-interactively (pubkey + stored TOTP), pubkey auth at the EPCC gateway
   succeeded, but the gateway cannot resolve the inner jump host
   `nextgenio-login1` (`channel 0: open failed: connect failed: Name or
   service not known`). The hostname has likely changed upstream. User needs
   to supply the new inner jump-host name (and ideally update
   `~/.local/bin/ts-{ssh,auth}` defaults), then retry. **Do not** brute-force
   candidate hostnames against the gateway — EPCC has fail2ban.
3. Discard the stale rmsnorm/silu CSVs and relaunch
   `run_blackhole_rmsnorm_silu.sh`. Status: **blocked on toolchain.** Hugepage and mesh-descriptor blockers are gone, but every kernel JIT now hits a deterministic `lto1: internal compiler error` in sfpi gcc 15.1.0 because ttnn 0.65.0 ships slim-LTO firmware objects. Wrapper-based LTO bypass attempt has been reverted (originals restored from `.lto.bak`). Real fix needs either a working sfpi gcc (sysadmin) or downgrading ttnn to a release with fat-LTO/non-LTO firmware objects (`pip install --force-reinstall ttnn==0.64.x`).
4. Pull/train/demo locally and commit new CSVs. Status: pending phase 3.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `$HOME/.config/claude` missing | Checked alternate config path | Used `$HOME/.claude` as the Claude source tree (carried over from prior session). |
| Remote `HugePages_Total: 0` | `tt-smi -r 0 1` and direct check | Stopped before relaunch; sysadmin/user must run `sudo /opt/tenstorrent/bin/hugepages-setup.sh`. |
| `ts-ssh` Permission denied (keyboard-interactive) | Ran one-shot `ts-ssh ... 'grep HugePages_Total ...'` | ControlMaster lapsed; ran `ts-auth` instead (pubkey + stored TOTP). |
| `channel 0: open failed: connect failed: Name or service not known` | `ts-auth` (twice, plain + `TS_AUTH_DEBUG=1`) | Gateway cannot resolve inner hop `nextgenio-login1`. Need user to provide updated jump-host name; refused to brute-force candidates against gateway because of fail2ban. |
