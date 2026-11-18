# Findings

## Recovery Context
- Planning catchup detected prior Claude session `50ee0691-5d11-4762-841d-2b37787c7e2d` with 103 unsynced messages.
- That work was then carried forward in opencode session `ses_22664831affesQbs7rsPtwMPBi` (slug `quiet-knight`, agent `Sisyphus - Ultraworker`). That session has now been migrated into the current Claude Code session.
- `planning/HANDOFF.md` is the authoritative handoff: GEMM and attention sweeps + XGBoost models are committed and validated; rmsnorm and silu remain blocked because the EPCC Tenstorrent host has `HugePages_Total: 0` after a reboot.

## Continuation Recipe (from HANDOFF.md, restated)
1. Confirm hugepages on remote.
2. `rm -f /tmp/syssim-prof/data/{rmsnorm,silu}_tt_bh_p150b_data.csv` (the existing rows are all `-1.0`).
3. Relaunch `run_blackhole_rmsnorm_silu.sh mini 10 30` via `ts-ssh ... 'cd /tmp/syssim-prof && nohup setsid bash run_blackhole_rmsnorm_silu.sh mini 10 30 > logs/nohup.out 2>&1 < /dev/null & echo PID=$!'`.
4. Local `MIN_ROWS=50 bash scripts/pull_and_train_blackhole.sh`.
5. Local `OMP_NUM_THREADS=1 SYSSIM_HW_NAME=tt_bh_p150b RLSYSIM_MODEL_DIR=data/trained_models PYTHONPATH=. python examples/trace_blackhole.py`.
6. Atomic commits for new CSVs only.

## Dump Artifacts (from prior session, already on disk)
- Claude session jsonl: `planning/claude_dump/session_50ee0691-5d11-4762-841d-2b37787c7e2d/session.jsonl`.
- User Claude skills: `planning/claude_dump/skills/user_claude_skills`.
- Project Claude skills: `planning/claude_dump/skills/project_claude_skills`.
- Claude plugin skills: `planning/claude_dump/skills/claude_plugins`.
- Opencode session export (taken during this migration): full JSON in opencode db; can be re-exported with `opencode export ses_22664831affesQbs7rsPtwMPBi`.

## Update (toolchain LTO ICE — real blocker)
After clearing all the surface blockers (stuck python, stale hugetlbfs files, mesh descriptor, Lustre cache symlink), the actual underlying issue is a deterministic gcc bug:
- **Symptom:** every kernel-JIT build for rmsnorm/silu fails at the link step with
  `lto1: internal compiler error: in lto_read_decls, at lto/lto-common.cc:1981` from `/opt/tenstorrent/sfpi/compiler/bin/.../lto1` (gcc 15.1.0). The same ICE shows up in `/tmp/syssim-prof/logs/sweep.log` from 2026-04-29 — this was pre-existing the moment the post-reboot venv was set up.
- **Why it bites here:** the ttnn 0.65.0 wheel ships its prebuilt firmware objects in `runtime/hw/lib/blackhole/{noc,substitutes,tdma_xmov,tmu-crt0}.o` as **slim-LTO** files (only `.gnu.lto*` bytecode + `__gnu_lto_slim` marker, no machine code). Linking them requires `lto1`, which deterministically ICEs.
- **Bypass attempt — fails:** wrapper at `runtime/sfpi/compiler/bin/riscv-tt-elf-g++` that strips `-flto*`/`-fwpa*` and adds `-Wno-error=array-bounds` got past the ICE on per-file compiles, but the slim-LTO objects can't be linked without LTO. Recompiling them from sources (`tt_metal/hw/firmware/src/tt-1xx/blackhole/noc.c`, `tt_metal/hw/firmware/src/tt-1xx/tdma_xmov.c`, `tt_metal/hw/toolchain/{substitutes.cpp,tmu-crt0.S}`) without LTO produced fat objects — but then `ncrisc.elf` ran `0x8e8 bytes` past its `0x600` byte segment limit. The kernel's link-script regions assume LTO-shrunk code; non-LTO code overflows.
- **Restoration done:** original prebuilt `.o` files restored from `.lto.bak` siblings (LTO marker count back to expected). The `runtime/sfpi` shadow tree was removed so the toolchain path resolution is back to default. The system is back to the post-hugepages-cleanup state (no destructive changes left).

## Real fix is one of these (need sysadmin or upstream)
1. Replace sfpi gcc 15.1.0 with a non-buggy build (Tenstorrent issue tracker: <https://github.com/tenstorrent/sfpi>) — sudo on `nextgenio-amd02`.
2. Downgrade ttnn to a version that ships fat-LTO or non-LTO firmware objects (e.g. try `pip install --force-reinstall ttnn==0.64.x` in `/tmp/tt-syssim-env`). Could be done by `bobbysong` interactively without sudo.
3. Wait for an upstream sfpi/tt-metal fix.

## Update (device-back-online attempt)
- After user reported "device online", `ts-auth` succeeded — `ControlMaster socket live at /Users/cydia2001/.ssh/cm/bobbysong@nextgenio-amd02:22 (persists ~4h)`. So the EPCC routing problem from earlier resolved itself.
- Reachability to `nextgenio-amd02` confirmed: staging files all present (`/tmp/syssim-prof/run_blackhole_rmsnorm_silu.sh`, `/tmp/tt-syssim-env/bin/activate`, the four CSVs incl. the stale rmsnorm/silu pair).
- **Hugepages still NOT set up.** Concrete evidence:
  - `grep HugePages_Total /proc/meminfo` → `0`
  - `sysctl vm.nr_hugepages` → `0`
  - `systemctl status tenstorrent-hugepages.service` → `Loaded: ... enabled; preset: enabled`, `Active: inactive (dead) since Tue 2026-04-28 10:20:54 BST; 1 week 1 day ago` (it ran once successfully but its effect has been cleared).
- `sudo -n /opt/tenstorrent/bin/hugepages-setup.sh` → `sudo: a password is required`. `bobbysong`'s sudoers list grants only DAOS systemctl ops + ipmctl + pmem unmount — nothing covering tenstorrent-hugepages or generic hugepage configuration.
- Resume is therefore still blocked at exactly the handoff's documented step. Need sysadmin (or whoever has root on `nextgenio-amd02`) to run **one** of:
  - `sudo systemctl start tenstorrent-hugepages.service` (preferred — the unit already exists and is enabled).
  - `sudo /opt/tenstorrent/bin/hugepages-setup.sh`.
- Once they confirm `vm.nr_hugepages > 0`, run `/loop` of: discard stale CSVs → relaunch sweep → pull/train → demo → atomic commit (sequence already prepared in `task_plan.md`).

## Continuation Check (this migration)
- First tried `ts-ssh bobbysong@tenstorrent-epcc.uaru-bull.ts.net 'grep HugePages_Total /proc/meminfo'`. Returned `Permission denied (keyboard-interactive)` because the previous ControlMaster had expired and `ssh-askpass` is missing on this macOS host.
- Then ran `ts-auth` (which uses the stored TOTP seed at `~/.config/claude-topo-auth/totp.secret` plus pubkey at `~/.ssh/id_ten`). Pubkey auth at the EPCC gateway succeeded, but the inner ProxyJump step failed:
  ```
  channel 0: open failed: connect failed: Name or service not known
  stdio forwarding failed
  ```
  The gateway can no longer resolve `nextgenio-login1` (the second jump host hard-coded in `~/.local/bin/ts-auth` and `~/.local/bin/ts-ssh` as `TS_JUMP_HOSTS`). Retried twice — same error both times, so it's not a transient TOTP/race condition.
- Net effect: I cannot reach `nextgenio-amd02` at all right now; the hugepage state remains unknown.

## New Blocker (added by this migration)
- **EPCC gateway cannot resolve any internal hostname for `-W` channel-open.** Confirmed pubkey auth completes (`Authenticated using "publickey" with partial success`), TOTP step (driven via `ts-auth` with stored seed) completes, and *then* gateway returns `channel 0: open failed: connect failed: Name or service not known` for the Direct-TCPIP forward.
- User confirmed the intended chain is `ssh -J bobbysong@gateway.epcc.ed.ac.uk,bobbysong@nextgenio-login1 bobbysong@nextgenio-amd02` and that the stored TOTP secret `VVGUYGG5LNMO75L2I6NDEUYQ44` matches what's in `~/.config/claude-topo-auth/totp.secret`.
- Tried inner names `nextgenio-login1`, `nextgenio-login1.epcc.ed.ac.uk`, and even `nextgenio-amd02` directly via single-hop — all yield the same `Name or service not known` from the gateway.
- Refused to enumerate other candidate hostnames against the gateway — EPCC has fail2ban.
- Likely causes (need user / EPCC-side action):
  - Internal DNS on `gateway.epcc.ed.ac.uk` is broken / unset.
  - Pubkey authorized_keys forced-command rejects destinations matching this user's request.
  - EPCC service maintenance window.
- Recommended next step for the user: from a real terminal, run
  `ssh -v -J bobbysong@gateway.epcc.ed.ac.uk,bobbysong@nextgenio-login1 bobbysong@nextgenio-amd02 'echo ok'`
  to confirm the chain still works for them. If it does, dump the verbose log and we can compare against ts-auth's behavior. If it doesn't, escalate to `epcc-support@epcc.ed.ac.uk`.

## Known Landmines (carried over from HANDOFF.md)
- Hugepages do not auto-reserve on this host after reboot.
- TT-Metal SIGBUS during long sweeps — retry-loop launcher catches it.
- Chip 0 doesn't open under default mesh descriptor — `_get_tt_device` falls back 0 → 1.
- macOS arm64 + xgboost segfaults on small data without `OMP_NUM_THREADS=1`.
- EPCC gateway has fail2ban — back off after 2–3 failed `ts-ssh` attempts.
- `ts-ssh` ControlMaster expires after ~2–4 h; re-run `ts-auth` with `TS_TOTP_SECRET=...`.
- Lustre + sfpi gcc 15.1.0 → `lto1` ICE; keep venv and kernel cache off Lustre.
