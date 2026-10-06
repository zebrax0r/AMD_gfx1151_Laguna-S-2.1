# Laguna S 2.1 inference on AMD Strix Halo (gfx1151)

One-click deployment of [Poolside Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)
via **llama.cpp / HIP**, serving an OpenAI-compatible endpoint for
[qwen-code](https://github.com/QwenLM/qwen-code) or any other
OpenAI-compatible client. (qwen-code is the name of the CLI harness this
repo wires up by default — it's a general-purpose, model-agnostic coding
agent, unrelated to which model is actually serving requests. See
"Client harness" below.)

**Laguna S 2.1**: 118B total / 8B active MoE, conventional GQA +
sliding-window attention, July 2026. Adopted 2026-09-19 after an earlier
model on this hardware (a hybrid-attention architecture) ran into a class
of immature-GPU-kernel problems that Laguna's more conventional attention
mechanism avoids entirely — see "Known bugs" below for what's actually
been hit on this exact box.

Target hardware: an AMD Strix Halo APU (GPU arch **gfx1151**, e.g. Ryzen AI
Max/Max+ 300 series), **128GB unified memory** (HP Z2 Mini G1a), no
dedicated VRAM. Until 2026-10-06 this repo was tuned as if the box had
96GB — a 32GB BIOS VRAM carve-out was hiding a quarter of the RAM from
Linux. See "System prep §2".

This is the sibling of [AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B](https://github.com/zebrax0r/AMD_MI210_Bunya_LLM_tools_Qwen3.8-27B),
rebuilt from scratch for a single-APU workstation instead of a SLURM/MI210
cluster. It deliberately does **not** use SGLang or vLLM — see "Why
llama.cpp" below.

## Why llama.cpp (not SGLang, not vLLM)

- **SGLang's official AMD support is CDNA-only** (gfx942/gfx950 —
  MI300/MI350-class datacenter cards). RDNA3.5/gfx1151 enablement exists
  only as an in-progress upstream PR (kernel build support) plus
  unofficial community forks — not something to build production on yet.
- **vLLM doesn't officially support gfx1151 either.** Running it here
  would mean building against ROCm *nightly* plus manual patches (e.g.
  missing `amdsmi` support), with only community toolboxes filling the
  gap. Even where it works, vLLM's real advantage — continuous batching
  across many concurrent requests — doesn't help a single-user,
  single-stream interactive workload like this one; it's built to win at
  high-concurrency serving, not single-session latency.
- **Ollama** (a separate, pre-existing deployment already present on this
  box, port 11434) vendors its own bundled llama.cpp, which lags upstream
  — on Strix Halo specifically it's missing Wave32 flash-attention and
  graphics-queue improvements present in a current standalone llama.cpp
  build, independently measured at ~56% slower on this hardware class.
  Using it here would be a downgrade from a hand-tuned direct build.
- **llama.cpp**, built directly from source with HIP/ROCm flags tuned for
  this exact GPU, is the only stack with both genuine upstream support for
  Laguna's architecture and a mature gfx1151 backend. One fork is used
  elsewhere in this repo's history (Poolside's own, for DFlash speculative
  decoding) and hit a real bug — not adopted, mainline is what's actually
  in production. See "Known bugs" below.

## Known bugs (this exact hardware)

This repo's defaults exist specifically to mitigate these. They are **not
fixed**, just worked around — expect to rebuild against llama.cpp master
periodically as fixes land upstream.

| Issue | Symptom | This repo's mitigation |
|---|---|---|
| Poolside `llama.cpp` fork (branch `laguna`), DFlash speculative decoding | Reproducibly hangs during draft-model memory measurement — `"dflash requires ctx_other to be set"` followed by a hang, not a clean failure. Tested with and without `-fa`, same result. A real bug in that fork's draft-model memory-measurement path, not a flag/config issue on our end (confirmed by reading the fork's own source) | Not adopted. `SPEC_TYPE=""` — no speculative decoding currently. The ~26 tok/s baseline (no spec decoding) is already a strong result, so this isn't a blocker — see "Expected performance" |
| [llama.cpp#28211](https://github.com/ggml-org/llama.cpp/issues/28211) | Upstream report: HIP/gfx1151 can give silently wrong logits on prompts longer than `n_ubatch` (no crash, just bad output) | Verified NOT reproduced on this exact model/build at `UBATCH_SIZE=2048` up to 4088 tokens tested (needle-in-haystack, exact retrieval). If you push well past that, re-verify with the same kind of test before trusting the output |
| [llama.cpp#24437](https://github.com/ggml-org/llama.cpp/issues/24437) | `GGML_HIP_ROCWMMA_FATTN=ON` causes up to -41% prefill throughput on gfx1151 at 8K+ context, worsening with context length | Build compiles this flag **OFF** — a deliberate divergence from some "known-good Strix Halo" community recipes that set it ON |
| [lemonade-sdk#3160](https://github.com/lemonade-sdk/lemonade/issues/3160) | Progressive generation corruption under sustained/concurrent load on ROCm-nightly gfx1151, recovers only on full reload | `PARALLEL=2` since 2026-10-06 for the multi-user gateway, after a 20-round 2-at-a-time soak showed no corruption (was `PARALLEL=1`); `install-watchdog` for automated selftest-gated restarts, `restart` if output degrades, `PARALLEL=1` to go back to single-slot |

## Expected performance

Measured directly on this exact box, Q4_K_M, **no speculative decoding**
(DFlash hits a real bug in Poolside's fork — see above): **~26 tok/s
sustained** across three separate generations (26.85 / 26.25 / 25.86
tok/s), no degradation over long (3000-token) generations. Prompt
processing similarly strong. Output verified correct: math, tool-calling
(`finish_reason: "tool_calls"` with well-formed arguments), code
generation, and a needle-in-haystack retrieval test at 4088 tokens of
context (exact match, confirming no silent corruption at the ubatch
boundary — see "Known bugs" above).

**GPU utilization, confirmed via `amd-smi` during real generation
(2026-09-19)**: 96-100% `GFX_ACTIVITY`, GFX clock averaging 2829MHz
against a 2900MHz ceiling (97.6% of max), edge/GFX temperature 66°C,
zero throttle residency (`PROCHOT`/`SPL`/`SPPT`/`THM_GFX`/`THM_SOC` all
0). Socket power sits around 63-66W — genuinely low, not a sign of
underutilization: the GPU is maxed on clock, fully busy, cool, and
unthrottled the entire time. `GFX_ACTIVITY` is a busy signal (compute
units not idle), not an intensity signal — Laguna activates only 8B of
its 118B total params per token and runs on mature, shared llama.cpp
attention/FFN kernels, so it does genuinely less work per token than a
model with a larger active-parameter count or novel custom kernels would.
There is no free performance being left on the table here: clocks are
already at their ceiling, temperatures are nowhere near thermal
slowdown, and no BIOS/overdrive tweak was found (researched directly —
this box's BIOS has no exposed CPU/GPU clock controls, and Strix Halo's
`amdgpu` driver reports most overdrive/power-cap controls as
"not supported on the given system" even with `ppfeaturemask` unlocked).
The one real, working memory-tuning lever for this hardware class — GTT
allocation boot params — is already applied (see "System prep" below).

**`UBATCH_SIZE=2048`.** GPU compute-buffer memory scales with ubatch size
roughly independent of context length — at `CTX_SIZE=131072` (an earlier,
lower value than the current default), ubatch=8192 used 81.4GB/82GB GTT
(446MB free, dangerously tight), while ubatch=2048 used only 77.2GB
(4.7GB free) for the *same* context. Verified this doesn't risk
silently-wrong-logits behavior (llama.cpp#28211): a 4088-token prompt
(well past 2048) correctly retrieved an exact marker string in a
needle-in-haystack test, no corruption.

**`CTX_SIZE=262144` — found empirically, not assumed.** Retested
2026-10-06 after the BIOS carve-out fix raised the GTT limit from 80GiB
to 116GiB (see "System prep §2"). GTT used at load, UD-Q4_K_M, ubatch
2048:

| `CTX_SIZE` | GTT used | GTT free | Result |
|---|---|---|---|
| 163840 | 77.0GiB | 39.0GiB | loads (the old default) |
| 262144 | 81.9GiB | 34.1GiB | **loads; exact needle retrieval at 247K tokens** |
| 524288 | 94.9GiB | 21.1GiB | loads; not correctness-tested |
| 786432 | 107.9GiB | 8.1GiB | loads, but system RAM nearly exhausted |
| 1048576 | — | — | OOM allocating the KV cache |

The 262144 needle test: a 247,003-token prompt with one marker line 60%
of the way in, retrieved exactly (`finish_reason: stop`). It also shows
the practical limit is time, not memory: prefill took 49m40s (83 tok/s
average, falling steadily as the context fills) and generation at that
depth ran at ~3.9 tok/s, against ~26 tok/s on a short context. A coding
session that grows to that size gradually is fine; pasting 250K tokens
in one go is not. That's why the default stops at 262144 even though
512K fits: a bigger window costs memory headroom and buys context you'd
rarely want to wait for. (Before the fix, 262144 OOMed and 163840 was
the ceiling, verified with a ~130K-token needle in 14m18s.)
`CLIENT_CTX_SIZE` (196,608) and `QWEN_SESSION_TOKEN_LIMIT` (229,376)
keep the same 75% / 87.5% ratios to `CTX_SIZE` as before.

### If you're using `qwen-code`

Its full agentic mode sends a large system/tool-definition prompt
(measured ~8,000-20,000+ tokens on this setup, depending on loaded
skills/tools) and can make several sequential LLM round-trips per user
turn (tool calls, reasoning steps), each reprocessing a large chunk of
that context with only partial cache reuse. The model also reasons at
length before answering (verbose `reasoning_content`, often consuming a
large chunk of the token budget before reaching final `content`) — this
is a model-behavior trait, not a bug, but it means real interactive
requests take longer than raw tok/s alone suggests. Watch
`./serve-laguna.sh status` or tail `logs/laguna.log` to confirm it's
actively generating rather than stalled. `qwen --bare` skips
auto-discovery/tool loading for a lighter, faster session if you don't
need the full agentic toolset.

**`wire-qwen-code` caps a single turn's output** at
`QWEN_CODE_MAX_OUTPUT_TOKENS` (default 10,000) via
`generationConfig.samplingParams.max_tokens`. Without this, `qwen-code`
defaults to the model's *declared* output limit — effectively unbounded.
Confirmed directly: a real turn generated 12,000+ tokens at a healthy,
stable pace (no stall or corruption in the server logs) and was still
going when `qwen-code`'s own 15-minute stream-lifetime cap
(`QWEN_STREAM_MAX_LIFETIME_MS`, default 900000ms) killed the connection,
discarding the entire in-flight response. Hitting the token cap instead
gives a clean `finish_reason: length` you can ask it to continue from,
rather than losing the whole response to a timeout.

### Client harness: qwen-code vs. alternatives

`qwen-code` (QwenLM's fork of Gemini CLI) is what this repo wires up by
default via `wire-qwen-code`, and it works correctly with Laguna —
tool-calling, reasoning/content split, and code generation all confirmed
via `--jinja`. But it's explicitly positioned as the "native" agent for
Qwen-family models specifically, not model-agnostic by design. An
independent comparison running the same 12 coding tasks through local
CLI agents found qwen-code completing 10/12 (one partial, one failed)
vs. Aider 12/12 and OpenCode 11/12 — directionally consistent with
qwen-code carrying assumptions that don't generalize as cleanly to a
non-Qwen backend. **OpenCode** (provider-agnostic, 75+ backends including
any OpenAI-compatible endpoint) is worth trying as a side-by-side
alternative against this same server — no server-side changes needed,
just point it at `http://<SERVER_HOST>:8000/v1`. Not yet set up in this
repo; noted here as a live follow-up, not a recommendation to switch yet.

## Other models

Qwen3.8-Flash-Next was tested on this box on 2026-10-06 (fits at
UD-IQ4_XS; 32–38 tok/s with its MTP draft head; exact needle retrieval at
241K tokens) but needs a newer llama.cpp than production uses. Halogen
(a closed-source engine for the same model) was looked at but not run.
Details, commands and a side-by-side with Laguna:
[docs/ALTERNATIVE_MODELS.md](docs/ALTERNATIVE_MODELS.md).

## Quickstart

```bash
./serve-laguna.sh init          # dirs + API key
./serve-laguna.sh probe         # GPU/ROCm/GTT preflight — read the warnings
./serve-laguna.sh build         # clone+build llama.cpp from latest master
./serve-laguna.sh download      # fetch ~73.1GB Q4_K_M (3 shards, no mmproj)
./serve-laguna.sh check         # bounded smoke-load test
./serve-laguna.sh serve         # launch + wait for /health, prints connection banner
./serve-laguna.sh wire-qwen-code   # point qwen-code CLI at this server
```

## Subcommands

See `./serve-laguna.sh` with no args, or `docs/TROUBLESHOOTING.md`.

## Configuration

Copy `laguna-env.example` to `laguna.env` (done automatically on first run)
and edit. Every default is documented inline with which bug or measurement
it's based on.

## System prep (manual — do before `build`)

### 1. ROCm

This machine already has a working ROCm install via AMD's dedicated
Strix-Halo/workstation apt channel, confirmed present:

```
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
```
(`/etc/apt/sources.list.d/amdrocm-stable.sources`, keyring at
`/etc/apt/keyrings/amdrocm.gpg`). It installs gfx1151-specific packages
(`amdrocm-core10.0-gfx1151`, `amdrocm-blas10.0-gfx1151`, etc., apt-versioned
`10.0.0-4`) directly targeting `ubuntu2604` — i.e. Ubuntu 26.04 is a first-
class target on this channel, unlike the general `repo.radeon.com` apt repo
used by older guides. `hipconfig --version` reports the underlying HIP
runtime as `7.15.x` — consistent with ROCm 7.14+ having native gfx1151
support (no `HSA_OVERRIDE_GFX_VERSION` needed). `rocminfo` confirms the
`gfx1151` agent (`AMD Radeon 8060S Graphics`, 40 CUs) is visible.

If you're setting this up on a fresh machine and this repo isn't already
configured, check whether AMD's install docs point you at
`stable.repo.amd.com` for your distro before falling back to
`repo.radeon.com`'s general apt repo (which may lack a component for a very
new Ubuntu release — pin to `noble` in that case) or a
[TheRock](https://github.com/ROCm/TheRock) nightly tarball extracted to a
side-by-side prefix with `ROCM_PATH` set in `laguna.env` accordingly.

**Do not run `amdgpu-install` / install `amdgpu-dkms`.** The `amdgpu`
kernel module is already loaded (in-tree, mainline). Installing the DKMS
driver package would fight with it — install ROCm's userspace packages
only, as was already done here.

### 2. GTT / kernel memory tuning

**Applied on this box 2026-10-06** (replacing an 80GiB setting from
2026-09-19). Two separate settings decide how much memory the GPU gets,
and both matter:

**a. BIOS: shrink the VRAM carve-out to 512MB.** The BIOS "UMA Frame
Buffer Size" setting (HP: F10 at boot) permanently reserves RAM as
"VRAM" before Linux boots. This box shipped with **32GB** reserved, so
Linux saw only ~92GiB of its 128GB — which is why this repo originally
described it as a 96GB machine. llama.cpp allocates from GTT, not that
carve-out, so it was mostly wasted. Set it to the 512MB minimum. Check
with `cat /sys/class/drm/card1/device/mem_info_vram_total` (should read
536870912) and `grep MemTotal /proc/meminfo` (~123GiB). `probe` warns if
the carve-out is over 1GB.

**b. Kernel: raise the GTT limit.** The GPU claims system RAM
dynamically via GTT; the driver's unconfigured default is roughly 50%
of RAM.

```bash
sudoedit /etc/default/grub
# Add to GRUB_CMDLINE_LINUX_DEFAULT (keep existing params):
#   amdgpu.gttsize=118784 ttm.pages_limit=30408704
sudo update-grub
sudo reboot
```

This lets the GPU use up to ~116GiB of the ~123GiB Linux sees, leaving
~7GiB minimum for the OS even at full GPU load (more in practice — the
limit is a cap, not a reservation). `amdgpu.gttsize` is in MiB and is
deprecated in favour of `ttm.pages_limit`, but `probe` still looks for
it; `ttm.pages_limit` is `GiB × 262144` (4KiB pages). Recompute both
together if you change it. Some 128GB Strix Halo guides use 124GiB
(`126976` / `32505856`); 116GiB was chosen here to keep more OS
headroom. The order of a and b doesn't matter — the GTT value is a
ceiling, so setting it higher than current RAM is harmless at boot.

This is the real, working memory-tuning lever on this hardware class —
by contrast, direct research into GPU clock/power overclocking on
gfx1151 and this specific machine's BIOS turned up nothing usable: the
BIOS has no exposed CPU/GPU clock/power settings (only third-party,
unofficial "BIOS mod" routes exist, not attempted here), and even with
the `amdgpu.ppfeaturemask` OverDrive-unlock boot param set, this chip's
driver reports `power_cap`/clock-overdrive controls as unsupported. See
"Expected performance" above for the live telemetry confirming there's
no throttling or clock headroom being left unused anyway.

## qwen-code CLI wiring

```bash
./serve-laguna.sh wire-qwen-code
```

Writes `~/.qwen/.env` (or a project-local `.qwen/.env` if you set
`QWEN_ENV_TARGET=project` / `QWEN_PROJECT_DIR` in `laguna.env`) with
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`. Alternatively, run
`qwen` and use the interactive `/auth` → Custom Provider flow with the same
three values (see the command's output for exact values, redacted here).

**Important**: `qwen-code`'s `~/.qwen/settings.json` (`security.auth`) takes
precedence over `.env` if it already names a provider — writing `.env`
alone can silently have no effect. `wire-qwen-code` patches
`settings.json` too: it backs up the existing file, adds/updates a provider
entry keyed by a stable id (`laguna-gfx1151`) without deleting any other
providers already configured (e.g. a separate Ollama-based setup — this
box already had one, on port 11434, discovered while building this repo),
and sets that entry as the default (`security.auth`). It also removes a
legacy `qwen38-gfx1151` provider entry this repo used before it was
renamed. Safe to rerun.

It also writes `generationConfig.contextWindowSize: $CLIENT_CTX_SIZE` into
that provider entry — note `CLIENT_CTX_SIZE`, not `CTX_SIZE`. Without this
field at all, `qwen-code` assumes the model's *advertised* native context
(Laguna's is 1,048,576) rather than what this server is actually
configured to serve, and paces its own auto-compaction against that wrong
number — meaning it keeps growing the conversation until it hits a hard
`400 ... exceeds the available context size` error instead of compacting
proactively. Reporting the *exact* real `CTX_SIZE` isn't enough either —
confirmed directly on this repo's earlier deployment: `qwen-code`'s own
token counts are estimates, and a single large step can jump past its
compaction trigger before compaction runs. `CLIENT_CTX_SIZE` is
deliberately lower than `CTX_SIZE` (currently 196608 vs. 262144) — the gap
is the safety margin.

**Even that margin isn't fully reliable on its own** — confirmed directly:
a 16,384-token margin was completely consumed in one turn. `qwen-code`'s
proactive compaction doesn't reliably trigger before a single
large-enough addition. So `wire-qwen-code` also sets three more (global,
not provider-specific) `settings.json` fields: `tools.truncateToolOutputThreshold`
and `tools.truncateToolOutputLines` (lowered from qwen-code's stock 25,000
chars / 1,000 lines to 8,000 / 300 — this bounds how much a single tool
call can add; the stock default still let a *batch* of several fanned-out
tool calls add up to more than the whole `CLIENT_CTX_SIZE` margin), and
`model.sessionTokenLimit` (143,360 — a deterministic backstop that blocks
sending the next message outright once the recorded prompt is already over
budget, independent of token-count estimation entirely). These require a
fresh `qwen` session to take effect, not just a Ctrl+Y retry.

**Ordering matters for `sessionTokenLimit`** — it must sit *above*
`CLIENT_CTX_SIZE`, not below it. Confirmed by getting this wrong first on
an earlier deployment: setting it below `CLIENT_CTX_SIZE` fired the hard
block *before* proactive compaction ever got a chance to run, so every
session hit a wall requiring manual `/compress`/`/clear` instead of
compacting quietly. Current value (143,360) sits between `CLIENT_CTX_SIZE`
(122,880, soft compaction trigger) and the real `CTX_SIZE` (163,840, hard
server limit), functioning as a true last resort. If you see
`Session token limit exceeded` from `qwen-code` itself, that's this
backstop working as intended, not a bug — `/compress` or `/clear` as it
suggests.

**This backstop is a value to revisit, not a fact to accept.** Confirmed
directly 2026-09-19: a real session hit `Session token limit exceeded:
115,063 tokens > 114,688 limit` — a genuinely large session, not a bug —
but the real `CTX_SIZE` at the time was 131,072, meaning 16,009 tokens of
already-safe server capacity were sitting unused behind an artificial
backstop that had simply never been revisited since it was first set. If
you hit this error, before just starting a new session, check whether
`QWEN_SESSION_TOKEN_LIMIT` actually has room to move up toward the real
`CTX_SIZE` — it often does.

We deliberately did **not** reach for llama-server's `--context-shift`
here, even though it's designed for exactly this (discard old context
instead of hard-failing). Its discard mechanism needs to partially
truncate the KV cache — safe for Laguna's conventional attention, but this
repo treats that as a case-by-case risk to re-evaluate per architecture
rather than a default to lean on, since some hybrid/recurrent
architectures elsewhere can't have their memory partially truncated
safely at all. The client-side layered defenses above don't depend on
that assumption either way.

If you change `CTX_SIZE`, `CLIENT_CTX_SIZE`, `QWEN_TOOL_OUTPUT_THRESHOLD`,
`QWEN_TOOL_OUTPUT_LINES`, or `QWEN_SESSION_TOKEN_LIMIT`, rerun
`wire-qwen-code` (on every machine running `qwen-code` against this
server, laptops included) so the client's config stays in sync.

## Multi-user gateway (LiteLLM)

Added 2026-10-06. Each person gets their own API key; you can see how
many tokens each one used, revoke anyone instantly, and add rate limits
or budgets later — without anyone sharing the single llama-server key.

```
 users (own key) ──> LiteLLM :4000 ──(internal key)──> llama-server 127.0.0.1:8000
                       └─ Postgres (keys + per-request token counts)
```

- **Network:** port 4000 is opened (ufw) to the LAN (`10.49.56.0/23`) and
  the Cisco VPN pool (`172.18.0.0/16`) only. llama-server is bound to
  `127.0.0.1` (`HOST` in `laguna.env`), so nothing reaches it except the
  gateway. Plain HTTP: keys cross the network unencrypted, which is
  acceptable inside the VPN but put TLS (e.g. Caddy) in front before
  exposing it anywhere less trusted. In practice the Cisco VPN only lets
  port 22 through to this box (tested 2026-10-06: 4000 and every other
  port tried was dropped upstream), so remote users connect over an SSH
  tunnel instead — see [Connecting over the VPN](#connecting-over-the-vpn-ssh-tunnel).
- **Model name:** users always ask for `local-coder`. llama-server serves
  one model at a time and ignores the requested name, so switching
  Laguna ⇄ Flash-Next changes nothing for users.
- **Concurrency:** `PARALLEL=2`, `CTX_SIZE=524288` → two users at once,
  262,144 tokens each. Measured ~20 tok/s per user with both generating
  (~26 alone), ~37 tok/s total; a 20-round 2-at-a-time soak produced no
  wrong or garbled output (see `laguna-env.example`). Each key defaults to
  3 requests at a time — IDE clients send background requests alongside
  the main one, and a limit of 1 made them fail with 429 `max_parallel_requests`
  while the server sat idle. Requests beyond the 2 slots queue in
  llama-server rather than failing; a long prefill can hold a slot for
  many minutes. One busy user can occupy both slots — tighten per user with
  `./gateway.sh set-limits <name> --parallel 1` if that becomes a problem.
- **Privacy:** LiteLLM logs token counts per request, not prompts or
  responses (`store_prompts_in_spend_logs: false`, verified).
- **Secrets:** `.secrets/gateway.env` (gitignored, mode 600) holds the
  admin key (`LITELLM_MASTER_KEY` — never hand it out), the Postgres
  password and the salt key. This repo is public; never commit it.

### Setup (once)

```bash
./gateway.sh init                      # LiteLLM venv + Prisma client + secrets
sudo ./gateway/setup-root.sh           # Postgres, ufw rules, systemd linger
sudo ./gateway/setup-tunnel-root.sh    # tunnel-only SSH account for VPN users
./serve-laguna.sh serve                # llama-server (now loopback-only, 2 slots)
./gateway.sh serve                     # LiteLLM on :4000
./gateway.sh install-services          # start both at boot (systemd --user)
./serve-laguna.sh install-watchdog     # selftest every 2h, restart only on failure
```

`setup-root.sh` takes `LAN_SUBNET=` / `VPN_SUBNET=` overrides if your
networks differ.

### Day to day

```bash
./gateway.sh add-user alice                    # prints base URL + key + model, once
./gateway.sh add-user bob --tpm 100000 --rpm 30 --budget 5
./gateway.sh list-users                        # names, limits, key hints
./gateway.sh set-limits bob --parallel 2 --tpm none   # change limits in place
./gateway.sh usage 7                           # tokens per user, last 7 days
./gateway.sh revoke-user alice                 # immediate 401 for that key
./gateway.sh status

sudo ./gateway/tunnel-key.sh add alice 'ssh-ed25519 AAAA...'   # let alice tunnel in
sudo ./gateway/tunnel-key.sh list
sudo ./gateway/tunnel-key.sh remove alice
```

Use the same name for a person in both places. Fully revoking someone is
`tunnel-key.sh remove <name>` plus `gateway.sh revoke-user <name>`.

Budgets: costs are set to 0 in `gateway/litellm-config.yaml`, so
`--budget` does nothing until you give tokens a nominal price there (e.g.
`0.000001` = "$1" per million tokens), after which `--budget 5` means
roughly 5M tokens. Usage lags ~1 minute (LiteLLM writes logs in batches).

Switching the served model: `systemctl --user stop llm-gateway llm-server`,
`./gateway.sh install-services flashnext.env`, then
`systemctl --user start llm-gateway`. Flash-Next is still configured for
one slot (`PARALLEL=1` in `flashnext.env`) — two slots haven't been tested
on it, and system RAM is tighter.

### What a user needs

Give them the three lines `add-user` prints: base URL
(`http://10.49.56.223:4000/v1`), their key, and model `local-coder`. Any
OpenAI-compatible client works. For `qwen-code`, from a clone of this
repo on their machine:

```bash
SERVER_HOST=10.49.56.223 CLIENT_PORT=4000 CLIENT_API_KEY=sk-... \
  CLIENT_MODEL_NAME=local-coder ./serve-laguna.sh wire-qwen-code
```

Allow generous output limits: Laguna sometimes reasons at length before
answering (its reasoning arrives in `reasoning_content`), so a client
capped at a few hundred tokens can get an empty answer with
`finish_reason: length`. `wire-qwen-code` sets 10,000.

You (on the server box) should use a key from `add-user` too, rather than
the direct llama-server key, so your own usage shows up in `usage`.

### Connecting over the VPN (SSH tunnel)

Port 4000 isn't reachable through the Cisco VPN, so VPN users forward a
local port over SSH. `setup-tunnel-root.sh` creates one shared account,
`llm-tunnel`, that can do nothing except forward to `127.0.0.1:4000`: no
password, shell, TTY, or any other forwarding. Each user is one line in
`/etc/ssh/llm-tunnel/authorized_keys`, managed with `tunnel-key.sh`. Their
LiteLLM key still controls and meters what they can do once connected.

To onboard someone:

1. They send their SSH **public** key (`~/.ssh/id_ed25519.pub`; make one
   with `ssh-keygen -t ed25519` if needed).
2. On the server: `./gateway.sh add-user <name>` and
   `sudo ./gateway/tunnel-key.sh add <name> '<their public key>'`.
3. On their machine, open the tunnel and leave it running (it prints
   nothing while it's working):

   ```bash
   ssh -N -L 4000:127.0.0.1:4000 llm-tunnel@10.49.56.223
   ```

4. Point the client at the tunnel's local end, **not** the server IP:
   base URL `http://127.0.0.1:4000/v1`. For `qwen-code`:

   ```bash
   SERVER_HOST=127.0.0.1 CLIENT_PORT=4000 CLIENT_API_KEY=sk-... \
     CLIENT_MODEL_NAME=local-coder ./serve-laguna.sh wire-qwen-code
   ```

Removing a tunnel key doesn't drop tunnels that are already open; `sudo
pkill -u llm-tunnel sshd` cuts them all (every tunnel user's session).

## Client-only setup (a laptop or other machine that doesn't run the server)

This repo is meant to be cloned on both the machine that runs the server
and any client machine that just runs `qwen-code` against it — that keeps
`CTX_SIZE`/`SERVER_HOST`/etc. in sync via `git pull` instead of manually
re-copying config by hand.

On the client machine:

```bash
git clone <this-repo-url>
cd <repo-dir>
cp laguna-env.example laguna.env
```

Edit `laguna.env`: set `SERVER_HOST` to the server machine's LAN IP or
hostname (leave everything else — `PORT`, `CTX_SIZE`, etc. — matching the
server's actual config). **`git pull` only updates the tracked
`laguna-env.example` template — it does NOT touch your local, gitignored
`laguna.env`.** After every pull, diff your `laguna.env` against the
current `laguna-env.example` and manually carry over anything that
changed (`SERVER_HOST` is the one value that should differ between the
two — everything else should match). Forgetting this is a real, confirmed
failure mode: it silently leaves `SERVER_HOST` unset/stale, `wire-qwen-code`
then defaults the base URL to `127.0.0.1`, and `qwen-code` fails with
`connect ECONNREFUSED 127.0.0.1:8000` even though the actual server is
healthy — easy to misdiagnose as a server-side problem when it's purely a
client-config gap.

Then fetch a copy of the API key — **don't paste it through a chat session
with an AI assistant to relay it**, that happened twice while building this
repo and both times the key had to be rotated as a result:

```bash
mkdir -p .secrets
ssh <user>@<server-host> cat <path-to-repo-on-server>/.secrets/api_key > .secrets/api_key
chmod 600 .secrets/api_key
```

Do **not** run `./serve-laguna.sh init` on the client — that generates a
*new*, different key, which won't match what the server actually accepts.
Only `wire-qwen-code` needs to run here:

```bash
./serve-laguna.sh wire-qwen-code
```

No `build`/`download` needed on the client — `wire-qwen-code` only reads
config and the key file, it doesn't touch the model or the llama.cpp
checkout.

## Troubleshooting

See `docs/TROUBLESHOOTING.md`.
