# Alternative models on this box

Laguna S 2.1 is still the production default. This page records what
else was tested after the BIOS carve-out fix (README "System prep §2")
raised the GPU's usable memory from ~80GiB to ~116GiB, and what you'd
need to switch.

## Qwen3.8-Flash-Next (tested 2026-10-06)

125B total / 6B active MoE, plus a 51B-parameter n-gram embedding table
and a 4B MTP draft head. Hybrid attention (Gated DeltaNet linear
attention + sparse full attention every 4th layer) — the same broad
family that caused kernel trouble for an earlier model on this box, so
it was tested specifically for that. GGUF architecture name: `qwen4exp`.

**Files** (from `unsloth/Qwen3.8-Flash-Next-GGUF`, ~98GB total):

- `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`
  (93.7GB). The largest quant that fits; UD-Q4_K_XL (111GB) leaves too
  little room for KV cache and the OS.
- `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.1GB, self-contained head).

**Needs a newer llama.cpp than production.** The production build
(`911f6cd`, 2026-09-18) can load `qwen4exp` but predates its MTP support
(`c061df1`, llama.cpp#29761, 2026-10-01). Tested with master `50569eb`,
built with the same flags as `cmd_build` into a separate git worktree
(`vendor/llama.cpp-master`) so production wasn't touched. Unsloth's MTP
README says mainline can't use these heads — that was true when it was
written and is no longer.

**Running it** (config in `flashnext.env`; same port, API key and
qwen-code limits as Laguna):

```bash
./serve-laguna.sh stop                                    # only one model fits at a time
ENV_FILE=flashnext.env ./serve-laguna.sh serve
ENV_FILE=flashnext.env ./serve-laguna.sh wire-qwen-code   # then start a fresh `qwen`
```

Switch back with `./serve-laguna.sh stop && ./serve-laguna.sh serve &&
./serve-laguna.sh wire-qwen-code` (no `ENV_FILE` = `laguna.env`). The
equivalent raw command:

```bash
llama-server --model models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --ctx-size 262144 --ubatch-size 2048 --batch-size 2048 --parallel 1 \
  -ngl 999 --jinja --flash-attn auto \
  --spec-type draft-mtp --spec-draft-model models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
```

**Results at `--ctx-size 262144`** (its native maximum):

| | no MTP | with MTP |
|---|---|---|
| GTT used at load | 71.8GiB | 78.4GiB |
| System RAM available after load | ~19GiB | ~8–10GiB |
| Generation, short context | 22 tok/s | 32–38 tok/s (draft acceptance 57–72%) |
| Math / code / tool-call checks | all correct | all correct |

GTT use is well under the 94GB file size — part of the model (most
likely the n-gram embedding table, which is only ever looked up, never
multiplied) stays in ordinary system RAM. That's why "RAM available"
is the tighter number: with MTP on, only ~8GiB is left for everything
else on the box. Don't run anything big alongside it.

**Long context** — needle-in-haystack, 240,817-token prompt, marker
placed 60% of the way in: exact retrieval (`ZEBRA-7741-QUARTZ`,
`finish_reason: stop`). Prefill took 26m02s (154 tok/s average);
generation at that depth dropped to ~15 tok/s. No corruption, no
kernel errors — the hybrid-attention problems seen with the earlier
model did not show up here.

**Comparison with Laguna S 2.1 (UD-Q4_K_M, production build):**

| | Laguna S 2.1 | Flash-Next + MTP |
|---|---|---|
| Generation, short context | ~26 tok/s | 32–38 tok/s |
| Needle at ~245K tokens | exact retrieval | exact retrieval |
| Prefill for that prompt | 49m40s (83 tok/s avg) | 26m02s (154 tok/s avg) |
| Generation at that depth | ~3.9 tok/s | ~15 tok/s |
| GTT used at 256K context | 81.9GiB | 78.4GiB |
| Max context that loads | 768K (tight); 1M OOMs | 256K (model's own limit) |
| llama.cpp build | production | needs master |
| Track record on this box | weeks in production | one afternoon of testing |

Flash-Next is faster on both prefill and generation, and the gap widens
as context grows — at ~245K tokens it prefills ~1.9× faster and
generates ~4× faster, which is what hybrid linear attention is for. The reasons not to
switch production yet are maturity, not measurements: it needs a
llama.cpp master build that hasn't been soaked, and nothing here has
run it in a real multi-hour coding session. A sensible next step is to
run it under `qwen-code` for a few real sessions on the master build
before changing defaults.

## Halogen (not tested)

Halogen is a closed-source inference engine (shipped as a Docker image),
not a model. It currently runs only Qwen3.8-Flash-Next, from its own
`.hgn` checkpoint format (~118GB download, separate from the GGUFs
above). Community reports on 128GB Strix Halo boxes: ~87GiB at 262K
context, ~40–43 tok/s decode, 1,200+ tok/s prefill — the prefill figure
would be ~8× what llama.cpp managed here at long context, which is the
main reason to try it. Its setup guide requires the same 512MB BIOS
carve-out this repo now uses and kernel 6.18.4+ with
`CONFIG_HSA_AMD_SVM` (this box runs 7.0). Not tried because it means
another 118GB download and running closed-source code; worth it if
long-prompt prefill speed becomes the bottleneck.
