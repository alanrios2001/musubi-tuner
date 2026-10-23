# LoRA-GA implementation review — 2026-10-22

## References and mathematical findings

Reviewed the [LoRA-GA paper](https://arxiv.org/html/2407.05000v2), the authors'
[factor initialization](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py#L159),
[gradient collection](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/utils/lora_ga_utils/lora_ga_utils.py#L37),
and [PEFT's adapter conversion](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py#L1002).
PEFT is a reference, not a new runtime dependency.

The port matches the authors' implementation for:

- FP32 randomized SVD with `q=min(4*r,min(shape))`, four power iterations.
- All three supported disjoint singular-vector selections. `ArB2r` uses the
  first `r` right singular vectors for A and the next `r` left vectors for B.
- `stable` factor multiplier `out_features**0.25 / sqrt(gamma)`, gamma 16 by
  default, and the supported `unit` and `gd` alternatives. This follows the
  reference source and Algorithm 1; the paper uses differing gamma notation
  in its prose, so changing the divisor to gamma would not match the code.
- Arithmetic averaging of per-batch loss gradients, without optimizer updates.
  The streamed estimator accumulates BF16-operand products in FP32 instead of
  repeatedly rounding the accumulator to BF16.
- Compensation `Wc=BF16(FP32(W)-s*B0@A0)` followed by the existing FP8 quantizer.
  The effective scaling is Musubi's `s=alpha/rank`. The paper also discusses
  rsLoRA's `alpha/sqrt(rank)`; enabling rsLoRA implicitly would change the requested
  Musubi architecture. The reference factor code supports the existing scaling.
- Export `[A;A0]`, `[B,-B0]`, doubled rank and doubled alpha, preserving `s` and
  the update relative to the original model. Raw Accelerate state stays rank r.

Executing the extracted official initialization method against the port passed
all nine supported direction/scale combinations, compensation, and three-batch
gradient collection. A separate dense-SVD oracle and autograd test confirms that
the first-order adapter update follows the best rank-2r gradient projection
on a small matrix where the randomized SVD spans its full dimension.

## Confirmed corrections

1. **SVD precision isolation.** A float32 input did not prevent an outer autocast
   context from reducing intermediate products to BF16. Reproduction changed
   CPU factors by up to 0.00766 and caused CUDA QR to fail with
   `"geqrf_cuda" not implemented for 'BFloat16'`. CUDA TF32 could also reduce the
   precision of products when explicitly enabled by the caller. Initialization
   now disables both during SVD and restores their original settings on success
   or failure. The ordinary training path is unchanged. The normal initialization
   caller already ran outside autocast; this is not evidence that the user's
   existing cache was produced under the failing context.
2. **Cached metric attribution.** Reusing initialization previously printed
   historical gradient/SVD times alongside a new total time. Current estimation
   times and `estimated_batches_this_run` now report zero on reuse. Historical
   timing, peak VRAM and storage counters move to `loraga/cache_origin/*`, including
   when a resumed state's sidecar itself originated from a cached run. The
   initialization tensor format and compatibility identity are unchanged.

## Interpreting the supplied run

The cached run recorded BF16 output RMSE 0.018299 (relative RMS error 1.458%),
ordinary FP8 RMSE 0.021513, and compensated FP8 plus initial adapter RMSE
0.032034, giving an output error ratio of 1.489. Target-weight RMSE increased by
only 5.31%; this does not predict the increase in model-output error or image quality.

BF16 storage of the compensated matrix and separate BF16 matrix products prevent
exact cancellation in floating-point arithmetic. Independently quantizing the
original and compensated backbones introduces different errors, which PEFT also
explicitly warns about. A native-layer parity test now verifies that the output
probe equals actual Musubi BF16 LoRA execution and native scaled FP8 execution
on a reduced architecture. Thus the measured deviation must not be hidden by
dropping compensation or changing the probe to a merged-weight calculation.

The configured 2x error gate is an engineering limit, not an established Klein
quality tolerance. A single cached batch cannot establish dataset-wide equivalence.
These fixes do not reduce the reported 1.489 ratio, require new gradient estimation,
or justify claiming an improvement over conventional LoRA. Existing default-path
initialization caches and checkpoints remain usable.

## Verification scope

`tests/test_loraga_audit.py` adds independent first-step gradient projection,
BF16 gradient accumulation with/without checkpoint recomputation, native output
probe parity, cache metric provenance, and CPU/CUDA SVD precision/restoration tests.
Existing tests cover compensation, factor conversion, native loading and resume.

Recorded review results: 12 audit cases passed, including the CUDA SVD precision
and exception-restoration cases. The broader regression command with
`CUDA_VISIBLE_DEVICES=-1` passed 70 cases and skipped 24 hardware-dependent cases.
These counts overlap; they are not 82 distinct tests. Ruff and `git diff --check`
passed. The extracted official-reference check passed again after the corrections.

An existing user training process was left running. Review validation uses small
fixtures and the CPU regression suite, not a second pretrained 4B training run.
The remaining acceptance work is held-out multi-batch error measurements and paired
training/generated-image comparisons. Do not interpret these tests as a full-model
image-quality result or an 8 GB capacity guarantee.
