# Runtime reproducibility verification (not yet executed on the research server)

These scripts provide a **repeatable way to test**, not evidence that the complete
SGT experiments have already been reproduced. The 2026-11-13 release snapshot passed
static syntax and split-manifest checks; runtime imports and GPU operations require
the original research environment and prepared graph data.

## 1. Environment

The historical `SGT/environment.yml` pins Python 3.7, PyTorch 1.9.1+cu111,
Torchvision 0.10.1+cu111 and DGL 0.4.3.post2. Those versions document the
original workflow, **not a verified modern or clean-room installation**. On
an existing compatible machine:

```bash
conda activate cgt
bash tools/smoke_all_datasets.sh check
```

The command runs the release manifest verifier, imports PyTorch, DGL,
histocartography, sklearn and the compiled Cython wrapper, and imports the
SGT model. Save the report/log with the release. If installation fails,
record the actual package versions using:

```bash
python --version
python -m pip freeze > environment_runtime_pip_freeze.txt
conda list --explicit > environment_runtime_conda_explicit.txt
nvidia-smi > environment_runtime_gpu.txt
```

Do not silently update dependency versions: changes can affect model behavior.
Never claim this installation has passed until it is executed on the target machine.

## 2. Configure local paths

```bash
cp tools/runtime_paths.example.sh tools/runtime_paths.local.sh
# Edit tools/runtime_paths.local.sh to your **actual existing prepared graph directories**.
source tools/runtime_paths.local.sh
```

`tools/runtime_paths.local.sh` is machine-specific and should not be committed.
Required layouts:

- `$OUT/tissue_graphs/{train,val,test}` — BRACS CGT graphs
- `$OUT_MORPH/tissue_graphs/{train,val,test}` — BRACS SGT graphs
- `$BACH_OUT/fold0/tissue_graphs/{train,val,test}` — BACH fold 0
- `$ISIC_OUT/tissue_graphs/{train,val,test}` — ISIC
- `$CAM_OUT/tissue_graphs/{train,val,test}` — CAMELYON16

Use graph data prepared from the frozen manifests in `splits/`. In particular,
BACH fold 0 is only an execution smoke test, not the complete five-fold study.

## 3. Short training smoke tests

```bash
bash tools/smoke_all_datasets.sh smoke
```

This executes one short (three-batch, one-epoch) training/evaluation smoke test
for **CGT and SGT on each of the four datasets**, with checkpoint selection
set to validation; model selection on the test set is prohibited by default.
Results are saved as tab-separated PASS/FAIL records and full logs in
`runtime_verification_logs/` (not committed by default).

To exercise all *five* controlled variants for all four datasets:

```bash
MODELS='cgt sgt sgt_concat sgt_mshuf sgt_pshuf' bash tools/smoke_all_datasets.sh smoke
```

Note: `train.py --smoke_batches 3` also runs a tiny test-set evaluation for
**software verification**. The smoke-test result must **not** be reported as a
scientific held-out test result. Full controlled experiments use the frozen
splits and validation-selected checkpoints.

## 4. What remains to be verified

- A clean environment installation from the pinned YAML.
- Correct patient/lesion/slide sample identifiers in the prepared graph folders.
- A complete seeded training run and independent inference per dataset.
- Agreement with the deposited predictions and checkpoint provenance.
- Original graph-generation inputs including the licensed stain-normalization target.

No source version, device-specific wheel or run result is invented in this patch.
