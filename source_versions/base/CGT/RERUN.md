# Re-running CGT (patched copy)

Patched copy of github.com/wang-kang-6/CGT (commit 4e6ce02). Every change in the code is
marked `[rerun-fix]` or `[rerun-addition]`. `git log -p` shows the full diff.

## 0. Shortcut used for the SGT revision: run CGT on the existing HACT graphs

`run_cgt_bracs.sh` runs this code on the tissue graphs already used by SGT (same graphs, split and labels), so the
two models are compared on identical input and no graphs have to be rebuilt from the images.
`core/convert_hact_graphs.py` only appends the normalised centroid (x/width, y/height) to the 512-d node features,
as CGT's own build_tg.py does. Steps: `check`, `convert`, `weights`, `infer`, `train`, `summary` (see the script header).
Results per run are written to `runs_<val_split>_seed<s>/logs/<id>/test_results.txt`, now including macro F1 and the
best validation epoch; `core/summarize_cgt_runs.py` reports the checkpoint with the best validation weighted F1.
The data loader keeps tissue graphs on the CPU with DGL >= 0.5, which refuses GPU features on a CPU graph.

## SGT on this code base: SGT = CGT + MASE + SRE

Each SGT module extends one CGT module and is marked `[sgt]` in the code. With both off (no flags), the model
is CGT exactly as released, so CGT is also the editor's "model without MASE/SRE" (CGT uses no weighted sampling).

| SGT module | CGT module it extends | what is added | flag | parameters |
|---|---|---|---|---|
| MASE, Morphology-Aware Structural Embedding | connectivity (degree) embedding with Local Connectivity Aggregation | embeddings of binned absolute log area and binned convexity (16 bins each) are added to the degree embedding, so they enter wherever CGT uses it: the input (weight 2) and the neighbour sum before every layer | `--sgt_mase` | 17,476 |
| SRE, Spatial Relationship Encoder | connectivity bias in self-attention (per-head bias of the binned Euclidean distance) | a per-head bias of the hop distance between two regions in the tissue graph (0-19 hops; CGT masks >= 20) | `--sgt_sre` | 132 |

CGT has 8,434,149 parameters; SGT has 8,451,757 (+0.2%). `--morph real` and `--morph shuffle` are the earlier
names of `--sgt_mase` and `--sgt_mase --mase_shuffle` and run the same code, so earlier `MORPH=real` runs are
CGT + MASE runs.

Presets (`MODEL=...` in `run_cgt_bracs.sh`; all but `cgt` need the graphs from `convert_morph`/`weights_morph`):

| MODEL | flags | role | run folders |
|---|---|---|---|
| `cgt` | none | baseline (= no MASE, no SRE) | `runs_val_seed<s>` |
| `sgt` | `--sgt_mase --sgt_sre` | proposed model | `runs_sgtfull_val_seed<s>` |
| `cgt_mase` | `--sgt_mase` | ablation | `runs_cgtmase_val_seed<s>` |
| `cgt_sre` | `--sgt_sre` | ablation | `runs_cgtsre_val_seed<s>` |
| `sgt_mshuf` | `+ --mase_shuffle` | randomised morphology: bins permuted within each graph | `runs_sgtmshuf_val_seed<s>` |
| `sgt_pshuf` | `+ --sre_shuffle` | randomised structure: hop matrix permuted within each graph | `runs_sgtpshuf_val_seed<s>` |
| `sgt_concat` | `--sgt_sre --morph concat --ffn_dim 259` | parameter-matched concatenation: raw area and convexity through one linear layer instead of MASE (8,454,345 parameters) | `runs_sgtconcat_val_seed<s>` |

Checks and statistics:
* `bash run_cgt_bracs.sh smoke`: trains every preset for one epoch on 3 batches, checks that each added module
  gets a finite gradient, and prints the parameter counts next to the expected ones.
* `MODEL=<m> bash run_cgt_bracs.sh predict`: reloads the checkpoint chosen on validation and writes the per-graph test
  predictions (`test_predictions.csv`) next to `test_results.txt`; it also works for runs made before this step existed.
* `A='<runs glob>' B='<runs glob>' NAMES="CGT SGT" bash run_cgt_bracs.sh compare`: per-seed table, seed-paired
  t-test (df = seeds - 1) with 95% interval and Cohen's d_z, and a paired bootstrap over the test graphs
  (10,000 resamples, the same resample for every run) for weighted F1, macro F1 and accuracy.

## 1. Environment (once)

```bash
cd CGT
conda env create -f environment.yml     # creates env "cgt" (python 3.7, torch 1.9.1+cu111, dgl 0.4.3)
conda activate cgt
```
If download.pytorch.org is slow, install the two torch wheels first from a mirror, then rerun
`conda env update -f environment.yml`. A C compiler (gcc) must be on the server for Cython.

## 2. Data

BRACS RoI set, **previous_version** (3163 / 602 / 626 RoIs, the split used in the paper), laid out as
`<BRACS>/train/0_N/*.png ... <BRACS>/test/6_IC/*.png`. If your own experiments use the latest BRACS
version, point `BRACS` at that instead so the comparison uses the same split.

## 3. Run

```bash
export BRACS=/path/to/BRACS_RoI/previous_version
export OUT=/path/to/cgt_data
nvidia-smi                     # pick a free GPU on the shared server
export GPU=2

bash run_cgt.sh check          # 1 minute: torch/GPU/dgl/histocartography/Cython
nohup bash run_cgt.sh tg      > rerun_tg.log 2>&1 &     # tissue graphs, hours
bash run_cgt.sh weights        # connectivity weights, minutes
bash run_cgt.sh infer          # released checkpoint on test -> README claims 66.9 weighted F1
nohup bash run_cgt.sh train   > rerun_train.log 2>&1 &  # full training
```
Training outputs go to `runs_<valsplit>_seed<seed>/logs/<uuid>/`: three checkpoints,
`test_results.txt`, per-checkpoint classification reports, `history.csv` and curve PNGs.

For a clean protocol (checkpoint selection on val, not test) and several seeds:
```bash
for s in 0 1 2; do VAL_SPLIT=val SEED=$s bash run_cgt.sh train; done
```

## What was broken in the released code

| File | Problem | Fix |
|---|---|---|
| build_tg.py | `if not self._valid_image(...)` is inverted, so every valid image is skipped | restored HACT-Net logic (skip existing, keep valid sizes) |
| build_tg.py | stain target `../data/target.png` doesn't exist relative to `core/` | path resolved from the script folder |
| build_tg.py | split guessed from any "train"/"val" in the full path | `--split` argument |
| train/inference/dataloader/grapher | `CUDA_VISIBLE_DEVICES='0,1,2,3'` hard-coded, overriding your GPU choice | removed; set `GPU=` |
| train/inference, wrapper | import `pytorch_lightning`, `torch_geometric` (unused, not in env) | removed |
| train.py | `plt.show()` blocks when a desktop display is attached | figures saved to PNG |
| train.py | `mlflow.pytorch.log_model` called even with `--logger none` | only with mlflow |
| train.py | validation loader reads the **test** split | `--val_split` (default `test` = original) |
| dataloader.py | `--in_ram` crashes (weights never loaded) | fixed |
| environment.yml | `sklearn` no longer installable; torchvision unpinned; malformed `-f` lines; no Cython | rewritten |
| (missing) | no script for the `weights/*.pt` connectivity matrices | `build_weights.py` (see below) |

## About `build_weights.py`

The authors share `weights/*.pt` via a form. The paper defines the connectivity bias from the
Euclidean distance between node centroids. In the released checkpoint only embedding rows 1..138 of
`w_encoder` are non-zero, so the official `w` values span 0..137. This matches
`w = floor(100 * dist(normalised centroids))` (max 100·√2 ≈ 141); those centroids are the last two
node features. It is a reconstruction: `run_cgt.sh infer` is the check. If the released checkpoint
reproduces about 66.9 weighted F1 with these weights, they match. If it falls far short, try
`python build_weights.py --tg_root $OUT --rounding round --overwrite` or use the official files.

Paper hyperparameters differ slightly from code defaults: the paper says 100 epochs and lr 1e-3
(Table II also lists a peak lr of 2e-4), while the code defaults are 200 epochs, lr 1e-4 and StepLR.
The script keeps the code defaults. The paper reports 66.54 ± 0.43 weighted F1 on the original split.
