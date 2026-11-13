# ISIC 2018 with the BRACS protocol (sgt2_update_isic)

Same models (CGT, SGT = CGT + MASE + SRE, concatenation, randomised morphology, randomised structure), same
training recipe (100 epochs, Adam 1e-4, weight decay 1e-3, StepLR(10, 0.8), batch 4, no weighted sampling,
random initialisation), seeds 0-4, checkpoint chosen on validation weighted F1 (test-rule checkpoint also kept).

Data: the 10,015 ISIC 2018 Task 3 training images (HAM10000) on the fixed lesion-grouped split
`isic2018_lesion_split.csv` (train 7,047 / val 1,468 / test 1,500; no lesion and no near-duplicate pair in two
sets). The test set is the one of the earlier ISIC results (1,500 images).

Graphs: the BRACS builder (`build_tg.TgBuilding`, as for BACH): colour-merged superpixels, ResNet-34 region
features, region adjacency graph, area and solidity per region. The Vahadane H&E stain normalisation is off by
default, because dermoscopy is not H&E stained (`STAIN_NORM=1` applies it as for BRACS/BACH).
MASE bins use the training nodes' log-area statistics, as for BRACS.

```bash
G=/workdir1.8t/fei27/CGT/GNNMD
export ISIC_IMAGES=$G/sgt_impl_ISIC1/data/ISIC2018_Task3_Training_Input
export ISIC_SPLIT=$G/sgt_impl_ISIC1/isic2018_lesion_split.csv
export ISIC_GRAPHS=$G/isic_graphs ISIC_OUT=$G/isic_data OMP_NUM_THREADS=8
DATASET=isic TRIAL=2 GPU=0 bash run_cgt_bracs.sh isic_graphs            # trial (use separate trial folders)
for k in 0 1; do DATASET=isic SHARD=$k NSHARDS=2 GPU=$k nohup bash run_cgt_bracs.sh isic_graphs > isic_graphs_$k.log 2>&1 & done
DATASET=isic bash run_cgt_bracs.sh isic_split                            # after both shards finished
DATASET=isic EPOCHS=100 GPU=0 bash run_cgt_bracs.sh smoke                # 7 x OK, parameters as printed
EPOCHS=100 nohup bash tools/run_isic.sh > isic_jobs.log 2>&1 &           # 25 runs
for M in sgt sgt_concat sgt_mshuf sgt_pshuf cgt; do DATASET=isic EPOCHS=100 MODEL=$M bash run_cgt_bracs.sh predict; done
DATASET=isic EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare         # CGT vs SGT
A="$PWD/runs_sgtfull_isic_e100_val_seed*" B="$PWD/runs_sgtconcat_isic_e100_val_seed*" NAMES='SGT concat' \
  DATASET=isic EPOCHS=100 PERCLASS=1 bash run_cgt_bracs.sh compare      # and sgtmshuf, sgtpshuf; CGT vs concat
```
Run folders: `runs_<tag>isic_e100_val_seed<s>`, logs `log_isic_e100_<model>_s<s>.txt`.
