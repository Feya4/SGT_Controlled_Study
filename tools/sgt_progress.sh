#!/usr/bin/env bash
# Progress of the SGT, control and fine-tuning runs in CGT_sgt2 (read-only).  Install: cp tools/sgt_progress.sh ~/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/SGT" || exit 1
echo "=== $(date '+%m-%d %H:%M') | CPU load $(cut -d' ' -f1-3 /proc/loadavg) ($(nproc --all) cores)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/GPU /'
else
  echo "nvidia-smi not available"
fi
echo "=== running now (logs written in the last 20 minutes)"
for f in log_sgt_e*_gpu*.txt log_ctl_e*.txt log_ft_*.txt log_c[37]_e*.txt log_bach_e*.txt; do
  [ -f "$f" ] || continue
  [ -n "$(find "$f" -mmin -20)" ] || continue   # skip logs of stopped or finished queues
  sp=$(tail -c 400 "$f" | tr '\r' '\n' | grep -a 'batch' | tail -1 | grep -o '[0-9.]*batch/s\|[0-9.]*s/batch')
  awk -v f="$f" -v sp="$sp" '/^=====/{m=$2; s=$4; n=0; bv=0; bt=0; init=""; E=200; for(i=1;i<=NF;i++) if($i=="--epochs") E=$(i+1); ft=($0 ~ /--init_from/) ? " ft" : ""}
    /^before training/{init=$0}
    /val weighted F1/{n++; split($0,a,"val weighted F1: "); v=a[2]+0; if(v>bv)bv=v}
    /CGT rule/{split($0,b,"rule[)]: "); t=b[2]+0; if(t>bt)bt=t}
    END{if(m=="") {printf "%-34s waiting to start\n", f; exit}
        printf "%-34s %-10s%s seed %s | epoch %3d/%d | best val wF1 %.4f | best test wF1 %.4f | %s\n", f, m, ft, s, n, E, bv, bt, sp;
        if(init!="") printf "%-34s   (%s)\n", "", init}' "$f"
done
echo "=== finished seeds (test wF1 with checkpoint chosen on: validation | test = CGT rule)"
for d in runs_*_val_seed*; do
  case "$d" in *bach_f[0-9]_*) continue ;; esac            # BACH: counted per model below
  r=$(ls "$d"/logs/*/test_results.txt 2>/dev/null | head -1); [ -n "$r" ] || continue
  v=$(grep -a '^best_val_weighted_f1_score' "$r" | grep -o 'weighted_f1=[0-9.]*' | head -1 | cut -d= -f2)
  t=$(grep -a '^best_test_weighted_f1_score' "$r" | grep -o 'weighted_f1=[0-9.]*' | head -1 | cut -d= -f2)
  printf "%-32s validation rule %s | CGT rule %s\n" "$d" "$v" "${t:--}"
done
ls -d runs_*bach_f[0-9]_*_val_seed* >/dev/null 2>&1 && echo "=== BACH (5 folds x 5 seeds per model; per-fold test wF1, validation rule; pooled scores: compare step)"
for m in $(ls -d runs_*bach_f[0-9]_*_val_seed* 2>/dev/null | sed 's/bach_f[0-9]_.*//' | sort -u); do
  cat "$m"bach_f[0-9]_*_val_seed*/logs/*/test_results.txt 2>/dev/null | grep -a '^best_val_weighted_f1_score' |
    grep -o 'weighted_f1=[0-9.]*' | grep -v best | cut -d= -f2 |
    awk -v m="${m#runs_}" '{n++; s+=$1; s2+=$1*$1} END{printf "%-14s %2d/25 runs finished", (m==""?"cgt":m), n;
      if(n) printf " | mean test wF1 over finished folds x seeds %.4f +- %.4f", s/n, (n>1?sqrt((s2-s*s/n)/(n-1)):0); print ""}'
done
