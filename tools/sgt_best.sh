#!/usr/bin/env bash
# Best results of every run in CGT_sgt2 (read-only), from the training logs, using only the first E epochs
# (a run with the same seed and E epochs is identical to these first E epochs: the LR schedule does not
# depend on the total). Per run: best validation wF1 and its epoch, the test wF1 at that epoch (validation
# rule, unbiased), the best test wF1 (CGT rule, optimistic); then mean +- SD per model.
#   bash ~/sgt_best.sh            # three classes, all epochs so far
#   bash ~/sgt_best.sh 3 100      # three classes, first 100 epochs only
#   bash ~/sgt_best.sh bach 100   # BACH: one line per fold and seed (f<k>s<s>); the mean is over folds x seeds
# Last column: whether the run's saved checkpoints (best over ALL epochs) come from within the first E epochs;
# if 'yes', macro F1, accuracy and per-class scores of the saved checkpoints are also the E-epoch ones.
cd "${S2:-/workdir1.8t/fei27/CGT/GNNMD/CGT_sgt2}" || exit 1
C="${1:-3}"; E="${2:-100000}"
if [ "$C" = bach ]; then PAT="log_bach_e*_f[0-9]_s[0-9].txt"; else PAT="log_c${C}_e*_s[0-9].txt"; fi
for f in $PAT; do
  [ -f "$f" ] || continue
  awk -v E="$E" -v F="$f" '/^=====/{m=$2; s=$4; n=0; bv=-1; tv=0; bt=-1; be=-1; bte=-1; av=-1; ae=-1; at=-1; ate=-1; pend=0}
       /val weighted F1/{split($0,a,"val weighted F1: "); v=a[2]+0; pend=1}
       /CGT rule/{split($0,b,"rule[)]: "); t=b[2]+0
                  if(pend){ if(n<E){ if(v>bv){bv=v; tv=t; be=n}; if(t>bt){bt=t; bte=n} }
                            if(v>av){av=v; ae=n}; if(t>at){at=t; ate=n}; n++; pend=0 } }
       END{if(match(F, /_f[0-9]_s[0-9]/)) s = substr(F, RSTART + 1, RLENGTH - 1)   # BACH: fold and seed
           if(m!="" && n>0) printf "%s %s %d %.4f %.4f %d %.4f %s\n", m, s, n, bv, tv, be, bt,
             ((ae<E && ate<E) ? "yes" : (n<E ? "running" : "no"))}' "$f"
done | sort -k1,1 -k2,2 | awk -v E="$E" '
  function sd(s, s2, n) { return n > 1 ? sqrt((s2 - s * s / n) / (n - 1)) : 0 }
  BEGIN { lim = (E < 100000) ? "first " E " epochs" : "all epochs so far"
          printf "using %s\n%-11s %4s %8s | %-21s | %-11s | %-10s | %s\n", lim, "model", "seed", "epochs", "best val wF1 (epoch)", "test@bestval", "best test", "saved ckpt within limit" }
  { done = ($3 < E) ? $3 : E
    printf "%-11s %4s %8s | %.4f (epoch %3d)   | %.4f      | %.4f     | %s\n", $1, $2, done "/" ((E < 100000) ? E : 200), $4, $6, $5, $7, $8
    k = $1; c[k]++; v[k] += $4; v2[k] += $4 * $4; t[k] += $5; t2[k] += $5 * $5; b[k] += $7; b2[k] += $7 * $7 }
  END { print "--- mean +- SD over seeds (" lim ")"
        for (k in c) printf "%-11s %d seeds | best val %.4f +- %.4f | test at best val %.4f +- %.4f | best test (CGT rule) %.4f +- %.4f\n", k, c[k], v[k]/c[k], sd(v[k], v2[k], c[k]), t[k]/c[k], sd(t[k], t2[k], c[k]), b[k]/c[k], sd(b[k], b2[k], c[k]) }'
