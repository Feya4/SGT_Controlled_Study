# Original stain-normalization target

`target.png` is required for reproducing the BRACS/BACH/CAMELYON16 tissue-graph preprocessing (and any ISIC run where stain normalization is enabled). It is deliberately **not** redistributed in this public code package because the right to redistribute the reference image has not been established.

The original `CGT_sgt2.zip` shared by the project owner contains `CGT/core/data/target.png`. On a licensed research machine, extract that original file to `SGT/core/data/target.png`, or set `SGT_STAIN_TARGET` to its absolute location. **Do not replace it with a different image** and claim an exact reproduction. See `REPRODUCIBILITY.md`.
