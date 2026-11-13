# Archive audit report

## Provenance and duplication

- Input A: `source_versions.zip` — 63 non-directory files; preserved byte for byte.
- Input B: `SGT_CAMELYON16_scripts_and_logs.zip` — 10 non-directory files; all **10/10** exactly match the corresponding entries under `camelyon_update/`.
- Separate duplicate copies of input B included in the output: **0**.
- Historical version trees preserved without combining distinct versions.

## Folder inventory

| Folder | Files |
| --- | ---: |
| `bach_update/` | 14 |
| `base/` | 24 |
| `bracs_update/` | 6 |
| `camelyon_update/` | 10 |
| `isic_update/` | 9 |

| **Total** | **63** |

## Static tests

- Python `.py` files parsed using Python `ast`: **33** checked; **0** syntax/decoding failures.
- Shell `.sh` files tested with `bash -n`: **15** checked; **0** failures.
- These checks **do not** import external dependencies, compile Cython, run medical preprocessing, load datasets, start GPU training, validate numerical output, or verify that experimental checkpoints match the paper.

## Portability and study-specific notes

- A number of the original scripts and logs refer to local `/workdir...` paths and need adaptation to other machines.
- No fixed split files or trained model weights were introduced into this historical archive.
- The 10 CAMELYON16 log/script files are preserved only as historic artifacts; these are not the same as a validated current five-seed prediction release.
- Do not conflate code availability with the ability to independently reproduce all experimental measurements.
- Consider publishing the historical archive separately from the maintained training code, verified splits and prediction/checkpoint artifacts.
