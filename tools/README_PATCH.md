# SGT runtime verification patch

Copy these files into the **root of the current GitHub repository**, maintaining
`tools/` paths. This patch is deliberately small and does **not** replace the
SGT model, dataset manifests or historical training recipe.

The `tools/sgt_progress.sh` helper now resolves the repository root from its
own location rather than using the original author server path.

See `RUNTIME_VERIFICATION.md` to execute and record environment/import tests
and per-dataset short training smoke tests on your research server. This does
not claim the tests have been run.
