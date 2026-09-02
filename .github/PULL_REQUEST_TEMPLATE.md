## What this changes

<!-- One or two sentences. -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -m "not gpu and not dataset"` passes
- [ ] No absolute paths added (paths go through `lbmnca.paths`)
- [ ] No configuration read with a silent default (`.get()` / `getattr(..., default)`)

## Effect on reported results

<!--
Does this change training or evaluation behaviour? If so, say which numbers in
the thesis it invalidates. If it is purely a portability or documentation fix,
write "none".
-->
