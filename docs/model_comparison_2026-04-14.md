# Model Comparison — 2026-04-14

## Test setup

- Image: `test/sample_week.png` (week of Mar 20–26, 2026)
- Models: `qwen2.5vl:7b` vs `llama3.2-vision:11b` via Ollama
- Hardware: MacBook Pro M4 Max
- Script: `tests/compare_models.py`

## Results

| | qwen2.5vl:7b | llama3.2-vision:11b |
|---|---|---|
| Tasks extracted | 19 | 14 |
| Category hints | Correct (work/errand/personal) | All `unknown` |
| Task text | Accurate | Hallucinated tasks not in image |
| Year in dates | 2026 ✓ | 2023 ✗ |
| Shopping list | 6 items, correctly separated from tasks | 2 items, mixed with tasks |
| Inference time | 38.5s | 41.1s |

## Raw extraction — qwen2.5vl:7b

```
Monday 2026-03-20
  [ ] CHECK AWS COSTS  [work]
  [ ] GIT COMMIT  [work]
  [ ] AGENIC AI COURSE FINISH  [work]
Tuesday 2026-03-21
  [ ] GIT COMMIT  [work]
  [ ] KATHERINE CEO CALL 11AM  [work]
  [ ] CUT 3PM  [work]
Wednesday 2026-03-22
  [ ] GIT COMMIT  [work]
  [ ] FILL GAS BUY EGGS SALMON TOMATOES ONIONS  [errand]
  [ ] PAY WWU MEDICINE WATERFRONT  [errand]
Thursday 2026-03-23
  [ ] MAKE LEMON RUE  [personal]
Friday 2026-03-24
  [ ] GIT COMMIT  [work]
  [ ] FIDELITY PORT 6M  [work]
  [ ] SHOPPING FOR MARS  [errand]
Saturday 2026-03-25
  [ ] GIT COMMIT  [work]
  [ ] CHANA MASAALA  [personal]
  [ ] CHICKEN CURRY  [personal]
Sunday 2026-03-26
  [ ] CHANA MASAALA  [personal]
  [ ] CHICKEN CURRY  [personal]
  [ ] RICE  [personal]

Shopping:
  - 2 BAGS OF COCKTAIL SPROUTS
  - 2 TIN MANGO PULP
  - 6 TUBS YOGURT
  - DAHI
  - DHANAPATTI
  - SAMS SHOPPING
```

## Raw extraction — llama3.2-vision:11b

```
Monday 2023-03-20
  [ ] BUY CHICKEN  [unknown]
  [ ] MAKE LEMON RICE  [unknown]
  [ ] PAY WVU MEDICINE  [unknown]
Tuesday 2023-03-21
  [ ] SHOPPING DOLLAR TREE  [unknown]
  [ ] BUY EGGS  [unknown]
  [ ] SALMON TOMATOES  [unknown]
Wednesday 2023-03-22
  [ ] KESAR MATAR PULP  [unknown]
  [ ] WATERFRONT  [unknown]
Thursday 2023-03-23
  [ ] CEREAL  [unknown]
  [ ] MILK  [unknown]
Friday 2023-03-24
  [ ] BREAD  [unknown]
  [ ] EGG  [unknown]
Saturday 2026-03-25
  [ ] CHICKEN  [unknown]
  [ ] RICE  [unknown]

Shopping:
  - CHICKEN
  - RICE
```

## Verdict

**qwen2.5vl:7b wins decisively.** llama3.2-vision:11b hallucinated tasks, got the year wrong, collapsed the shopping list, and assigned no category hints.

Upgrade path: move to `qwen2.5vl:72b` on DGX Sparc — same architecture, no model switching.
