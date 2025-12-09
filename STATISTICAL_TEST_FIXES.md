# Statistical Test Fixes: Corrected P-value Calculation

## Issues Identified

### 1. **Incorrect Feature Space Sampling**
- **Problem:** Code was sampling from `range(32768)` but SAE `hidden_dim = 8192`
- **Impact:** Random circuits could include non-existent features, leading to incorrect comparisons
- **Fix:** Changed to sample from `range(sae_hidden_dim)` where `sae_hidden_dim = 8192`

### 2. **Wrong Null Distribution**
- **Problem:** Null distribution was comparing discovered circuit to permutations of ONE random circuit
- **Impact:** This doesn't test the right hypothesis - we want to know if discovered circuits are different from random
- **Fix:** Null distribution now compares random circuit pairs (random vs random, many times)

### 3. **One-Sided P-value When Two-Sided Needed**
- **Problem:** P-value only tested if `observed >= random`, so when `observed = 0.0`, all random scores `>= 0.0` → p = 1.0
- **Impact:** Cannot detect if discovered circuits are significantly DIFFERENT from random (either direction)
- **Fix:** Implemented two-sided test that detects significant differences in either direction

### 4. **P-value Calculation Logic**
- **Problem:** When observed = 0.0 and random mean ≈ 0.0001, the one-sided test always gives p = 1.0
- **Impact:** Cannot distinguish "no overlap" from "random-like behavior"
- **Fix:** Two-sided test uses `|score - mean| >= |observed - mean|` to detect any significant deviation

## Changes Made

### File: `src/circuits/similarity.py`

#### 1. `compute_random_similarity_distribution()` - Fixed

**Before:**
- Compared discovered circuit to permutations of one random circuit
- Hardcoded `max_feature_id = 32768`

**After:**
- Creates null distribution by comparing random circuit pairs (random vs random)
- Accepts `max_feature_id` parameter (default 8192, matching SAE hidden_dim)
- Each permutation creates TWO random circuits and compares them
- This properly tests: "What is the similarity between two random circuits?"

**Key Change:**
```python
# OLD: Compare discovered circuit to permutations of one random circuit
score = compute_circuit_similarity(circuit1, random_circuit)

# NEW: Compare two random circuits (null distribution)
random_circuit1 = create_random_circuit(...)
random_circuit2 = create_random_circuit(...)
score = compute_circuit_similarity(random_circuit1, random_circuit2)
```

#### 2. `compute_significance()` - Fixed

**Before:**
- One-sided test only: `count_extreme = sum(1 for score in random_scores if score >= observed_score)`
- When `observed = 0.0`, all random scores `>= 0.0` → p = 1.0

**After:**
- Two-sided test by default: `abs(score - mean) >= abs(observed - mean)`
- Detects significant differences in either direction
- Optional one-sided test for specific hypotheses

**Key Change:**
```python
# OLD: One-sided
count_extreme = sum(1 for score in random_scores if score >= observed_score)

# NEW: Two-sided
abs_deviation_observed = abs(observed_score - mean)
count_extreme = sum(1 for score in random_scores 
                   if abs(score - mean) >= abs_deviation_observed)
```

### File: `scripts/evaluate_circuits.py`

#### Changes:
1. Set `sae_hidden_dim = 8192` (matching actual SAE configuration)
2. Updated to use `max_feature_id=sae_hidden_dim` in `compute_random_similarity_distribution()`
3. Added informative print statements about null distribution
4. Use two-sided test: `compute_significance(observed_sim, null_dist, two_sided=True)`

## What the Test Now Does

### Null Hypothesis (H₀):
The discovered circuit is like a random circuit (no meaningful structure)

### Alternative Hypothesis (H₁):
The discovered circuit is significantly different from random (has meaningful structure)

### Test Procedure:
1. **Observed:** Compare discovered circuit to one random circuit → `observed_sim`
2. **Null Distribution:** Compare 1000 pairs of random circuits → `null_dist`
3. **P-value:** Two-sided test - Is `observed_sim` significantly different from `null_dist`?

### Interpretation:
- **P < 0.05 (two-sided):** Discovered circuit is significantly different from random
- **P ≥ 0.05:** Cannot reject null hypothesis (circuit may be like random)

## Expected Impact on Results

### Before Fix:
- Most categories: `observed = 0.0`, `p = 1.0` (always non-significant)
- Unethical: `observed = 0.0020`, `p = 0.072` (marginally significant)

### After Fix:
- **Null distribution will be different:** Random vs random similarities (not discovered vs random)
- **P-values will be more meaningful:** Two-sided test can detect both high and low similarities
- **Better statistical power:** Proper null distribution allows detection of significant differences

### What to Expect:
1. **Null distribution mean:** Should be very small (~0.0001-0.0002) - random circuits rarely overlap
2. **Observed similarities:** May still be 0.0 for most categories (no overlap with random)
3. **P-values:** Will now properly test if observed is significantly different from null distribution
4. **Significant results:** Categories with observed similarity far from null mean (either direction) will be significant

## Key Insight

**The critical fix:** The null distribution should represent "what similarity do we expect between two random circuits?" not "what similarity do we get when comparing discovered circuit to random permutations?"

This is the correct statistical test for answering: "Is the discovered circuit significantly different from what we'd expect by chance?"

## Next Steps

1. **Re-run evaluation script** with fixed code
2. **Compare new results** to previous results
3. **Interpret p-values correctly:**
   - P < 0.05: Circuit is significantly different from random (has structure)
   - P ≥ 0.05: Cannot conclude circuit is different from random
4. **For research question:** If circuits are significantly different from random, then we can meaningfully compare them to each other to answer monolithic vs. modular

## Technical Details

### Two-Sided Test Formula:
```
p_value = (count of |random_score - mean| >= |observed - mean| + 1) / (n_permutations + 1)
```

This counts how many random scores are as extreme or more extreme than the observed score, in either direction.

### Why Two-Sided?
- We want to know if discovered circuits are **different** from random (not just greater)
- Low similarity (0.0) can be significant if it's significantly lower than expected
- High similarity can also be significant if it's significantly higher than expected
- Two-sided test is more conservative and appropriate for exploratory analysis

