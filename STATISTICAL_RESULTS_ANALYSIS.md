# Statistical Results Analysis: Circuit vs Random Baseline

## Overview

This document analyzes the statistical test results comparing discovered refusal circuits to random baselines. The test answers: **"Are the discovered circuits significantly different from random chance?"**

## What the Test Measures

### Test Design

**Null Hypothesis (H₀):** The discovered circuit is indistinguishable from a random circuit (no meaningful structure)

**Alternative Hypothesis (H₁):** The discovered circuit is significantly different from random (has meaningful structure)

**Test Method:** Permutation test with 1,000 random permutations

**Similarity Metric:** Jaccard similarity between circuits (0-1 scale):
- **0.0** = No overlap (completely different circuits)
- **1.0** = Identical circuits
- **0.4-0.6** = Moderate overlap (some shared features)

### How Similarity is Computed

From `src/circuits/similarity.py`:
```python
# Combined similarity = 40% node overlap + 30% edge overlap + 30% importance correlation
combined_similarity = 0.4 * node_similarity + 0.3 * edge_similarity + 0.3 * abs(importance_corr)
```

**Node Similarity:** Jaccard index on top 100 nodes (intersection/union)
**Edge Similarity:** Jaccard index on top 100 edges
**Importance Correlation:** Pearson correlation of importance scores for overlapping nodes

## Key Findings from LLaMA Results

### Pattern 1: Most Categories Show Zero Overlap (9/10 categories)

**Categories:** Deception, Harassment, Harmful, Hate, Illegal, Privacy, Self Harm, Sexual, Violence

**Results:**
- **Observed Similarity:** 0.0 (no overlap with random baseline)
- **Random Mean:** ~0.0001-0.0002 (extremely small)
- **P-value:** 1.0 (NOT significantly different from random)
- **Z-score:** Negative (~-0.25 to -0.31)

**Interpretation:**
1. **Zero Overlap is Expected:** Random circuits have essentially no overlap with each other (expected behavior)
2. **P-value = 1.0 is Problematic:** This means ALL 1,000 random permutations had similarity ≥ 0.0, so the observed circuit is NOT significantly different from random
3. **Statistical Power Issue:** When both observed and random distributions are near-zero, the test cannot distinguish "meaningful structure" from "random structure"

**What This Means:**
- The discovered circuits have **no measurable overlap** with random circuits (good sign)
- BUT the test **cannot prove** they're significantly different because random circuits also have no overlap
- This is a **limitation of the test design** when dealing with sparse, non-overlapping circuits

### Pattern 2: Unethical Category Shows Marginal Signal (1/10 categories)

**Category:** Unethical

**Results:**
- **Observed Similarity:** 0.0020 (slightly higher than random)
- **Random Mean:** 0.000147
- **Random Std:** 0.000538
- **P-value:** 0.072 (marginally significant, but p > 0.05)
- **Z-score:** 3.46 (positive, indicating observed > random)

**Interpretation:**
1. **Positive Signal:** The Unethical circuit shows measurable overlap with the random baseline
2. **Not Statistically Significant:** p = 0.072 > 0.05 threshold (would need p < 0.05 for significance)
3. **Effect Size:** Observed is ~13.7x the random mean (0.0020 / 0.000147), suggesting a real effect
4. **Normal Approximation Disagrees:** p_normal = 0.00027 (highly significant), but empirical p = 0.072 (not significant)

**What This Means:**
- The Unethical circuit **might** have meaningful structure
- But the evidence is **not strong enough** to reject the null hypothesis at α = 0.05
- The discrepancy between empirical and normal p-values suggests the distribution may not be normal

## Implications for the Research Question

### Research Question: Monolithic vs. Modular Refusal

**Goal:** Determine if refusal circuits are:
- **Monolithic:** Shared circuits across categories (high similarity between categories)
- **Modular:** Category-specific circuits (low similarity between categories)

### Current Results Indicate:

**⚠️ Critical Limitation:** The statistical tests show that discovered circuits are **not significantly different from random baselines** for 9/10 categories.

**This means:**
1. **Cannot Answer Research Question:** If circuits aren't significantly different from random, we can't compare them to each other meaningfully
2. **Possible Explanations:**
   - **Circuits are poorly discovered:** The discovery process may not be finding meaningful patterns
   - **Test is underpowered:** The similarity metric or test design may not be sensitive enough
   - **Circuits are truly sparse:** Real circuits may have such low overlap that they appear random
   - **Threshold too high:** Using importance threshold 0.05 may filter out too many features

### What Needs to Happen Next:

1. **Verify Circuit Discovery:** Check if circuits actually contain meaningful features (not empty or too small)
2. **Lower Threshold:** Try lower importance thresholds (e.g., 0.01, 0.001) to capture more features
3. **Alternative Tests:** Consider testing circuit **functionality** (faithfulness/completeness) rather than structure
4. **Cross-Category Comparison:** Compare circuits **to each other** (not just to random) to answer monolithic vs. modular

## Statistical Interpretation Guide

### Understanding P-values

**P-value = 1.0:**
- **Meaning:** 100% of random permutations had similarity ≥ observed
- **Interpretation:** Observed result is **not significantly different** from random
- **In this context:** Cannot reject null hypothesis (circuit is not significantly different from random)

**P-value = 0.072:**
- **Meaning:** 7.2% of random permutations had similarity ≥ observed
- **Interpretation:** Marginally significant, but **not significant** at α = 0.05
- **In this context:** Weak evidence against null hypothesis, but not strong enough

**P-value < 0.05:**
- **Meaning:** < 5% of random permutations had similarity ≥ observed
- **Interpretation:** **Statistically significant** - reject null hypothesis
- **In this context:** Circuit is significantly different from random (has meaningful structure)

### Understanding Z-scores

**Z-score = (observed - mean) / std**

- **Positive Z-score:** Observed > random mean (circuit has more overlap than expected)
- **Negative Z-score:** Observed < random mean (circuit has less overlap than expected)
- **|Z| > 2:** Typically considered significant (but use p-value for final decision)

**In these results:**
- Most categories: Z ≈ -0.25 to -0.31 (slightly below random, but not significantly)
- Unethical: Z = 3.46 (significantly above random, but p-value still > 0.05 due to empirical calculation)

### Understanding Confidence Intervals

**95% CI for Random Distribution:**
- **Range:** [ci_lower, ci_upper]
- **Meaning:** 95% of random permutations fall within this range
- **Interpretation:** If observed falls outside this range, it's unusual (but check p-value for significance)

**In these results:**
- All observed values (except Unethical) fall within or very close to the CI
- Unethical observed (0.0020) is well above CI upper bound (~0.0012), suggesting it's unusual

## Visualization Recommendations

### Table 1: Summary Statistics by Category

**Recommended Format:** Markdown table or LaTeX table

**Columns:**
1. **Category** (string)
2. **Observed Similarity** (float, 4 decimals)
3. **Random Mean** (float, scientific notation or 6 decimals)
4. **Random Std** (float, scientific notation or 6 decimals)
5. **Z-Score** (float, 2 decimals)
6. **P-Value (Empirical)** (float, 4 decimals)
7. **95% CI Lower** (float, scientific notation)
8. **95% CI Upper** (float, scientific notation)
9. **Significant?** (boolean: p < 0.05)
10. **Effect Size** (ratio: observed / random_mean, 2 decimals)

**Sorting:** By p-value (ascending) or by observed similarity (descending)

**Highlighting:**
- **Bold** categories with p < 0.05 (statistically significant)
- **Italic** categories with 0.05 ≤ p < 0.10 (marginally significant)
- **Gray** categories with p = 1.0 (not significant)

### Table 2: Effect Size Comparison

**Purpose:** Show how much larger observed is compared to random

**Columns:**
1. **Category**
2. **Observed**
3. **Random Mean**
4. **Effect Size Ratio** (observed / random_mean)
5. **Interpretation** (e.g., "13.7x random", "No difference", "Below random")

**Visualization:** Bar chart or heatmap showing effect size ratios

### Table 3: Statistical Test Results Summary

**Purpose:** Quick overview of test outcomes

**Format:** Summary statistics
- **Total Categories:** 10
- **Significant (p < 0.05):** 0
- **Marginally Significant (0.05 ≤ p < 0.10):** 1 (Unethical)
- **Not Significant (p ≥ 0.10):** 9
- **Zero Overlap:** 9
- **Positive Signal:** 1 (Unethical)

### Visualization Options

#### Option 1: Statistical Summary Table (Primary Recommendation)

```
┌──────────────┬──────────────┬──────────────┬──────────┬──────────┬──────────────┬──────────────┐
│ Category     │ Observed     │ Random Mean  │ Z-Score  │ P-Value   │ 95% CI       │ Significant? │
├──────────────┼──────────────┼──────────────┼──────────┼──────────┼──────────────┼──────────────┤
│ Unethical    │ 0.0020       │ 1.47e-04     │ 3.46     │ 0.0720    │ [-9.09e-04,   │ No (marginal)│
│              │              │              │          │           │  1.20e-03]    │              │
├──────────────┼──────────────┼──────────────┼──────────┼──────────┼──────────────┼──────────────┤
│ Deception    │ 0.0000       │ 1.41e-04     │ -0.27    │ 1.0000    │ [-8.80e-04,   │ No           │
│              │              │              │          │           │  1.16e-03]    │              │
│ ...          │ ...          │ ...          │ ...      │ ...       │ ...          │ ...          │
└──────────────┴──────────────┴──────────────┴──────────┴──────────┴──────────────┴──────────────┘
```

#### Option 2: Effect Size Bar Chart

**X-axis:** Categories (sorted by effect size)
**Y-axis:** Effect Size Ratio (observed / random_mean)
**Reference line:** y = 1.0 (no effect)
**Color coding:**
- Green: p < 0.05
- Yellow: 0.05 ≤ p < 0.10
- Red: p ≥ 0.10

#### Option 3: P-value Forest Plot

**X-axis:** P-value (log scale)
**Y-axis:** Categories
**Vertical line:** p = 0.05 (significance threshold)
**Points:** P-values for each category
**Error bars:** 95% CI (if applicable)

#### Option 4: Z-score Distribution

**Histogram or violin plot** showing:
- Distribution of z-scores across categories
- Reference line at z = 0 (no effect)
- Shaded regions for |z| > 2 (typically significant)

### Recommended Table Structure (Markdown)

```markdown
| Category | Observed | Random Mean (×10⁻⁴) | Random Std (×10⁻⁴) | Z-Score | P-Value | 95% CI Lower (×10⁻⁴) | 95% CI Upper (×10⁻⁴) | Significant? | Effect Size |
|----------|----------|---------------------|---------------------|---------|---------|----------------------|----------------------|--------------|-------------|
| **Unethical** | 0.0020 | 1.47 | 5.38 | 3.46 | **0.0720** | -9.09 | 12.02 | No (marginal) | 13.7× |
| Deception | 0.0000 | 1.41 | 5.21 | -0.27 | 1.0000 | -8.80 | 11.61 | No | 0.0× |
| Harassment | 0.0000 | 1.65 | 5.73 | -0.29 | 1.0000 | -9.59 | 12.89 | No | 0.0× |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

**Notes:**
- Use scientific notation (×10⁻⁴) for very small numbers to improve readability
- Bold significant or marginally significant results
- Include effect size as a ratio for easy interpretation

## Key Takeaways

1. **Most circuits show zero overlap with random baselines** - This is expected but makes statistical testing difficult
2. **Only Unethical shows marginal signal** - p = 0.072, not significant at α = 0.05
3. **Test may be underpowered** - When both observed and random are near-zero, the test cannot distinguish meaningful structure
4. **Cannot answer research question yet** - Need to verify circuit discovery or use alternative tests
5. **Consider functional tests** - Faithfulness/completeness metrics may be more informative than structural similarity

## Next Steps

1. **Verify circuit contents:** Check if circuits are empty or too small at threshold 0.05
2. **Try lower thresholds:** Test with importance thresholds 0.01, 0.001 to capture more features
3. **Cross-category comparison:** Compare circuits to each other (not just random) to answer monolithic vs. modular
4. **Functional evaluation:** Use faithfulness/completeness metrics to validate circuit importance
5. **Alternative similarity metrics:** Consider other ways to measure circuit similarity (e.g., functional similarity)

