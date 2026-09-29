# Core evaluation pipeline

This repository evaluates a prepared FactorAtlas dataset; it does not generate images or distribute model weights.

## Input contract

The data directory contains `metadata.json`, renderer-relative image files, and `<tag>_features.npz`. The canonical product is 8 shapes x 12 hues x 10 patterns x 12 contexts x 2 seeds = 23,040 records.

## Core method

For factor value v, calibration C, validation V, and test T:

    c_v = normalize(mean(x_i for i in C with y_i = v))
    d_v = normalize(c_v - mean(c_u for u != v))
    q_v(alpha) = normalize(t_v + alpha * d_v)

`t_v` is the frozen native-text embedding. Select alpha only on validation macro-mAP over 0.00, 0.05, ..., 8.00; break ties by validation top-1, then smaller alpha. Test data never chooses directions, prompts, or alpha.

Report D with `c_v`, A with `t_v`, and R with `q_v(alpha)`, plus cyclic-wrong, 64-random-direction, and text-only OVR controls. Scores are image-wise top-1 and query-wise one-vs-rest macro-mAP.

The implementation is `experiments/access_gap/factor_access_protocol.py` and `experiments/access_gap/factor_access_audit.py`.
