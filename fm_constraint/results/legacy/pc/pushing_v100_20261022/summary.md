# pushing: 30 contexts x 4 rollouts, 95% CI = bootstrap over contexts (5000x)
track_d = 0.1 m, a_max = 0.00833 m, baseline = baseline

| run | success % | Δ success %p vs base | entropy | score | mean dist (m) | steps | runaway eps % | Δ runaway %p | gap>d steps % | max gap (m) | a>a_max steps % | jerk (/a_max) | plan ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 91.7 [87.5, 95.8] | – | 0.509 | 0.713 | 0.0401 | 235.4 | 1.7 [0.0, 4.2] | – | 0.17 | 0.0384 | 16.78 | 0.185 | 114.7 |
| pc | 98.3 [95.8, 100.0] | +6.7 [+1.7, +11.7] | 0.229 | 0.606 | 0.0310 | 227.6 | 0.0 [0.0, 0.0] | -1.7 [-4.2, +0.0] | 0.00 | 0.0320 | 0.00 | 0.105 | 179.6 |
| pc-a1 | 95.8 [92.5, 99.2] | +4.2 [-0.8, +9.2] | 0.241 | 0.600 | 0.0343 | 228.6 | 0.8 [0.0, 2.5] | -0.8 [-3.3, +1.7] | 0.01 | 0.0297 | 0.00 | 0.094 | 175.8 |
| pc-noCBF | 95.0 [91.7, 98.3] | +3.3 [-0.8, +7.5] | 0.264 | 0.607 | 0.0363 | 202.5 | 0.8 [0.0, 2.5] | -0.8 [-3.3, +1.7] | 0.25 | 0.0341 | 8.62 | 0.107 | 130.2 |

Δ: paired difference on the same resampled contexts; an interval that contains 0 means no detectable difference.
