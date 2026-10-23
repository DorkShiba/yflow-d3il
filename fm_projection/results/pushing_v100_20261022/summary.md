# pushing: 30 contexts x 4 rollouts, 95% CI = bootstrap over contexts (5000x)
track_d = 0.1 m, a_max = 0.00833 m, baseline = baseline

| run | success % | Δ success %p vs base | entropy | score | mean dist (m) | steps | runaway eps % | Δ runaway %p | gap>d steps % | max gap (m) | a>a_max steps % | jerk (/a_max) | plan ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 91.7 [87.5, 95.8] | – | 0.509 | 0.713 | 0.0401 | 235.4 | 1.7 [0.0, 4.2] | – | 0.17 | 0.0384 | 16.78 | 0.185 | 114.7 |
| final | 92.5 [87.5, 96.7] | +0.8 [-5.8, +6.7] | 0.552 | 0.739 | 0.0424 | 255.9 | 1.7 [0.0, 4.2] | +0.0 [-2.5, +2.5] | 0.28 | 0.0351 | 0.00 | 0.171 | 127.2 |
| every | 91.7 [86.7, 95.8] | +0.0 [-5.8, +5.8] | 0.535 | 0.726 | 0.0475 | 252.4 | 4.2 [0.8, 8.3] | +2.5 [-1.7, +7.5] | 0.47 | 0.0364 | 0.00 | 0.159 | 184.3 |

Δ: paired difference on the same resampled contexts; an interval that contains 0 means no detectable difference.
