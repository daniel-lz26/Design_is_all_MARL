### MAPPO vs baselines on random_10, 2 robots

> Scenario `random_10`, layout `arena_13x6ft`, 2 robots, travel noise sigma 0.1, seeds 2000-2019.
> MAPPO policies trained on `random_10` (never on `random_10`); 3 training seeds. `MAPPO (all seeds)` pools their episodes (3 x 20).
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 25.0 ± 4.9 | 51.8 ± 6.5 | 445.0 ± 11.7 | 10.0 | 100 |
| FIFO/FIFO | 23.5 ± 5.1 | 52.2 ± 6.3 | 454.8 ± 11.4 | 10.0 | 100 |
| FIFO/EDD | 23.5 ± 4.9 | 52.4 ± 6.2 | 452.6 ± 12.8 | 10.0 | 100 |
| FIFO/SPT | 25.5 ± 5.1 | 52.3 ± 6.3 | 452.7 ± 12.7 | 10.0 | 100 |
| EDD/FIFO | 24.0 ± 5.1 | 49.7 ± 5.9 | 465.4 ± 11.8 | 10.0 | 100 |
| EDD/EDD | 23.5 ± 4.6 | 49.7 ± 6.0 | 465.7 ± 11.6 | 10.0 | 100 |
| EDD/SPT | 26.0 ± 4.4 | 49.6 ± 5.8 | 464.8 ± 10.5 | 10.0 | 100 |
| NEAREST/FIFO | 27.0 ± 5.7 | 45.6 ± 6.8 | 448.3 ± 11.8 | 10.0 | 100 |
| NEAREST/EDD | 26.5 ± 5.9 | 45.5 ± 6.7 | 451.3 ± 11.3 | 10.0 | 100 |
| NEAREST/SPT | 29.0 ± 5.7 | 45.8 ± 6.6 | 452.0 ± 12.3 | 10.0 | 100 |
| MIN_SLACK/FIFO | 22.0 ± 4.5 | 51.1 ± 5.9 | 451.4 ± 10.9 | 10.0 | 100 |
| MIN_SLACK/EDD | 21.5 ± 4.1 | 51.0 ± 5.9 | 451.8 ± 12.0 | 10.0 | 100 |
| MIN_SLACK/SPT | 25.5 ± 4.4 | 50.7 ± 5.6 | 452.4 ± 13.0 | 10.0 | 100 |
| MAPPO (all seeds) | 33.7 ± 3.1 | 47.9 ± 3.2 | 457.8 ± 7.1 | 10.0 | 100 |
| MAPPO s0 | 33.0 ± 5.9 | 48.1 ± 6.3 | 460.8 ± 12.3 | 10.0 | 100 |
| MAPPO s1 | 32.0 ± 5.2 | 48.2 ± 5.6 | 457.3 ± 13.6 | 10.0 | 100 |
| **MAPPO s2** | 36.0 ± 6.3 | 47.3 ± 6.1 | 455.4 ± 13.8 | 10.0 | 100 |
| MAPPO robots (s1) + EDD machines | 28.0 ± 6.4 | 46.7 ± 5.3 | 456.9 ± 13.7 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
