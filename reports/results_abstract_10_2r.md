### MAPPO vs baselines on abstract_10, 2 robots

> Scenario `abstract_10`, layout `arena_13x6ft`, 2 robots, travel noise sigma 0.1, seeds 1000-1019.
> MAPPO policies trained on `random_10` (never on `abstract_10`); 3 training seeds. `MAPPO (all seeds)` pools their episodes (3 x 20).
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 25.5 ± 2.8 | 46.2 ± 1.8 | 470.5 ± 7.4 | 10.0 | 100 |
| FIFO/FIFO | 21.0 ± 1.4 | 47.0 ± 2.6 | 467.6 ± 4.5 | 10.0 | 100 |
| FIFO/EDD | 20.5 ± 1.0 | 47.3 ± 2.6 | 466.2 ± 3.9 | 10.0 | 100 |
| FIFO/SPT | 21.0 ± 1.4 | 47.0 ± 2.6 | 467.6 ± 4.5 | 10.0 | 100 |
| EDD/FIFO | 24.5 ± 2.4 | 41.9 ± 1.6 | 487.7 ± 4.7 | 10.0 | 100 |
| EDD/EDD | 24.5 ± 2.4 | 41.8 ± 1.6 | 486.3 ± 5.3 | 10.0 | 100 |
| EDD/SPT | 24.5 ± 2.4 | 41.9 ± 1.6 | 487.7 ± 4.7 | 10.0 | 100 |
| NEAREST/FIFO | 29.5 ± 3.2 | 39.9 ± 1.4 | 483.2 ± 6.5 | 10.0 | 100 |
| NEAREST/EDD | 29.5 ± 3.2 | 39.8 ± 1.4 | 482.9 ± 6.6 | 10.0 | 100 |
| NEAREST/SPT | 29.5 ± 3.2 | 39.9 ± 1.4 | 483.2 ± 6.5 | 10.0 | 100 |
| MIN_SLACK/FIFO | 22.5 ± 2.1 | 45.3 ± 2.2 | 464.7 ± 6.3 | 10.0 | 100 |
| MIN_SLACK/EDD | 22.5 ± 2.1 | 45.3 ± 2.1 | 466.3 ± 5.5 | 10.0 | 100 |
| MIN_SLACK/SPT | 22.5 ± 2.1 | 45.3 ± 2.2 | 464.7 ± 6.3 | 10.0 | 100 |
| MAPPO (all seeds) | 30.7 ± 1.5 | 40.2 ± 1.0 | 479.2 ± 3.4 | 10.0 | 100 |
| MAPPO s0 | 31.0 ± 3.0 | 40.4 ± 1.4 | 480.9 ± 5.8 | 10.0 | 100 |
| MAPPO s1 | 31.0 ± 3.0 | 39.9 ± 2.3 | 476.4 ± 6.1 | 10.0 | 100 |
| MAPPO s2 | 30.0 ± 2.6 | 40.3 ± 2.0 | 480.4 ± 7.0 | 10.0 | 100 |
| **MAPPO robots (s1) + EDD machines** | 31.0 ± 3.0 | 39.8 ± 2.3 | 475.8 ± 6.3 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
