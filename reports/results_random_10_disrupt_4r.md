### MAPPO vs baselines on random_10_disrupt, 4 robots

> Scenario `random_10_disrupt`, layout `arena_13x6ft`, 4 robots, travel noise sigma 0.1, seeds 2000-2019.
> MAPPO families `mappo_4r`, `mappo_4r_disrupt` trained on `random_10`, `random_10_disrupt` (never on `random_10_disrupt`). `(all seeds)` rows pool each family's training seeds.
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.
> Disruptions on: blocked and delayed trips, machine breakdowns, robot outages (rates are scenario design choices, not measurements).

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 29.5 ± 5.6 | 52.9 ± 8.6 | 471.7 ± 21.6 | 10.0 | 100 |
| FIFO/FIFO | 26.5 ± 6.1 | 53.8 ± 9.7 | 474.7 ± 22.2 | 10.0 | 100 |
| FIFO/EDD | 25.5 ± 6.2 | 53.9 ± 9.3 | 469.9 ± 21.5 | 10.0 | 100 |
| FIFO/SPT | 32.0 ± 5.6 | 53.0 ± 8.8 | 472.9 ± 24.3 | 10.0 | 100 |
| EDD/FIFO | 27.0 ± 6.3 | 53.0 ± 9.4 | 475.5 ± 21.8 | 10.0 | 100 |
| EDD/EDD | 25.5 ± 6.2 | 53.6 ± 9.5 | 470.2 ± 22.4 | 10.0 | 100 |
| EDD/SPT | 30.0 ± 5.9 | 52.7 ± 8.7 | 476.8 ± 23.9 | 10.0 | 100 |
| NEAREST/FIFO | 27.5 ± 6.2 | 53.0 ± 9.6 | 475.4 ± 22.3 | 10.0 | 100 |
| NEAREST/EDD | 26.5 ± 5.7 | 53.1 ± 9.4 | 470.7 ± 21.9 | 10.0 | 100 |
| NEAREST/SPT | 30.5 ± 6.0 | 52.3 ± 8.6 | 475.6 ± 24.1 | 10.0 | 100 |
| MIN_SLACK/FIFO | 25.5 ± 6.0 | 54.1 ± 9.6 | 475.8 ± 21.9 | 10.0 | 100 |
| MIN_SLACK/EDD | 24.0 ± 6.0 | 54.6 ± 9.5 | 467.5 ± 21.9 | 10.0 | 100 |
| MIN_SLACK/SPT | 29.5 ± 6.0 | 53.7 ± 8.9 | 478.3 ± 24.1 | 10.0 | 100 |
| mappo_4r (all seeds) | 33.5 ± 2.8 | 55.0 ± 5.1 | 474.7 ± 12.2 | 10.0 | 100 |
| mappo_4r s0 | 34.0 ± 5.6 | 55.1 ± 9.5 | 474.0 ± 22.9 | 10.0 | 100 |
| mappo_4r s1 | 33.0 ± 5.3 | 55.0 ± 9.7 | 475.4 ± 23.6 | 10.0 | 100 |
| mappo_4r s2 | 33.5 ± 5.1 | 54.8 ± 9.7 | 474.8 ± 22.4 | 10.0 | 100 |
| mappo_4r robots (s2) + EDD machines | 26.0 ± 5.8 | 53.9 ± 9.8 | 467.5 ± 22.4 | 10.0 | 100 |
| mappo_4r_disrupt (all seeds) | 34.0 ± 2.9 | 54.9 ± 5.0 | 470.5 ± 12.1 | 10.0 | 100 |
| mappo_4r_disrupt s0 | 32.5 ± 6.0 | 54.9 ± 9.3 | 460.7 ± 20.7 | 10.0 | 100 |
| **mappo_4r_disrupt s1** | 35.0 ± 5.1 | 54.8 ± 9.5 | 476.3 ± 23.4 | 10.0 | 100 |
| mappo_4r_disrupt s2 | 34.5 ± 5.1 | 55.1 ± 9.4 | 474.4 ± 23.5 | 10.0 | 100 |
| mappo_4r_disrupt robots (s1) + EDD machines | 27.5 ± 6.2 | 53.3 ± 9.4 | 467.6 ± 22.4 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
