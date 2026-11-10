### MAPPO vs baselines on abstract_10, 4 robots

> Scenario `abstract_10`, layout `arena_13x6ft`, 4 robots, travel noise sigma 0.1, seeds 1000-1019.
> MAPPO families `mappo_4r`, `mappo_4r_disrupt` trained on `random_10`, `random_10_disrupt` (never on `abstract_10`). `(all seeds)` rows pool each family's training seeds.
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 43.5 ± 4.4 | 26.9 ± 1.3 | 449.8 ± 3.8 | 10.0 | 100 |
| FIFO/FIFO | 43.0 ± 5.9 | 26.1 ± 0.9 | 449.1 ± 3.3 | 10.0 | 100 |
| FIFO/EDD | 39.5 ± 5.6 | 26.2 ± 1.0 | 451.4 ± 1.6 | 10.0 | 100 |
| FIFO/SPT | 43.0 ± 5.9 | 26.1 ± 0.9 | 449.1 ± 3.3 | 10.0 | 100 |
| EDD/FIFO | 43.0 ± 5.9 | 26.1 ± 0.9 | 448.0 ± 2.7 | 10.0 | 100 |
| EDD/EDD | 40.0 ± 5.3 | 26.3 ± 1.0 | 451.3 ± 1.6 | 10.0 | 100 |
| EDD/SPT | 43.0 ± 5.9 | 26.1 ± 0.9 | 448.0 ± 2.7 | 10.0 | 100 |
| NEAREST/FIFO | 43.5 ± 5.7 | 26.2 ± 0.9 | 449.8 ± 3.3 | 10.0 | 100 |
| NEAREST/EDD | 38.5 ± 6.1 | 26.0 ± 0.8 | 451.2 ± 0.9 | 10.0 | 100 |
| NEAREST/SPT | 43.5 ± 5.7 | 26.2 ± 0.9 | 449.8 ± 3.3 | 10.0 | 100 |
| MIN_SLACK/FIFO | 43.0 ± 5.1 | 26.5 ± 0.8 | 451.9 ± 3.9 | 10.0 | 100 |
| MIN_SLACK/EDD | 38.0 ± 5.6 | 26.1 ± 0.8 | 451.2 ± 0.9 | 10.0 | 100 |
| MIN_SLACK/SPT | 43.0 ± 5.1 | 26.5 ± 0.8 | 451.9 ± 3.9 | 10.0 | 100 |
| mappo_4r (all seeds) | 43.8 ± 2.6 | 27.5 ± 0.6 | 451.8 ± 2.0 | 10.0 | 100 |
| mappo_4r s0 | 43.0 ± 5.9 | 27.0 ± 1.1 | 450.0 ± 3.3 | 10.0 | 100 |
| mappo_4r s1 | 43.0 ± 5.5 | 27.9 ± 1.2 | 453.3 ± 3.6 | 10.0 | 100 |
| **mappo_4r s2** | 45.5 ± 2.8 | 27.7 ± 1.0 | 452.0 ± 4.1 | 10.0 | 100 |
| mappo_4r robots (s2) + EDD machines | 41.5 ± 3.5 | 26.1 ± 0.8 | 451.1 ± 1.2 | 10.0 | 100 |
| mappo_4r_disrupt (all seeds) | 42.0 ± 3.0 | 27.1 ± 0.5 | 449.0 ± 1.6 | 10.0 | 100 |
| mappo_4r_disrupt s0 | 40.0 ± 5.3 | 27.3 ± 1.0 | 449.3 ± 3.6 | 10.0 | 100 |
| mappo_4r_disrupt s1 | 43.0 ± 5.9 | 27.0 ± 1.0 | 448.9 ± 2.7 | 10.0 | 100 |
| mappo_4r_disrupt s2 | 43.0 ± 5.9 | 27.0 ± 1.0 | 448.9 ± 2.7 | 10.0 | 100 |
| mappo_4r_disrupt robots (s1) + EDD machines | 40.0 ± 5.3 | 26.3 ± 1.0 | 451.3 ± 1.6 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
