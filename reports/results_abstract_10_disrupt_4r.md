### MAPPO vs baselines on abstract_10_disrupt, 4 robots

> Scenario `abstract_10_disrupt`, layout `arena_13x6ft`, 4 robots, travel noise sigma 0.1, seeds 1000-1019.
> MAPPO families `mappo_4r`, `mappo_4r_disrupt` trained on `random_10`, `random_10_disrupt` (never on `abstract_10_disrupt`). `(all seeds)` rows pool each family's training seeds.
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.
> Disruptions on: blocked and delayed trips, machine breakdowns, robot outages (rates are scenario design choices, not measurements).

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 29.0 ± 5.7 | 56.4 ± 9.3 | 496.0 ± 12.9 | 10.0 | 100 |
| FIFO/FIFO | 22.0 ± 4.9 | 54.3 ± 9.8 | 493.2 ± 18.4 | 10.0 | 100 |
| FIFO/EDD | 22.0 ± 4.9 | 54.2 ± 9.8 | 495.6 ± 19.0 | 10.0 | 100 |
| FIFO/SPT | 22.0 ± 4.9 | 54.3 ± 9.8 | 493.2 ± 18.4 | 10.0 | 100 |
| EDD/FIFO | 23.0 ± 5.3 | 53.7 ± 9.8 | 494.0 ± 18.9 | 10.0 | 100 |
| EDD/EDD | 22.5 ± 4.8 | 53.4 ± 9.3 | 495.4 ± 17.2 | 10.0 | 100 |
| EDD/SPT | 23.0 ± 5.3 | 53.7 ± 9.8 | 494.0 ± 18.9 | 10.0 | 100 |
| NEAREST/FIFO | 24.5 ± 5.1 | 53.1 ± 10.1 | 495.0 ± 20.1 | 10.0 | 100 |
| NEAREST/EDD | 23.5 ± 4.6 | 52.9 ± 9.5 | 497.2 ± 18.9 | 10.0 | 100 |
| NEAREST/SPT | 24.5 ± 5.1 | 53.1 ± 10.1 | 495.0 ± 20.1 | 10.0 | 100 |
| MIN_SLACK/FIFO | 21.5 ± 4.9 | 53.5 ± 9.5 | 494.7 ± 19.9 | 10.0 | 100 |
| MIN_SLACK/EDD | 21.5 ± 4.9 | 53.6 ± 9.2 | 496.7 ± 18.9 | 10.0 | 100 |
| MIN_SLACK/SPT | 21.5 ± 4.9 | 53.5 ± 9.5 | 494.7 ± 19.9 | 10.0 | 100 |
| mappo_4r (all seeds) | 32.3 ± 2.8 | 56.7 ± 4.7 | 497.2 ± 10.4 | 10.0 | 100 |
| mappo_4r s0 | 30.5 ± 5.6 | 55.8 ± 9.1 | 496.2 ± 19.8 | 10.0 | 100 |
| **mappo_4r s1** | 33.5 ± 5.3 | 57.0 ± 9.2 | 495.6 ± 20.3 | 10.0 | 100 |
| mappo_4r s2 | 33.0 ± 4.8 | 57.4 ± 8.5 | 499.7 ± 18.3 | 10.0 | 100 |
| mappo_4r robots (s2) + EDD machines | 22.0 ± 4.7 | 55.5 ± 8.9 | 498.7 ± 18.9 | 10.0 | 100 |
| mappo_4r_disrupt (all seeds) | 31.3 ± 2.4 | 55.2 ± 5.0 | 494.1 ± 10.3 | 10.0 | 100 |
| mappo_4r_disrupt s0 | 32.0 ± 3.9 | 55.8 ± 9.9 | 492.8 ± 19.6 | 10.0 | 100 |
| mappo_4r_disrupt s1 | 31.5 ± 4.4 | 55.1 ± 9.3 | 494.3 ± 19.1 | 10.0 | 100 |
| mappo_4r_disrupt s2 | 30.5 ± 4.9 | 54.7 ± 9.0 | 495.3 ± 19.5 | 10.0 | 100 |
| mappo_4r_disrupt robots (s1) + EDD machines | 23.5 ± 4.6 | 53.0 ± 9.1 | 494.5 ± 15.8 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
