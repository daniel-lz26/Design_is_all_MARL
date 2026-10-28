### Baselines on abstract_10

> Scenario `abstract_10`, layout `arena_13x6ft`, 2 robots, travel noise sigma 0.1, seeds 1000-1019.
> **Inputs contain PLACEHOLDER values** (station coordinates, release and process times). Do not quote these numbers as final.
> On-time % is over all orders, not only delivered ones.

| Policy (robot/machine) | on-time % | mean tardiness (s) | makespan (s) | delivered | all delivered % |
|---|---|---|---|---|---|
| Random | 25.0 ± 2.8 | 45.3 ± 1.6 | 471.3 ± 7.0 | 10.0 | 100 |
| FIFO/FIFO | 21.0 ± 1.4 | 47.0 ± 2.6 | 467.6 ± 4.5 | 10.0 | 100 |
| FIFO/EDD | 20.5 ± 1.0 | 47.3 ± 2.6 | 466.2 ± 3.9 | 10.0 | 100 |
| FIFO/SPT | 21.0 ± 1.4 | 47.0 ± 2.6 | 467.6 ± 4.5 | 10.0 | 100 |
| EDD/FIFO | 24.5 ± 2.4 | 41.9 ± 1.6 | 487.7 ± 4.7 | 10.0 | 100 |
| EDD/EDD | 24.5 ± 2.4 | 41.8 ± 1.6 | 486.3 ± 5.3 | 10.0 | 100 |
| EDD/SPT | 24.5 ± 2.4 | 41.9 ± 1.6 | 487.7 ± 4.7 | 10.0 | 100 |
| NEAREST/FIFO | 29.5 ± 3.2 | 39.9 ± 1.4 | 483.2 ± 6.5 | 10.0 | 100 |
| **NEAREST/EDD** | 29.5 ± 3.2 | 39.8 ± 1.4 | 482.9 ± 6.6 | 10.0 | 100 |
| NEAREST/SPT | 29.5 ± 3.2 | 39.9 ± 1.4 | 483.2 ± 6.5 | 10.0 | 100 |
| MIN_SLACK/FIFO | 22.5 ± 2.1 | 45.3 ± 2.2 | 464.7 ± 6.3 | 10.0 | 100 |
| MIN_SLACK/EDD | 22.5 ± 2.1 | 45.3 ± 2.1 | 466.3 ± 5.5 | 10.0 | 100 |
| MIN_SLACK/SPT | 22.5 ± 2.1 | 45.3 ± 2.2 | 464.7 ± 6.3 | 10.0 | 100 |

Mean ± 95% CI over 20 seeds. Bold = highest on-time %.
