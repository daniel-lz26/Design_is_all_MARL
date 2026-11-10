# PPO smoke check (250k decisions: 50k behaviour cloning + 200k PPO)

100 test seeds (20000-20099), on-time %. "BC start" is the behaviour-cloned actor before PPO.
Differences are paired by seed with a bootstrap p-value. Trained on this PC (Ryzen 5 3600),
one torch thread per run; MAPPO took 6.4 min, IPPO 4.9 min.

| Scenario | Best rule | BC start | MAPPO smoke | IPPO smoke | MAPPO - BC | IPPO - BC |
|---|---|---|---|---|---|---|
| cell_random_10 | BOTTLENECK/SPT 49.7 | 49.5 | 51.3 | 50.3 | +1.8 ± 1.5 (p=0.016) | +0.8 ± 1.4 (p=0.290) |
| cell_random_10_disrupt | SPT/SPT 26.8 | 25.9 | 28.4 | 26.6 | +2.5 ± 1.7 (p=0.004) | +0.7 ± 1.6 (p=0.398) |
| cell_10_disrupt | SPT/EDD 15.9 | 10.4 | 16.6 | 15.6 | +6.2 ± 1.6 (p<0.001) | +5.1 ± 1.4 (p<0.001) |

Behaviour cloning alone (BOTTLENECK/SPT teacher, 50k decisions): 100% match on the training
demonstrations; on cell_random_10 test seeds 49.5% vs the teacher's 49.7%, with 93 of 100
episodes identical. Learning curves: smoke_learning_curves.png.
