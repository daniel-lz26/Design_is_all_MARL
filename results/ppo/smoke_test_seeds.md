# PPO smoke checks (250k decisions: 50k behaviour cloning + 200k PPO)

100 test seeds (20000-20099), on-time %. "BC start" is the behaviour-cloned actor before PPO.
Differences are paired by seed with a bootstrap p-value. One torch thread per run on Daniel's PC
(Ryzen 5 3600), sharing the CPU with other jobs. Learning curves (v2):
smoke_learning_curves.png.

## Observation v2 (env-v1.2): current

MAPPO took 6.8 min and IPPO 5.3 min.

| Scenario | Best rule | Best auction | BC start | MAPPO smoke | IPPO smoke | MAPPO - BC | IPPO - BC |
|---|---|---|---|---|---|---|---|
| cell_random_10 | BOTTLENECK/SPT 49.7 | AUCTION/SPT 51.9 | 49.5 | 50.0 | 51.4 | +0.5 ± 1.3 (p=0.49) | +1.9 ± 2.0 (p=0.065) |
| cell_random_10_disrupt | SPT/SPT 26.8 | AUCTION/SPT 26.4 | 26.0 | 28.5 | 28.4 | +2.5 ± 1.7 (p=0.005) | +2.4 ± 1.8 (p=0.012) |
| cell_10_disrupt | SPT/EDD 15.9 | AUCTION/EDD 11.5 | 10.2 | 17.1 | 14.9 | +6.9 ± 1.4 (p<0.001) | +4.7 ± 1.4 (p<0.001) |

Behaviour cloning on v2: 100% match on the training demonstrations; 49.5% vs the teacher's
49.7% on cell_random_10.

## Observation v1 (env-v1.1): superseded

MAPPO took 6.4 min and IPPO 4.9 min. Best auction here is the old AUCTION (now AUCTION_v0).

| Scenario | Best rule | BC start | MAPPO smoke | IPPO smoke | MAPPO - BC | IPPO - BC |
|---|---|---|---|---|---|---|
| cell_random_10 | BOTTLENECK/SPT 49.7 | 49.5 | 51.3 | 50.3 | +1.8 ± 1.5 (p=0.016) | +0.8 ± 1.4 (p=0.290) |
| cell_random_10_disrupt | SPT/SPT 26.8 | 25.9 | 28.4 | 26.6 | +2.5 ± 1.7 (p=0.004) | +0.7 ± 1.6 (p=0.398) |
| cell_10_disrupt | SPT/EDD 15.9 | 10.4 | 16.6 | 15.6 | +6.2 ± 1.6 (p<0.001) | +5.1 ± 1.4 (p<0.001) |
