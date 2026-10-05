# HSICN: Hierarchical Swarm Intelligence with Chaos Network for the VRPTW

Reference implementation of **HSICN**, a cyclic collaborative framework for the Vehicle Routing
Problem with Time Windows (VRPTW), as described in:

> M. Soumya Krishnan and Vimina E. R., *Hierarchical Swarm Intelligence with Chaos Network (HSICN):
> A Cyclic Collaborative Framework for Solving Vehicle Routing Problem with Time Windows*,
> Scientific Reports (under review).

## Method in brief

HSICN organizes seven modules in two levels and activates them in a fixed cycle (`m = t mod 7`)
around a shared Collaborative Elite Archive:

| Level | Modules |
|---|---|
| Population-based | Ant Colony Optimization (elite-guided), Particle Swarm Optimization (chaotic inertia), Artificial Bee Colony |
| Refinement | Enhanced Local Search, chaos-controlled ruin-and-recreate, Route Minimization with an ejection pool, Ejection Chain |

Every candidate solution is repaired to full feasibility before evaluation, and it is accepted only
if it uses fewer vehicles, or the same number of vehicles and a shorter distance (hierarchical
objective of the Solomon and Gehring–Homberger benchmarks). A final intensification phase follows
the cyclic search.

## Requirements

Python 3.9 or later with NumPy and Matplotlib:

```
pip install -r requirements.txt
```

The experiments in the paper were run with Python 3.13.7, NumPy 2.3.3 and Matplotlib 3.10.6.

## Benchmark instances

The Solomon (100 customers) and Gehring–Homberger (200 customers) instances are available from the
SINTEF TOP repository: https://www.sintef.no/projectweb/top/vrptw/  
Place the instance files (e.g. `R101.txt`, `C1_2_1.txt`) in one folder.

## Usage

```
# all instances in ./solomon, five seeds (as in the paper)
python hsicn.py --instances ./solomon --output ./results --seeds 1,2,3,4,5

# selected instances, one seed, four parallel processes
python hsicn.py --instances ./solomon --only R101,C201 --seeds 1 --workers 4
```

| Option | Meaning (default) |
|---|---|
| `--instances` | folder with the instance files (`./solomon`) |
| `--output` | output folder (`./results`) |
| `--iterations` | number of cyclic iterations T (`100`) |
| `--seeds` | comma-separated random seeds (`1`) |
| `--only` | comma-separated instance names to run (all) |
| `--workers` | number of parallel processes (`1`) |

All parameters are those reported in the paper (Table 2); a run with a given seed is fully
reproducible.

## Output

| File | Content |
|---|---|
| `summary.csv` | one line per run: instance, seed, vehicles, total distance, feasibility, run time |
| `route_details/<instance>_s<seed>_routes.txt` | total distance, number of vehicles and every route |
| `routes/<instance>_s<seed>_routes.png` | plot of the solution |
| `logs/<instance>_s<seed>_log.txt` | improvements found during the run |

The solutions reported in the paper, the per-run results and an independent feasibility checker
are provided in the supplementary material of the article (Supplementary Data 1).

## Citation

If you use this code, please cite the article above.

## License

Released under the MIT License (see `LICENSE`).
