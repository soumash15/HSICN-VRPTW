"""Reference solver used in the HSICN paper: solve Solomon / Gehring-Homberger instances with PyVRP (Hybrid Genetic Search; Wouda, Lan & Kool 2024)
under the hierarchical objective: a large fixed cost per vehicle makes fleet size primary
and distance secondary. Times and distances are scaled by 1000 and rounded to integers
(PyVRP works with integers); reported TD is recomputed in double precision.

Usage: python pyvrp_run.py <instance_folder> <output_folder> <time_limit_s> <INST1,INST2,...> [seeds]
Example (as in the paper): python pyvrp_run.py ./solomon ./results_hgs 170 R101,C201 1,2,3
"""
import sys, time, csv, os, math
import pyvrp
from pyvrp import Model
from pyvrp.stop import MaxRuntime
S = 1000

def parse(path):
    lines = open(path).read().splitlines(); rows = []; nv = cap = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "VEHICLE":
            for j in range(i + 1, i + 5):
                q = lines[j].split()
                if len(q) == 2 and all(x.isdigit() for x in q):
                    nv, cap = int(q[0]), int(q[1]); break
        p = ln.split()
        if len(p) == 7:
            try: rows.append([float(x) for x in p])
            except ValueError: pass
    rows.sort(key=lambda r: r[0]); return rows, nv, cap

def solve(path, runtime, seed):
    rows, nv, cap = parse(path)
    m = Model()
    d0 = rows[0]
    depot = m.add_depot(m.add_location(x=int(d0[1]), y=int(d0[2])), tw_early=0, tw_late=int(d0[5] * S))
    m.add_vehicle_type(num_available=nv, capacity=[cap], fixed_cost=10**9, tw_early=0, tw_late=int(d0[5] * S))
    locs = [depot]
    for r in rows[1:]:
        locs.append(m.add_client(m.add_location(x=int(r[1]), y=int(r[2])), delivery=[int(r[3])],
                                 tw_early=int(round(r[4] * S)), tw_late=int(round(r[5] * S)),
                                 service_duration=int(round(r[6] * S))))
    xy = [(r[1], r[2]) for r in rows]
    L = m.locations
    for i, a in enumerate(L):
        for j, b in enumerate(L):
            if i != j:
                # round distance UP so that integer feasibility implies real-valued feasibility
                d = math.ceil(math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) * S)
                m.add_edge(a, b, distance=d, duration=d)
    res = m.solve(stop=MaxRuntime(runtime), seed=seed, display=False)
    sol = res.best
    routes = [[int(v.idx) + 1 for v in r.schedule() if v.is_client()] for r in sol.routes()]
    td = sum(math.hypot(xy[0][0]-xy[r[0]][0], xy[0][1]-xy[r[0]][1]) + math.hypot(xy[r[-1]][0]-xy[0][0], xy[r[-1]][1]-xy[0][1]) +
             sum(math.hypot(xy[a][0]-xy[b][0], xy[a][1]-xy[b][1]) for a, b in zip(r, r[1:])) for r in routes)
    return routes, td, sol.is_feasible()

if __name__ == "__main__":
    inst_dir, out_dir, runtime = sys.argv[1], sys.argv[2], float(sys.argv[3])
    names = sys.argv[4].split(","); seeds = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else "1").split(",")]
    os.makedirs(os.path.join(out_dir, "route_details"), exist_ok=True)
    out_csv = os.path.join(out_dir, "summary_pyvrp.csv"); new = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline=""); w = csv.writer(fh)
    if new: w.writerow(["instance", "seed", "nv", "td", "feasible", "time_s"])
    files = {os.path.splitext(f)[0].upper(): f for f in os.listdir(inst_dir) if f.lower().endswith(".txt")}
    for name in names:
        for sd in seeds:
            t = time.time()
            routes, td, feas = solve(os.path.join(inst_dir, files[name.upper()]), runtime, sd)
            with open(os.path.join(out_dir, "route_details", f"{name}_s{sd}_routes.txt"), "w") as f:
                f.write(f"Instance: {name}\nTotal Distance: {td:.2f}\nNumber of Vehicles: {len(routes)}\n\n")
                for k, r in enumerate(routes, 1):
                    f.write(f"Vehicle {k}:\n  Route: 0 -> " + " -> ".join(map(str, r)) + " -> 0\n")
            w.writerow([name, sd, len(routes), round(td, 2), feas, round(time.time() - t, 1)]); fh.flush()
            print(name, sd, len(routes), round(td, 2), feas, flush=True)
