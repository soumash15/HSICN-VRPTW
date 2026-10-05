#!/usr/bin/env python3
"""
HSICN -- Hierarchical Swarm Intelligence with Chaos Network for the
Vehicle Routing Problem with Time Windows (VRPTW).

Reference implementation of the method described in
  M. Soumya Krishnan and Vimina E. R., "Hierarchical Swarm Intelligence with Chaos
  Network (HSICN): A Cyclic Collaborative Framework for Solving Vehicle Routing
  Problem with Time Windows", Scientific Reports (under review).

Framework
  Phase 0  Solomon I1 construction + enhanced local search (ELS).
  Phase 1  T cyclic iterations; at iteration t the module m = t mod 7 is applied:
           0 ACO (elite-guided), 1 PSO (chaotic inertia), 2 ABC, 3 ELS,
           4 chaos-controlled ruin-and-recreate, 5 route minimization with an
           ejection pool, 6 ejection chain.  Every candidate is repaired to full
           feasibility and accepted only if F(s) < F(S_best), i.e. fewer vehicles,
           or the same number of vehicles and a shorter distance.  The three best
           solutions form the Collaborative Elite Archive.
  Phase 2  Final intensification (ELS, route minimization, ruin-and-recreate).

Objective: F(s) = TD + alpha*K + P(s), alpha = 1e4 (hierarchical: vehicles first,
then distance).  Distances are Euclidean in double precision; travel time equals
distance; service times and the depot due date are respected.

Feasibility tests for insertion, relocate, exchange and 2-opt* moves run in O(1)
using forward earliest-start and backward latest-start times.

Usage
  python hsicn.py --instances ./solomon --output ./results --seeds 1,2,3,4,5
  python hsicn.py --instances ./solomon --only R101,C201 --seeds 1 --workers 4
"""

import math
import random
import time
import os
import csv
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-7
BIG_M = 10000.0          # alpha in F(s) = Z2 + alpha*K + P(s)


# --------------------------------------------------------------------------
# Instance parsing
# --------------------------------------------------------------------------
def parse_instance(file_path):
    """Parse a Solomon / Gehring-Homberger VRPTW file.
    Returns coords, demand, ready, due, service (lists indexed by node), capacity, max_vehicles."""
    lines = open(file_path).read().splitlines()
    cap = nv = None
    rows = []
    for i, ln in enumerate(lines):
        p = ln.split()
        if ln.strip().upper() == "VEHICLE":
            for j in range(i + 1, min(i + 5, len(lines))):
                q = lines[j].split()
                if len(q) == 2 and all(x.isdigit() for x in q):
                    nv, cap = int(q[0]), int(q[1])
                    break
        if len(p) == 7:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    rows.sort(key=lambda r: r[0])
    n = int(rows[-1][0]) + 1
    coords = [(0.0, 0.0)] * n
    demand = [0] * n
    ready = [0.0] * n
    due = [0.0] * n
    service = [0.0] * n
    for r in rows:
        k = int(r[0])
        coords[k] = (r[1], r[2])
        demand[k] = int(r[3])
        ready[k], due[k], service[k] = r[4], r[5], r[6]
    return coords, demand, ready, due, service, cap, nv


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class HSICNSolver:
    def __init__(self, coords, demand, ready, due, service, capacity, max_vehicles, seed=None):
        self.coords = coords
        self.dem = demand
        self.rd = ready
        self.du = due
        self.sv = service
        self.cap = capacity
        self.max_vehicles = max_vehicles
        self.n = len(coords) - 1
        self.D = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in coords] for a in coords]
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        self.pso_params = dict(swarm_size=30, w_max=0.9, w_min=0.4, max_iterations=30)
        self.aco_params = dict(num_ants=30, alpha=1.0, beta=3.0, rho=0.1, pheromone_init=1.0,
                               tau_min=0.01, tau_max=10.0, elite_weight=2.0, Q=1.0)
        self.abc_params = dict(colony_size=30, limit=50, max_cycles=30)
        self.local_search_prob = 0.3
        self.chaos_r = 3.99
        self.chaos_state = self.rng.random()
        self.pher = np.full((self.n + 1, self.n + 1), self.aco_params["pheromone_init"])
        self._i1_cache = None
        self.rr_T0, self.rr_Tf = 20.0, 1.0

    # ------------------------------------------------------------------ basics
    def _chaos_next(self):
        self.chaos_state = self.chaos_r * self.chaos_state * (1 - self.chaos_state)
        if self.chaos_state <= 0 or self.chaos_state >= 1:      # guard numerical collapse
            self.chaos_state = self.rng.random()
        return self.chaos_state

    def route_dist(self, r):
        D = self.D
        if not r:
            return 0.0
        d = D[0][r[0]] + D[r[-1]][0]
        for a, b in zip(r, r[1:]):
            d += D[a][b]
        return d

    def total_dist(self, routes):
        return sum(self.route_dist(r) for r in routes)

    def route_ok(self, r):
        """Capacity + time windows + return to depot before depot due date."""
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        if sum(dem[c] for c in r) > self.cap:
            return False
        t, prev = 0.0, 0
        for c in r:
            t = t + sv[prev] + D[prev][c]
            if t > du[c] + EPS:
                return False
            if t < rd[c]:
                t = rd[c]
            prev = c
        return t + sv[prev] + D[prev][0] <= du[0] + EPS

    def solution_ok(self, routes):
        seen = [c for r in routes for c in r]
        if len(seen) != self.n or set(seen) != set(range(1, self.n + 1)):
            return False
        return all(self.route_ok(r) for r in routes)

    def rdata(self, r):
        """Forward earliest start A, backward latest start Z and cumulative load L on [0]+r+[0]."""
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        full = [0] + r + [0]
        m = len(full)
        A = [0.0] * m
        L = [0] * m
        t, load = 0.0, 0
        for k in range(1, m):
            a, b = full[k - 1], full[k]
            t = t + sv[a] + D[a][b]
            if t < rd[b]:
                t = rd[b]
            A[k] = t
            load += dem[b]
            L[k] = load
        Z = [0.0] * m
        Z[m - 1] = du[0]
        for k in range(m - 2, -1, -1):
            a, b = full[k], full[k + 1]
            z = Z[k + 1] - sv[a] - D[a][b]
            Z[k] = du[a] if du[a] < z else z
        return full, A, Z, L

    def best_insertion(self, u, datas):
        """Cheapest feasible insertion of u; datas = list of rdata tuples. Returns (cost, ri, pos)."""
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        best = (float("inf"), None, None)
        for ri, (full, A, Z, L) in enumerate(datas):
            if L[-1] + dem[u] > self.cap:
                continue
            for k in range(len(full) - 1):
                a, b = full[k], full[k + 1]
                cost = D[a][u] + D[u][b] - D[a][b]
                if cost >= best[0]:
                    continue
                tu = A[k] + sv[a] + D[a][u]
                if tu > du[u] + EPS:
                    continue
                if tu < rd[u]:
                    tu = rd[u]
                if tu + sv[u] + D[u][b] <= Z[k + 1] + EPS:
                    best = (cost, ri, k)
        return best

    # ------------------------------------------------------------ fitness (Eq. 3)
    def evaluate(self, routes):
        """F(s) = Z2 + alpha*K + P(s). Solutions passed here are repaired, so P(s)=0
        unless something upstream failed; infeasible solutions are penalised."""
        td = 0.0
        pen = 0.0
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        for r in routes:
            td += self.route_dist(r)
            load = sum(dem[c] for c in r)
            if load > self.cap:
                pen += 5000.0 * (load - self.cap)
            t, prev = 0.0, 0
            for c in r:
                t = t + sv[prev] + D[prev][c]
                if t > du[c]:
                    pen += 100.0 * (t - du[c])
                t = max(t, rd[c])
                prev = c
            t = t + sv[prev] + D[prev][0]
            if t > du[0]:
                pen += 100.0 * (t - du[0])
        served = sorted(c for r in routes for c in r)
        if served != list(range(1, self.n + 1)):
            pen += 1e7
        feasible = pen == 0.0
        return td + BIG_M * len(routes) + pen, td, feasible

    # ------------------------------------------------------------ repair
    def repair(self, routes):
        seen = set()
        clean = []
        for r in routes:
            nr = []
            for c in r:
                if c != 0 and c not in seen and 1 <= c <= self.n:
                    nr.append(c)
                    seen.add(c)
            if nr:
                clean.append(nr)
        ejected = [c for c in range(1, self.n + 1) if c not in seen]
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        fixed = []
        for r in clean:
            # capacity: eject largest-demand customers until the load fits
            while sum(dem[c] for c in r) > self.cap:
                c = max(r, key=lambda x: dem[x])
                r.remove(c)
                ejected.append(c)
            # time windows: eject the first customer served late, repeat
            while r and not self.route_ok(r):
                t, prev, bad = 0.0, 0, None
                for i, c in enumerate(r):
                    t = t + sv[prev] + D[prev][c]
                    if t > du[c] + EPS:
                        bad = i
                        break
                    t = max(t, rd[c])
                    prev = c
                if bad is None:            # only the depot return is late
                    bad = len(r) - 1
                ejected.append(r.pop(bad))
            if r:
                fixed.append(r)
        if ejected:
            ejected.sort(key=lambda c: du[c] - rd[c])          # tightest windows first
            datas = [self.rdata(r) for r in fixed]
            for u in ejected:
                cost, ri, k = self.best_insertion(u, datas)
                if ri is None:
                    fixed.append([u])
                    datas.append(self.rdata([u]))
                else:
                    fixed[ri].insert(k, u)
                    datas[ri] = self.rdata(fixed[ri])
        return fixed

    # ------------------------------------------------------ Solomon I1 (cached)
    def solomon_i1(self):
        if self._i1_cache is not None:
            return deepcopy(self._i1_cache)
        D, du = self.D, self.du
        unvisited = set(range(1, self.n + 1))
        routes = []
        while unvisited:
            seed = max(unvisited, key=lambda c: 0.7 * D[0][c] + 0.3 / (1.0 + du[c]))
            route = [seed]
            unvisited.discard(seed)
            while True:
                data = [self.rdata(route)]
                best = (float("inf"), None, None)
                for u in unvisited:
                    cost, ri, k = self.best_insertion(u, data)
                    if ri is not None and cost < best[0]:
                        best = (cost, u, k)
                if best[1] is None:
                    break
                route.insert(best[2], best[1])
                unvisited.discard(best[1])
            routes.append(route)
        self._i1_cache = deepcopy(routes)
        return routes

    # ------------------------------------------------------ Enhanced Local Search
    def _intra(self, r):
        """2-opt and Or-opt (segments 1-3) inside one route, first improvement."""
        # Distance deltas are computed in O(1) (symmetric Euclidean distances);
        # the O(L) feasibility test is run only for improving candidates.
        D = self.D
        improved = True
        while improved:
            improved = False
            m = len(r)
            full = [0] + r + [0]
            # 2-opt: reverse r[i..j]  (full indices i+1 .. j+1)
            for i in range(m - 1):
                p, a = full[i], full[i + 1]
                for j in range(i + 1, m):
                    b, q = full[j + 1], full[j + 2]
                    delta = D[p][b] + D[a][q] - D[p][a] - D[b][q]
                    if delta < -1e-6:
                        nr = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                        if self.route_ok(nr):
                            r, improved = nr, True
                            break
                if improved:
                    break
            if improved:
                continue
            # Or-opt: move segment of length 1-3 to another position
            for seg in (1, 2, 3):
                if m <= seg:
                    continue
                for i in range(m - seg + 1):
                    s0, s1 = r[i], r[i + seg - 1]
                    p, q = full[i], full[i + seg + 1]
                    rem = D[p][q] - D[p][s0] - D[s1][q]
                    rest = r[:i] + r[i + seg:]
                    fr = [0] + rest + [0]
                    for j in range(len(rest) + 1):
                        if j == i:
                            continue
                        x, y = fr[j], fr[j + 1]
                        delta = rem + D[x][s0] + D[s1][y] - D[x][y]
                        if delta < -1e-6:
                            nr = rest[:j] + r[i:i + seg] + rest[j:]
                            if self.route_ok(nr):
                                r, improved = nr, True
                                break
                    if improved:
                        break
                if improved:
                    break
        return r

    def _inter_pass(self, routes):
        """One sweep of relocate, exchange and 2-opt* over all route pairs.
        Applies the first improving move found for each pair. Returns True if anything changed."""
        D, rd, du, sv, dem, cap = self.D, self.rd, self.du, self.sv, self.dem, self.cap
        changed = False
        R = len(routes)
        datas = [self.rdata(r) for r in routes]
        order = list(range(R))
        self.rng.shuffle(order)
        for x in order:
            for y in order:
                if x == y or not routes[x] or not routes[y]:
                    continue
                f1, A1, Z1, L1 = datas[x]
                f2, A2, Z2, L2 = datas[y]
                move = None
                # ---- relocate customer from x into y
                for p in range(1, len(f1) - 1):
                    u = f1[p]
                    if L2[-1] + dem[u] > cap:
                        continue
                    gain = D[f1[p - 1]][u] + D[u][f1[p + 1]] - D[f1[p - 1]][f1[p + 1]]
                    for k in range(len(f2) - 1):
                        a, b = f2[k], f2[k + 1]
                        cost = D[a][u] + D[u][b] - D[a][b]
                        if cost - gain >= -1e-6:
                            continue
                        tu = A2[k] + sv[a] + D[a][u]
                        if tu > du[u] + EPS:
                            continue
                        tu = max(tu, rd[u])
                        if tu + sv[u] + D[u][b] <= Z2[k + 1] + EPS:
                            move = ("rel", p, k)
                            break
                    if move:
                        break
                # ---- exchange (swap) u in x with v in y
                if move is None and x < y:
                    for p in range(1, len(f1) - 1):
                        u = f1[p]
                        a1, b1 = f1[p - 1], f1[p + 1]
                        for q in range(1, len(f2) - 1):
                            v = f2[q]
                            if L1[-1] - dem[u] + dem[v] > cap or L2[-1] - dem[v] + dem[u] > cap:
                                continue
                            a2, b2 = f2[q - 1], f2[q + 1]
                            delta = (D[a1][v] + D[v][b1] - D[a1][u] - D[u][b1] +
                                     D[a2][u] + D[u][b2] - D[a2][v] - D[v][b2])
                            if delta >= -1e-6:
                                continue
                            tv = max(rd[v], A1[p - 1] + sv[a1] + D[a1][v])
                            if tv > du[v] + EPS or tv + sv[v] + D[v][b1] > Z1[p + 1] + EPS:
                                continue
                            tu = max(rd[u], A2[q - 1] + sv[a2] + D[a2][u])
                            if tu > du[u] + EPS or tu + sv[u] + D[u][b2] > Z2[q + 1] + EPS:
                                continue
                            move = ("exc", p, q)
                            break
                        if move:
                            break
                # ---- 2-opt* : tails exchange
                if move is None and x < y:
                    tot1, tot2 = L1[-1], L2[-1]
                    for i in range(len(f1) - 1):
                        a, a_n = f1[i], f1[i + 1]
                        for j in range(len(f2) - 1):
                            if (i == 0 and j == 0) or (i == len(f1) - 2 and j == len(f2) - 2):
                                continue
                            b, b_n = f2[j], f2[j + 1]
                            delta = D[a][b_n] + D[b][a_n] - D[a][a_n] - D[b][b_n]
                            if delta >= -1e-6:
                                continue
                            if L1[i] + (tot2 - L2[j]) > cap or L2[j] + (tot1 - L1[i]) > cap:
                                continue
                            t1 = max(rd[b_n], A1[i] + sv[a] + D[a][b_n])
                            if t1 > Z2[j + 1] + EPS:
                                continue
                            t2 = max(rd[a_n], A2[j] + sv[b] + D[b][a_n])
                            if t2 > Z1[i + 1] + EPS:
                                continue
                            move = ("2os", i, j)
                            break
                        if move:
                            break
                if move is None:
                    continue
                kind, p, q = move
                r1, r2 = routes[x], routes[y]
                if kind == "rel":
                    u = r1.pop(p - 1)
                    r2.insert(q, u)
                elif kind == "exc":
                    r1[p - 1], r2[q - 1] = r2[q - 1], r1[p - 1]
                else:
                    n1 = f1[1:p + 1] + f2[q + 1:-1]
                    n2 = f2[1:q + 1] + f1[p + 1:-1]
                    routes[x], routes[y] = n1, n2
                datas[x] = self.rdata(routes[x])
                datas[y] = self.rdata(routes[y])
                changed = True
        return changed

    def enhanced_local_search(self, routes, max_iter=10):
        routes = [r[:] for r in routes if r]
        for _ in range(max_iter):
            before = self.total_dist(routes)
            routes = [self._intra(r) for r in routes]
            self._inter_pass(routes)
            routes = [r for r in routes if r]
            if before - self.total_dist(routes) < 1e-6:
                break
        return routes

    # ------------------------------------------------------ Route Minimization
    def route_minimization(self, routes, max_iterations=20):
        routes = [r[:] for r in routes if r]
        for _ in range(max_iterations):
            improved = False
            # (a) merge two routes end-to-end when feasible
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i != j:
                        comb = routes[i] + routes[j]
                        if self.route_ok(comb):
                            routes = [r for k, r in enumerate(routes) if k not in (i, j)] + [comb]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # (b) eliminate a whole route by feasible re-insertion of its customers
            for idx in sorted(range(len(routes)), key=lambda k: len(routes[k])):
                others = [r[:] for k, r in enumerate(routes) if k != idx]
                datas = [self.rdata(r) for r in others]
                ok = True
                for u in sorted(routes[idx], key=lambda c: self.du[c] - self.rd[c]):
                    cost, ri, k = self.best_insertion(u, datas)
                    if ri is None:
                        ok = False
                        break
                    others[ri].insert(k, u)
                    datas[ri] = self.rdata(others[ri])
                if ok:
                    routes = others
                    improved = True
                    break
            if not improved:
                break
        return routes

    # ------------------------------------------------------ Ejection-pool route elimination
    def _ep_perturb(self, routes, datas, moves=10):
        """Random feasible relocations/exchanges to diversify the partial solution."""
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        rng = self.rng
        for _ in range(moves):
            if len(routes) < 2:
                return
            x, y = rng.sample(range(len(routes)), 2)
            if not routes[x]:
                continue
            u = rng.choice(routes[x])
            nx = [c for c in routes[x] if c != u]
            cost, ri, k = self.best_insertion(u, [datas[y]])
            if ri is None:
                continue
            routes[y].insert(k, u)
            routes[x] = nx
            datas[x] = self.rdata(routes[x])
            datas[y] = self.rdata(routes[y])

    def _insert_eject(self, v, routes, pcount, kmax_len=20):
        """Insert v into some route, ejecting 1 (or 2 for short routes) other customers so that
        the route stays feasible; choose the option with the smallest sum of penalty counters."""
        best = None                      # (psum, dist_delta, ri, new_route, ejected)
        rng = self.rng
        for ri, r in enumerate(routes):
            m = len(r)
            for k in range(m + 1):
                ins = r[:k] + [v] + r[k:]
                for e in range(m + 1):
                    if e == k:
                        continue
                    ej = ins[e]
                    nr = ins[:e] + ins[e + 1:]
                    ps = pcount[ej]
                    if best is not None and ps > best[0]:
                        continue
                    if self.route_ok(nr):
                        dd = self.route_dist(nr) - self.route_dist(r)
                        if best is None or (ps, dd) < (best[0], best[1]):
                            best = (ps, dd, ri, nr, [ej])
        if best is not None:
            return best
        for ri, r in enumerate(routes):
            m = len(r)
            if m > kmax_len:
                continue
            for k in range(m + 1):
                ins = r[:k] + [v] + r[k:]
                for e1 in range(m + 1):
                    if e1 == k:
                        continue
                    for e2 in range(e1 + 1, m + 1):
                        if e2 == k:
                            continue
                        ps = pcount[ins[e1]] + pcount[ins[e2]]
                        if best is not None and ps > best[0]:
                            continue
                        nr = [c for i, c in enumerate(ins) if i != e1 and i != e2]
                        if self.route_ok(nr):
                            dd = self.route_dist(nr) - self.route_dist(r)
                            if best is None or (ps, dd) < (best[0], best[1]):
                                best = (ps, dd, ri, nr, [ins[e1], ins[e2]])
        return best

    def ejection_pool_elimination(self, routes, max_steps=300, max_attempts=3, time_limit=10.0):
        """Try to remove routes one at a time (Nagata & Braysy, 2009, simplified).
        A route is removed and its customers placed in an ejection pool (EP). Customers are taken
        from the EP and inserted at a feasible position; if none exists, they are inserted with
        ejection of up to two other customers (chosen by minimum penalty counter), which go back
        to the EP. Every intermediate route is feasible. If the EP empties within the step
        budget, the solution has one fewer route."""
        best = [r[:] for r in routes if r]
        fails = 0
        t_end = time.time() + time_limit
        while len(best) > 1 and fails < max_attempts and time.time() < t_end:
            S = [r[:] for r in best]
            drop = self.rng.randrange(len(S))
            EP = S.pop(drop)
            self.rng.shuffle(EP)
            datas = [self.rdata(r) for r in S]
            pcount = [1] * (self.n + 1)
            steps = 0
            while EP and steps < max_steps and time.time() < t_end:
                steps += 1
                v = EP.pop()
                cost, ri, k = self.best_insertion(v, datas)
                if ri is not None:
                    S[ri].insert(k, v)
                    datas[ri] = self.rdata(S[ri])
                    continue
                pcount[v] += 1
                res = self._insert_eject(v, S, pcount)
                if res is None:
                    EP.insert(0, v)
                    continue
                _, _, ri, nr, ejected = res
                S[ri] = nr
                datas[ri] = self.rdata(nr)
                EP.extend(ejected)
                self._ep_perturb(S, datas)
            if not EP:
                S = [r for r in S if r]
                if self.solution_ok(S):
                    best = S
                    fails = 0
                    continue
            fails += 1
        return best

    # ------------------------------------------------------ Ejection chain
    def ejection_chain(self, routes):
        routes = [r[:] for r in routes if r]
        if len(routes) < 2:
            return routes
        L = self.rng.randint(2, min(5, len(routes)))
        idxs = self.rng.sample(range(len(routes)), L)
        ejected = []
        for ri in idxs:
            if routes[ri]:
                c = self.rng.choice(routes[ri])
                routes[ri].remove(c)
                ejected.append(c)
        routes = [r for r in routes if r]
        datas = [self.rdata(r) for r in routes]
        for u in ejected:
            cost, ri, k = self.best_insertion(u, datas)
            if ri is None:
                routes.append([u])
                datas.append(self.rdata([u]))
            else:
                routes[ri].insert(k, u)
                datas[ri] = self.rdata(routes[ri])
        return routes

    # ------------------------------------------------------ perturbation operators
    def _perturb(self, routes, w):
        rng = self.rng
        new = [r[:] for r in routes]
        ops = max(1, int(round(abs(w) * (1 + rng.random()) * 3)))
        for _ in range(ops):
            op = rng.choice(["swap", "insert", "shift", "reverse"])
            if op == "swap" and new:
                r = rng.choice(new)
                if len(r) > 1:
                    i, j = rng.sample(range(len(r)), 2)
                    r[i], r[j] = r[j], r[i]
            elif op == "insert" and len(new) > 1:
                i1, i2 = rng.sample(range(len(new)), 2)
                if new[i1]:
                    c = rng.choice(new[i1])
                    new[i1].remove(c)
                    new[i2].insert(rng.randrange(len(new[i2]) + 1), c)
            elif op == "shift" and new:
                r = rng.choice(new)
                if len(r) > 2:
                    c = r.pop(rng.randrange(len(r)))
                    r.insert(rng.randrange(len(r) + 1), c)
            elif op == "reverse" and new:
                r = rng.choice(new)
                if len(r) > 3:
                    i = rng.randrange(len(r) - 2)
                    j = rng.randrange(i + 2, len(r))
                    r[i:j] = r[i:j][::-1]
        return self.repair([r for r in new if r])

    # ------------------------------------------------------ PSO
    def pso(self, init):
        p = self.pso_params
        S = p["swarm_size"]
        parts = [deepcopy(init) if i == 0 else self.solomon_i1() for i in range(S)]
        pf = [self.evaluate(x)[0] for x in parts]
        pbest = deepcopy(parts)
        g = int(np.argmin(pf))
        gbest, gf = deepcopy(pbest[g]), pf[g]
        for _ in range(p["max_iterations"]):
            w = p["w_min"] + (p["w_max"] - p["w_min"]) * self._chaos_next()
            for i in range(S):
                if self.rng.random() < 0.2:
                    parts[i] = self._perturb(parts[i], w)
                if self.rng.random() < self.local_search_prob:
                    parts[i] = self.enhanced_local_search(parts[i], max_iter=3)
                f = self.evaluate(parts[i])[0]
                if f < pf[i]:
                    pbest[i], pf[i] = deepcopy(parts[i]), f
                    if f < gf:
                        gbest, gf = deepcopy(parts[i]), f
        return gbest

    # ------------------------------------------------------ ACO (with elite deposit)
    def _ant(self):
        D, rd, du, sv, dem = self.D, self.rd, self.du, self.sv, self.dem
        a, b = self.aco_params["alpha"], self.aco_params["beta"]
        unv = set(range(1, self.n + 1))
        routes = []
        while unv:
            route, cur, load, t = [], 0, 0, 0.0
            while True:
                cands, probs = [], []
                for c in unv:
                    if load + dem[c] > self.cap:
                        continue
                    tc = t + sv[cur] + D[cur][c]
                    if tc > du[c] + EPS:
                        continue
                    tc = max(tc, rd[c])
                    if tc + sv[c] + D[c][0] > du[0] + EPS:
                        continue
                    h = (1.0 / (1.0 + D[cur][c])) ** b
                    probs.append((self.pher[cur][c] ** a) * h * (1.0 + 1.0 / (1.0 + du[c])))
                    cands.append((c, tc))
                if not cands:
                    break
                pr = np.array(probs)
                pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(pr), 1.0 / len(pr))
                c, tc = cands[int(self.nprng.choice(len(cands), p=pr))]
                route.append(c)
                unv.discard(c)
                load += dem[c]
                cur, t = c, tc
            if not route:                   # cannot happen on valid instances
                routes.extend([[c] for c in unv])
                break
            routes.append(route)
        return routes

    def _deposit(self, routes, amount):
        for r in routes:
            full = [0] + r + [0]
            for u, v in zip(full, full[1:]):
                self.pher[u][v] += amount

    def aco(self, iterations, elite):
        ap = self.aco_params
        # Elite archive guidance: delta tau = elite_weight * Q / TD(S_k)   
        for f, sol in elite:
            self._deposit(sol, ap["elite_weight"] * ap["Q"] / max(self.total_dist(sol), 1e-6))
        np.clip(self.pher, ap["tau_min"], ap["tau_max"], out=self.pher)
        best, bf = None, float("inf")
        for _ in range(iterations):
            ants = [self._ant() for _ in range(ap["num_ants"])]
            scored = sorted(((self.evaluate(r)[0], r) for r in ants), key=lambda x: x[0])
            if scored[0][0] < bf:
                bf, best = scored[0]
            self.pher *= (1 - ap["rho"])
            for f, r in scored:
                self._deposit(r, ap["Q"] / max(self.total_dist(r), 1e-6))
            for f, r in scored[:5]:
                self._deposit(r, ap["elite_weight"] * ap["Q"] / max(self.total_dist(r), 1e-6))
            np.clip(self.pher, ap["tau_min"], ap["tau_max"], out=self.pher)
        return best

    # ------------------------------------------------------ ABC
    def abc(self, init):
        p = self.abc_params
        C = p["colony_size"]
        food = [deepcopy(init) if i == 0 else self.solomon_i1() for i in range(C)]
        fit = [self.evaluate(x)[0] for x in food]
        trials = [0] * C
        b = int(np.argmin(fit))
        best, bf = deepcopy(food[b]), fit[b]
        for _ in range(p["max_cycles"]):
            for i in range(C):                                   # employed bees
                new = self._perturb(food[i], 0.5) if self.rng.random() < 0.5 else self.ejection_chain(food[i])
                if self.rng.random() < 0.3:
                    new = self.enhanced_local_search(new, max_iter=2)
                f = self.evaluate(new)[0]
                if f < fit[i]:
                    food[i], fit[i], trials[i] = new, f, 0
                else:
                    trials[i] += 1
            arr = np.array(fit)
            pr = 1.0 / (1.0 + arr - arr.min() + 1e-6)
            pr = pr / pr.sum()
            for _ in range(C):                                   # onlooker bees
                i = int(self.nprng.choice(C, p=pr))
                new = self._perturb(food[i], 0.5)
                f = self.evaluate(new)[0]
                if f < fit[i]:
                    food[i], fit[i], trials[i] = new, f, 0
            for i in range(C):                                   # scout bees
                if trials[i] >= p["limit"]:
                    food[i] = self.solomon_i1()
                    fit[i] = self.evaluate(food[i])[0]
                    trials[i] = 0
            k = int(np.argmin(fit))
            if fit[k] < bf:
                best, bf = deepcopy(food[k]), fit[k]
        return best

    # ------------------------------------------------------ Chaos-enhanced search
    # ------------------------------------------------------ Chaos-controlled ruin-and-recreate
    def _neighbours(self):
        if not hasattr(self, "_nbr"):
            self._nbr = [sorted(range(1, self.n + 1), key=lambda j: self.D[i][j]) for i in range(self.n + 1)]
        return self._nbr

    def _ruin(self, routes, c_bar):
        """SISR-style string removal (Christiaens & Vanden Berghe 2020): remove strings of
        consecutive customers from routes close to a random seed customer."""
        rng = self.rng
        nbr = self._neighbours()
        L_max = 10
        avg_len = max(1.0, sum(len(r) for r in routes) / max(1, len(routes)))
        ls_max = min(L_max, avg_len)
        ks_max = 4.0 * c_bar / (1.0 + ls_max) - 1.0
        k_s = int(rng.uniform(1, ks_max + 1))
        where = {}
        for ri, r in enumerate(routes):
            for c in r:
                where[c] = ri
        seed = rng.randint(1, self.n)
        removed, ruined = [], set()
        for c in [seed] + nbr[seed]:
            if len(ruined) >= k_s:
                break
            if c in removed or c not in where:
                continue
            ri = where[c]
            if ri in ruined:
                continue
            r = routes[ri]
            lmax = min(len(r), ls_max)
            l = int(rng.uniform(1, lmax + 1))
            pos = r.index(c)
            start = rng.randint(max(0, pos - l + 1), min(pos, len(r) - l))
            seg = r[start:start + l]
            routes[ri] = r[:start] + r[start + l:]
            removed.extend(seg)
            ruined.add(ri)
        return [r for r in routes if r], removed

    def _recreate(self, routes, removed, blink=0.01):
        """Greedy insertion with blinks; customers ordered randomly, by demand, by distance
        from the depot or by time-window width. Opens a new route only if no feasible position."""
        rng, D, rd, du, sv, dem = self.rng, self.D, self.rd, self.du, self.sv, self.dem
        key = rng.choice(["random", "demand", "far", "tw"])
        if key == "random":
            rng.shuffle(removed)
        elif key == "demand":
            removed.sort(key=lambda c: -dem[c])
        elif key == "far":
            removed.sort(key=lambda c: -D[0][c])
        else:
            removed.sort(key=lambda c: du[c] - rd[c])
        datas = [self.rdata(r) for r in routes]
        for u in removed:
            best = (float("inf"), None, None)
            for ri, (full, A, Z, L) in enumerate(datas):
                if L[-1] + dem[u] > self.cap:
                    continue
                for k in range(len(full) - 1):
                    if rng.random() < blink:
                        continue
                    a, b = full[k], full[k + 1]
                    cost = D[a][u] + D[u][b] - D[a][b]
                    if cost >= best[0]:
                        continue
                    tu = A[k] + sv[a] + D[a][u]
                    if tu > du[u] + EPS:
                        continue
                    if tu < rd[u]:
                        tu = rd[u]
                    if tu + sv[u] + D[u][b] <= Z[k + 1] + EPS:
                        best = (cost, ri, k)
            if best[1] is None:
                routes.append([u])
                datas.append(self.rdata([u]))
            else:
                routes[best[1]].insert(best[2], u)
                datas[best[1]] = self.rdata(routes[best[1]])
        return routes

    def chaos_ruin_recreate(self, routes, iterations=300):
        """Chaos-controlled ruin-and-recreate: the logistic map sets the average number of
        removed customers (5-15) and supplies the simulated-annealing acceptance numbers. Fleet size may
        never increase; the best solution found is returned."""
        cur = [r[:] for r in routes if r]
        cur_nv, cur_td = len(cur), self.total_dist(cur)
        best, best_nv, best_td = [r[:] for r in cur], cur_nv, cur_td
        for i in range(iterations):
            x = self._chaos_next()
            c_bar = 5.0 + 10.0 * x
            cand, removed = self._ruin([r[:] for r in cur], c_bar)
            cand = self._recreate(cand, removed)
            nv, td = len(cand), self.total_dist(cand)
            # chaotic simulated annealing: T decays geometrically from T0 to Tf, and the
            # logistic map supplies the acceptance random number (SISR acceptance rule)
            T = self.rr_T0 * (self.rr_Tf / self.rr_T0) ** (i / iterations)
            thr = -T * math.log(max(self._chaos_next(), 1e-12))
            if nv < cur_nv or (nv == cur_nv and td < cur_td + thr):
                cur, cur_nv, cur_td = cand, nv, td
                if (nv, td) < (best_nv, best_td - 1e-9):
                    best, best_nv, best_td = [r[:] for r in cand], nv, td
        return best

    # ------------------------------------------------------ main loop (Algorithm 1)
    def solve(self, max_iterations=100, log=None):
        log = log if log is not None else []
        t0 = time.time()
        init = self.enhanced_local_search(self.solomon_i1(), max_iter=10)
        init = self.repair(init)
        best = deepcopy(init)
        bf, btd, _ = self.evaluate(best)
        elite = [(bf, deepcopy(best))]
        history = []
        stagnation = 0
        log.append(f"Init (I1+ELS): NV={len(best)} TD={btd:.2f}")
        for it in range(max_iterations):
            phase = it % 7
            if phase == 0:
                cur = self.aco(20, elite)
            elif phase == 1:
                cur = self.pso(best)
            elif phase == 2:
                cur = self.abc(best)
            elif phase == 3:
                cur = self.enhanced_local_search(best, 15)
            elif phase == 4:
                cur = self.enhanced_local_search(self.chaos_ruin_recreate(best, 5000), 5)
            elif phase == 5:
                cur = self.enhanced_local_search(
                    self.ejection_pool_elimination(self.route_minimization(best)), 10)
            else:
                cur = self.enhanced_local_search(self.ejection_chain(best), 5)
            cur = self.repair(cur)
            cf, ctd, feas = self.evaluate(cur)
            if feas and cf < bf - 1e-6:
                best, bf, btd = deepcopy(cur), cf, ctd
                stagnation = 0
                elite.append((cf, deepcopy(cur)))
                elite = sorted(elite, key=lambda e: e[0])[:3]
                log.append(f"  iter {it:3d} phase {phase}: NV={len(best)} TD={btd:.2f}")
            else:
                stagnation += 1
            history.append((len(best), btd))
            if stagnation > 50:
                self.pher[:] = self.aco_params["pheromone_init"]
                self.chaos_state = self.rng.random()
                stagnation = 0
                log.append(f"  restart at iter {it}")
        # Final intensification (Phase 2)
        fin = self.enhanced_local_search(best, 50)
        fin = self.ejection_pool_elimination(self.route_minimization(fin), max_steps=2000, max_attempts=5, time_limit=60.0)
        fin = self.enhanced_local_search(fin, 20)
        fin = self.enhanced_local_search(self.chaos_ruin_recreate(fin, 100000), 20)
        fin = self.repair(fin)
        ff, ftd, feas = self.evaluate(fin)
        if feas and ff < bf:
            best, bf, btd = fin, ff, ftd
            log.append(f"  final intensification: NV={len(best)} TD={btd:.2f}")
        assert self.solution_ok(best), "internal error: final solution infeasible"
        log.append(f"Final: NV={len(best)} TD={btd:.2f} time={time.time()-t0:.1f}s")
        return best, btd, history


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def save_routes(path, name, solver, routes, td, seed):
    with open(path, "w") as f:
        f.write(f"Instance: {name}\nSeed: {seed}\n{'='*80}\n")
        f.write(f"Total Distance: {td:.2f}\nNumber of Vehicles: {len(routes)}\n")
        f.write(f"Total Customers: {len({c for r in routes for c in r})}\n")
        f.write(f"Feasible (capacity, time windows, depot return, each customer once): "
                f"{'Yes' if solver.solution_ok(routes) else 'No'}\n{'='*80}\n\n")
        for k, r in enumerate(routes, 1):
            f.write(f"Vehicle {k}:\n  Route: 0 -> " + " -> ".join(map(str, r)) + " -> 0\n")
            f.write(f"  Distance: {solver.route_dist(r):.2f}\n")
            f.write(f"  Load: {sum(solver.dem[c] for c in r)}/{solver.cap}\n\n")


def plot_routes(path, name, solver, routes, td):
    plt.figure(figsize=(8, 7))
    xs = [c[0] for c in solver.coords]
    ys = [c[1] for c in solver.coords]
    plt.scatter(xs[1:], ys[1:], s=12, c="lightgray")
    plt.scatter([xs[0]], [ys[0]], s=100, c="red", marker="s")
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(routes):
        full = [0] + r + [0]
        plt.plot([xs[i] for i in full], [ys[i] for i in full], lw=1.2, color=cmap(k % 20))
    plt.title(f"{name}  NV={len(routes)}  TD={td:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def run_one(args):
    inst_path, out_dir, iters, seed = args
    name = os.path.splitext(os.path.basename(inst_path))[0].upper()
    coords, dem, rd, du, sv, cap, nv = parse_instance(inst_path)
    s = HSICNSolver(coords, dem, rd, du, sv, cap, nv, seed=seed)
    log = [f"HSICN  instance={name}  seed={seed}  iterations={iters}"]
    t0 = time.time()
    routes, td, hist = s.solve(iters, log)
    el = time.time() - t0
    tag = f"{name}_s{seed}"
    os.makedirs(os.path.join(out_dir, "route_details"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "routes"), exist_ok=True)
    save_routes(os.path.join(out_dir, "route_details", f"{tag}_routes.txt"), name, s, routes, td, seed)
    with open(os.path.join(out_dir, "logs", f"{tag}_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    plot_routes(os.path.join(out_dir, "routes", f"{tag}_routes.png"), name, s, routes, td)
    return dict(instance=name, seed=seed, nv=len(routes), td=round(td, 2),
                feasible=s.solution_ok(routes), time_s=round(el, 1))


def main():
    ap = argparse.ArgumentParser(description="HSICN for the VRPTW")
    ap.add_argument("--instances", default="./solomon", help="folder with instance files")
    ap.add_argument("--output", default="./results")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--seeds", default="1", help="comma-separated seeds, e.g. 1,2,3,4,5")
    ap.add_argument("--only", default=None, help="comma-separated instance names, e.g. R101,C201")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.instances) if f.lower().endswith(".txt"))
    if a.only:
        want = {x.strip().upper() for x in a.only.split(",")}
        files = [f for f in files if os.path.splitext(f)[0].upper() in want]
    seeds = [int(x) for x in a.seeds.split(",")]
    jobs = [(os.path.join(a.instances, f), a.output, a.iterations, sd) for f in files for sd in seeds]
    os.makedirs(a.output, exist_ok=True)
    summary = os.path.join(a.output, "summary.csv")
    done = set()
    if os.path.exists(summary):
        done = {(r["instance"], int(r["seed"])) for r in csv.DictReader(open(summary))}
    jobs = [j for j in jobs if (os.path.splitext(os.path.basename(j[0]))[0].upper(), j[3]) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {summary})")
    new_file = not os.path.exists(summary)
    fh = open(summary, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["instance", "seed", "nv", "td", "feasible", "time_s"])
    if new_file:
        w.writeheader()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for res in pool.imap_unordered(run_one, jobs):
                print(res, flush=True)
                w.writerow(res)
                fh.flush()
    else:
        for j in jobs:
            res = run_one(j)
            print(res, flush=True)
            w.writerow(res)
            fh.flush()
    fh.close()


if __name__ == "__main__":
    main()
