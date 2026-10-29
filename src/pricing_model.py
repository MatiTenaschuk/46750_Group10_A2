"""Question 2: the pricing problem M^DP

    python -m src.pricing_model                 # base case, cap only (the first of the three experiments)
    python -m src.pricing_model --method SOS1   # the same with SOS1 instead of big-M (to validate big-M)


--------------------------------------------
M^DP = M^C  +  posted prices x_t  +  every member's KKT conditions.
    * Everything M^C already has (members' variables and constraints, connection point, cap, objective) is built by
      the parent class CommunityModel (Question 1). 
    * What is new is added in PricingModel.build():
          prices x_t .............................  (decision variables), 0 <= x_t <= x_max
          multipliers of every member ............  (Lagrangian), 
          stationarity conditions ................ 
          complementarity (big-M or SOS1) ........  
    * Validation (validate_dp) like the professor's `validate`, adapted to members.

Notation: report <-> code
    x_t                         self.x[t]
    l, d+, d-, pv               v["load"], v["dev_pos"], v["dev_neg"], v["pv"]        (v = self.member_vars[i])
    p^ch, p^dis, e              v["charge"], v["discharge"], v["soc"]
    q_{i,t} (net consumption)   self.net[i][t]   
    lambda, theta, omega, mu, psi, zeta, sigma, nu, upsilon, xi, phi    self.duals[i]["lam"], ["theta"], ...
"""
import argparse
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src.data_loader import load_community
from src.community_model import CommunityModel, add_member


class PricingModel(CommunityModel):
    """M^DP: the manager posts one price x_t per hour (the same for every member) and every member reacts optimally.

    method   : 'bigM' -> every pair 0 <= a, b >= 0 becomes  a <= M_P u,  b <= M_D (1 - u),  u binary
               'SOS1' -> {a, b} is a special ordered set of type 1 (no M to choose)
    x_max    : price bound 0 <= x_t <= x_max (paper, eq. 2f). Default: max_t p_t + (1-beta) tau^imp + alpha, the
               highest internal price M^C can ever have (cap multiplier gamma_t <= alpha), so M^C's prices are allowed.
    M_dual   : M_D, one constant for all multipliers (DKK/kWh). M_P is computed per pair from the data.
    """

    def __init__(self, com, method="bigM", x_max=None, M_dual=None, time_limit=600, mip_gap=1e-4,
                 name="DP", verbose=False):
        # M^DP has the connection point of M^C: one meter, the community cap, the discounted tariffs
        super().__init__(com, meter="community", use_cap=True, use_discount=True, name=name, verbose=verbose)
        if method not in ("bigM", "SOS1"):
            raise ValueError("method must be 'bigM' or 'SOS1'")
        self.method = method
        contract = com.contract
        if x_max is None:
            x_max = (max(com.energy_price) + (1 - contract.tariff_discount_ratio) * com.import_tariff
                     + contract.penalty_DKK_per_kWh)
        self.x_max = float(x_max)
        # Multipliers are bounded by a few times the price bound: psi <= 2 c^L, zeta <= x_max, mu and omega
        # <= x_max + 2 c^L, sigma (value of stored energy) <= x_max / eta, xi and phi by differences of sigma.
        self.M_dual = float(M_dual) if M_dual is not None else 4 * self.x_max
        self.time_limit, self.mip_gap = time_limit, mip_gap

    def build(self):
        super().build()          # M^C: members, connection point, cap, internal balance, objective 
        m, T = self.m, self.T

        # --- Upper level: the posted prices, the same for every member  -----------------------------
        self.x = m.addVars(T, lb=0, ub=self.x_max, name="x")

        # --- Lower level: every member's KKT conditions  ---------------------------------------------
        self.duals, self.pairs = {}, []
        for d in self.com.members:
            self._add_member_kkt(d)

        # --- Complementarity, linearized  ------------------------------------------------------------
        self._add_complementarity()

        # The objective is NOT changed: it is M^C's . The prices x_t are not in it, the members'
        # payments x_t q_{i,t} are transfers in the community. x_t acts only through x_t - lambda = 0.
        m.update()
        return self

    def _add_member_kkt(self, d):
        """Multipliers, stationarity and the list of complementarity pairs of one member."""
        m, T, i = self.m, self.T, d.consumer_id
        v = self.member_vars[i]
        last = T[-1]
        D = {}   # this member's multipliers

        # ---- multipliers: free for equalities, >= 0 for inequalities (dual feasibility) -------------------------
        D["lam"] = m.addVars(T, lb=-GRB.INFINITY, name=f"{i}_lambda")     # balance q = l + p^ch - pv - p^dis
        D["theta"] = m.addVars(T, lb=-GRB.INFINITY, name=f"{i}_theta")    # deviation link l - l^ref - d+ + d- = 0
        D["om_up"] = m.addVars(T, lb=0, name=f"{i}_omega_up")             # l <= L^max
        D["om_lo"] = m.addVars(T, lb=0, name=f"{i}_omega_lo")             # l >= L^min
        D["psi_p"] = m.addVars(T, lb=0, name=f"{i}_psi_plus")             # d+ >= 0
        D["psi_m"] = m.addVars(T, lb=0, name=f"{i}_psi_minus")            # d- >= 0
        D["mu"] = m.addVar(lb=0, name=f"{i}_mu")                          # daily minimum
        if d.has_pv:
            D["z_up"] = m.addVars(T, lb=0, name=f"{i}_zeta_up")           # pv <= PV^max   (zeta, not rho!!!)
            D["z_lo"] = m.addVars(T, lb=0, name=f"{i}_zeta_lo")           # pv >= 0
        if d.has_battery:
            D["sig"] = m.addVars(T, lb=-GRB.INFINITY, name=f"{i}_sigma")  # state-of-charge dynamics
            D["nu_up"] = m.addVars(T, lb=0, name=f"{i}_nu_up")            # p^ch <= P^ch
            D["nu_lo"] = m.addVars(T, lb=0, name=f"{i}_nu_lo")            # p^ch >= 0
            D["ups_up"] = m.addVars(T, lb=0, name=f"{i}_ups_up")          # p^dis <= P^dis
            D["ups_lo"] = m.addVars(T, lb=0, name=f"{i}_ups_lo")          # p^dis >= 0
            D["xi_up"] = m.addVars(T, lb=0, name=f"{i}_xi_up")            # e <= E  (xi, not kappa, kappa = cap shares)
            D["xi_lo"] = m.addVars(T, lb=0, name=f"{i}_xi_lo")            # e >= 0
            D["phi"] = m.addVar(lb=0, name=f"{i}_phi")                    # end of day e_23 >= E_0
        self.duals[i] = D

        # ---- stationarity: one condition per variable, add up the numbers in front of it  -------------
        cL = d.linear_disutility
        for t in T:
            m.addConstr(self.x[t] - D["lam"][t] == 0, name=f"{i}_stat_q[{t}]")              # THE change from A1
            m.addConstr(cL - D["theta"][t] - D["psi_p"][t] == 0, name=f"{i}_stat_dplus[{t}]")
            m.addConstr(cL + D["theta"][t] - D["psi_m"][t] == 0, name=f"{i}_stat_dminus[{t}]")
            m.addConstr(D["lam"][t] + D["theta"][t] - D["mu"] + D["om_up"][t] - D["om_lo"][t] == 0,
                        name=f"{i}_stat_l[{t}]")
            if d.has_pv and d.pv_available[t] > 0:
                m.addConstr(d.pv_marginal_cost - D["lam"][t] + D["z_up"][t] - D["z_lo"][t] == 0,
                            name=f"{i}_stat_pv[{t}]")
            # When PV^max_t = 0 (no PV system, or night), 0 <= pv <= 0 fixes pv at 0: it is not a decision, so it gets
            # no condition. Keeping it would make both pv limits binding with only zeta_up - zeta_lo determined
            # the two multipliers could then grow without limit and always hit M_D.
            if d.has_battery:
                eta_ch, eta_dis = d.battery_charging_efficiency, d.battery_discharging_efficiency
                m.addConstr(D["lam"][t] - eta_ch * D["sig"][t] + D["nu_up"][t] - D["nu_lo"][t] == 0,
                            name=f"{i}_stat_ch[{t}]")
                m.addConstr(-D["lam"][t] + D["sig"][t] / eta_dis + D["ups_up"][t] - D["ups_lo"][t] == 0,
                            name=f"{i}_stat_dis[{t}]")
                if t < last:
                    m.addConstr(D["sig"][t] - D["sig"][t + 1] + D["xi_up"][t] - D["xi_lo"][t] == 0,
                                name=f"{i}_stat_e[{t}]")
                else:
                    m.addConstr(D["sig"][t] + D["xi_up"][t] - D["xi_lo"][t] - D["phi"] == 0,
                                name=f"{i}_stat_e[{t}]")

        # ---- complementarity pairs (a, b, M_P), either the constraint is binding (a = 0) or its multiplier is 0 --
        # M_P is the largest value a can take, from the data, times 1.5 (a bit above the true
        # maximum, so that a binding big-M constraint in validate_dp really means M too small).
        def M(slack_max):
            return 1.5 * float(slack_max) + 1e-3

        P = self.pairs
        for t in T:
            rng = d.load_upper_kWh[t] - d.load_lower_kWh[t]
            P.append((f"{i}: L^max-l [{t}]", d.load_upper_kWh[t] - v["load"][t], D["om_up"][t], M(rng)))
            P.append((f"{i}: l-L^min [{t}]", v["load"][t] - d.load_lower_kWh[t], D["om_lo"][t], M(rng)))
            P.append((f"{i}: d+ [{t}]", v["dev_pos"][t], D["psi_p"][t], M(d.load_upper_kWh[t] - d.reference_load[t])))
            P.append((f"{i}: d- [{t}]", v["dev_neg"][t], D["psi_m"][t], M(d.reference_load[t] - d.load_lower_kWh[t])))
            if d.has_pv and d.pv_available[t] > 0:
                P.append((f"{i}: PV^max-pv [{t}]", d.pv_available[t] - v["pv"][t], D["z_up"][t], M(d.pv_available[t])))
                P.append((f"{i}: pv [{t}]", v["pv"][t], D["z_lo"][t], M(d.pv_available[t])))
            if d.has_battery:
                P.append((f"{i}: P^ch-p^ch [{t}]", d.battery_max_charge_kW - v["charge"][t], D["nu_up"][t],
                          M(d.battery_max_charge_kW)))
                P.append((f"{i}: p^ch [{t}]", v["charge"][t], D["nu_lo"][t], M(d.battery_max_charge_kW)))
                P.append((f"{i}: P^dis-p^dis [{t}]", d.battery_max_discharge_kW - v["discharge"][t], D["ups_up"][t],
                          M(d.battery_max_discharge_kW)))
                P.append((f"{i}: p^dis [{t}]", v["discharge"][t], D["ups_lo"][t], M(d.battery_max_discharge_kW)))
                P.append((f"{i}: E-e [{t}]", d.battery_capacity_kWh - v["soc"][t], D["xi_up"][t],
                          M(d.battery_capacity_kWh)))
                P.append((f"{i}: e [{t}]", v["soc"][t], D["xi_lo"][t], M(d.battery_capacity_kWh)))
        # daily pairs (counted once, not per hour)
        P.append((f"{i}: sum l - E^min", gp.quicksum(v["load"][t] for t in T) - d.min_daily_energy_kWh, D["mu"],
                  M(sum(d.load_upper_kWh) - d.min_daily_energy_kWh)))
        if d.has_battery:
            P.append((f"{i}: e_23 - E_0", v["soc"][last] - d.battery_initial_soc_kWh, D["phi"],
                      M(d.battery_capacity_kWh - d.battery_initial_soc_kWh)))

    def _add_complementarity(self):
        """Big-M (Fortuny-Amat) or SOS1 for every pair, as in the professor's PriceMakerMPEC."""
        m = self.m
        self.u = {}
        for name, a, b, M_P in self.pairs:
            if self.method == "bigM":
                u = self.u[name] = m.addVar(vtype=GRB.BINARY, name=f"u[{name}]")
                m.addConstr(a <= M_P * u)                    # u = 0 -> a = 0: the constraint is binding
                m.addConstr(b <= self.M_dual * (1 - u))      # u = 1 -> b = 0: its multiplier is zero
            else:
                s = m.addVar(lb=0, name=f"s[{name}]")        # SOS1 members must be variables: slack s = a
                m.addConstr(s == a)
                m.addSOS(GRB.SOS_TYPE1, [s, b])

    # --- solve -------------------------------------------------------------------------------------------------
    def solve(self):
        if not hasattr(self, "m"):
            self.build()
        self.m.Params.TimeLimit = self.time_limit      # a MILP can take time, stop after time_limit seconds ...
        self.m.Params.MIPGap = self.mip_gap            # or when the solution is proven within mip_gap of the best
        self.m.optimize()
        ok = self.m.Status == GRB.OPTIMAL or (self.m.Status == GRB.TIME_LIMIT and self.m.SolCount > 0)
        if not ok:
            raise RuntimeError(f"Model {self.name}: no solution (Gurobi status {self.m.Status}; 3 = infeasible)")
        res = self._collect()                          # same keys as R, I, C (Question 1)
        T, ids = self.T, self.ids
        res["status"] = "OPTIMAL" if self.m.Status == GRB.OPTIMAL else "TIME_LIMIT"
        res["mip_gap"] = self.m.MIPGap
        res["solve_time"] = self.m.Runtime
        res["n_binaries"] = self.m.NumBinVars
        res["x"] = np.array([self.x[t].X for t in T])                                   # posted prices
        res["mu"] = {i: self.duals[i]["mu"].X for i in ids}
        # what each member pays the manager (x_t q_{i,t}) and its total cost (payment + PV + disutility)
        res["payment"] = {i: float(res["x"] @ res["net"][i]) for i in ids}
        res["member_cost"] = {i: res["payment"][i] + res["own_cost"][i] for i in ids}
        return res


# --- validation -------------------------------------------------------------------------------------------
def member_alone(d, x, extra=None):
    """Member d's own LP at the posted prices x, built with the same add_member as Question 1.
    Returns the model, its variables and its cost expression (payment + PV + disutility)."""
    m = gp.Model(f"{d.consumer_id} alone")
    m.Params.OutputFlag = 0
    v, c, net, own_cost = add_member(m, d, d.consumer_id)
    cost = gp.quicksum(x[t] * net[t] for t in range(d.n_hours)) + own_cost
    m.setObjective(cost, GRB.MINIMIZE)
    return m, v, cost


def indifferent_member_hours(com, x, tol=1e-7, min_range=1e-3):
    """Member-hours where the member's best response is not unique at the prices x. Fix the member's cost at its optimum, 
    then find the smallest and largest load it could choose.
    Returns {(member, hour): (min load, max load)}."""
    out = {}
    for d in com.members:
        m, v, cost = member_alone(d, x)
        m.optimize()
        best = m.ObjVal
        m.addConstr(cost <= best + tol)                     # stay optimal for the member
        for t in range(d.n_hours):
            bounds = []
            for sense in (GRB.MINIMIZE, GRB.MAXIMIZE):
                m.setObjective(v["load"][t], sense)
                m.optimize()
                bounds.append(m.ObjVal)
            if bounds[1] - bounds[0] > min_range:            # ignore ranges created by the solver's tolerances
                out[d.consumer_id, t] = (round(bounds[0], 3), round(bounds[1], 3))
    return out


def summarize_indifference(model, res, ind, path=None):
    """Show the indifferent member-hours and measure the risk of the optimistic assumption. What if indifferent 
    members broke their ties against the manager?
 
    Prints (1) a table of indifferent hours per member, by slope class and asset class, (2) per hour how many
    members are indifferent, the extra import if all of them chose their highest optimal load in that hour and the
    resulting import above the cap. Each hour is taken on its own (an upper bound for that hour, not a scenario for
    the whole day)."""
    com, T = model.com, list(model.T)
    cap = com.contract.capacity_limit_kW
    alpha = com.contract.penalty_DKK_per_kWh
    rows = []
    for (i, t), (lo, hi) in ind.items():
        chosen = model.member_vars[i]["load"][t].X
        rows.append({"member": i, "hour": t, "min_load": lo, "max_load": hi, "chosen_load": round(chosen, 3),
                     "room_up": max(0.0, hi - chosen)})
    df = pd.DataFrame(rows, columns=["member", "hour", "min_load", "max_load", "chosen_load", "room_up"])
    if path is not None:
        df.to_csv(path, index=False)
 
    # (1) indifferent hours per member: rows = slope class (L1 most flexible ... L4 stiffest), columns = assets
    def assets(d):
        return {(False, False): "no asset", (True, False): "PV", (False, True): "battery",
                (True, True): "PV + battery"}[(d.has_pv, d.has_battery)]
    counts = {(f"L{d.slope_class} (c^L={d.linear_disutility})", assets(d)): int((df["member"] == d.consumer_id).sum())
              for d in com.members}
    table = pd.Series(counts).unstack().reindex(columns=["no asset", "PV", "battery", "PV + battery"]).fillna(0)
    print(f"Indifferent member-hours: {len(df)} in total. Hours per member (out of {len(T)}):")
    print(table.astype(int).to_string())
 
    # (2) per hour: the risk of the optimistic tie-break
    print("\nhour  indifferent  extra import if ties go up [kWh]  import above cap then [kWh]")
    worst_total = 0.0
    for t in T:
        d_t = df[df["hour"] == t]
        if d_t.empty:
            continue
        extra = d_t["room_up"].sum()
        above = max(0.0, res["P_imp"][t] + extra - cap[t])
        worst_total += above
        print(f"{t:>4} {len(d_t):>12} {extra:>32.2f} {above:>28.2f}")
    print(f"If every hour went wrong on its own: up to {worst_total:.1f} kWh above the cap in total, "
          f"i.e. up to {alpha * worst_total:.0f} DKK of penalty (alpha = {alpha:g} DKK/kWh), "
          f"against a community cost of {res['cost']:.2f} DKK.")
    return df


def _val(e):
    return e.X if isinstance(e, gp.Var) else e.getValue()


def validate_dp(model, res, res_C=None, tol=1e-5):
    """Like the homeworks solution, adapted to M^DP. Prints a verdict."""
    ok = True
    print(f"1. status: {res['status']}, MIP gap {res['mip_gap']:.1e}, {res['n_binaries']} binaries, "
          f"{res['solve_time']:.1f} s")
    # 2. complementarity: a * b = 0 for every pair (relative to the size of a and b)
    worst = max(abs(_val(a) * _val(b)) for _, a, b, _ in model.pairs)
    ok &= worst <= tol
    print(f"2. complementarity: max |a*b| = {worst:.1e}  ->  {'OK' if worst <= tol else 'VIOLATED'}")
    # 3. big-M: no disjunctive constraint may be binding (otherwise M, not the problem, decided the solution)
    if model.method == "bigM":
        ra = max(_val(a) / M_P for _, a, _, M_P in model.pairs)
        rb = max(_val(b) / model.M_dual for _, _, b, _ in model.pairs)
        m_ok = ra < 1 - 1e-6 and rb < 1 - 1e-6
        ok &= m_ok
        print(f"3. big-M: largest a/M_P = {ra:.3f}, largest b/M_D = {rb:.3f}  ->  {'OK' if m_ok else 'M TOO SMALL'}")
    # 4. every member re-solved alone at the posted prices: its schedule in M^DP must cost what its optimum costs
    worst_gap, different = 0.0, []
    for d in model.com.members:
        m, v, cost = member_alone(d, res["x"])
        m.optimize()
        gap = res["member_cost"][d.consumer_id] - m.ObjVal
        worst_gap = max(worst_gap, abs(gap))
        if max(abs(v["load"][t].X - model.member_vars[d.consumer_id]["load"][t].X) for t in model.T) > 1e-4:
            different.append(d.consumer_id)
    m_ok = worst_gap <= 1e-4
    ok &= m_ok
    print(f"4. members alone at the posted prices: largest cost difference {worst_gap:.1e} DKK  ->  "
          f"{'OK' if m_ok else 'A MEMBER IS NOT OPTIMAL'}"
          + (f"; another optimal schedule exists for {different} (indifferent)" if different else ""))
    # 5. against M^C: same cost? same price shape (prices may differ by a constant level, see our tiny example)?
    if res_C is not None:
        dc = res["cost"] - res_C["cost"]
        shape = (res["x"] - res["x"].mean()) - (res_C["price"] - res_C["price"].mean())
        print(f"5. against M^C: cost DP - C = {dc:.4f} DKK; largest difference in price shape "
              f"{np.abs(shape).max():.3f} DKK/kWh; mean price DP {res['x'].mean():.3f} vs C {res_C['price'].mean():.3f}")
    if not ok:
        print("VERDICT: solution NOT validated")
    elif res["status"] != "OPTIMAL":
        print(f"VERDICT: a valid M^DP solution (every member reacts optimally), but NOT proven optimal: the solver "
              f"stopped at its time limit with a gap of {res['mip_gap']:.1e}. Do not use it as the result.")
    else:
        print("VERDICT: solution validated")
    return ok

def check_C_prices(com, res_C, method="bigM"):
    """Post M^C's internal prices (the duals of Q1) as x_t and solve M^DP with x fixed. It's expected the same cost as M^C, 
    because M^C's KKT conditions split into every member's KKT conditions at x = the duals of C."""
    model = PricingModel(com, method=method)
    model.build()
    for t in model.T:
        model.x[t].LB = model.x[t].UB = float(res_C["price"][t])
    res = model.solve()
    same = abs(res['cost'] - res_C['cost']) <= 1e-4
    print(f"6. Post model C's internal prices (Q1) as x_t and let every member react on its own:\n"
          f"   community cost {res['cost']:.2f} DKK, under central dispatch {res_C['cost']:.2f} DKK  ->  "
          + ("OK: C's prices alone reproduce central dispatch" if same
             else "NOT the same: C's prices do not reproduce central dispatch"))
    return res


# --- run --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bigM", choices=["bigM", "SOS1"])
    ap.add_argument("--day", default="d1")
    ap.add_argument("--time-limit", type=float, default=600)
    args = ap.parse_args()

    com = load_community("base", day=args.day)
    res_C = CommunityModel.C(com).solve()
    model = PricingModel(com, method=args.method, time_limit=args.time_limit)
    res = model.solve()

    print(f"=== M^DP, cap only ({args.method}), day {args.day} ===")
    print(f"community cost: DP {res['cost']:.2f} DKK, C {res_C['cost']:.2f} DKK")
    print("hour  p_t   x_t(DP)  x_t(C)   import DP  import C   cap")
    cap = com.contract.capacity_limit_kW
    for t in model.T:
        print(f"{t:>4} {com.energy_price[t]:5.2f} {res['x'][t]:8.3f} {res_C['price'][t]:8.3f} "
              f"{res['P_imp'][t]:10.2f} {res_C['P_imp'][t]:9.2f} {cap[t]:6.2f}")
    validate_dp(model, res, res_C)
    check_C_prices(com, res_C, method=args.method)
    out = Path("results/q2")
    out.mkdir(parents=True, exist_ok=True)
    print()
    ind = indifferent_member_hours(com, res["x"])
    summarize_indifference(model, res, ind, path=out / f"indifferent_{args.method}_{args.day}.csv")
    print(f"full list saved in results/q2/indifferent_{args.method}_{args.day}.csv")
    pd.DataFrame({"hour": list(model.T), "p": com.energy_price, "x_DP": res["x"], "x_C": res_C["price"],
                  "P_imp_DP": res["P_imp"], "P_imp_C": res_C["P_imp"], "cap": cap}
                 ).to_csv(out / f"cap_only_{args.method}_{args.day}.csv", index=False)
    print(f"saved results/q2/cap_only_{args.method}_{args.day}.csv")
 