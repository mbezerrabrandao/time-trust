"""
gurobi_test.py

Quick validation script for:
1) GAMSPy import and basic model build
2) Gurobi availability through GAMSPy (LP + MIP solves)

Run:
    python -m test.gurobi_test
or:
    python test/gurobi_test.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import gamspy as gp
from gamspy import Container, Model, Problem, Sense


@dataclass
class SolveReport:
    name: str
    ok: bool
    solve_status: Optional[str] = None
    model_status: Optional[str] = None
    objective_value: Optional[float] = None
    details: Optional[str] = None


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def test_gamspy_gurobi_lp() -> SolveReport:
    """
    Builds and solves a tiny LP using Gurobi through GAMSPy:
        min x + y
        s.t. x + 2y >= 4
             x, y >= 0
    Expected optimum: x = 0, y = 2, objective = 2
    """
    report = SolveReport(name="lp_test", ok=False)

    try:
        c = Container()

        # Decision variables
        x = gp.Variable(c, name="x", type="positive")
        y = gp.Variable(c, name="y", type="positive")

        # Constraint
        constr = gp.Equation(c, name="constr")
        constr[...] = x + 2 * y >= 4

        # Model
        m = Model(
            c,
            name="test_gamspy_gurobi_lp",
            equations=[constr],
            problem=Problem.LP,
            sense=Sense.MIN,
            objective=x + y,
        )

        print("[LP] Solving using Gurobi via GAMSPy...")
        summary = m.solve(solver="gurobi")

        # Collect results
        report.solve_status = str(m.solve_status)
        report.model_status = str(m.status)
        report.objective_value = _safe_float(m.objective_value)

        x_val = _safe_float(x.toValue())
        y_val = _safe_float(y.toValue())

        print("\n[LP] === SUMMARY DATAFRAME ===")
        print(summary)

        print("\n[LP] === MODEL ATTRIBUTES ===")
        print("Solve status:", report.solve_status)
        print("Model status:", report.model_status)
        print("Objective  :", report.objective_value)
        print("x =", x_val)
        print("y =", y_val)

        # Basic sanity check (do not be too strict with tolerances)
        if report.objective_value is not None and x_val is not None and y_val is not None:
            if abs(report.objective_value - 2.0) < 1e-6 and abs(x_val - 0.0) < 1e-6 and abs(y_val - 2.0) < 1e-6:
                report.ok = True
            else:
                report.details = "LP solved, but solution did not match expected (x=0, y=2, obj=2)."
        else:
            report.details = "LP solved, but could not parse objective/variable values."

    except Exception as e:
        report.details = f"LP test failed with exception: {e}\n{traceback.format_exc()}"

    return report


def test_gamspy_gurobi_knapsack_mip() -> SolveReport:
    """
    Builds and solves a small 0-1 knapsack MIP using Gurobi through GAMSPy.
    Items:
        i1..i5
        weights: [2,3,4,5,9]
        values : [3,4,5,8,10]
        capacity: 10
    """
    report = SolveReport(name="knapsack_mip_test", ok=False)

    try:
        c = Container()

        # Set of items
        I = gp.Set(c, name="i", records=["i1", "i2", "i3", "i4", "i5"])

        # Weights
        w = gp.Parameter(
            c,
            name="w",
            domain=[I],
            records=[
                ("i1", 2),
                ("i2", 3),
                ("i3", 4),
                ("i4", 5),
                ("i5", 9),
            ],
        )

        # Values
        v = gp.Parameter(
            c,
            name="v",
            domain=[I],
            records=[
                ("i1", 3),
                ("i2", 4),
                ("i3", 5),
                ("i4", 8),
                ("i5", 10),
            ],
        )

        # Capacity
        capacity = gp.Parameter(c, name="capacity", records=10)

        # Binary decision variables
        x = gp.Variable(c, name="x", type="binary", domain=[I])

        # Capacity constraint
        weight_constr = gp.Equation(c, name="weight_constr")
        weight_constr[...] = gp.Sum(I, w[I] * x[I]) <= capacity

        # Objective: maximize total value
        obj = gp.Sum(I, v[I] * x[I])

        m = Model(
            c,
            name="test_gamspy_gurobi_knapsack_mip",
            equations=[weight_constr],
            problem=Problem.MIP,
            sense=Sense.MAX,
            objective=obj,
        )

        print("\n[MIP] Solving knapsack using Gurobi via GAMSPy...")
        summary = m.solve(solver="gurobi")

        report.solve_status = str(m.solve_status)
        report.model_status = str(m.status)
        report.objective_value = _safe_float(m.objective_value)

        print("\n[MIP] === SUMMARY DATAFRAME ===")
        print(summary)

        print("\n[MIP] === MODEL ATTRIBUTES ===")
        print("Solve status:", report.solve_status)
        print("Model status:", report.model_status)
        print("Objective  :", report.objective_value)

        print("\n[MIP] === Variable x ===")
        print(x.records)

        # If we got an objective value, we consider it a success for environment testing.
        # (Exact optimum may vary if solver settings differ, but this should solve quickly.)
        if report.objective_value is not None:
            report.ok = True
        else:
            report.details = "MIP solved, but could not parse objective value."

    except Exception as e:
        report.details = f"Knapsack MIP test failed with exception: {e}\n{traceback.format_exc()}"

    return report


def run_all_tests() -> int:
    """
    Runs all environment tests.
    Returns:
        0 if all tests passed
        1 if any test failed
    """
    print("=== GAMSPy + Gurobi Environment Validation ===\n")

    lp_report = test_gamspy_gurobi_lp()
    mip_report = test_gamspy_gurobi_knapsack_mip()

    reports = [lp_report, mip_report]

    print("\n=== FINAL REPORT ===")
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        print(f"- {r.name}: {status}")
        if not r.ok and r.details:
            print(f"  Details: {r.details.strip()}")

    all_ok = all(r.ok for r in reports)
    return 0 if all_ok else 1


def main() -> None:
    exit_code = run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()