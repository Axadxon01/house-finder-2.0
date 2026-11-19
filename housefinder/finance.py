"""Pure financial helpers: mortgage, rental yield, appreciation scenarios."""

from __future__ import annotations

import pandas as pd


def monthly_payment(principal: float, annual_rate_pct: float, years: int) -> float:
    n = int(years * 12)
    if principal <= 0 or n <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def amortization(principal: float, annual_rate_pct: float, years: int) -> pd.DataFrame:
    """Yearly totals of interest and principal paid, and remaining balance."""
    payment = monthly_payment(principal, annual_rate_pct, years)
    r = annual_rate_pct / 100 / 12
    balance = principal
    rows = []
    for year in range(1, int(years) + 1):
        interest_y = principal_y = 0.0
        for _ in range(12):
            interest = balance * r
            principal_part = min(payment - interest, balance)
            balance -= principal_part
            interest_y += interest
            principal_y += principal_part
        rows.append({"year": year, "interest": interest_y, "principal": principal_y,
                     "balance": max(balance, 0.0)})
    return pd.DataFrame(rows)


def rental_metrics(price: float, monthly_rent: float, expenses_pct: float, vacancy_pct: float,
                   down_payment_pct: float, annual_rate_pct: float, years: int,
                   closing_costs_pct: float = 3.0) -> dict:
    """Standard investor metrics. Expenses are a % of collected rent."""
    gross_annual = monthly_rent * 12
    collected = gross_annual * (1 - vacancy_pct / 100)
    noi = collected * (1 - expenses_pct / 100)
    loan = price * (1 - down_payment_pct / 100)
    debt_service = monthly_payment(loan, annual_rate_pct, years) * 12
    cash_invested = price * down_payment_pct / 100 + price * closing_costs_pct / 100
    cash_flow = noi - debt_service
    return {
        "gross_yield": gross_annual / price * 100 if price else 0.0,
        "cap_rate": noi / price * 100 if price else 0.0,
        "noi": noi,
        "debt_service": debt_service,
        "cash_flow": cash_flow,
        "cash_on_cash": cash_flow / cash_invested * 100 if cash_invested else 0.0,
        "cash_invested": cash_invested,
    }


def projection(price: float, scenarios: dict[str, float], years: int) -> pd.DataFrame:
    """Value per year under constant annual appreciation rates (in %)."""
    rows = []
    for year in range(0, int(years) + 1):
        row = {"year": year}
        for name, rate in scenarios.items():
            row[name] = price * (1 + rate / 100) ** year
        rows.append(row)
    return pd.DataFrame(rows)


def roi(gain: float, cost: float) -> float:
    return (gain - cost) / cost * 100 if cost > 0 else 0.0
