"""
scripts/check_claude_costs.py

Query the token_usage table and print a formatted spend summary for the
current (or a specified) calendar month.

Usage:
    Python scripts/check_claude_costs.py                  # current month
    Python scripts/check_claude_costs.py --month 2026-03  # specific month
    Python scripts/check_claude_costs.py --all            # all time
    Python scripts/check_claude_costs.py --csv            # CSV output

Output example:

  Kinesio-Track — Claude API Cost Report
  Month: April 2026
  ──────────────────────────────────────────────────
  Call type          Calls    Input tok   Output tok    Cost (USD)
  ──────────────────────────────────────────────────
  initial_plan         42      126,000       84,000       $0.76
  adapt_plan          138       82,800       13,800       $0.46
  red_flag              7        4,200          700       $0.02
  feedback            312       62,400        3,120       $0.23
  ──────────────────────────────────────────────────
  TOTAL               499      275,400      101,620       $1.47
  Budget              $100.00
  Remaining           $98.53  (98.5% available)
  ──────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def report(year: int, month: int | None, all_time: bool, as_csv: bool) -> None:
    from app.db.postgres import create_db_pool, get_db_context
    from app.core.config import settings

    await create_db_pool()

    async with get_db_context() as db:
        from sqlalchemy import extract, func, select
        from app.models.token_usage import TokenUsage

        query = select(
            TokenUsage.call_type,
            func.count().label("call_count"),
            func.sum(TokenUsage.input_tokens).label("input_tokens"),
            func.sum(TokenUsage.output_tokens).label("output_tokens"),
            func.sum(TokenUsage.cost_usd).label("cost_usd"),
        ).group_by(TokenUsage.call_type)

        if not all_time:
            query = query.where(
                extract("year",  TokenUsage.called_at) == year,
            )
            if month is not None:
                query = query.where(
                    extract("month", TokenUsage.called_at) == month,
                )

        result = await db.execute(query)
        rows   = result.all()

    if not rows:
        print("No token usage records found for the specified period.")
        return

    # Aggregate
    total_calls  = sum(int(r.call_count   or 0) for r in rows)
    total_input  = sum(int(r.input_tokens  or 0) for r in rows)
    total_output = sum(int(r.output_tokens or 0) for r in rows)
    total_cost   = sum(float(r.cost_usd   or 0) for r in rows)
    budget       = settings.MONTHLY_TOKEN_BUDGET_USD

    if as_csv:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["call_type", "calls", "input_tokens", "output_tokens", "cost_usd"])
        for r in sorted(rows, key=lambda x: float(x.cost_usd or 0), reverse=True):
            writer.writerow([
                r.call_type,
                int(r.call_count   or 0),
                int(r.input_tokens  or 0),
                int(r.output_tokens or 0),
                f"{float(r.cost_usd or 0):.6f}",
            ])
        writer.writerow(["TOTAL", total_calls, total_input, total_output, f"{total_cost:.6f}"])
        print(out.getvalue())
        return

    # Pretty print
    if all_time:
        period_label = "All time"
    elif month is None:
        period_label = f"Year {year}"
    else:
        period_label = datetime(year, month, 1).strftime("%B %Y")

    w = 54
    print(f"\n  Kinesio-Track — Claude API Cost Report")
    print(f"  Period: {period_label}")
    print(f"  {'─' * w}")
    print(f"  {'Call type':<20} {'Calls':>6}  {'Input tok':>11}  {'Output tok':>11}  {'Cost':>10}")
    print(f"  {'─' * w}")

    for r in sorted(rows, key=lambda x: float(x.cost_usd or 0), reverse=True):
        cost = float(r.cost_usd or 0)
        print(
            f"  {r.call_type:<20} {int(r.call_count or 0):>6}"
            f"  {int(r.input_tokens or 0):>11,}"
            f"  {int(r.output_tokens or 0):>11,}"
            f"  ${cost:>9.2f}"
        )

    print(f"  {'─' * w}")
    print(
        f"  {'TOTAL':<20} {total_calls:>6}"
        f"  {total_input:>11,}"
        f"  {total_output:>11,}"
        f"  ${total_cost:>9.2f}"
    )

    if not all_time and month is not None:
        remaining = budget - total_cost
        pct_used  = (total_cost / budget * 100) if budget > 0 else 0.0
        pct_avail = 100.0 - pct_used

        print(f"  {'─' * w}")
        print(f"  Budget             ${budget:>9.2f}")
        print(f"  Remaining          ${remaining:>9.2f}  ({pct_avail:.1f}% available)")

        if pct_used >= 100:
            print(f"  ⚠  BUDGET EXCEEDED by ${abs(remaining):.2f}!")
        elif pct_used >= 80:
            print(f"  ⚠  WARNING: {pct_used:.1f}% of monthly budget consumed.")

    print(f"  {'─' * w}\n")


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser(description="Print Claude API cost report.")
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        metavar="YYYY-MM",
        help="Month to report (default: current month).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_time",
        help="Report all-time spend instead of a single month.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output CSV instead of formatted table.",
    )
    args = parser.parse_args()

    if args.month:
        try:
            dt = datetime.strptime(args.month, "%Y-%m")
            year, month = dt.year, dt.month
        except ValueError:
            print("ERROR: --month must be in YYYY-MM format.", file=sys.stderr)
            sys.exit(1)
    else:
        year  = now.year
        month = now.month

    asyncio.run(
        report(
            year=year,
            month=None if args.all_time else month,
            all_time=args.all_time,
            as_csv=args.csv,
        )
    )