"""
Run a Pandera schema as a gate and write the result to data/reports/.

Hard gate: any schema failure. The report lists every failure case (lazy
validation) and `gate` raises GateFailed, which scripts turn into a nonzero
exit — that is what GitHub Actions keys on in P5.
Soft checks (completeness, gaps) are added to the same report as warnings.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa

from shef.config import REPORTS


class GateFailed(RuntimeError):
    pass


def gate(df: pd.DataFrame, schema: pa.DataFrameSchema, stage: str, warnings: list[str] | None = None,
         stats: dict | None = None) -> pd.DataFrame:
    report = {
        "stage": stage,
        "schema": schema.name,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df),
        "passed": True,
        "failures": [],
        "warnings": warnings or [],
        "stats": stats or {},
    }
    try:
        out = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        report["passed"] = False
        report["failure_count"] = len(cases)
        report["failures"] = (
            cases.groupby(["column", "check"], dropna=False).size().reset_index(name="n")
            .astype({"column": str, "check": str}).to_dict("records")
        )
        report["failure_examples"] = cases.head(20).astype(str).to_dict("records")
        _write(stage, report)
        raise GateFailed(f"[{stage}] {len(cases)} failure(s): {report['failures']}") from None
    _write(stage, report)
    return out


def _write(stage: str, report: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"validation_{stage}.json").write_text(json.dumps(report, indent=2, default=str))


def gaps(ts: pd.Series, freq: pd.Timedelta = pd.Timedelta(hours=1)) -> list[dict]:
    """Every jump larger than one step in a sorted UTC series."""
    s = ts.sort_values().reset_index(drop=True)
    d = s.diff()
    idx = d[d > freq].index
    return [{"after": str(s[i - 1]), "before": str(s[i]), "missing_hours": int(d[i] / freq) - 1} for i in idx]


def completeness_by_year(df: pd.DataFrame, col: str, ts_col: str = "ts_utc") -> dict[int, float]:
    g = df.groupby(df[ts_col].dt.year)[col]
    return {int(y): round(float(v), 5) for y, v in g.apply(lambda s: s.notna().mean()).items()}
