from datetime import date

import pandas as pd
import pytest

from shef.timestamps import add_calendar, central_day_hours, is_dst_transition_day, issue_time_utc, require_utc


def utc(*args):
    return pd.Timestamp(*args, tz="UTC")


@pytest.mark.parametrize("target, expected", [
    (date(2025, 3, 9), utc("2025-03-08 16:00")),   # issued Sat 3/8, still CST
    (date(2025, 3, 10), utc("2025-03-09 15:00")),  # issued on spring-forward day, CDT
    (date(2025, 11, 2), utc("2025-11-01 15:00")),  # issued Sat 11/1, still CDT
    (date(2025, 11, 3), utc("2025-11-02 16:00")),  # issued on fall-back day, CST
    (date(2025, 7, 1), utc("2025-06-30 15:00")),
])
def test_issue_time_is_10am_central_on_d_minus_1(target, expected):
    assert issue_time_utc(target) == expected


@pytest.mark.parametrize("day, hours", [(date(2025, 3, 9), 23), (date(2025, 11, 2), 25), (date(2025, 7, 1), 24)])
def test_central_day_length(day, hours):
    assert central_day_hours(day) == hours


def test_dst_flag_covers_whole_central_day():
    ts = pd.Series(pd.date_range("2025-11-01 00:00", "2025-11-04 00:00", freq="h", tz="UTC"))
    flag = is_dst_transition_day(ts)
    local_day = ts.dt.tz_convert("America/Chicago").dt.date
    assert flag[local_day == date(2025, 11, 2)].all()
    assert flag[local_day == date(2025, 11, 2)].sum() == 25
    assert not flag[local_day != date(2025, 11, 2)].any()


def test_calendar_uses_central_clock():
    # 03:00 UTC on Tue 1 Jul 2025 is 22:00 CDT on Mon 30 Jun.
    df = add_calendar(pd.DataFrame({"ts_utc": [utc("2025-07-01 03:00")]}))
    assert (df.loc[0, "hour"], df.loc[0, "dayofweek"], df.loc[0, "month"]) == (22, 0, 6)


def test_tz_naive_rejected():
    with pytest.raises(ValueError):
        require_utc(pd.Series(pd.date_range("2025-01-01", periods=2, freq="h")))
