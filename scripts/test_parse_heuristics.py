#!/usr/bin/env python3
"""Heuristic unit checks for RAID-SQL v2 parse helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raid_sql.parse import (  # noqa: E402
    apply_select_order_plan,
    extract_select_order_plan,
    inject_distinct_if_asked,
    is_incomplete_sql,
    needs_join_repair,
    needs_semantic_set_op_repair,
    normalize_count_number_aggregate_first,
    normalize_fname_lname_order,
    normalize_group_by_select_order,
    normalize_how_many_aggregate_first,
    normalize_how_many_each_keys_first,
    normalize_select_star_order_by,
    prefer_aggregate_first_select,
    question_suggests_multi_table,
    reorder_select_for_group_by,
    select_order_heuristic_variants,
    strip_count_from_frequency_sort,
    vote_by_bag_key_then_order,
)


def main() -> int:
    # legacy keys-first is now a no-op (Spider column-order analysis)
    src = "SELECT count(Club_ID) , Manufacturer FROM club GROUP BY Manufacturer"
    assert reorder_select_for_group_by(src) == src.strip().rstrip(";")

    # metric-leading question → aggregate first
    q_avg = "What is the average age of pilots for each plane name?"
    out = normalize_group_by_select_order(
        "SELECT plane_name, avg(age) FROM PilotSkills GROUP BY plane_name",
        q_avg,
    )
    assert out.upper().index("AVG") < out.upper().index("PLANE_NAME")
    # entity-leading show/list: do not force agg-first
    assert normalize_group_by_select_order(
        "SELECT Manufacturer, COUNT(*) FROM club GROUP BY Manufacturer",
        "Show different manufacturers and the number of clubs",
    ).startswith("SELECT Manufacturer")

    assert prefer_aggregate_first_select(
        "SELECT Country , COUNT(*) FROM singer GROUP BY Country"
    ).upper().startswith("SELECT COUNT")

    # How many (no each) → aggregate first
    q_hm = (
        "How many courses do teachers teach at most? "
        "Also find the id of the teacher who teaches the most."
    )
    assert normalize_how_many_aggregate_first(
        "SELECT teacher_id , COUNT(class_id) FROM Classes "
        "GROUP BY teacher_id ORDER BY COUNT(class_id) DESC LIMIT 1",
        q_hm,
    ).upper().startswith("SELECT COUNT")
    # How many … each → do not force
    assert normalize_how_many_aggregate_first(
        "SELECT Manufacturer, COUNT(*) FROM club GROUP BY Manufacturer",
        "How many clubs use each manufacturer?",
    ).startswith("SELECT Manufacturer")
    # How many … each with agg-first pred → keys first
    assert normalize_how_many_each_keys_first(
        "SELECT COUNT(*), Manufacturer FROM club GROUP BY Manufacturer",
        "How many clubs use each manufacturer?",
    ).upper().startswith("SELECT MANUFACTURER")

    # Select_order plan apply
    assert extract_select_order_plan(
        "Select_order: count(*), driver_id\nSQL: SELECT driver_id, count(*) FROM t"
    ) == ["count(*)", "driver_id"]
    assert apply_select_order_plan(
        "SELECT driver_id, COUNT(*) FROM t GROUP BY driver_id",
        ["count(*)", "driver_id"],
    ).upper().startswith("SELECT COUNT")

    # Count the number / find the number of → aggregate first
    assert normalize_count_number_aggregate_first(
        "SELECT Racing_Series , count(Driver_ID) FROM driver GROUP BY Racing_Series",
        "Count the number of drivers that have raced in each series.",
    ).upper().startswith("SELECT COUNT")
    assert normalize_count_number_aggregate_first(
        "SELECT product_type_code , avg(product_price) FROM Products "
        "GROUP BY product_type_code",
        "What is the average price of products for each product type?",
    ).startswith("SELECT product_type_code")  # not a count-number cue

    # SC bag vote prefers finalize order among same bag
    picked = vote_by_bag_key_then_order(
        [
            ("SELECT id, COUNT(*) FROM t GROUP BY id", "bagA", True),
            ("SELECT COUNT(*), id FROM t GROUP BY id", "bagA", True),
            ("SELECT COUNT(*), id FROM t GROUP BY id", "bagA", True),
        ],
        "Count the number of rows for each id.",
        prefer_order=normalize_count_number_aggregate_first,
    )
    assert picked.upper().startswith("SELECT COUNT")
    variants = select_order_heuristic_variants(
        "SELECT Racing_Series , count(Driver_ID) FROM driver GROUP BY Racing_Series",
        "Count the number of drivers that have raced in each series.",
    )
    assert len(variants) >= 2

    # distinct injector
    assert inject_distinct_if_asked(
        "SELECT customer_details FROM customers",
        "Give me the distinct customer details.",
    ).upper().startswith("SELECT DISTINCT")
    assert (
        inject_distinct_if_asked(
            "SELECT DISTINCT customer_details FROM customers",
            "Give me the distinct customer details.",
        )
        == "SELECT DISTINCT customer_details FROM customers"
    )

    # fname,lname → lname,fname
    assert (
        normalize_fname_lname_order(
            "SELECT T1.fname, T1.lname FROM Artists AS T1"
        )
        == "SELECT T1.lname, T1.fname FROM Artists AS T1"
    )

    # SELECT * ORDER BY → projected column
    assert (
        normalize_select_star_order_by(
            "SELECT * FROM Channels ORDER BY Channel_Details ASC"
        )
        == "SELECT Channel_Details FROM Channels ORDER BY Channel_Details ASC"
    )
    assert normalize_select_star_order_by("SELECT * FROM Students") == "SELECT * FROM Students"

    # frequency-sort: drop COUNT from SELECT
    q_freq = "Sort the student answer texts in descending order of their frequency of occurrence."
    assert (
        strip_count_from_frequency_sort(
            "SELECT Student_Answer_Text, COUNT(*) FROM Student_Answers "
            "GROUP BY Student_Answer_Text ORDER BY COUNT(*) DESC",
            q_freq,
        )
        == "SELECT Student_Answer_Text FROM Student_Answers "
        "GROUP BY Student_Answer_Text ORDER BY COUNT(*) DESC"
    )
    # do not strip when question asks how many
    assert "COUNT" in strip_count_from_frequency_sort(
        "SELECT Driver_ID, COUNT(*) FROM vehicle_driver GROUP BY Driver_ID "
        "ORDER BY COUNT(*) DESC LIMIT 1",
        "How many vehicles has a driver driven at most, and what is the driver id?",
    )

    # incomplete
    assert is_incomplete_sql("SELECT Club_ID FROM player)")
    assert not is_incomplete_sql(
        "SELECT Name FROM club WHERE Club_ID NOT IN (SELECT Club_ID FROM player)"
    )

    # intersect semantic
    q = (
        "Show the country of players with earnings more than 1400000 "
        "and players with earnings less than 1100000."
    )
    assert needs_semantic_set_op_repair(
        q, "SELECT Country FROM player WHERE Earnings > 1400000 OR Earnings < 1100000"
    )
    assert not needs_semantic_set_op_repair(
        q,
        "SELECT Country FROM player WHERE Earnings > 1400000 "
        "INTERSECT SELECT Country FROM player WHERE Earnings < 1100000",
    )

    # join gate
    assert question_suggests_multi_table(
        "List the names of players and their clubs"
    )
    assert needs_join_repair(
        "List the names of players and their clubs",
        "SELECT Name FROM player",
        predicted_class='"NON-NESTED"',
        has_foreign_keys=True,
    )
    assert not needs_join_repair(
        "List the names of players and their clubs",
        "SELECT T1.Name FROM player AS T1 JOIN club AS T2 ON T1.Club_ID = T2.Club_ID",
        predicted_class='"NON-NESTED"',
        has_foreign_keys=True,
    )
    assert not needs_join_repair(
        "How many singers do we have?",
        "SELECT COUNT(*) FROM singer",
        predicted_class='"EASY"',
        has_foreign_keys=False,
    )

    print("OK: test_parse_heuristics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
