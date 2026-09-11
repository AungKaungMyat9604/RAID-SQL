"""RAID-SQL v2 pipeline: RAG-first → slim link → classify → generate → EG repair → SC."""

from __future__ import annotations

from typing import Any, Optional

from config import Settings, get_settings
from raid_sql import prompts
from raid_sql import prompts_din
from raid_sql.execute import execute_query
from raid_sql.llm import _is_network_error, create_chat_client
from raid_sql.metrics import ExampleMetrics, RunLedger, StageMetrics
from raid_sql.parse import (
    apply_select_order_plan,
    extract_label,
    extract_schema_links,
    extract_select_order_plan,
    extract_sql,
    extract_sub_questions,
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
    select_order_heuristic_variants,
    strip_count_from_frequency_sort,
    vote_by_bag_key_then_order,
)
from raid_sql.retrieve import FewShotRetriever, format_retrieved_demos
from raid_sql.schema import SpiderSchemaStore, sample_db_values
from raid_sql.spider_data import resolve_db_path


class RAIDPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        schema_store: Optional[SpiderSchemaStore] = None,
        retriever: Optional[FewShotRetriever] = None,
        ledger: Optional[RunLedger] = None,
        split: str = "dev",
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = create_chat_client(self.settings)
        self.schema = schema_store or SpiderSchemaStore(
            self.settings.spider_dir / "tables.json"
        )
        self.retriever = retriever or FewShotRetriever(self.settings)
        self.ledger = ledger
        self.split = split
        self._P = prompts_din if self.settings.use_din_static_banks else prompts

    def _record(self, index: int, stage: Any) -> None:
        if self.ledger is not None:
            self.ledger.append_stage(index, stage)

    def _finalize_sql(self, sql: str, question: str = "") -> str:
        s = (sql or "").strip()
        s = normalize_group_by_select_order(s, question)
        s = normalize_how_many_aggregate_first(s, question)
        s = normalize_how_many_each_keys_first(s, question)
        s = normalize_count_number_aggregate_first(s, question)
        s = inject_distinct_if_asked(s, question)
        s = normalize_fname_lname_order(s)
        s = normalize_select_star_order_by(s)
        s = strip_count_from_frequency_sort(s, question)
        return s

    _SELECT_ORDER_INSTRUCTION = (
        "# After reasoning, emit one line Select_order: <cols in Spider EX order>, "
        "then SQL.\n"
        "# SELECT order: How many / Count the number / what is the number WITHOUT "
        "'each' → aggregates (COUNT/…) FIRST then keys; WITH 'each' → key THEN "
        "aggregate. Otherwise follow attribute order in the question.\n"
    )

    def _has_foreign_keys(self, db_id: str) -> bool:
        fk = self.schema.find_foreign_keys_mysql_like(db_id) or ""
        return bool(fk.strip().strip("[]").strip())

    def schema_linking_prompt(self, question: str, db_id: str, extra: str = "") -> str:
        instruction = (
            "# Find the schema_links for generating SQL queries for each question "
            "based on the database schema and Foreign keys.\n"
        )
        fields = self.schema.find_fields_mysql_like(db_id)
        foreign_keys = (
            "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n"
        )
        return (
            instruction
            + self._P.schema_linking_prompt
            + fields
            + foreign_keys
            + (extra + "\n" if extra else "")
            + f'Q: "{question}"\nA: Let’s think step by step.'
        )

    def classification_prompt(self, question: str, db_id: str, schema_links: str) -> str:
        instruction = (
            "# For the given question, classify it as EASY, NON-NESTED, or NESTED "
            "based on nested queries and JOIN.\n"
            "\nif need nested queries: predict NESTED\n"
            "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
            "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
        )
        fields = self.schema.find_fields_mysql_like(db_id)
        fields += "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n\n"
        return (
            instruction
            + fields
            + self._P.classification_prompt
            + f'Q: "{question}\nschema_links: {schema_links}\nA: Let’s think step by step.'
        )

    def easy_prompt(self, question: str, db_id: str, schema_links: str, extra: str = "") -> str:
        instruction = (
            self._SELECT_ORDER_INSTRUCTION
            + "# Use the schema links and retrieved demos to generate the SQL query.\n"
        )
        fields = self.schema.find_fields_mysql_like(db_id) + "\n"
        return (
            instruction
            + fields
            + (extra + "\n" if extra else "")
            + self._P.easy_prompt
            + f'Q: "{question}\nSchema_links: {schema_links}\n'
            f"Select_order:\nSQL:"
        )

    def medium_prompt(self, question: str, db_id: str, schema_links: str, extra: str = "") -> str:
        instruction = (
            self._SELECT_ORDER_INSTRUCTION
            + "# Use schema links, Foreign_keys, and retrieved demos to generate SQL.\n"
        )
        fields = self.schema.find_fields_mysql_like(db_id)
        fields += "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n\n"
        return (
            instruction
            + fields
            + (extra + "\n" if extra else "")
            + self._P.medium_prompt
            + f'Q: "{question}\nSchema_links: {schema_links}\n'
            f"A: Let’s think step by step. Select_order:"
        )

    def hard_prompt(
        self,
        question: str,
        db_id: str,
        schema_links: str,
        sub_questions: str,
        extra: str = "",
    ) -> str:
        instruction = (
            self._SELECT_ORDER_INSTRUCTION
            + "# Use intermediate representation, schema links, and retrieved demos "
            "to generate nested / set-op SQL.\n"
        )
        fields = self.schema.find_fields_mysql_like(db_id)
        fields += "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n"
        sub_q = sub_questions or "the nested sub-question required by the question"
        stepping = (
            f'\nA: Let\'s think step by step. "{question}" can be solved by knowing '
            f'the answer to the following sub-question "{sub_q}".'
        )
        return (
            instruction
            + fields
            + "\n"
            + (extra + "\n" if extra else "")
            + self._P.hard_prompt
            + f'Q: "{question}"'
            + f"\nschema_links: {schema_links}"
            + stepping
            + f'\nThe SQL query for the sub-question "{sub_q}" is'
            + "\nSelect_order:"
        )

    def text_debug_prompt(self, question: str, db_id: str, sql: str) -> str:
        fields = self.schema.find_fields_mysql_like(db_id)
        fields += "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n"
        fields += "Primary_keys = " + self.schema.find_primary_keys_mysql_like(db_id)
        return (
            self._P.DEBUG_INSTRUCTION
            + fields
            + f"#### Question: {question}\n#### SQLite SQL QUERY\n{sql}\n"
            f"#### SQLite FIXED SQL QUERY\nSELECT"
        )

    def eg_repair_prompt(
        self,
        question: str,
        db_id: str,
        sql: str,
        error: str,
        values: str,
        extra_demos: str = "",
    ) -> str:
        fields = self.schema.find_fields_mysql_like(db_id)
        fields += "Foreign_keys = " + self.schema.find_foreign_keys_mysql_like(db_id) + "\n"
        fields += "Primary_keys = " + self.schema.find_primary_keys_mysql_like(db_id)
        return (
            self._P.EG_REPAIR_INSTRUCTION
            + fields
            + (extra_demos + "\n" if extra_demos else "")
            + (values + "\n" if values else "")
            + f"#### Question: {question}\n"
            f"#### SQLite error: {error}\n"
            f"#### Broken SQL:\n{sql}\n"
            f"#### FIXED SQL QUERY\nSELECT"
        )

    def run_one(
        self,
        *,
        index: int,
        question: str,
        db_id: str,
        gold_sql: str = "",
    ) -> ExampleMetrics:
        settings = self.settings
        ex = ExampleMetrics(
            index=index,
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
        )
        data_dir = settings.spider_dir

        try:
            db_path = resolve_db_path(data_dir, db_id, split=self.split)
        except FileNotFoundError as exc:
            ex.predicted_sql = "SELECT"
            sm = StageMetrics(stage="resolve_db", ok=False, error=str(exc)[:500])
            ex.stages.append(sm)
            self._record(index, sm)
            return ex

        values = ""
        if settings.include_db_values:
            values = sample_db_values(
                db_path,
                limit_per_column=settings.db_value_limit,
                question=question,
            )

        has_fks = self._has_foreign_keys(db_id)
        retrieved_items: list[dict[str, Any]] = []
        retrieved_block = ""

        # 0) Optional early retrieve (used only after classify for generate in v2)
        # Schema-link gets question-matched DB values (slim cost path).
        link_extra = values

        # 1) Schema linking
        link_prompt = self.schema_linking_prompt(question, db_id, extra=link_extra)
        try:
            link_text, m_link = self.llm.generate(
                link_prompt,
                stage="schema_link",
                stop_sequences=["Q:"],
                est_latency_ms=self.ledger.priors.get("schema_link") if self.ledger else 1500.0,
            )
        except Exception as exc:  # noqa: BLE001
            m_link = StageMetrics(
                stage="schema_link", model=self.llm.model, ok=False, error=str(exc)[:500]
            )
            link_text = "Schema_links: []"
            ex.stages.append(m_link)
            self._record(index, m_link)
            # DNS/network outage: stop immediately (don't burn 10+ more failed calls)
            if _is_network_error(exc):
                ex.predicted_sql = "SELECT"
                ex.predicted_class = "NESTED"
                ex.exec_ok = False
                return ex
        ex.stages.append(m_link)
        self._record(index, m_link)
        schema_links = extract_schema_links(link_text)

        # 2) Classification
        cls_prompt = self.classification_prompt(question, db_id, schema_links)
        try:
            cls_text, m_cls = self.llm.generate(
                cls_prompt,
                stage="classify",
                stop_sequences=["Q:"],
                est_latency_ms=self.ledger.priors.get("classify") if self.ledger else 1500.0,
            )
        except Exception as exc:  # noqa: BLE001
            m_cls = StageMetrics(
                stage="classify", model=self.llm.model, ok=False, error=str(exc)[:500]
            )
            cls_text = 'Label: "NESTED"'
            if _is_network_error(exc):
                ex.stages.append(m_cls)
                self._record(index, m_cls)
                ex.predicted_sql = "SELECT"
                ex.predicted_class = "NESTED"
                ex.exec_ok = False
                return ex
        ex.stages.append(m_cls)
        self._record(index, m_cls)
        predicted_class = extract_label(cls_text)
        ex.predicted_class = predicted_class
        sub_questions = extract_sub_questions(cls_text)

        is_easy = "EASY" in predicted_class and "NON-NESTED" not in predicted_class
        is_non_nested = "NON-NESTED" in predicted_class
        hardness = "EASY" if is_easy else ("NON-NESTED" if is_non_nested else "NESTED")
        prefer_skeleton = (
            "GROUP BY"
            if is_easy
            else (
                "JOIN"
                if is_non_nested
                else "INTERSECT|EXCEPT|NOT IN| IN "
            )
        )
        demo_style = "easy" if is_easy else ("medium" if is_non_nested else "hard")

        # RAG for generate (class + skeleton aware)
        if settings.use_rag:
            retrieved_items, m_ret = self.retriever.retrieve(
                question,
                top_k=settings.fewshot_top_k,
                prefer_hardness=hardness,
                prefer_skeleton=prefer_skeleton,
                est_latency_ms=self.ledger.priors.get("retrieve") if self.ledger else 50.0,
            )
            ex.stages.append(m_ret)
            self._record(index, m_ret)
            demos = format_retrieved_demos(retrieved_items, style=demo_style)
            retrieved_block = "\n".join(x for x in (demos, values) if x).strip()
        else:
            retrieved_block = values

        def generate_once(temperature: float, stage: str) -> tuple[str, Any]:
            if is_easy:
                prompt = self.easy_prompt(
                    question, db_id, schema_links, extra=retrieved_block
                )
            elif is_non_nested:
                prompt = self.medium_prompt(
                    question, db_id, schema_links, extra=retrieved_block
                )
            else:
                prompt = self.hard_prompt(
                    question,
                    db_id,
                    schema_links,
                    sub_questions,
                    extra=retrieved_block,
                )
            text, metrics = self.llm.generate(
                prompt,
                stage=stage,
                temperature=temperature,
                stop_sequences=["Q:"],
                est_latency_ms=self.ledger.priors.get(stage) if self.ledger else 2000.0,
            )
            plan = extract_select_order_plan(text)
            sql = extract_sql(text)
            if plan:
                sql = apply_select_order_plan(sql, plan)
            sql = self._finalize_sql(sql, question)
            return sql, metrics

        # 3) Generate (greedy)
        try:
            sql, m_gen = generate_once(0.0, "generate")
        except Exception as exc:  # noqa: BLE001
            m_gen = StageMetrics(
                stage="generate", model=self.llm.model, ok=False, error=str(exc)[:500]
            )
            sql = "SELECT"
            ex.stages.append(m_gen)
            self._record(index, m_gen)
            if _is_network_error(exc):
                ex.predicted_sql = "SELECT"
                ex.exec_ok = False
                return ex
        ex.stages.append(m_gen)
        self._record(index, m_gen)

        debug_stops = ["#", "\n####"]
        join_demo_cache = ""

        def run_text_debug(cur_sql: str) -> str:
            dbg_prompt = self.text_debug_prompt(question, db_id, cur_sql)
            dbg_text, metrics = self.llm.generate(
                dbg_prompt,
                stage="text_debug",
                max_output_tokens=settings.debug_max_output_tokens,
                stop_sequences=debug_stops,
                est_latency_ms=self.ledger.priors.get("text_debug")
                if self.ledger
                else 1500.0,
            )
            ex.stages.append(metrics)
            self._record(index, metrics)
            fixed = self._finalize_sql(extract_sql("SELECT " + dbg_text), question)
            if is_incomplete_sql(fixed) and not is_incomplete_sql(cur_sql):
                return cur_sql
            return fixed

        if settings.use_text_debug and sql:
            try:
                sql = run_text_debug(sql)
            except Exception as exc:  # noqa: BLE001
                m_dbg = StageMetrics(
                    stage="text_debug",
                    model=self.llm.model,
                    ok=False,
                    error=str(exc)[:500],
                )
                ex.stages.append(m_dbg)
                self._record(index, m_dbg)

        def _join_demos() -> str:
            nonlocal join_demo_cache
            if join_demo_cache or not settings.use_rag:
                return join_demo_cache
            items, m_j = self.retriever.retrieve(
                question,
                top_k=min(4, settings.fewshot_top_k),
                prefer_hardness="NON-NESTED",
                prefer_skeleton="JOIN",
                est_latency_ms=self.ledger.priors.get("retrieve") if self.ledger else 50.0,
            )
            ex.stages.append(m_j)
            self._record(index, m_j)
            join_demo_cache = format_retrieved_demos(items, style="medium")
            return join_demo_cache

        def repair_once(cur_sql: str, err: str, *, join_mode: bool = False) -> str:
            extra = _join_demos() if join_mode else ""
            repair_prompt = self.eg_repair_prompt(
                question, db_id, cur_sql, str(err), values, extra_demos=extra
            )
            repair_text, m_rep = self.llm.generate(
                repair_prompt,
                stage="repair",
                max_output_tokens=settings.debug_max_output_tokens,
                stop_sequences=debug_stops,
                est_latency_ms=self.ledger.priors.get("repair")
                if self.ledger
                else 1500.0,
            )
            ex.stages.append(m_rep)
            self._record(index, m_rep)
            return self._finalize_sql(extract_sql("SELECT " + repair_text), question)

        def needs_exec_repair(result: dict[str, Any], cur_sql: str) -> tuple[bool, str, bool]:
            """Return (need_repair, error_msg, join_mode)."""
            if is_incomplete_sql(cur_sql):
                return True, "incomplete or truncated SQL", False
            if not result["success"]:
                return True, str(result.get("error") or "execution failed"), False
            if (
                settings.repair_on_empty
                and result["success"]
                and result["row_count"] == 0
            ):
                join_mode = needs_join_repair(
                    question,
                    cur_sql,
                    predicted_class=predicted_class,
                    has_foreign_keys=has_fks,
                )
                msg = (
                    "empty result; fix JOINs using Foreign_keys and WHERE/GROUP BY"
                    if join_mode
                    else "empty result"
                )
                return True, msg, join_mode
            if needs_semantic_set_op_repair(question, cur_sql):
                return (
                    True,
                    "question asks for values shared by BOTH conditions; "
                    "use INTERSECT (not OR / single-sided WHERE)",
                    False,
                )
            if needs_join_repair(
                question,
                cur_sql,
                predicted_class=predicted_class,
                has_foreign_keys=has_fks,
            ):
                return (
                    True,
                    "multi-table question missing JOIN; use Foreign_keys to join tables",
                    True,
                )
            return False, "", False

        # 4) Execution-guided repair
        if settings.use_execution_repair:
            for _ in range(settings.max_repair_attempts):
                result, m_sql = execute_query(
                    db_path,
                    sql,
                    est_latency_ms=self.ledger.priors.get("sqlite") if self.ledger else 5.0,
                )
                ex.stages.append(m_sql)
                self._record(index, m_sql)
                do_repair, err, join_mode = needs_exec_repair(result, sql)
                if not do_repair:
                    ex.exec_ok = True
                    break
                try:
                    sql = repair_once(sql, err, join_mode=join_mode)
                except Exception as exc:  # noqa: BLE001
                    m_rep = StageMetrics(
                        stage="repair",
                        model=self.llm.model,
                        ok=False,
                        error=str(exc)[:500],
                    )
                    ex.stages.append(m_rep)
                    self._record(index, m_rep)
            else:
                result, m_sql = execute_query(
                    db_path,
                    sql,
                    est_latency_ms=self.ledger.priors.get("sqlite") if self.ledger else 5.0,
                )
                ex.stages.append(m_sql)
                self._record(index, m_sql)
                ex.exec_ok = bool(result["success"]) and not is_incomplete_sql(sql)

        # 5) Self-consistency for non-easy
        if settings.use_self_consistency and not is_easy and settings.self_consistency_k > 1:
            candidates: list[tuple[str, Optional[str], bool]] = []
            seen_sql: set[str] = set()

            def _push_sc_sql(cand_sql: str, *, record: bool) -> None:
                norm = " ".join((cand_sql or "").split()).lower()
                if norm in seen_sql:
                    return
                seen_sql.add(norm)
                r, m_sql = execute_query(db_path, cand_sql)
                if record:
                    ex.stages.append(m_sql)
                    self._record(index, m_sql)
                ok = bool(r["success"]) and not is_incomplete_sql(cand_sql)
                candidates.append(
                    (cand_sql, r.get("bag_key") if ok else None, ok)
                )

            r0, m0 = execute_query(db_path, sql)
            ex.stages.append(m0)
            self._record(index, m0)
            ok0 = bool(r0["success"]) and not is_incomplete_sql(sql)
            seen_sql.add(" ".join((sql or "").split()).lower())
            candidates.append((sql, r0.get("bag_key") if ok0 else None, ok0))
            # Inject SELECT-order heuristic variants so agg-first / keys-first
            # do not split the vote (bag_key merges them).
            for variant in select_order_heuristic_variants(sql, question)[1:]:
                _push_sc_sql(variant, record=False)

            for _ in range(settings.self_consistency_k - 1):
                try:
                    alt, m_sc = generate_once(
                        settings.self_consistency_temperature, "sc_sample"
                    )
                except Exception as exc:  # noqa: BLE001
                    m_sc = StageMetrics(
                        stage="sc_sample",
                        model=self.llm.model,
                        ok=False,
                        error=str(exc)[:500],
                    )
                    alt = sql
                ex.stages.append(m_sc)
                self._record(index, m_sc)
                if settings.use_text_debug:
                    try:
                        alt = run_text_debug(alt)
                    except Exception:  # noqa: BLE001
                        pass
                alt = self._finalize_sql(alt, question)
                _push_sc_sql(alt, record=True)
                for variant in select_order_heuristic_variants(alt, question)[1:]:
                    _push_sc_sql(variant, record=False)

            sql = self._finalize_sql(
                vote_by_bag_key_then_order(
                    candidates, question, prefer_order=self._finalize_sql
                ),
                question,
            )
            ex.exec_ok = any(ok for _, _, ok in candidates)

            need_post = (
                not ex.exec_ok
                or is_incomplete_sql(sql)
                or needs_semantic_set_op_repair(question, sql)
                or needs_join_repair(
                    question,
                    sql,
                    predicted_class=predicted_class,
                    has_foreign_keys=has_fks,
                )
            )
            if settings.use_execution_repair and need_post:
                join_mode = needs_join_repair(
                    question,
                    sql,
                    predicted_class=predicted_class,
                    has_foreign_keys=has_fks,
                )
                if is_incomplete_sql(sql):
                    err = "incomplete or truncated SQL"
                elif needs_semantic_set_op_repair(question, sql):
                    err = (
                        "question asks for values shared by BOTH conditions; "
                        "use INTERSECT (not OR / single-sided WHERE)"
                    )
                elif join_mode:
                    err = "multi-table question missing JOIN; use Foreign_keys to join tables"
                else:
                    err = "all self-consistency candidates failed execution"
                try:
                    sql = repair_once(sql, err, join_mode=join_mode)
                except Exception as exc:  # noqa: BLE001
                    m_rep = StageMetrics(
                        stage="repair",
                        model=self.llm.model,
                        ok=False,
                        error=str(exc)[:500],
                    )
                    ex.stages.append(m_rep)
                    self._record(index, m_rep)
                r_final, m_final = execute_query(db_path, sql)
                ex.stages.append(m_final)
                self._record(index, m_final)
                ex.exec_ok = bool(r_final["success"]) and not is_incomplete_sql(sql)

        ex.predicted_sql = self._finalize_sql(sql, question).replace("\n", " ").strip()
        return ex
