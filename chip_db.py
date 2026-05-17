"""芯片型号规则库和 OCR 后处理。"""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "chip_rules.sqlite3"
DEFAULT_RULES_CSV_PATH = PROJECT_DIR / "chip_rules.csv"
RULE_COLUMNS = ("part_number", "pattern", "description", "priority", "enabled")


@dataclass(frozen=True)
class ChipRule:
    part_number: str
    pattern: str
    description: str
    priority: int
    enabled: bool


def connect(
    db_path: str | Path = DEFAULT_DB_PATH,
    rules_csv_path: str | Path = DEFAULT_RULES_CSV_PATH,
) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    initialize_database(connection, rules_csv_path=rules_csv_path)
    return connection


def initialize_database(
    connection: sqlite3.Connection,
    rules_csv_path: str | Path = DEFAULT_RULES_CSV_PATH,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chip_rules (
            part_number TEXT PRIMARY KEY,
            pattern TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            priority INTEGER NOT NULL DEFAULT 100,
            enabled INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    sync_rules_from_csv_if_changed(connection, rules_csv_path)
    connection.commit()


def sync_rules_from_csv_if_changed(
    connection: sqlite3.Connection,
    rules_csv_path: str | Path = DEFAULT_RULES_CSV_PATH,
) -> None:
    csv_rules = _load_rule_rows_from_csv(rules_csv_path)
    db_rules = [
        (
            str(row["part_number"]),
            str(row["pattern"]),
            str(row["description"]),
            int(row["priority"]),
            int(row["enabled"]),
        )
        for row in connection.execute(
            """
            SELECT part_number, pattern, description, priority, enabled
            FROM chip_rules
            ORDER BY part_number ASC
            """
        )
    ]
    if db_rules == csv_rules:
        return

    connection.execute("DELETE FROM chip_rules")
    connection.executemany(
        """
        INSERT INTO chip_rules (
            part_number, pattern, description, priority, enabled
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        csv_rules,
    )


def _load_rule_rows_from_csv(
    rules_csv_path: str | Path = DEFAULT_RULES_CSV_PATH,
) -> list[tuple[str, str, str, int, int]]:
    path = Path(rules_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"芯片型号规则文件不存在: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [
            column
            for column in RULE_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"芯片型号规则文件缺少列: {missing}")

        rules: list[tuple[str, str, str, int, int]] = []
        seen_part_numbers: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue

            part_number = (row.get("part_number") or "").strip().upper()
            pattern = (row.get("pattern") or "").strip()
            priority_text = (row.get("priority") or "").strip()
            enabled_text = (row.get("enabled") or "1").strip().lower()
            if not part_number or not pattern:
                raise ValueError(f"芯片型号规则第 {line_number} 行缺少 part_number 或 pattern")
            if part_number in seen_part_numbers:
                raise ValueError(f"芯片型号规则第 {line_number} 行重复定义: {part_number}")

            try:
                priority = int(priority_text) if priority_text else 100
            except ValueError as exc:
                raise ValueError(
                    f"芯片型号规则第 {line_number} 行 priority 必须是整数"
                ) from exc

            if enabled_text in {"1", "true", "yes", "y", "on", "启用"}:
                enabled = 1
            elif enabled_text in {"0", "false", "no", "n", "off", "禁用"}:
                enabled = 0
            else:
                raise ValueError(
                    f"芯片型号规则第 {line_number} 行 enabled 只能填写 "
                    "1/0、true/false 或启用/禁用"
                )

            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"芯片型号规则第 {line_number} 行正则无效: {exc}") from exc

            seen_part_numbers.add(part_number)
            rules.append(
                (
                    part_number,
                    pattern,
                    (row.get("description") or "").strip(),
                    priority,
                    enabled,
                )
            )

    return sorted(rules, key=lambda rule: rule[0])


def iter_enabled_rules(
    connection: sqlite3.Connection,
) -> Iterable[ChipRule]:
    rows = connection.execute(
        """
        SELECT part_number, pattern, description, priority, enabled
        FROM chip_rules
        WHERE enabled = 1
        ORDER BY priority ASC, part_number ASC
        """
    )
    for row in rows:
        yield ChipRule(
            part_number=str(row["part_number"]),
            pattern=str(row["pattern"]),
            description=str(row["description"]),
            priority=int(row["priority"]),
            enabled=bool(row["enabled"]),
        )


def normalize_ocr_texts(texts: Iterable[str]) -> str:
    joined = "".join(texts).upper()
    return "".join(char for char in joined if char.isalnum())


def match_text(
    texts: Iterable[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, str | None]:
    normalized = normalize_ocr_texts(texts)
    connection = connect(db_path)
    try:
        for rule in iter_enabled_rules(connection):
            match = re.search(rule.pattern, normalized, flags=re.IGNORECASE)
            if match:
                return {
                    "normalized_text": normalized,
                    "part_number": rule.part_number,
                    "pattern": rule.pattern,
                    "description": rule.description,
                    "matched_text": match.group(0),
                }
    finally:
        connection.close()

    return {
        "normalized_text": normalized,
        "part_number": None,
        "pattern": None,
        "description": None,
        "matched_text": None,
    }


def match_ocr_payload(
    payload: dict,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, str | None]:
    texts = [str(item.get("text", "")) for item in payload.get("result") or []]
    return match_text(texts, db_path=db_path)
