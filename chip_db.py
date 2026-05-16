"""芯片型号规则库和 OCR 后处理。"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "chip_rules.sqlite3"


@dataclass(frozen=True)
class ChipRule:
    part_number: str
    pattern: str
    description: str
    priority: int
    enabled: bool


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    initialize_database(connection)
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
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
    rules: list[tuple[str, str, str, int, int]] = [
        # (part_number, pattern, description, priority, enabled)
        ("SN74LS00N", "74[0-9A-Z]{2}00", "Quad 2-input NAND gate", 10, 1),
        ("SN74LS266N", "74[0-9A-Z]{2}266", "4-bit synchronous binary counter", 10, 1),
        ("SN74LS138N", "74[0-9A-Z]{2}138", "3-to-8 line decoder", 10, 1),
        ("SN74LS161N", "74[0-9A-Z]{2}161", "4-bit synchronous up/down counter", 10, 1),
        ("STC89C52RC", "89C52", "8051 microcontroller", 10, 1),
        # 继续在此添加更多规则...
    ]

    connection.executemany(
        """
        INSERT OR IGNORE INTO chip_rules (
            part_number, pattern, description, priority, enabled
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rules,
    )
    connection.commit()


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
            match = re.search(rule.pattern, normalized)
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
