from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sqlite3


class DescriptionStore:
    def __init__(
        self,
        path: str | Path,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = read_only

        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(self.path)

            self.connection = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self._create_table()

    def _create_table(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS descriptions (
                panoid TEXT NOT NULL,
                subimage_index INTEGER NOT NULL,
                prompt_index INTEGER NOT NULL,
                description TEXT NOT NULL,
                PRIMARY KEY (panoid, subimage_index, prompt_index)
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()

    def add(
        self,
        panoid: str,
        subimage_index: int,
        prompt_index: int,
        description: str,
    ) -> None:
        self.add_many([
            (panoid, subimage_index, prompt_index, description),
        ])

    def add_many(
        self,
        descriptions: Iterable[tuple[str, int, int, str]],
    ) -> None:
        if self.read_only:
            raise RuntimeError("DescriptionStore was opened in read-only mode")

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO descriptions (
                    panoid,
                    subimage_index,
                    prompt_index,
                    description
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT (panoid, subimage_index, prompt_index)
                DO UPDATE SET description = excluded.description
                """,
                descriptions,
            )

    def existing_keys(
        self,
        panoids: Iterable[str],
    ) -> set[tuple[str, int, int]]:
        panoids = list(panoids)

        if not panoids:
            return set()

        placeholders = ",".join("?" for _ in panoids)

        rows = self.connection.execute(
            f"""
            SELECT panoid, subimage_index, prompt_index
            FROM descriptions
            WHERE panoid IN ({placeholders})
            """,
            panoids,
        ).fetchall()

        return set(rows)

    def get(
        self,
        panoid: str,
        subimage_index: int,
        prompt_index: int,
    ) -> str:
        row = self.connection.execute(
            """
            SELECT description
            FROM descriptions
            WHERE panoid = ?
              AND subimage_index = ?
              AND prompt_index = ?
            """,
            (panoid, subimage_index, prompt_index),
        ).fetchone()

        if row is None:
            raise KeyError(
                "No description found for "
                f"panoid={panoid!r}, "
                f"subimage_index={subimage_index}, "
                f"prompt_index={prompt_index}"
            )

        return row[0]

    def get_all(
        self,
        panoid: str,
        subimage_index: int,
    ) -> dict[int, str]:
        rows = self.connection.execute(
            """
            SELECT prompt_index, description
            FROM descriptions
            WHERE panoid = ?
              AND subimage_index = ?
            ORDER BY prompt_index
            """,
            (panoid, subimage_index),
        ).fetchall()

        return {
            prompt_index: description
            for prompt_index, description in rows
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DescriptionStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()