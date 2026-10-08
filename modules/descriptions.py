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
                timeout=60.0,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(
                self.path,
                timeout=60.0,
            )
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
        panoids = list(dict.fromkeys(panoids))

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


    def get_many(
        self,
        crop_keys: Iterable[tuple[str, int]],
        prompt_indices: Iterable[int],
    ) -> dict[tuple[str, int], list[str]]:
        """Load the requested prompt descriptions for each panorama crop."""
        crop_keys = list(dict.fromkeys(crop_keys))
        prompt_indices = list(dict.fromkeys(prompt_indices))

        if not crop_keys:
            return {}
        if not prompt_indices:
            raise ValueError("prompt_indices cannot be empty")

        panoids = list(dict.fromkeys(panoid for panoid, _ in crop_keys))
        pano_placeholders = ",".join("?" for _ in panoids)
        prompt_placeholders = ",".join("?" for _ in prompt_indices)

        rows = self.connection.execute(
            f"""
            SELECT panoid, subimage_index, prompt_index, description
            FROM descriptions
            WHERE panoid IN ({pano_placeholders})
              AND prompt_index IN ({prompt_placeholders})
            ORDER BY panoid, subimage_index, prompt_index
            """,
            [*panoids, *prompt_indices],
        ).fetchall()

        requested_keys = set(crop_keys)
        descriptions: dict[tuple[str, int], dict[int, str]] = {key: {} for key in crop_keys}

        for panoid, subimage_index, prompt_index, description in rows:
            key = (panoid, subimage_index)
            if key in requested_keys:
                descriptions[key][prompt_index] = description

        missing = [
            (panoid, subimage_index, prompt_index)
            for panoid, subimage_index in crop_keys
            for prompt_index in prompt_indices
            if prompt_index not in descriptions[(panoid, subimage_index)]
        ]
        if missing:
            preview = ", ".join(
                f"{panoid}[crop {subimage_index}, prompt {prompt_index + 1}]"
                for panoid, subimage_index, prompt_index in missing[:5]
            )
            raise KeyError(
                f"Missing {len(missing)} requested description(s) in {self.path}: {preview}"
            )

        return {
            key: [descriptions[key][prompt_index] for prompt_index in prompt_indices]
            for key in crop_keys
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DescriptionStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
