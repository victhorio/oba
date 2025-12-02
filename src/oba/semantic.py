import sqlite3

import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]
from ag.embeddings.openai import OpenAIEmbeddings
from attrs import define, field

from oba.vault import notes_index_build, read_note


@define
class EmbeddingsIndex:
    model: OpenAIEmbeddings
    vectors: dict[int, list[float]] = field(factory=dict)
    index: dict[int, str] = field(factory=dict)


async def index_create(vault_path: str, model: OpenAIEmbeddings) -> tuple[EmbeddingsIndex, float]:
    """
    Based on the notes index, creates a map of {note name -> embedding} for the given vault.
    """

    notes_index = notes_index_build(vault_path)
    notes_content: list[tuple[str, str]] = [
        (note_name, read_note(vault_path, note_name)) for note_name in notes_index
    ]

    embeddings = await model.embed(inputs=[contents for _, contents in notes_content])

    index = EmbeddingsIndex(model=model)
    for i, ((note_name, _), vector) in enumerate(zip(notes_content, embeddings.vectors)):
        index.vectors[i] = vector
        index.index[i] = note_name

    return index, embeddings.dollar_cost


def conn_create() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    return conn


def embeddings_store(conn: sqlite3.Connection, index: EmbeddingsIndex) -> None:
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(embedding float[1536])")

    with conn:
        for rowid, vector in index.vectors.items():
            conn.execute(
                "INSERT INTO embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(vector)),
            )


async def notes_search(
    conn: sqlite3.Connection,
    index: EmbeddingsIndex,
    query_text: str,
    k: int = 5,
) -> tuple[list[str], float]:
    embeddings = await index.model.embed(inputs=[query_text])
    query_vector = embeddings.vectors[0]

    indexes = _vector_query(conn, query_vector, k)
    return [index.index[i] for i in indexes], embeddings.dollar_cost


def _vector_query(
    conn: sqlite3.Connection,
    query: list[float],
    k: int,
) -> list[int]:
    rows = conn.execute(
        """
        SELECT rowid, vec_distance_cosine(embedding, ?) as distance
        FROM embeddings
        ORDER BY distance
        LIMIT ?
        """,
        (sqlite_vec.serialize_float32(query), k),
    ).fetchall()

    return [row["rowid"] for row in rows]
