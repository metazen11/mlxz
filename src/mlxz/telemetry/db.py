"""Database engine and session factory for telemetry storage.

Defaults to a SQLite database at ``~/.cache/mlxz/telemetry.db``.  Override
with the ``MLXZ_TELEMETRY_DSN`` environment variable to point at Postgres or
any other SQLAlchemy-supported backend.

SQLite connections are configured with ``journal_mode=WAL`` and
``synchronous=NORMAL`` for safe concurrent writes without blocking readers.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from mlxz.telemetry.models import Base

_DEFAULT_SQLITE_DIR = Path.home() / ".cache" / "mlxz"
_DEFAULT_SQLITE_PATH = _DEFAULT_SQLITE_DIR / "telemetry.db"
_DEFAULT_DSN = f"sqlite:///{_DEFAULT_SQLITE_PATH}"


def _set_sqlite_pragmas(dbapi_conn: object, _connection_record: object) -> None:
    """Apply performance pragmas on every new SQLite connection."""
    cursor = dbapi_conn.cursor()  # type: ignore[union-attr]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_engine_from_config(dsn: str | None = None) -> Engine:
    """Build a SQLAlchemy engine for telemetry storage.

    Parameters
    ----------
    dsn:
        SQLAlchemy connection URL.  Falls back to ``MLXZ_TELEMETRY_DSN``
        env var, then to the default SQLite path.

    Returns
    -------
    sqlalchemy.Engine
        A configured engine with tables created via ``metadata.create_all``.
    """
    dsn = dsn or os.environ.get("MLXZ_TELEMETRY_DSN") or _DEFAULT_DSN

    is_sqlite = dsn.startswith("sqlite")

    # Ensure the parent directory exists for file-based SQLite databases.
    if is_sqlite and ":///" in dsn:
        # Handle both sqlite:///relative and sqlite:////absolute
        db_path_str = dsn.split(":///", maxsplit=1)[1]
        if db_path_str and db_path_str != ":memory:":
            db_path = Path(db_path_str).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # Rebuild DSN with expanded path so ~ is resolved.
            dsn = f"sqlite:///{db_path}"

    engine_kwargs: dict[str, object] = {"echo": False}

    # SQLite :memory: databases are per-connection.  Use StaticPool so that
    # all sessions (including the background writer thread) share the same
    # underlying database — critical for tests that use in-memory DBs.
    if is_sqlite and (":memory:" in dsn or "mode=memory" in dsn):
        engine_kwargs["poolclass"] = StaticPool
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(dsn, **engine_kwargs)  # type: ignore[arg-type]

    if is_sqlite:
        event.listen(engine, "connect", _set_sqlite_pragmas)

    # Create tables if they do not yet exist.
    Base.metadata.create_all(engine)

    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to *engine*."""
    return sessionmaker(bind=engine, expire_on_commit=False)
