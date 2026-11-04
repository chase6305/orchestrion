"""Opt-in SQLite execution journal with one process owner and conservative recovery.

The journal stores final data and dispatch intents, never executable Python or
plans to replay. Commits use WAL/FULL on fixed SQLite releases and rollback/EXTRA
otherwise. Use a local filesystem with working SQLite locks; this is not a
distributed command ownership service.
"""

from __future__ import annotations

import json
import os
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from .model import NodeResult, NodeStatus, RunStatus, _name, _positive_int

try:
    import sqlite3
except ImportError:
    # Some embedded Python builds omit the optional _sqlite3 extension. Keeping
    # the journal optional must not prevent in-memory workflows from importing.
    sqlite3 = None

_APPLICATION_ID = 0x4F524348
_VERSION = 1


def _wal_reset_fixed(version: tuple) -> bool:
    """SQLite's 2026 WAL-reset fix, including its documented backport branches.

    https://www.sqlite.org/wal.html#the_wal_reset_bug
    """
    return (
        version >= (3, 51, 3)
        or (3, 50, 7) <= version < (3, 51, 0)
        or (3, 44, 6) <= version < (3, 45, 0)
    )


class JournalError(RuntimeError):
    """Durable state could not be read or committed; stop admitting device work."""


class JournalInUse(JournalError):
    """Another journal owner already holds this database's lifetime lock."""


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), ensure_ascii=True)


def _run_status(nodes) -> str:
    statuses = {NodeStatus(node["status"]) for node in nodes}
    if NodeStatus.UNKNOWN in statuses:
        return RunStatus.NEEDS_ATTENTION.value
    if statuses.intersection({NodeStatus.FAILED, NodeStatus.TIMED_OUT}):
        return RunStatus.FAILED.value
    if NodeStatus.CANCELLED in statuses:
        return RunStatus.CANCELLED.value
    return RunStatus.SUCCEEDED.value


class SQLiteJournal:
    """Caller-owned durable history for one WorkflowEngine, on one thread.

    Keep this context outside the engine context. Closing an active engine's
    journal is rejected. A journal binds to at most one engine; open a new journal
    after closing the previous engine and journal to recover its device domain.
    The ``.owner`` sidecar is a SQLite lifetime lock and must not be deleted while
    an owner is alive. Process exit releases the lock automatically.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *,
        max_events_per_run: int = 1000,
        journal_mode: str = "auto",
    ) -> None:
        if sqlite3 is None:
            raise JournalError("this Python build has no SQLite support")
        _positive_int(max_events_per_run, "max_events_per_run")
        if not isinstance(journal_mode, str) or journal_mode not in {
            "auto",
            "wal",
            "delete",
        }:
            raise ValueError("journal_mode must be auto, wal or delete")
        wal_supported = _wal_reset_fixed(sqlite3.sqlite_version_info)
        if journal_mode == "wal" and not wal_supported:
            raise JournalError("WAL mode requires SQLite with the WAL-reset fix")
        mode = (
            ("wal" if wal_supported else "delete")
            if journal_mode == "auto"
            else journal_mode
        )
        if str(path) == ":memory:":
            raise ValueError("a durable journal requires a filesystem path")
        self.path = str(Path(path).expanduser().resolve())
        self._event_limit = max_events_per_run
        self._pid = os.getpid()
        self._owner = None
        self._bound = False
        self._db = None
        self._lock = None
        connection_options = {"timeout": 0, "isolation_level": None}
        if hasattr(sqlite3, "LEGACY_TRANSACTION_CONTROL"):
            connection_options["autocommit"] = sqlite3.LEGACY_TRANSACTION_CONTROL
        try:
            # A separate database lets the real journal commit while ownership
            # stays exclusive across commits, connections, and process crashes.
            self._lock = sqlite3.connect(self.path + ".owner", **connection_options)
            try:
                self._lock.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as exc:
                raise JournalInUse(
                    "journal already has an owner: " + self.path
                ) from exc
            self._db = sqlite3.connect(self.path, **connection_options)
            self._db.row_factory = sqlite3.Row
            # Initialize rollback-mode databases atomically before changing mode;
            # an unrelated database must be rejected without changing its header.
            self._db.execute("PRAGMA synchronous=EXTRA")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._initialize()
            selected = self._db.execute("PRAGMA journal_mode=" + mode).fetchone()[0]
            if selected != mode:
                raise JournalError("SQLite cannot enable journal mode: " + mode)
            self.journal_mode = selected
            sync = "FULL" if mode == "wal" else "EXTRA"
            self._db.execute("PRAGMA synchronous=" + sync)
            self.synchronous = self._db.execute("PRAGMA synchronous").fetchone()[0]
            if self.synchronous != (2 if mode == "wal" else 3):
                raise JournalError("SQLite cannot enable durable synchronization")
        except (OSError, sqlite3.Error, JournalError) as exc:
            if self._db is not None:
                self._db.close()
                self._db = None
            if self._lock is not None:
                self._lock.close()
                self._lock = None
            if isinstance(exc, JournalError):
                raise
            raise JournalError("cannot open journal: " + self.path) from exc

    def _initialize(self) -> None:
        app = self._db.execute("PRAGMA application_id").fetchone()[0]
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        tables = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if app == _APPLICATION_ID and version == _VERSION:
            return
        if app or version or tables:
            raise JournalError("unsupported or unrelated journal database")
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE runs ("
                "run_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE, "
                "fingerprint TEXT, status TEXT NOT NULL, submitted_at REAL NOT NULL, "
                "node_count INTEGER NOT NULL CHECK(node_count>0), "
                "finished_at REAL)"
            )
            db.execute(
                "CREATE TABLE nodes ("
                "run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, "
                "node_id TEXT NOT NULL, action TEXT NOT NULL, action_version TEXT, status TEXT NOT NULL, "
                "output TEXT, error TEXT, started_at REAL, finished_at REAL, "
                "feedback_dropped INTEGER NOT NULL DEFAULT 0, claims TEXT NOT NULL, "
                "lease_active INTEGER NOT NULL DEFAULT 0, checkpoint TEXT, "
                "reconciled_at REAL, reconciliation_evidence TEXT, "
                "PRIMARY KEY(run_id, node_id))"
            )
            db.execute(
                "CREATE TABLE events ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, "
                "node_id TEXT, kind TEXT NOT NULL, timestamp REAL NOT NULL, detail TEXT)"
            )
            db.execute("CREATE INDEX events_run ON events(run_id, sequence)")
            db.execute("PRAGMA application_id={}".format(_APPLICATION_ID))
            db.execute("PRAGMA user_version={}".format(_VERSION))

    @property
    def closed(self) -> bool:
        return self._db is None

    def _check_open(self) -> None:
        if self.closed:
            raise JournalError("journal is closed")
        if os.getpid() != self._pid:
            raise JournalInUse("an inherited journal cannot be used after fork")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._check_open()
        try:
            with self._db:
                # Connection context managers alone do not begin a transaction
                # for DDL. Schema creation and header versions must be atomic too.
                self._db.execute("BEGIN IMMEDIATE")
                yield self._db
        except (sqlite3.Error, OSError) as exc:
            raise JournalError("journal transaction failed: " + str(exc)) from exc

    def _bind(self, engine) -> None:
        self._check_open()
        if self._bound:
            raise JournalInUse(
                "journal is already bound; open a new journal to recover"
            )
        self._owner = weakref.ref(engine)
        self._bound = True

    def _event(self, db, run_id, node_id, kind, detail=None) -> None:
        db.execute(
            "INSERT INTO events(run_id,node_id,kind,timestamp,detail) VALUES(?,?,?,?,?)",
            (run_id, node_id, kind, time.time(), _json(detail)),
        )
        # Indexed per-run history, independent of potentially interleaved run IDs.
        db.execute(
            "DELETE FROM events WHERE run_id=? AND sequence <= ("
            "SELECT sequence FROM events WHERE run_id=? "
            "ORDER BY sequence DESC LIMIT 1 OFFSET ?)",
            (run_id, run_id, self._event_limit),
        )

    def _admit(self, run, compiled, fingerprint) -> None:
        with self._transaction() as db:
            db.execute(
                "INSERT INTO runs(run_id,idempotency_key,fingerprint,status,submitted_at,node_count) "
                "VALUES(?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.idempotency_key,
                    fingerprint,
                    "running",
                    time.time(),
                    len(compiled),
                ),
            )
            db.executemany(
                "INSERT INTO nodes(run_id,node_id,action,action_version,status,claims) "
                "VALUES(?,?,?,?,?,?)",
                (
                    (
                        run.run_id,
                        item.node.node_id,
                        item.node.action,
                        item.action_version,
                        "pending",
                        _json({claim.resource: claim.units for claim in item.claims}),
                    )
                    for item in compiled
                ),
            )
            self._event(db, run.run_id, None, "admitted")

    def _start(self, run_id: str, result: NodeResult) -> None:
        self._start_many(((run_id, result),))

    def _start_many(self, records) -> None:
        """Commit a dispatch barrier before any of its adapters may enter."""
        with self._transaction() as db:
            for run_id, result in records:
                cursor = db.execute(
                    "UPDATE nodes SET status='running', started_at=?, lease_active=1 "
                    "WHERE run_id=? AND node_id=?",
                    (result.started_at, run_id, result.node_id),
                )
                if cursor.rowcount != 1:
                    raise JournalError(
                        "missing durable operation record before dispatch"
                    )
                self._event(db, run_id, result.node_id, "dispatch_intent")

    def _settle(self, run_id: str, result: NodeResult) -> None:
        self._settle_many(run_id, (result,))

    def _settle_many(self, run_id: str, results) -> None:
        """Commit one synchronous graph transition, including its skipped nodes."""
        with self._transaction() as db:
            for result in results:
                cursor = db.execute(
                    "UPDATE nodes SET status=?, output=?, error=?, started_at=?, "
                    "finished_at=?, feedback_dropped=?, lease_active=? "
                    "WHERE run_id=? AND node_id=?",
                    (
                        result.status.value,
                        _json(result.output),
                        result.error,
                        result.started_at,
                        result.finished_at,
                        result.feedback_dropped,
                        int(result.status == NodeStatus.UNKNOWN),
                        run_id,
                        result.node_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise JournalError("missing durable operation result record")
                self._event(db, run_id, result.node_id, "settled", result.status.value)

    def _finish(self, run_id: str, status: RunStatus) -> None:
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
                (status.value, time.time(), run_id),
            )
            if cursor.rowcount != 1:
                raise JournalError("missing durable run record")
            self._event(db, run_id, None, "finished", status.value)

    def _checkpoint(self, run_id: str, node_id: str, data: Any) -> None:
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE nodes SET checkpoint=? WHERE run_id=? AND node_id=?",
                (_json(data), run_id, node_id),
            )
            if cursor.rowcount != 1:
                raise JournalError("missing durable checkpoint record")
            self._event(db, run_id, node_id, "checkpoint", data)

    def _reconcile(self, run_id: str, node_id: str, evidence: str) -> None:
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE nodes SET lease_active=0, reconciled_at=?, reconciliation_evidence=? "
                "WHERE run_id=? AND node_id=? AND status='unknown' AND lease_active=1",
                (time.time(), evidence, run_id, node_id),
            )
            if cursor.rowcount != 1:
                raise JournalError("missing durable reconciliation record")
            self._event(db, run_id, node_id, "reconciled", {"evidence": evidence})

    def _prune_keys(self, keys) -> None:
        with self._transaction() as db:
            db.executemany(
                "UPDATE runs SET idempotency_key=NULL, fingerprint=NULL "
                "WHERE idempotency_key=?",
                ((key,) for key in keys),
            )

    def _load(self) -> list:
        """Load keyed history and all unfinished/unreconciled device work."""
        self._check_open()
        try:
            ids = self._db.execute(
                "SELECT run_id FROM runs WHERE idempotency_key IS NOT NULL "
                "OR finished_at IS NULL OR EXISTS ("
                "SELECT 1 FROM nodes WHERE nodes.run_id=runs.run_id AND ("
                "lease_active=1 OR status IN ('running','cancelling') OR "
                "(status='unknown' AND (reconciled_at IS NULL OR "
                "reconciliation_evidence IS NULL OR trim(reconciliation_evidence)='')))) "
                "ORDER BY rowid"
            ).fetchall()
            return [self.get_run(row[0]) for row in ids]
        except sqlite3.Error as exc:
            raise JournalError("cannot load journal") from exc

    def _recover(self, records: list) -> None:
        """Commit conservative recovery after the engine validates all leases."""
        with self._transaction() as db:
            for record in records:
                if record["finished_at"] is not None:
                    continue
                for node in record["nodes"]:
                    status = NodeStatus(node["status"])
                    if status.terminal:
                        continue
                    status = (
                        NodeStatus.UNKNOWN
                        if node["lease_active"]
                        else (NodeStatus.CANCELLED)
                    )
                    node.update(
                        status=status.value,
                        error="process interrupted; device result requires reconciliation"
                        if node["lease_active"]
                        else "process interrupted before dispatch",
                        finished_at=time.time(),
                    )
                    db.execute(
                        "UPDATE nodes SET status=?, error=?, finished_at=? "
                        "WHERE run_id=? AND node_id=?",
                        (
                            node["status"],
                            node["error"],
                            node["finished_at"],
                            record["run_id"],
                            node["node_id"],
                        ),
                    )
                record["status"] = _run_status(record["nodes"])
                record["finished_at"] = time.time()
                db.execute(
                    "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
                    (record["status"], record["finished_at"], record["run_id"]),
                )
                self._event(db, record["run_id"], None, "recovered", record["status"])

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Read a defensive JSON-compatible record without contacting devices."""
        self._check_open()
        _name(run_id, "run ID")
        try:
            row = self._db.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            record = dict(row)
            record["nodes"] = []
            for row in self._db.execute(
                "SELECT * FROM nodes WHERE run_id=? ORDER BY rowid", (run_id,)
            ):
                node = dict(row)
                node["output"] = json.loads(node["output"]) if node["output"] else None
                node["claims"] = json.loads(node["claims"])
                node["checkpoint"] = (
                    json.loads(node["checkpoint"]) if node["checkpoint"] else None
                )
                node["lease_active"] = bool(node["lease_active"])
                record["nodes"].append(node)
            if len(record["nodes"]) != record["node_count"]:
                raise JournalError(
                    "durable run has missing operation records: " + run_id
                )
            return record
        except (sqlite3.Error, ValueError) as exc:
            raise JournalError("cannot read run: " + run_id) from exc

    def events(self, run_id: str, *, after: int = 0) -> list:
        """Read bounded durable audit events; progress feedback stays in memory."""
        self._check_open()
        _name(run_id, "run ID")
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a nonnegative integer")
        try:
            return [
                dict(row, detail=json.loads(row["detail"]))
                for row in self._db.execute(
                    "SELECT * FROM events WHERE run_id=? AND sequence>? "
                    "ORDER BY sequence",
                    (run_id, after),
                )
            ]
        except (sqlite3.Error, ValueError) as exc:
            raise JournalError("cannot read journal events") from exc

    def prune_completed_runs(self, keep_last: int = 1000) -> int:
        """Delete old settled, unkeyed records, preserving all unresolved leases."""
        if (
            isinstance(keep_last, bool)
            or not isinstance(keep_last, int)
            or keep_last < 0
        ):
            raise ValueError("keep_last must be a nonnegative integer")
        with self._transaction() as db:
            cursor = db.execute(
                "DELETE FROM runs WHERE run_id IN ("
                "SELECT run_id FROM runs WHERE finished_at IS NOT NULL "
                "AND idempotency_key IS NULL AND NOT EXISTS ("
                "SELECT 1 FROM nodes WHERE nodes.run_id=runs.run_id AND lease_active=1) "
                "ORDER BY rowid DESC LIMIT -1 OFFSET ?)",
                (keep_last,),
            )
            return cursor.rowcount

    def close(self) -> None:
        """Release ownership after the engine has closed and all workers drained."""
        if self.closed:
            return
        owner = None if self._owner is None else self._owner()
        if owner is not None and (not owner._closed or owner._runs or owner._workers):
            raise JournalInUse("close and drain the engine before closing its journal")
        self._db.close()
        self._db = None
        self._lock.close()
        self._lock = None

    def __enter__(self) -> "SQLiteJournal":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
