"""Bidirectional sync between PostgreSQL user_memories and Milvus mem0_memories.

Usage:
    cd backend
    venv/Scripts/python scripts/sync_pg_to_milvus.py --dry-run          # preview both directions
    venv/Scripts/python scripts/sync_pg_to_milvus.py                    # full bidirectional sync
    venv/Scripts/python scripts/sync_pg_to_milvus.py --pg-to-milvus     # PostgreSQL → Milvus only
    venv/Scripts/python scripts/sync_pg_to_milvus.py --milvus-to-pg     # Milvus → PostgreSQL only
    venv/Scripts/python scripts/sync_pg_to_milvus.py --user-id 4        # sync only user_id=4
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import SessionLocal
from app.models import UserMemory, User
from app.services.memory_client import get_memory


def get_milvus_contents(memory, user_id: str) -> dict[str, dict]:
    """Return {content_stripped: {id, memory, metadata}} for a user's Milvus memories."""
    try:
        results = memory.get_all(filters={"user_id": user_id})
        items = results.get("results", []) if isinstance(results, dict) else []
        out = {}
        for r in items:
            content = r.get("memory", "").strip()
            if content:
                out[content] = r
        return out
    except Exception:
        print(f"  WARNING: Failed to query Milvus for user_id={user_id}")
        return {}


def get_pg_contents(db, user_id: int) -> dict[str, UserMemory]:
    """Return {content_stripped: UserMemory} for a user's PG memories."""
    records = db.query(UserMemory).filter(UserMemory.user_id == user_id).all()
    out = {}
    for r in records:
        content = (r.content or "").strip()
        if content:
            out[content] = r
    return out


def sync_pg_to_milvus(memory, dry_run: bool = False, target_user_id: int | None = None) -> dict:
    """Copy PostgreSQL memories into Milvus."""
    db = SessionLocal()
    try:
        query = db.query(UserMemory)
        if target_user_id is not None:
            query = query.filter(UserMemory.user_id == target_user_id)
        pg_records = query.order_by(UserMemory.id).all()
    finally:
        db.close()

    if not pg_records:
        print("[PG → Milvus] No PostgreSQL records found.")
        return {"total": 0, "synced": 0, "skipped": 0, "failed": 0}

    # Group by user_id
    by_user: dict[int, list] = {}
    for rec in pg_records:
        by_user.setdefault(rec.user_id, []).append(rec)

    stats = {"total": len(pg_records), "synced": 0, "skipped": 0, "failed": 0}
    print(f"\n{'='*60}")
    print(f"[PG → Milvus] Syncing {stats['total']} records from PostgreSQL to Milvus...")

    for uid, records in by_user.items():
        uid_str = str(uid)
        milvus_map = get_milvus_contents(memory, uid_str)
        print(f"  user_id={uid}: {len(records)} PG records, {len(milvus_map)} Milvus records")

        for rec in records:
            content = (rec.content or "").strip()
            if content in milvus_map:
                stats["skipped"] += 1
                continue

            if dry_run:
                print(f"    [DRY-RUN] → Milvus: {content[:60]}...")
                stats["synced"] += 1
                continue

            try:
                memory.add(
                    [{"role": "user", "content": content}],
                    user_id=uid_str,
                    metadata={"source": rec.source},
                )
                stats["synced"] += 1
                milvus_map[content] = {}  # track to avoid double-add within batch
                print(f"    [OK] → Milvus: {content[:60]}...")
            except Exception as e:
                print(f"    [FAIL] → Milvus: {content[:60]}... -> {e}")
                stats["failed"] += 1

    print(f"[PG → Milvus] Done: total={stats['total']} synced={stats['synced']} "
          f"skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def sync_milvus_to_pg(memory, dry_run: bool = False, target_user_id: int | None = None) -> dict:
    """Copy Milvus memories into PostgreSQL."""
    db = SessionLocal()
    try:
        user_ids = [target_user_id] if target_user_id else [u.id for u in db.query(User).all()]
    finally:
        db.close()

    stats = {"total": 0, "synced": 0, "skipped": 0, "failed": 0}

    print(f"\n{'='*60}")
    print(f"[Milvus → PG] Syncing from Milvus to PostgreSQL...")

    for uid in user_ids:
        uid_str = str(uid)
        milvus_map = get_milvus_contents(memory, uid_str)

        db = SessionLocal()
        try:
            pg_map = get_pg_contents(db, uid)
        finally:
            db.close()

        print(f"  user_id={uid}: {len(milvus_map)} Milvus records, {len(pg_map)} PG records")
        stats["total"] += len(milvus_map)

        for content, milvus_rec in milvus_map.items():
            if content in pg_map:
                stats["skipped"] += 1
                continue

            source = "auto_extracted"
            metadata = milvus_rec.get("metadata")
            if isinstance(metadata, dict):
                source = metadata.get("source", "auto_extracted")

            if dry_run:
                print(f"    [DRY-RUN] → PG: user_id={uid} {content[:60]}...")
                stats["synced"] += 1
                continue

            db = SessionLocal()
            try:
                pg_mem = UserMemory(user_id=uid, content=content, source=source)
                db.add(pg_mem)
                db.commit()
                stats["synced"] += 1
                pg_map[content] = pg_mem
                print(f"    [OK] → PG: user_id={uid} {content[:60]}...")
            except Exception as e:
                print(f"    [FAIL] → PG: user_id={uid} {content[:60]}... -> {e}")
                stats["failed"] += 1
                db.rollback()
            finally:
                db.close()

    print(f"[Milvus → PG] Done: total={stats['total']} synced={stats['synced']} "
          f"skipped={stats['skipped']} failed={stats['failed']}")
    return stats


def run(dry_run: bool = False, target_user_id: int | None = None,
        pg_to_milvus: bool = True, milvus_to_pg: bool = True) -> None:
    memory = get_memory()
    if memory is None:
        print("ERROR: mem0 initialization failed. Check Ollama and Milvus.")
        return

    if dry_run:
        print("=" * 60)
        print("  DRY-RUN MODE — no writes will be performed")
        print("=" * 60)

    pg_stats = {"total": 0, "synced": 0, "skipped": 0, "failed": 0}
    milvus_stats = {"total": 0, "synced": 0, "skipped": 0, "failed": 0}

    if pg_to_milvus:
        pg_stats = sync_pg_to_milvus(memory, dry_run=dry_run, target_user_id=target_user_id)

    if milvus_to_pg:
        milvus_stats = sync_milvus_to_pg(memory, dry_run=dry_run, target_user_id=target_user_id)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SYNC SUMMARY")
    print(f"{'=' * 60}")
    if pg_to_milvus:
        print(f"  PG → Milvus:  total={pg_stats['total']} synced={pg_stats['synced']} "
              f"skipped={pg_stats['skipped']} failed={pg_stats['failed']}")
    if milvus_to_pg:
        print(f"  Milvus → PG:  total={milvus_stats['total']} synced={milvus_stats['synced']} "
              f"skipped={milvus_stats['skipped']} failed={milvus_stats['failed']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bidirectional sync between PostgreSQL and Milvus")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--user-id", type=int, default=None, help="Sync only a specific user_id")
    parser.add_argument("--pg-to-milvus", action="store_true", default=False, help="Only PostgreSQL → Milvus")
    parser.add_argument("--milvus-to-pg", action="store_true", default=False, help="Only Milvus → PostgreSQL")
    args = parser.parse_args()

    # If neither direction specified, do both
    pg_to_mv = args.pg_to_milvus or (not args.pg_to_milvus and not args.milvus_to_pg)
    mv_to_pg = args.milvus_to_pg or (not args.pg_to_milvus and not args.milvus_to_pg)

    run(dry_run=args.dry_run, target_user_id=args.user_id,
        pg_to_milvus=pg_to_mv, milvus_to_pg=mv_to_pg)
