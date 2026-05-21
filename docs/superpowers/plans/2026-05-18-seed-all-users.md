# Seed Memories for All Users

**Goal:** 修改 `scripts/seed_memories.py`，支持给数据库中所有用户各写入 2 条记忆。

**Files:**
- Modify: `scripts/seed_memories.py`

**Tasks:**
1. 导入 `SessionLocal` 和 `User`，在 `all_users=True` 时查询所有用户 ID。
2. 修改 `run()` 签名，增加 `all_users: bool = False`。
3. 循环 user_ids，为每个用户从 `SAMPLE_MEMORIES` 中取 2 条写入 mem0。
4. 更新 CLI，增加 `--all-users` flag。
5. 运行脚本验证。
