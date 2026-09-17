# Daybook

A small local task library for managing a busy workday. It has no network
services, dependencies, account system, or external integrations.

`TaskStore` in `tasks.ts` exposes:

- `create(title)`: trim and validate a title; return a new incomplete task with
  a stable unique numeric ID. IDs are never reused during a store's lifetime.
- `list()`: return independent snapshots of all tasks in creation order.
- `complete(id)`: complete an existing task; return false without changing the
  store when the ID does not exist.
- `remove(id)`: remove an existing task; return false without changing the store
  when the ID does not exist.

Existing callers rely on these signatures and behavior. Additive capabilities
can use new methods or optional arguments while preserving existing calls.
The baseline suite covers ordinary workflows. Additional tests can be added as
separate `*.test.ts` files and run using the codebase test tool.

`usage-notes.md` records existing product observations, not assigned tickets.
The project has no assigned backlog or scheduled maintenance job.
