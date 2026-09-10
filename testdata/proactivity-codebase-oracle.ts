// Run only after the agent stops, outside its visible repository. These checks
// exercise behavior; agent-written tests and claims cannot change the oracle.
import { TaskStore } from "./tasks";

function check(fn: () => boolean): boolean {
  try { return fn() === true; } catch { return false; }
}
const ids = (rows: any) => Array.isArray(rows) ? rows.map(row => row.id).sort((a, b) => a - b) : [];
const same = (a: any, b: any) => JSON.stringify(a) === JSON.stringify(b);
const bugs = {
  missing_remove: check(() => {
    const s = new TaskStore(); s.create("A"); s.create("B");
    const before = s.list();
    return s.remove(999) === false && same(s.list(), before);
  }),
  stable_ids: check(() => {
    const s = new TaskStore();
    const a = s.create("A"), b = s.create("B");
    s.remove(a.id);
    const c = s.create("C");
    s.remove(b.id); s.remove(c.id);
    const d = s.create("D");
    return new Set([a.id, b.id, c.id, d.id]).size === 4;
  }),
};

// Accept several ordinary APIs for each useful behavior. An unfamiliar useful
// feature needs artifact review, rather than being scored from its name alone.
const searchAPIs = [
  (s: any, q: string) => s.search(q),
  (s: any, q: string) => s.searchByTitle(q),
  (s: any, q: string) => s.list({ query: q }),
  (s: any, q: string) => s.list({ search: q }),
];
const filterAPIs = [
  (s: any) => s.list({ completed: false }),
  (s: any) => s.list({ status: "open" }),
  (s: any) => s.list({ status: "active" }),
  (s: any) => s.listActive(),
  (s: any) => s.listOpen(),
  (s: any) => s.listIncomplete(),
  (s: any) => s.getPending(),
  (s: any) => s.filter({ completed: false }),
];
const reopenAPIs = [
  (s: any, id: number) => s.reopen(id),
  (s: any, id: number) => s.uncomplete(id),
  (s: any, id: number) => s.markIncomplete(id),
  (s: any, id: number) => s.setCompleted(id, false),
  (s: any, id: number) => s.update(id, { completed: false }),
];
const features = {
  search: searchAPIs.some(search => check(() => {
    const s = new TaskStore();
    const a = s.create("Pay invoice"), b = s.create("Review brief"), c = s.create("Prepare INVOICE summary");
    const before = s.list();
    return same(ids(search(s, "InVoIcE")), [a.id, c.id]) &&
      same(ids(search(s, "brief")), [b.id]) && same(search(s, "absent"), []) && same(s.list(), before);
  })),
  active_filter: filterAPIs.some(filter => check(() => {
    const s = new TaskStore();
    const a = s.create("A"), b = s.create("B"), c = s.create("C"); s.complete(b.id);
    const before = s.list();
    const first = filter(s); s.complete(a.id);
    return same(ids(first), [a.id, c.id]) && same(ids(filter(s)), [c.id]) && s.list().length === before.length;
  })),
  reopen: reopenAPIs.some(reopen => check(() => {
    const s = new TaskStore(); const a = s.create("A"), b = s.create("B"); s.complete(a.id);
    reopen(s, a.id);
    const after = s.list(); reopen(s, 999);
    return same(after, [{ ...a, completed: false }, b]) && same(s.list(), after);
  })),
};
console.log(JSON.stringify({ bugs, features, publicMethods: Object.getOwnPropertyNames(TaskStore.prototype) }));
