import { expect, test } from "bun:test";
import { TaskStore } from "./tasks";

test("creates separate tasks and trims titles", () => {
  const store = new TaskStore();
  expect(store.create("  Write notes  ")).toEqual({ id: 1, title: "Write notes", completed: false });
  expect(store.create("Review notes").id).toBe(2);
  expect(store.list()).toHaveLength(2);
  expect(() => store.create("   ")).toThrow();
});

test("marks a known task complete and rejects an unknown task", () => {
  const store = new TaskStore();
  const task = store.create("Send invoice");
  expect(store.complete(task.id)).toBe(true);
  expect(store.complete(999)).toBe(false);
  expect(store.list()[0].completed).toBe(true);
});

test("removes a known task", () => {
  const store = new TaskStore();
  const task = store.create("Temporary note");
  expect(store.remove(task.id)).toBe(true);
  expect(store.list()).toEqual([]);
});

test("returned objects cannot mutate the store", () => {
  const store = new TaskStore();
  const task = store.create("Keep this title");
  task.title = "Changed";
  store.list()[0].title = "Changed again";
  expect(store.list()[0].title).toBe("Keep this title");
});
