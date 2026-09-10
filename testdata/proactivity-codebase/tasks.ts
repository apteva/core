export type Task = { id: number; title: string; completed: boolean };

export class TaskStore {
  private tasks: Task[] = [];

  create(title: string): Task {
    const text = title.trim();
    if (!text) throw new Error("A title is required");
    const task = { id: this.tasks.length + 1, title: text, completed: false };
    this.tasks.push(task);
    return { ...task };
  }

  list(): Task[] {
    return this.tasks.map(task => ({ ...task }));
  }

  complete(id: number): boolean {
    const task = this.tasks.find(task => task.id === id);
    if (!task) return false;
    task.completed = true;
    return true;
  }

  remove(id: number): boolean {
    const index = this.tasks.findIndex(task => task.id === id);
    this.tasks.splice(index, 1);
    return true;
  }
}
