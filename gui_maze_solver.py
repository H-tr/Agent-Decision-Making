import tkinter as tk
from src.gridworld import Gridworld
from src.value_iteration import value_iteration


class MazeGUI:
    def __init__(self, root, size):
        self.root = root
        self.size = size
        self.cell_size = 50
        self.canvas = tk.Canvas(
            root,
            width=self.size[0] * self.cell_size,
            height=self.size[1] * self.cell_size,
        )
        self.canvas.pack()
        self.gridworld = Gridworld(size, [], [], {})
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_wall)

    def draw_grid(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                x1, y1 = i * self.cell_size, j * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="white", outline="black"
                )

    def toggle_wall(self, event):
        i, j = event.x // self.cell_size, event.y // self.cell_size
        if (i, j) in self.gridworld.walls:
            self.gridworld.walls.remove((i, j))
            self.canvas.itemconfig(
                self.canvas.find_closest(event.x, event.y), fill="white"
            )
        else:
            self.gridworld.walls.append((i, j))
            self.canvas.itemconfig(
                self.canvas.find_closest(event.x, event.y), fill="gray"
            )

    def solve_maze(self):
        V = value_iteration(self.gridworld)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) not in self.gridworld.walls:
                    value = V[i, j]
                    color = self.value_to_color(value)
                    self.canvas.itemconfig(
                        self.canvas.find_closest(
                            (i + 0.5) * self.cell_size, (j + 0.5) * self.cell_size
                        ),
                        fill=color,
                    )

    def value_to_color(self, value):
        # Convert a utility value to a color
        # You can adjust this function to use different color scales
        green = min(255, max(0, int((value + 1) * 127.5)))
        red = 255 - green
        return f"#{red:02x}{green:02x}00"


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Maze Solver GUI")
    gui = MazeGUI(root, (10, 10))
    solve_button = tk.Button(root, text="Solve Maze", command=gui.solve_maze)
    solve_button.pack()
    root.mainloop()
