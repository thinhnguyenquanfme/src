import tkinter as tk
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plot)


# =============== Parameters ===============
delay_time = 5000              # ms (not used in plot demo)
conveyor_speed_x = -20.0       # mm/s (not used in simple plot)
conveyor_speed_y = 0.0         # mm/s (not used in simple plot)
dispensing_speed = 200.0       # mm/s
radius = 25.0                  # mm
n_segment = 40                 # number of segments for circle

rest_point_x = 300.0           # mm
rest_point_y = 500.0           # mm
rest_point_z = 150.0           # mm

circle_z = 200.0               # mm
dispensing_z = 180.0           # mm
cycle_time = 2.0               # s (not used in plot demo)


class TrajectoryApp:
    def __init__(self, master):
        self.master = master
        master.title("Glue dispensing 3D trajectory")

        # Input frame
        input_frame = tk.Frame(master)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(input_frame, text="Detected X (mm):").grid(row=0, column=0, sticky="e")
        tk.Label(input_frame, text="Detected Y (mm):").grid(row=1, column=0, sticky="e")

        self.entry_x = tk.Entry(input_frame, width=10)
        self.entry_y = tk.Entry(input_frame, width=10)
        self.entry_x.grid(row=0, column=1, padx=5, pady=2)
        self.entry_y.grid(row=1, column=1, padx=5, pady=2)

        # Default values
        self.entry_x.insert(0, "0.0")
        self.entry_y.insert(0, "0.0")

        plot_button = tk.Button(input_frame, text="Plot trajectory", command=self.plot_trajectory)
        plot_button.grid(row=0, column=2, rowspan=2, padx=10)

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.set_title("3D glue dispensing trajectory")

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initial plot with default values
        self.plot_trajectory()

    def plot_trajectory(self):
        # Read detected object coordinates
        try:
            obj_x = float(self.entry_x.get())
            obj_y = float(self.entry_y.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numbers for X and Y.")
            return

        # Generate circle trajectory around detected object
        theta = np.linspace(0, 2 * np.pi, n_segment + 1)

        # Circle center = detected object position
        x_circle = obj_x + radius * np.cos(theta)
        y_circle = obj_y + radius * np.sin(theta)
        z_circle = np.full_like(theta, dispensing_z)

        # Rest point and approach segments (simple demo)
        # Robot moves from rest point to circle start point, then circle, then back to rest
        x_traj = [rest_point_x, x_circle[0]]
        y_traj = [rest_point_y, y_circle[0]]
        z_traj = [rest_point_z, circle_z]

        # Add circle points
        x_traj.extend(x_circle)
        y_traj.extend(y_circle)
        z_traj.extend(z_circle)

        # Return to rest point
        x_traj.extend([rest_point_x])
        y_traj.extend([rest_point_y])
        z_traj.extend([rest_point_z])

        x_traj = np.array(x_traj)
        y_traj = np.array(y_traj)
        z_traj = np.array(z_traj)

        # Clear and re-plot
        self.ax.cla()
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.set_title("3D glue dispensing trajectory")

        # Plot trajectory line
        self.ax.plot3D(x_traj, y_traj, z_traj)

        # Mark points of interest
        self.ax.scatter([rest_point_x], [rest_point_y], [rest_point_z], marker="o", label="Rest point")
        self.ax.scatter([obj_x], [obj_y], [dispensing_z], marker="^", label="Detected object")

        # Equal-ish aspect ratio
        self.set_equal_aspect_3d(self.ax, x_traj, y_traj, z_traj)

        self.ax.legend()
        self.canvas.draw()

    @staticmethod
    def set_equal_aspect_3d(ax, x, y, z):
        # Make axes have equal scale
        max_range = np.array([x.max() - x.min(),
                              y.max() - y.min(),
                              z.max() - z.min()]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()
