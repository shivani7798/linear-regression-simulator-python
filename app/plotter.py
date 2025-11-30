import matplotlib
#matplotlib.use('TkAgg')
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend (for servers/headless)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class RegressionPlotter:
    def __init__(self, container):
        """Create a Matplotlib figure embedded into a Tkinter container."""
        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax_data = self.fig.add_subplot(121)
        self.ax_loss = self.fig.add_subplot(122)

        self.ax_data.set_title('Data & Fit')
        self.ax_data.set_xlabel('X')
        self.ax_data.set_ylabel('y')

        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('MSE')

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # interactive point storage
        self.points_x = []
        self.points_y = []

        # line and scatter references
        self.scatter = None
        self.line = None
        self.loss_line = None

    def set_points(self, X, y):
        self.points_x = list(X)
        self.points_y = list(y)
        self._redraw_data()

    def _redraw_data(self):
        self.ax_data.cla()
        self.ax_data.set_title('Data & Fit')
        self.ax_data.set_xlabel('X')
        self.ax_data.set_ylabel('y')
        if len(self.points_x) > 0:
            self.ax_data.scatter(self.points_x, self.points_y, label='Data')
        self.ax_data.legend()
        self.canvas.draw_idle()

    def plot_fit(self, m, b):
        if len(self.points_x) == 0:
            return
        x = np.array(self.points_x)
        xs = np.linspace(np.min(x) - 1, np.max(x) + 1, 200)
        ys = m * xs + b
        self.ax_data.plot(xs, ys, label=f'Fit: y={m:.2f}x+{b:.2f}')
        self.ax_data.legend()
        self.canvas.draw_idle()

    def plot_loss(self, losses):
        self.ax_loss.cla()
        self.ax_loss.set_title('Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('MSE')
        self.ax_loss.plot(np.arange(1, len(losses) + 1), losses)
        self.canvas.draw_idle()

    def clear(self):
        self.points_x = []
        self.points_y = []
        self.ax_data.cla()
        self.ax_loss.cla()
        self.canvas.draw_idle()