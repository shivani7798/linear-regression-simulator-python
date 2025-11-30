import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from simulator import LinearRegressionGD
from plotter import RegressionPlotter
from utils import generate_linear_data

class App:
    def __init__(self, root):
        self.root = root
        root.title('Linear Regression Simulator')
        root.geometry('1000x600')

        # left controls frame
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side='left', fill='y')

        ttk.Label(ctrl, text='Learning Rate').pack(pady=(4, 0))
        self.lr_var = tk.DoubleVar(value=0.01)
        ttk.Scale(ctrl, from_=0.0001, to=1.0, orient='horizontal', variable=self.lr_var).pack(fill='x')
        self.lr_entry = ttk.Entry(ctrl, textvariable=self.lr_var)
        self.lr_entry.pack(fill='x', pady=(2, 8))

        ttk.Label(ctrl, text='Epochs').pack(pady=(4, 0))
        self.epochs_var = tk.IntVar(value=200)
        ttk.Scale(ctrl, from_=10, to=2000, orient='horizontal', variable=self.epochs_var).pack(fill='x')
        self.epochs_entry = ttk.Entry(ctrl, textvariable=self.epochs_var)
        self.epochs_entry.pack(fill='x', pady=(2, 8))

        ttk.Button(ctrl, text='Randomize Data', command=self.randomize).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Clear Data', command=self.clear_data).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Train', command=self.train).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Compare LR', command=self.compare_lr).pack(fill='x', pady=4)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
        self.status = tk.StringVar(value='Ready')
        ttk.Label(ctrl, textvariable=self.status, wraplength=160).pack(pady=8)

        # right plot area
        plot_container = ttk.Frame(root)
        plot_container.pack(side='right', fill='both', expand=True)

        self.plotter = RegressionPlotter(plot_container)

        # initial data
        X, y = generate_linear_data(seed=0)
        self.X = list(X[:20])
        self.y = list(y[:20])
        self.plotter.set_points(self.X, self.y)

        # simple click binding to add point
        self.plotter.canvas.mpl_connect('button_press_event', self.on_click)

        # lock for training thread
        self._training = False

    def randomize(self):
        X, y = generate_linear_data(seed=None)
        self.X = list(X[:30])
        self.y = list(y[:30])
        self.plotter.set_points(self.X, self.y)
        self.status.set('Randomized data')

    def clear_data(self):
        self.X = []
        self.y = []
        self.plotter.clear()
        self.status.set('Cleared data')

    def on_click(self, event):
        if event.inaxes == self.plotter.ax_data:
            self.X.append(event.xdata)
            self.y.append(event.ydata)
            self.plotter.set_points(self.X, self.y)
            self.status.set(f'Added point: ({event.xdata:.2f}, {event.ydata:.2f})')

    def train(self):
        if self._training:
            messagebox.showinfo('Training', 'Training already in progress')
            return
        if len(self.X) < 2:
            messagebox.showwarning('Not enough data', 'Add at least 2 points to train')
            return

        lr = float(self.lr_var.get())
        epochs = int(self.epochs_var.get())

        self._training = True
        self.status.set('Training...')

        def worker():
            model = LinearRegressionGD(learning_rate=lr)
            model.set_data(self.X, self.y)

            # callback to update live plot every n epochs
            def cb(epoch, m, b, loss):
                if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
                    # update plot
                    self.plotter.set_points(self.X, self.y)
                    self.plotter.plot_fit(m, b)
                    # we won't plot loss live every callback to keep GUI responsive

            m, b, losses = model.train(epochs=epochs, callback=cb)

            # final update on main thread
            self.root.after(0, lambda: self._on_training_done(m, b, losses))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_training_done(self, m, b, losses):
        self.plotter.set_points(self.X, self.y)
        self.plotter.plot_fit(m, b)
        self.plotter.plot_loss(losses)
        self._training = False
        self.status.set(f'Training done. m={m:.4f}, b={b:.4f}, final loss={losses[-1]:.4f}')

    def compare_lr(self):
        if len(self.X) < 2:
            messagebox.showwarn