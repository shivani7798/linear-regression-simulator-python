import tkinter as tk


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
