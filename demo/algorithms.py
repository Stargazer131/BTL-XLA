from tkinter import ttk, messagebox
from master import GenericFrame
import tkinter as tk


class Negative(GenericFrame):
    pass


class Threshold(GenericFrame):
    pass


class PowerLaw(GenericFrame):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.c_entry = tk.Entry(self.top_panel)
        self.y_entry = tk.Entry(self.top_panel)

    def init_top_panel(self):
        super().init_top_panel()
        c_label = ttk.Label(self.top_panel, text="C: ")
        c_label.grid(row=0, column=1, padx=5, pady=5)
        self.c_entry.grid(row=0, column=2, padx=5, pady=5)
        y_label = ttk.Label(self.top_panel, text="Y: ")
        y_label.grid(row=0, column=3, padx=5, pady=5)
        self.y_entry.grid(row=0, column=4, padx=5, pady=5)

    def process_image(self):
        try:
            c = float(self.c_entry.get())
            y = float(self.y_entry.get())
            if c <= 0 or y <= 0:
                messagebox.showerror("Error", "Invalid input. Please enter number > 0")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            return

        self.parameters = [c, y]
        super().process_image()


class Filter(GenericFrame):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.k_entry = tk.Entry(self.top_panel)

    def init_top_panel(self):
        super().init_top_panel()
        k_label = ttk.Label(self.top_panel, text="Kernel size: ")
        k_label.grid(row=0, column=1, padx=5, pady=5)
        self.k_entry.grid(row=0, column=2, padx=5, pady=5)

    def process_image(self):
        try:
            k = int(self.k_entry.get())
            if k < 3 or k % 2 == 0:
                messagebox.showerror("Error", "Invalid input. Please enter an odd number and >= 3")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            return

        self.parameters = [k]
        super().process_image()


class MaxFilter(Filter):
    pass


class MinFilter(Filter):
    pass


class SimpleAverageFilter(Filter):
    pass


class WeightedAverageFilter(Filter):
    pass


class KNearestMeanFilter(GenericFrame):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.k_entry = tk.Entry(self.top_panel)
        self.threshold_entry = tk.Entry(self.top_panel)
        self.kernel_size_entry = tk.Entry(self.top_panel)

    def init_top_panel(self):
        super().init_top_panel()
        k_label = ttk.Label(self.top_panel, text="K Neighbour: ")
        k_label.grid(row=0, column=1, padx=5, pady=5)
        self.k_entry.grid(row=0, column=2, padx=5, pady=5)

        threshold_label = ttk.Label(self.top_panel, text="Threshold: ")
        threshold_label.grid(row=0, column=3, padx=5, pady=5)
        self.threshold_entry.grid(row=0, column=4, padx=5, pady=5)

        kernel_size_label = ttk.Label(self.top_panel, text="Kernel size: ")
        kernel_size_label.grid(row=0, column=5, padx=5, pady=5)
        self.kernel_size_entry.grid(row=0, column=6, padx=5, pady=5)

    def process_image(self):
        try:
            k = int(self.k_entry.get())
            threshold = int(self.threshold_entry.get())
            kernel_size = int(self.kernel_size_entry.get())
            if kernel_size < 3 or kernel_size % 2 == 0:
                messagebox.showerror("Error", "Invalid input. Please enter an kernel size with odd number and >= 3")
                return

            if threshold <= 0 or k <= 0:
                messagebox.showerror("Error", "Invalid input. Please enter threshold or k > 0")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")
            return

        self.parameters = [k, kernel_size, threshold]
        super().process_image()


class MedianFilter(Filter):
    pass

