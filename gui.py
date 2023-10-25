from tkinter import ttk, filedialog
import tkinter as tk
from PIL import Image, ImageTk
import converter
import utility


# base class

class GenericFrame:
    def __init__(self, algorithm):
        # main window
        self.root = tk.Tk()
        self.algorithm = algorithm
        self.parameters = []

        # top part
        self.top_panel = ttk.Frame(self.root, borderwidth=2, relief="solid")
        self.return_button = ttk.Button(self.top_panel, text="Return", command=self.go_back, cursor='hand2')

        # bottom part
        self.bottom_panel = ttk.Frame(self.root, borderwidth=2, relief="solid")
        self.left_panel = ttk.Frame(self.bottom_panel, borderwidth=2, relief="solid")
        self.input_image_label = ttk.Label(self.left_panel)
        self.right_panel = ttk.Frame(self.bottom_panel, borderwidth=2, relief="solid")
        self.output_image_label = ttk.Label(self.right_panel)

        self.input_image = Image.new("L", (1, 1))
        self.output_image = Image.new("L", (1, 1))

        self.window_width, self.window_height = 1500, 775
        self.x_position, self.y_position = 10, 0

        # set the maximum height and width for left and right image
        self.max_image_width = self.window_width // 2 - 20  # Adjust for padding and margins
        self.max_image_height = (self.window_height // 10) * 9 - 20  # Adjust for padding and margins

    def init_window(self):
        self.root.geometry(f"{self.window_width}x{self.window_height}+{self.x_position}+{self.y_position}")
        self.root.resizable(False, False)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=9)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.title(f'{self.algorithm} algorithm')

    def init_top_panel(self):
        self.top_panel.grid(row=0, column=0, sticky="nsew")
        self.return_button.grid(row=0, column=0, padx=5, pady=5)
        self.top_panel.grid_rowconfigure(0, weight=1)

    def init_bottom_panel(self):
        self.bottom_panel.grid(row=1, column=0, sticky="nsew")

        # Left Panel (Image Chooser)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.input_image_label.pack()
        open_button = ttk.Button(self.left_panel, text="Input Image", command=self.open_image, cursor='hand2')
        open_button.pack()
        self.left_panel.grid_columnconfigure(0, weight=1)

        # Right Panel (Processed Image)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.output_image_label.pack()

        process_button = ttk.Button(self.right_panel, text="Output Image", command=self.process_image, cursor='hand2')
        process_button.pack()
        self.right_panel.grid_columnconfigure(0, weight=1)

        # configure
        self.bottom_panel.grid_rowconfigure(0, weight=1)
        self.bottom_panel.grid_columnconfigure(0, weight=1)
        self.bottom_panel.grid_columnconfigure(1, weight=1)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.input_image = Image.open(file_path)
            self.input_image = self.input_image.convert("L")  # Convert the image to grayscale

            # Resize the image to fit within the maximum dimensions while maintaining its aspect ratio
            display_image = self.input_image.copy()
            display_image.thumbnail((self.max_image_width, self.max_image_height))

            display_image = ImageTk.PhotoImage(display_image)
            self.input_image_label.config(image=display_image)
            self.input_image_label.image = display_image

    def process_image(self):
        processed_image = self.input_image.copy()

        # get selected algorithm name
        selected_function = getattr(converter, self.algorithm)  # store the function
        processed_image = selected_function(processed_image, self.parameters)

        # Resize the image to fit within the maximum dimensions while maintaining its aspect ratio
        processed_image.thumbnail((self.max_image_width, self.max_image_height))

        processed_image = ImageTk.PhotoImage(processed_image)
        self.output_image_label.config(image=processed_image)
        self.output_image_label.image = processed_image

    def go_back(self):
        self.root.destroy()
        app = App()
        app.root.mainloop()

    def init_app(self):
        self.init_window()
        self.init_top_panel()
        self.init_bottom_panel()


class App:
    def __init__(self):
        self.algorithm_classes = utility.get_classes_from_file('gui.py')

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Algorithm Selector")

        # Create a frame to hold the elements and center it
        self.frame = tk.Frame(self.root, borderwidth=1, relief='solid')
        self.frame.pack(fill='both', expand=True)  # Configure fill and expand options

        # Create a label
        self.label = tk.Label(self.frame, text="Select an algorithm")
        self.label.grid(row=0, column=0, padx=10, pady=10)

        # Create a combobox
        algorithm_options = utility.get_available_algorithms()
        self.combobox = ttk.Combobox(self.frame, values=algorithm_options, state='readonly', cursor='hand2',
                                     font="Verdana 13 bold")
        self.combobox.grid(row=0, column=1, padx=10, pady=10)
        self.combobox.set(algorithm_options[0])

        # Create a button
        self.button = tk.Button(self.frame, text="Select", command=self.on_button_click,
                                width=30, height=3, cursor='hand2')
        self.button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        # Center the window on the screen
        window_width = 500
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    def on_button_click(self):
        selected_algorithm = self.combobox.get()
        for algorithm in self.algorithm_classes:
            algorithm_name = utility.split_camel_case(algorithm.__name__)
            if selected_algorithm == algorithm_name:
                self.root.destroy()
                running_frame = algorithm(algorithm_name)
                running_frame.init_app()
                running_frame.root.mainloop()


# start of image enhancer (chap 3)

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
        except ValueError:
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
                return
        except ValueError:
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
            if kernel_size < 3 or kernel_size % 2 == 0 or threshold <= 0 or k <= 0:
                return

        except ValueError:
            return

        self.parameters = [k, kernel_size, threshold]
        super().process_image()


class MedianFilter(Filter):
    pass


# end of image enhancer (chap 3)

# start of edge detection (chap 4)

class LaplacianFilter(GenericFrame):
    def __init__(self, algorithm):
        super().__init__(algorithm)
        values = ['filter', 'variant_filter', 'enhancer', 'variant_enhancer']
        self.type_combobox = ttk.Combobox(self.top_panel, values=values, state='readonly', cursor='hand2')
        self.type_combobox.set(values[0])

    def init_top_panel(self):
        super().init_top_panel()

        type_label = ttk.Label(self.top_panel, text="Kernel type: ")
        type_label.grid(row=0, column=1, padx=5, pady=5)
        self.type_combobox.grid(row=0, column=2, padx=5, pady=5)

    def process_image(self):
        self.parameters = self.type_combobox.get()
        super().process_image()


class Canny(GenericFrame):
    pass


class Otsu(GenericFrame):
    pass


class Isodata(GenericFrame):
    pass


class Triangle(GenericFrame):
    pass


class MaskOneDim(GenericFrame):
    pass


class Robert(GenericFrame):
    pass


class Sobel(GenericFrame):
    pass


class Prewitt(GenericFrame):
    pass

