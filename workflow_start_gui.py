import tkinter as tk
from tkinter import ttk
import os

class OCTAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Start")
        self.save_intermediate_steps = False
        self.display_intermediate_results = False
        self.oct_start = 1
        self.oct_registration_frame = 1
        self.oct_end = 100
        self.arcomo_number = 0

        self.create_widgets()

    def create_widgets(self):
        # Labels
        self.label_oct_registration_frame = ttk.Label(self.master, text="OCT Registration Frame:")
        self.label_oct_start = ttk.Label(self.master, text="OCT Start Frame:")
        self.label_oct_end = ttk.Label(self.master, text="OCT End Frame:")
        self.label_arcomo_number = ttk.Label(self.master, text="ArCoMo Number:")

        # Entry Widgets
        self.entry_oct_registration_frame = ttk.Entry(self.master)
        self.entry_oct_start = ttk.Entry(self.master)
        self.entry_oct_end = ttk.Entry(self.master)
        self.entry_arcomo_number = ttk.Entry(self.master)

        # Checkboxes
        self.save_intermediate_steps_var = tk.IntVar()
        self.check_save_intermediate_steps = ttk.Checkbutton(self.master,
                                                             text="Save Intermediate Steps",
                                                             variable=self.save_intermediate_steps_var)

        self.display_intermediate_results_var = tk.IntVar()
        self.check_display_intermediate_results = ttk.Checkbutton(self.master,
                                                                 text="Display Intermediate Results",
                                                                 variable=self.display_intermediate_results_var)

        # Button
        self.run_button = ttk.Button(self.master, text="Run Analysis", command=self.run_analysis)
        self.get_oct_frames_info_button = ttk.Button(self.master, text="Get OCT Frames Info", command=self.get_oct_frames_info)

        # Layout
        self.label_arcomo_number.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_arcomo_number.grid(row=0, column=1, padx=10, pady=5)

        self.get_oct_frames_info_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.label_oct_start.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_start.grid(row=2, column=1, padx=10, pady=5)

        self.label_oct_end.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_end.grid(row=3, column=1, padx=10, pady=5)

        self.label_oct_registration_frame.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.entry_oct_registration_frame.grid(row=4, column=1, padx=10, pady=5)

        self.check_save_intermediate_steps.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.check_display_intermediate_results.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.run_button.grid(row=7, column=0, columnspan=2, pady=10)


    def get_oct_frames_info(self):
        # Retrieve ArCoMo Number
        arcomo_number = self.entry_arcomo_number.get()

        if not arcomo_number.isdigit():
            # Display an error message or handle invalid input
            return

        arcomo_number = int(arcomo_number)

        # Construct the file path for oct_frames_info.txt
        file_path = f"ArCoMo_Data/ArCoMo{arcomo_number}/ArCoMo{arcomo_number}_oct_frames_info.txt"

        # Check if the file exists
        if os.path.exists(file_path):
            # Read information from the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    key, value = map(str.strip, line.split(':'))
                    key = key.lower()

                    if key == 'oct_start':
                        self.entry_oct_start.delete(0, tk.END)
                        self.entry_oct_start.insert(0, value)
                    elif key == 'oct_end':
                        self.entry_oct_end.delete(0, tk.END)
                        self.entry_oct_end.insert(0, value)
                    elif key == 'oct_registration':
                        self.entry_oct_registration_frame.delete(0, tk.END)
                        self.entry_oct_registration_frame.insert(0, value)

        else:
            # Display an error message or handle the case where the file does not exist
            return
        
    def run_analysis(self):
        # Retrieve values from entry widgets and checkboxes
        self.oct_registration_frame = self.entry_oct_registration_frame.get()
        self.oct_start = self.entry_oct_start.get()
        self.oct_end = self.entry_oct_end.get()
        self.arcomo_number = self.entry_arcomo_number.get()
        self.save_intermediate_steps = self.save_intermediate_steps_var.get()
        self.display_intermediate_results = self.display_intermediate_results_var.get()

        # Close the application window
        self.master.destroy()
