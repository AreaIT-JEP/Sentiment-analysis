import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import threading
import json
import os
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import nltk
import concurrent.futures
import pandas as pd
from functools import lru_cache

# --- NLTK and Cache Setup ---
# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Theme definitions (ONLY light theme remaining)
THEMES = {
    "light": {
        "bg": "#192536",
        "fg": "#ffffff",
        "accent": "#FFA801",
        "chart_bg": "#192536",
        "text": "#ffffff"
    }
}

# Create cache directory in user home folder
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sentiment_analyzer_cache")
if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
    except OSError:
        # Fall back to current directory if home directory isn't writable
        CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

# Cache sentiment analysis results for frequently occurring text
@lru_cache(maxsize=10000)
def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


# --- Main Application Class ---
class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.filename = None
        self.results = {}
        self.analyzer = SentimentIntensityAnalyzer()
        # Theme is always light now
        self.current_theme = "light" 
        self.analysis_running = False
        
        # Automatically set optimal workers based on CPU core count
        # Capped at 8 to prevent excessive overhead on very high-core machines, and minimum 2.
        self.max_workers = max(2, min(8, os.cpu_count() or 4)) 
        
        # Store the full data for review details lookup
        self.full_review_data = None 
        # Flag to control the initial informational popup for the Preview tab
        self._initial_preview_info_shown_for_current_file = False 
        
        # Initialize the preview context (right-click) menu
        self.preview_context_menu = tk.Menu(self.root, tearoff=0)

        self.setup_ui()
        self.configure_styles()
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        """Sets up the main user interface."""
        self.root.title("Sentiment Analyzer Pro")
        self.root.geometry("1200x800")
        self.root.minsize(1920, 1080)
        
        self.create_menu()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        file_frame = ttk.LabelFrame(control_frame, text="File Operations")
        file_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X)
        
        self.btn_open = ttk.Button(file_frame, text="Open CSV", command=self.open_file)
        self.btn_open.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_analyze = ttk.Button(file_frame, text="Analyze", command=self.start_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_export = ttk.Button(file_frame, text="Export Results", command=self.save_results, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5, pady=5)

        # Removed Advanced Options frame (threads control)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready - No file selected")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(status_frame, mode='determinate', length=200)
        self.progress.pack(side=tk.LEFT, padx=10)

        self.file_info_label = ttk.Label(status_frame, text="")
        self.file_info_label.pack(side=tk.RIGHT, padx=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="Preview")
        self.setup_preview_area()
        
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Table View")
        self.create_table()
        
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text="Chart View")
        self.setup_chart_area()
        
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        self.setup_summary_area()
        
        self.apply_theme()

    def setup_summary_area(self):
        """Sets up the summary area with overall statistics."""
        summary_container = ttk.Frame(self.summary_frame)
        summary_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.summary_title = ttk.Label(
            summary_container, 
            text="Analysis Summary", 
            style='Title.TLabel'
        )
        self.summary_title.pack(pady=10)
        
        stats_frame = ttk.Frame(summary_container)
        stats_frame.pack(fill=tk.X, pady=10)
        
        text_stats = ttk.LabelFrame(stats_frame, text="Statistics")
        text_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.stats_content = ttk.Frame(text_stats)
        self.stats_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        chart_frame = ttk.LabelFrame(stats_frame, text="Overall Sentiment")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.summary_chart_frame = ttk.Frame(chart_frame)
        self.summary_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        top_frame = ttk.LabelFrame(summary_container, text="Top Products")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        top_pos_frame = ttk.LabelFrame(top_frame, text="Most Positive")
        top_pos_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        top_neg_frame = ttk.LabelFrame(top_frame, text="Most Negative")
        top_neg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.top_pos_tree = ttk.Treeview(
            top_pos_frame,
            columns=("Product", "Positive"),
            show="headings",
            height=5
        )
        self.top_pos_tree.heading("Product", text="Product")
        self.top_pos_tree.heading("Positive", text="Positive (%)")
        self.top_pos_tree.column("Product", width=250)
        self.top_pos_tree.column("Positive", width=100)
        self.top_pos_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Negative
        self.top_neg_tree = ttk.Treeview(
            top_neg_frame,
            columns=("Product", "Negative"),
            show="headings",
            height=5
        )
        self.top_neg_tree.heading("Product", text="Product")
        self.top_neg_tree.heading("Negative", text="Negative (%)")
        self.top_neg_tree.column("Product", width=250)
        self.top_neg_tree.column("Negative", width=100)
        self.top_neg_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_menu(self):
        """Creates the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CSV...", command=self.open_file)
        file_menu.add_command(label="Export Results...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Start Analysis", command=self.start_analysis)
        analysis_menu.add_command(label="Clear Results", command=self.clear_results)
        analysis_menu.add_command(label="Clear Cache", command=self.clear_cache)
        
        # Removed View menu (as there is no theme toggle)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def setup_preview_area(self):
        """Sets up the preview area for CSV data."""
        preview_container = ttk.Frame(self.preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_label = ttk.Label(preview_container, text="No file loaded", style='Title.TLabel')
        self.preview_label.pack(fill=tk.X, padx=5, pady=5)
        
        table_frame = ttk.Frame(preview_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial columns (these will be replaced by actual CSV headers after file load)
        columns = ("Review Title", "Review", "Star", "Product") 
        self.preview_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=10
        )
        
        y_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        x_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Configure default columns (will be updated dynamically on file load)
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=200, minwidth=100) 
        
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Bind the right-click event for the context menu
        self.preview_tree.bind("<Button-3>", self.show_preview_context_menu)
        # Bind the left-click event for the review details popup
        self.preview_tree.bind("<Button-1>", self.on_preview_click)


    def show_preview_context_menu(self, event):
        """Displays a context menu when right-clicking on an item in the preview treeview."""
        item_id = self.preview_tree.identify_row(event.y)
        
        if item_id:
            self.preview_context_menu.delete(0, tk.END)

            values = self.preview_tree.item(item_id, 'values')
            headers = self.preview_tree["columns"]

            self._selected_preview_row_data = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    self._selected_preview_row_data[header] = values[i]

            self.preview_context_menu.add_command(
                label="Copy Review Text",
                command=lambda: self.copy_preview_review_text(self._selected_preview_row_data)
            )
            self.preview_context_menu.add_command(
                label="View Full Review",
                command=lambda: self.view_full_preview_review(self._selected_preview_row_data)
            )
            self.preview_context_menu.add_separator()
            self.preview_context_menu.add_command(
                label="Hide Row (Preview Only)",
                command=lambda: self.hide_preview_row(item_id)
            )

            try:
                self.preview_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.preview_context_menu.grab_release()

    def copy_preview_review_text(self, row_data):
        """Copies the 'Review' text from the selected preview row to clipboard."""
        review_text_col = self._find_column_name(row_data.keys(), ["review", "review text", "text"])
        review_text = row_data.get(review_text_col, "N/A") if review_text_col else "N/A"

        if review_text != "N/A":
            self.root.clipboard_clear()
            self.root.clipboard_append(review_text)
            messagebox.showinfo("Copied", "Review text copied to clipboard!")
        else:
            messagebox.showwarning("No Review", "No review text found for this row.")

    def view_full_preview_review(self, row_data):
        """Displays the full 'Review' text in a new popup window."""
        review_text_col = self._find_column_name(row_data.keys(), ["review", "review text", "text"])
        product_col = self._find_column_name(row_data.keys(), ["product", "product name"])
        review_title_col = self._find_column_name(row_data.keys(), ["review title", "title"])

        review_text = row_data.get(review_text_col, "N/A Review") if review_text_col else "N/A Review"
        product_name = row_data.get(product_col, "N/A Product") if product_col else "N/A Product"
        review_title = row_data.get(review_title_col, "N/A Title") if review_title_col else "N/A Title"

        self.show_review_details_popup(review_title, review_text, product_name)

    def hide_preview_row(self, item_id):
        """Hides (deletes) a selected row from the preview treeview."""
        if messagebox.askyesno("Hide Row", "Are you sure you want to hide this row from the preview?\n(This does NOT affect the original CSV file.)"):
            self.preview_tree.delete(item_id)

    # --- New/Updated Methods for Left-Click Review Details Popup ---
    def on_preview_click(self, event):
        """
        Handles left-click events on the preview treeview cells.
        If a 'Review Title' or 'Review' cell is clicked, it shows a popup with the full review text.
        """
        item_id = self.preview_tree.identify_row(event.y)
        column_id = self.preview_tree.identify_column(event.x)

        if not item_id or not column_id:
            return 

        col_index = int(column_id[1:]) - 1 
        
        tree_columns = self.preview_tree["columns"]
        clicked_column_header = tree_columns[col_index] if 0 <= col_index < len(tree_columns) else None

        if clicked_column_header and (clicked_column_header.lower() == "review title" or clicked_column_header.lower() == "review"):
            row_idx = self.preview_tree.index(item_id)

            if self.full_review_data is not None and row_idx < len(self.full_review_data):
                if isinstance(self.full_review_data, pd.DataFrame):
                    row_data = self.full_review_data.iloc[row_idx].to_dict()
                else: 
                    row_data = self.full_review_data[row_idx]
                
                review_title_col = self._find_column_name(row_data.keys(), ["review title", "title"])
                review_text_col = self._find_column_name(row_data.keys(), ["review", "review text", "text"])
                product_col = self._find_column_name(row_data.keys(), ["product", "product name"])

                review_title = row_data.get(review_title_col, "N/A Title") if review_title_col else "N/A Title"
                review_text = row_data.get(review_text_col, "N/A Review") if review_text_col else "N/A Review"
                product_name = row_data.get(product_col, "N/A Product") if product_col else "N/A Product"

                self.show_review_details_popup(review_title, review_text, product_name)
            else:
                messagebox.showwarning("Data Error", "Could not retrieve full review data. Please reload the file.")

    def _find_column_name(self, iterable_of_names, potential_names):
        """
        Helper to find a column name in an iterable (e.g., list of column names, dict keys)
        from a list of potential names (case-insensitive).
        Returns the actual column name found, or None if not found.
        """
        lower_names = {str(name).lower(): name for name in iterable_of_names}
        for p_name in potential_names:
            if p_name.lower() in lower_names:
                return lower_names[p_name.lower()]
        return None

    def show_review_details_popup(self, title, review_text, product_name):
        """Displays full review details in a Toplevel window."""
        top = tk.Toplevel(self.root)
        top.title(f"Review Details: {title[:50]}...") # Truncate title for window title
        top.transient(self.root) # Make it disappear when parent is closed
        top.grab_set() # Make it modal
        top.geometry("700x500")
        top.lift() # Bring to front
        top.attributes('-topmost', True) # Keep on top

        current_theme_colors = THEMES[self.current_theme]
        top.configure(bg=current_theme_colors['bg'])

        # Corrected: ttk.Frame background is set via ttk.Style.
        # Direct background/bg/foreground/fg are for tk.Frame, tk.Label, tk.Text.
        # Ttk widgets get styling from theme config.
        header_frame = ttk.Frame(top)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        # We don't use header_frame.configure(background=...) anymore. It inherits from TFrame style.
        
        ttk.Label(header_frame, text=f"Product: {product_name}", 
                  font=('Segoe UI', 10, 'bold'),
                  background=current_theme_colors['bg'], foreground=current_theme_colors['fg']
                 ).pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Title: {title}", 
                  font=('Segoe UI', 10, 'italic'),
                  background=current_theme_colors['bg'], foreground=current_theme_colors['fg']
                 ).pack(anchor=tk.W)

        # Corrected: ttk.Frame background is set via ttk.Style.
        text_frame = ttk.Frame(top)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        # We don't use text_frame.configure(background=...) anymore. It inherits from TFrame style.

        review_text_area = tk.Text(text_frame, wrap=tk.WORD, font=('Segoe UI', 10), relief=tk.FLAT,
                                   bg=current_theme_colors['bg'], fg=current_theme_colors['fg'])
        review_text_area.insert(tk.END, review_text)
        review_text_area.config(state=tk.DISABLED) 
        review_text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=review_text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        review_text_area.config(yscrollcommand=scrollbar.set)

        close_button = ttk.Button(top, text="Close", command=top.destroy)
        close_button.pack(pady=10)

        top.update_idletasks()
        top_width = top.winfo_width() if top.winfo_width() > 0 else 700
        top_height = top.winfo_height() if top.winfo_height() > 0 else 500
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (top_width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (top_height // 2)
        top.geometry(f"+{x}+{y}")


    def create_table(self):
        """Creates the results table."""
        table_container = ttk.Frame(self.table_frame, relief='groove', borderwidth=1)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        y_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL)
        x_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
        
        self.tree = ttk.Treeview(
            table_container, 
            columns=("Product", "Positive", "Negative", "Neutral", "Total"),
            show="headings",
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )
        
        y_scrollbar.config(command=self.tree.yview)
        x_scrollbar.config(command=self.tree.xview)
        
        self.tree.heading("Product", text="Product", command=lambda: self.sort_treeview(self.tree, "Product", False))
        self.tree.heading("Positive", text="Positive (%)", command=lambda: self.sort_treeview(self.tree, "Positive", True))
        self.tree.heading("Negative", text="Negative (%)", command=lambda: self.sort_treeview(self.tree, "Negative", True))
        self.tree.heading("Neutral", text="Neutral (%)", command=lambda: self.sort_treeview(self.tree, "Neutral", True))
        self.tree.heading("Total", text="Total Reviews", command=lambda: self.sort_treeview(self.tree, "Total", True))
        
        self.tree.column("Product", width=250, minwidth=150)
        self.tree.column("Positive", width=100, minwidth=80)
        self.tree.column("Negative", width=100, minwidth=80)
        self.tree.column("Neutral", width=100, minwidth=80)
        self.tree.column("Total", width=100, minwidth=80)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)
        
        search_frame = ttk.Frame(self.table_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_table)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)

    def filter_table(self, *args):
        """Filters the table based on search text."""
        search_text = self.search_var.get().lower()
        
        self.tree.delete(*self.tree.get_children())
        
        for product, data in self.results.items():
            if search_text in product.lower():
                self.tree.insert("", "end", values=(
                    product, 
                    f"{data['pos']}%", 
                    f"{data['neg']}%", 
                    f"{data['neu']}%",
                    data['total']
                ))

    def sort_treeview(self, treeview, col, numeric=False):
        """Sorts treeview when column header is clicked."""
        item_list = [(treeview.set(k, col), k) for k in treeview.get_children('')]
        
        if not hasattr(self, 'sort_reverse'):
            self.sort_reverse = {}
        if col not in self.sort_reverse:
            self.sort_reverse[col] = False
            
        if numeric:
            item_list.sort(key=lambda x: float(x[0].rstrip('%')) if x[0].rstrip('%') else 0, 
                           reverse=self.sort_reverse[col])
        else:
            item_list.sort(reverse=self.sort_reverse[col])
            
        self.sort_reverse[col] = not self.sort_reverse[col]
        
        for index, (val, k) in enumerate(item_list):
            treeview.move(k, '', index)
            
        for c in treeview["columns"]:
            if c != col:
                treeview.heading(c, text=treeview.heading(c, "text").rstrip(" ↑↓"))
        
        direction = " ↓" if self.sort_reverse[col] else " ↑"
        treeview.heading(col, text=treeview.heading(col, "text").rstrip(" ↑↓") + direction)

    def setup_chart_area(self):
        """Sets up the scrollable chart area."""
        chart_container = ttk.Frame(self.chart_frame)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(chart_container, highlightthickness=0)
        y_scrollbar = ttk.Scrollbar(chart_container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=y_scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handles mousewheel events for canvas scrolling."""
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/macOS
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def configure_styles(self):
        """Configures ttk styles for the application."""
        style = ttk.Style()
        
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TLabel', font=('Segoe UI', 9))
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Treeview', font=('Segoe UI', 10))
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    def apply_theme(self):
        """Applies the current theme (always light) to all UI elements."""
        theme = THEMES[self.current_theme] 
        
        style = ttk.Style()
        style.theme_use('default') 
        
        style.configure('.', background=theme['bg'], foreground=theme['fg'])
        style.configure('TButton', background=theme['accent'])
        style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        style.configure('TFrame', background=theme['bg'])
        style.configure('TLabelframe', background=theme['bg'])
        style.configure('TLabelframe.Label', background=theme['bg'], foreground=theme['fg'])
        style.configure('TNotebook', background=theme['bg'])
        style.configure('TNotebook.Tab', background=theme['bg'], foreground=theme['fg'])
        style.configure('Treeview', background=theme['bg'], fieldbackground=theme['bg'], foreground=theme['fg'])
        style.configure('Treeview.Heading', background=theme['accent'], foreground=theme['fg'])
        
        self.root.configure(bg=theme['bg'])
        self.canvas.configure(bg=theme['bg'])

    def open_file(self, filepath=None):
        """Opens a CSV file and displays a preview.
           If filepath is provided, opens that file. Otherwise, opens a file dialog.
        """
        if filepath is None:
            filename = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
            )
        else:
            filename = filepath

        if not filename:
            print("No file selected.")
            return

        try:
            self.clear_results() 
            self.preview_tree.delete(*self.preview_tree.get_children())
            
            if not os.path.isfile(filename):
                raise FileNotFoundError("The selected file does not exist.")
            
            try:
                self.full_review_data = pd.read_csv(filename, encoding='utf-8', on_bad_lines='skip')
                
                review_title_col_name = self._find_column_name(
                    self.full_review_data.columns, 
                    ["review title", "title", self.full_review_data.columns[0] if len(self.full_review_data.columns) > 0 else '']
                )
                review_text_col_name = self._find_column_name(
                    self.full_review_data.columns, 
                    ["review", "review text", "text", self.full_review_data.columns[1] if len(self.full_review_data.columns) > 1 else '']
                )
                star_col_name = self._find_column_name(
                    self.full_review_data.columns, 
                    ["star", "rating", self.full_review_data.columns[2] if len(self.full_review_data.columns) > 2 else '']
                )
                product_col_name = self._find_column_name(
                    self.full_review_data.columns, 
                    ["product", "product name", self.full_review_data.columns[3] if len(self.full_review_data.columns) > 3 else '']
                )
                
                display_columns = []
                if review_title_col_name: display_columns.append(review_title_col_name)
                if review_text_col_name: display_columns.append(review_text_col_name)
                if star_col_name: display_columns.append(star_col_name)
                if product_col_name: display_columns.append(product_col_name)

                if not display_columns:
                    if len(self.full_review_data.columns) >= 4:
                        display_columns = list(self.full_review_data.columns[:4])
                    else:
                        display_columns = list(self.full_review_data.columns)
                    messagebox.showwarning(
                        "Column Warning", 
                        "Could not find 'Review Title', 'Review', 'Star', or 'Product' columns by common names. "
                        "Displaying first available columns. Please ensure your CSV has relevant headers for full functionality."
                    )

                self.preview_tree["columns"] = display_columns
                for col in display_columns:
                    self.preview_tree.heading(col, text=col)
                    if col.lower() in [review_text_col_name.lower(), review_title_col_name.lower() if review_title_col_name else '']:
                        self.preview_tree.column(col, width=300, minwidth=150)
                    else:
                        self.preview_tree.column(col, width=100, minwidth=80)
                
                for i, (_, row) in enumerate(self.full_review_data.head(10).iterrows()):
                    values_for_tree = [str(row[col]) if col in row else '' for col in display_columns]
                    self.preview_tree.insert("", "end", values=values_for_tree)
                    
            except Exception as e:
                messagebox.showwarning("CSV Read Warning", f"Pandas failed to read CSV, attempting fallback mode for preview. Error: {e}")
                self.full_review_data = [] 
                
                with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader, None)
                    
                    if not header:
                        raise ValueError("CSV file is empty or has no header.")
                    
                    self.preview_tree["columns"] = header
                    for col in header:
                        self.preview_tree.heading(col, text=col)
                        self.preview_tree.column(col, width=150, minwidth=80)
                    
                    for i, row_list in enumerate(csv_reader):
                        row_dict = {header[j]: row_list[j] if j < len(row_list) else '' for j in range(len(header))}
                        self.full_review_data.append(row_dict)
                        if i < 10:
                            self.preview_tree.insert("", "end", values=row_list)

            file_size = os.path.getsize(filename) / (1024 * 1024)

            file_name = os.path.basename(filename)
            self.status_label.config(text=f"Loaded: {file_name}")
            self.file_info_label.config(text=f"Size: {file_size:.2f} MB")
            self.preview_label.config(text=f"Preview of {file_name}")
            self.btn_analyze.config(state=tk.NORMAL)
            self.filename = filename
            
            self.notebook.select(0)
            self.root.after(100, self.show_initial_preview_info_popup) 
            
            cache_file = self.get_cache_filename(filename)
            if os.path.exists(cache_file):
                cache_time = time.ctime(os.path.getmtime(cache_file))
                use_cache = messagebox.askyesno(
                    "Cache Available",
                    f"Analysis results for this file exist in cache (from {cache_time}).\n\nUse cached results?"
                )
                if use_cache:
                    self.load_cache(cache_file)

        except Exception as e:
            messagebox.showerror("Error", f"Error opening the file: {str(e)}")
            self.status_label.config(text="Error loading file")
            self._initial_preview_info_shown_for_current_file = False
            self.full_review_data = None 

    def show_initial_preview_info_popup(self):
        """Displays an initial information popup for the Preview tab (once per file load)."""
        if not hasattr(self, '_initial_preview_info_shown_for_current_file') or not self._initial_preview_info_shown_for_current_file:
            messagebox.showinfo(
                "Preview Tab Information",
                "This tab shows a preview of your CSV file. "
                "The first few rows are displayed to help you verify the data before analysis.\n\n"
                "**Left-click on a 'Review Title' or 'Review' cell** to view the full text in a popup.\n"
                "**Right-click on any row** for more options like copying the review or hiding the row from the preview."
            )
            self._initial_preview_info_shown_for_current_file = True


    def on_tab_change(self, event):
        """Handles tab change event to show initial info popup if Preview tab is selected."""
        selected_tab_id = self.notebook.select()
        selected_tab_text = self.notebook.tab(selected_tab_id, "text")
        if selected_tab_text == "Preview" and self.filename:
            self.show_initial_preview_info_popup()

    def get_cache_filename(self, csv_filename):
        """Generates a cache filename based on the CSV filename and modification time."""
        base_name = os.path.basename(csv_filename)
        mod_time = os.path.getmtime(csv_filename)
        cache_name = f"{base_name.replace('.', '_')}_{int(mod_time)}.json"
        return os.path.join(CACHE_DIR, cache_name)

    def load_cache(self, cache_file):
        """Loads analysis results from cache."""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            
            self.display_results(self.results)
            self.btn_export.config(state=tk.NORMAL)
            self.status_label.config(text="Loaded from cache")
            self.notebook.select(1)  # Switch to table view
        except Exception as e:
            messagebox.showerror("Cache Error", f"Failed to load cached results: {str(e)}")

    def start_analysis(self):
        """Starts sentiment analysis on the loaded file."""
        if self.analysis_running:
            messagebox.showinfo("Analysis in Progress", "Analysis is already running!")
            return
            
        print("Starting analysis...")
        self.analysis_running = True
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.status_label.config(text=f"Analyzing with {self.max_workers} threads...")
        
        threading.Thread(target=self.perform_analysis, daemon=True).start()

    def perform_analysis(self):
        """Performs sentiment analysis on the product reviews."""
        try:
            if not self.filename or not os.path.exists(self.filename):
                raise FileNotFoundError("The selected file does not exist or was moved.")
            
            # Use already loaded full_review_data if available, otherwise re-read
            if self.full_review_data is None:
                self.status_label.config(text="Re-reading file for analysis...")
                products = self._read_file_for_analysis_raw() 
            else:
                self.status_label.config(text="Preparing data for analysis from loaded data...")
                products = self._prepare_products_from_full_data()
            
            print(f"Products loaded: {len(products)}")
            
            if not products:
                raise ValueError("No valid products found in the file for analysis.")
                
            self.results = self.calculate_sentiments(products)
            
            try:
                cache_file = self.get_cache_filename(self.filename)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not cache results: {e}")
                
            self.root.after(0, self.display_results, self.results)
            self.root.after(0, lambda: self.btn_export.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_label.config(text="Analysis complete!"))
            self.root.after(0, lambda: self.notebook.select(1)) 
            
        except Exception as e:
            print(f"Analysis error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Analysis failed"))
            
        finally:
            self.root.after(0, lambda: self.btn_open.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL if self.filename else tk.DISABLED))
            self.analysis_running = False

    def _prepare_products_from_full_data(self):
        """Prepares the 'products' dictionary from self.full_review_data for analysis."""
        products = {}
        
        if isinstance(self.full_review_data, pd.DataFrame):
            df = self.full_review_data
            review_col = self._find_column_name(df.columns, ["review", "review text", "text", df.columns[1] if len(df.columns) > 1 else ''])
            product_col = self._find_column_name(df.columns, ["product", "product name", df.columns[3] if len(df.columns) > 3 else ''])
            
            if not review_col or not product_col:
                raise ValueError("Could not identify 'review' or 'product' columns in the DataFrame for analysis.")

            for product_name, group in df.groupby(product_col):
                reviews = group[review_col].dropna().tolist()
                reviews = [str(r).strip() for r in reviews if str(r).strip() and len(str(r).strip()) > 5]
                if reviews:
                    products[str(product_name)] = reviews 
        elif isinstance(self.full_review_data, list) and self.full_review_data: 
            header = list(self.full_review_data[0].keys())
            review_col = self._find_column_name(header, ["review", "review text", "text", header[1] if len(header) > 1 else ''])
            product_col = self._find_column_name(header, ["product", "product name", header[3] if len(header) > 3 else ''])

            if not review_col or not product_col:
                raise ValueError("Could not identify 'review' or 'product' columns in the list of dictionaries for analysis.")

            for row_dict in self.full_review_data:
                product_name = str(row_dict.get(product_col, '')).strip()
                review_text = str(row_dict.get(review_col, '')).strip()
                if product_name and review_text and len(review_text) > 5:
                    if product_name not in products:
                        products[product_name] = []
                    products[product_name].append(review_text)
        else:
            raise ValueError("No full review data loaded or data format is unrecognized for analysis.")

        min_reviews = 1
        filtered_products = {k: v for k, v in products.items() if len(v) >= min_reviews}
        return filtered_products

    def _read_file_for_analysis_raw(self):
        """Reads the CSV file directly to prepare for analysis (used if self.full_review_data is empty)."""
        products = {}
        total_rows = 0
        
        file_size = os.path.getsize(self.filename) / (1024 * 1024)
        
        if file_size > 100:
            chunk_size = 100_000
            with open(self.filename, 'r', encoding='utf-8', errors='replace') as f:
                total_rows = sum(1 for _ in f) - 1
            
            chunks_processed = 0
            for chunk in pd.read_csv(
                self.filename, 
                chunksize=chunk_size,
                encoding='utf-8',
                on_bad_lines='skip',
                low_memory=True
            ):
                review_col = self._find_column_name(chunk.columns, ["review", "review text", "text", chunk.columns[1] if len(chunk.columns) > 1 else ''])
                product_col = self._find_column_name(chunk.columns, ["product", "product name", chunk.columns[3] if len(chunk.columns) > 3 else ''])
                
                if not review_col or not product_col:
                    raise ValueError("Could not identify 'review' or 'product' columns in a chunk for analysis.")

                for _, row in chunk.iterrows():
                    try:
                        review_text = str(row[review_col]).strip()
                        product_name = str(row[product_col]).strip()
                        if product_name and review_text and len(review_text) > 5:
                            if product_name not in products:
                                products[product_name] = []
                            products[product_name].append(review_text)
                    except (KeyError, IndexError, ValueError):
                        continue
                
                chunks_processed += 1
                progress = 5 + (chunks_processed * chunk_size / total_rows) * 35
                self.update_progress(min(40, progress))
        else:
            df = pd.read_csv(self.filename, encoding='utf-8', on_bad_lines='skip')
            
            review_col = self._find_column_name(df.columns, ["review", "review text", "text", df.columns[1] if len(df.columns) > 1 else ''])
            product_col = self._find_column_name(df.columns, ["product", "product name", df.columns[3] if len(df.columns) > 3 else ''])
            
            if not review_col or not product_col:
                raise ValueError("Could not identify 'review' or 'product' columns in the file for analysis.")

            for product_name, group in df.groupby(product_col):
                reviews = group[review_col].dropna().tolist()
                reviews = [str(r).strip() for r in reviews if str(r).strip() and len(str(r).strip()) > 5]
                if reviews:
                    products[str(product_name)] = reviews
            
            self.update_progress(40)
        
        min_reviews = 1
        filtered_products = {k: v for k, v in products.items() if len(v) >= min_reviews}
        
        total_reviews = sum(len(v) for v in filtered_products.values())
        print(f"Processed {len(filtered_products)} products with {total_reviews} reviews for analysis.")
        
        return filtered_products

    def calculate_sentiments(self, products):
        """Calculates sentiment scores for each product in parallel."""
        results = {}
        total_products = len(products)
        processed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.root.after(0, lambda: self.status_label.config(text=f"Analyzing sentiment with {self.max_workers} threads..."))
            
            future_to_product = {
                executor.submit(self._analyze_product, product, reviews): product
                for product, reviews in products.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_product):
                product = future_to_product[future]
                try:
                    results[product] = future.result()
                    processed += 1
                    progress = 40 + (processed / total_products * 60)
                    self.update_progress(progress)
                except Exception as e:
                    print(f"Error analyzing {product}: {e}")
                    results[product] = {
                        'pos': 0.0,
                        'neg': 0.0,
                        'neu': 0.0,
                        'total': 0,
                        'error': str(e)
                    }
        
        return results

    def _analyze_product(self, product, reviews):
        """Analyzes sentiment for a single product's reviews."""
        counts = {'pos': 0, 'neg': 0, 'neu': 0, 'total': len(reviews)}
        
        batch_size = 100
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            for review in batch:
                try:
                    score = get_sentiment_score(review)
                    if score['compound'] > 0.05:
                        counts['pos'] += 1
                    elif score['compound'] < -0.05:
                        counts['neg'] += 1
                    else:
                        counts['neu'] += 1
                except Exception:
                    counts['total'] -= 1
        
        total = counts['total']
        if total <= 0:
            return {
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'total': 0,
                'compound': 0.0
            }
            
        compound = (counts['pos'] - counts['neg']) / total
        
        return {
            'pos': round((counts['pos'] / total * 100), 2) if total > 0 else 0.0,
            'neg': round((counts['neg'] / total * 100), 2) if total > 0 else 0.0,
            'neu': round((counts['neu'] / total * 100), 2) if total > 0 else 0.0,
            'total': total,
            'compound': round(compound, 4)
        }

    def update_progress(self, value):
        """Updates the progress bar on the main thread."""
        self.root.after(0, lambda: self.progress.config(value=value))

    def display_results(self, results):
        """Displays the sentiment results on the GUI."""
        self.tree.delete(*self.tree.get_children())
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        for widget in self.stats_content.winfo_children():
            widget.destroy()
        for widget in self.summary_chart_frame.winfo_children():
            widget.destroy()
        self.top_pos_tree.delete(*self.top_pos_tree.get_children())
        self.top_neg_tree.delete(*self.top_neg_tree.get_children())

        sorted_items = sorted(results.items(), key=lambda x: x[0])

        for product, data in sorted_items:
            self.tree.insert("", "end", values=(
                product, 
                f"{data['pos']}%", 
                f"{data['neg']}%", 
                f"{data['neu']}%",
                data['total']
            ))

        self.create_summary_statistics(results)
        self._create_charts_batch(sorted_items)

    def create_summary_statistics(self, results):
        """Creates and displays summary statistics."""
        if not results:
            return
            
        total_reviews = sum(data['total'] for data in results.values())
        weighted_pos = sum(data['pos'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        weighted_neg = sum(data['neg'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        weighted_neu = sum(data['neu'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        
        stats_list = [
            ("Total Products", f"{len(results)}"),
            ("Total Reviews", f"{total_reviews:,}"),
            ("Average Reviews per Product", f"{total_reviews / len(results):.1f}" if len(results) > 0 else "0"),
            ("Overall Positive", f"{weighted_pos:.1f}%"),
            ("Overall Negative", f"{weighted_neg:.1f}%"),
            ("Overall Neutral", f"{weighted_neu:.1f}%"),
        ]
        
        row = 0
        for label, value in stats_list:
            ttk.Label(self.stats_content, text=label + ":", font=('Segoe UI', 10)).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(self.stats_content, text=value, font=('Segoe UI', 10, 'bold')).grid(row=row, column=1, sticky="e", padx=5, pady=2)
            row += 1
            
        theme = THEMES[self.current_theme]
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=theme['chart_bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme['chart_bg'])
        
        categories = ['Positive', 'Negative', 'Neutral']
        values = [weighted_pos, weighted_neg, weighted_neu]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=categories, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': theme['chart_bg'], 'linewidth': 1}
        )
        
        for text in texts:
            text.set_color(theme['text'])
        for autotext in autotexts:
            autotext.set_color(theme['chart_bg'])
            autotext.set_fontweight('bold')
            
        ax.set_title('Overall Sentiment Distribution', color=theme['text'])
        fig.tight_layout()
        
        chart_canvas = FigureCanvasTkAgg(fig, self.summary_chart_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        top_positive = sorted(results.items(), key=lambda x: x[1]['pos'], reverse=True)[:10]
        for product, data in top_positive:
            self.top_pos_tree.insert("", "end", values=(product, f"{data['pos']}%"))
        
        top_negative = sorted(results.items(), key=lambda x: x[1]['neg'], reverse=True)[:10]
        for product, data in top_negative:
            self.top_neg_tree.insert("", "end", values=(product, f"{data['neg']}%"))

    def _create_charts_batch(self, items):
        """Creates charts in batches for better performance."""
        theme = THEMES[self.current_theme]
        
        items = [(product, data) for product, data in items if data['total'] > 0]
        
        items = sorted(items, key=lambda x: x[1]['total'], reverse=True)
        
        items = items[:50]
        
        batch_size = 5
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            
            for product, data in batch:
                card_frame = ttk.Frame(self.scrollable_frame, relief='groove', borderwidth=2)
                card_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)

                ttk.Label(
                    card_frame,
                    text=product,
                    style='Title.TLabel',
                    foreground=theme['accent']
                ).pack(anchor=tk.W, padx=10, pady=5)
                
                ttk.Label(
                    card_frame,
                    text=f"Total Reviews: {data['total']}",
                    foreground=theme['fg']
                ).pack(anchor=tk.W, padx=10)

                try:
                    categories = ['Positive', 'Negative', 'Neutral']
                    values = [data['pos'], data['neg'], data['neu']]
                    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

                    fig = Figure(figsize=(6, 3.5), dpi=100, facecolor=theme['chart_bg'])
                    ax = fig.add_subplot(111)
                    ax.set_facecolor(theme['chart_bg'])

                    bars = ax.bar(categories, values, color=colors)

                    for bar, value in zip(bars, values):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{value:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=9,
                            color=theme['text'],
                            fontweight='bold'
                        )

                    ax.set_title(f'Sentiment Distribution', color=theme['text'], fontsize=11)
                    ax.set_ylim(0, max(values) + 10 if max(values) < 90 else 100)
                    ax.set_ylabel("Percentage (%)", color=theme['text'], fontsize=9)
                    ax.tick_params(axis='x', colors=theme['text'])
                    ax.tick_params(axis='y', colors=theme['text'])
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()

                    chart_canvas = FigureCanvasTkAgg(fig, card_frame)
                    chart_canvas.draw()
                    chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                except Exception as e:
                    ttk.Label(
                        card_frame,
                        text=f"Error creating chart: {str(e)}",
                        foreground='red'
                    ).pack(fill=tk.X, padx=10, pady=10)

    def save_results(self):
        """Saves analysis results to a file."""
        if not self.results:
            messagebox.showinfo("No Results", "No analysis results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("JSON Files", "*.json")]
        )
        
        if not filename:
            return
        
        try:
            data = []
            for product, result in self.results.items():
                data.append({
                    'Product': product,
                    'Positive (%)': result['pos'],
                    'Negative (%)': result['neg'],
                    'Neutral (%)': result['neu'],
                    'Total Reviews': result['total']
                })
            
            df = pd.DataFrame(data)
            
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                df.to_csv(filename, index=False)
            elif ext == '.xlsx':
                df.to_excel(filename, index=False)
            elif ext == '.json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2)
            else:
                df.to_csv(filename, index=False)
                
            messagebox.showinfo("Export Successful", f"Results saved to {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save results: {str(e)}")

    def clear_results(self):
        """Clears all current analysis results."""
        self.results = {}
        self.tree.delete(*self.tree.get_children())
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        for widget in self.stats_content.winfo_children():
            widget.destroy()
        for widget in self.summary_chart_frame.winfo_children():
            widget.destroy()
        self.top_pos_tree.delete(*self.top_pos_tree.get_children())
        self.top_neg_tree.delete(*self.top_neg_tree.get_children())
            
        self.btn_export.config(state=tk.DISABLED)
        self.status_label.config(text="Results cleared")
        if hasattr(self, '_initial_preview_info_shown_for_current_file'):
            self._initial_preview_info_shown_for_current_file = False
        self.full_review_data = None 

    def clear_cache(self):
        """Clears the cached analysis results."""
        try:
            file_count = len([name for name in os.listdir(CACHE_DIR) if os.path.isfile(os.path.join(CACHE_DIR, name))])
            
            if file_count == 0:
                messagebox.showinfo("Cache Empty", "No cached results to clear.")
                return
                
            confirm = messagebox.askyesno("Clear Cache", f"Delete {file_count} cached result files?")
            if not confirm:
                return
                
            deleted = 0
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                        deleted += 1
                    except Exception as e:
                        print(f"Error deleting cache file {file_path}: {e}")
                        pass 
                        
            messagebox.showinfo("Cache Cleared", f"Deleted {deleted} cache files.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")

    def show_about(self):
        """Shows the about dialog."""
        about_text = """Sentiment Analyzer Pro v2.0
        
A professional tool for analyzing sentiment in product reviews.

Features:
• Fast CSV processing with pandas
• Multi-threaded sentiment analysis
• Interactive data visualization
• Light theme
• Result caching for improved performance

© 2025 Sentiment Analysis Inc."""

        messagebox.showinfo("About", about_text)

    def on_close(self):
        """Handles window close event."""
        if self.analysis_running:
            if not messagebox.askyesno("Quit", "Analysis is still running. Are you sure you want to quit?"):
                return
                
        self.root.destroy()