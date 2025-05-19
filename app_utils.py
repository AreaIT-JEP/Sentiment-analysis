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

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Theme definitions (only light theme)
THEMES = {
    "light": {
        "bg": "#192536",
        "fg": "#ffffff",
        "accent": "#FFA801",
        "chart_bg": "#192536",
        "text": "#ffffff"
    }
}

# Create cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".sentiment_analyzer_cache")
if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
    except OSError:
        CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

@lru_cache(maxsize=10000)
def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.filename = None
        self.results = {}
        self.analyzer = SentimentIntensityAnalyzer()
        self.current_theme = "light"
        self.analysis_running = False
        self.max_workers = max(2, min(8, os.cpu_count() or 4))

        self.setup_ui()
        self.configure_styles()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
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
        """Set up the summary area with overall statistics."""
        summary_container = ttk.Frame(self.summary_frame)
        summary_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        self.summary_title = ttk.Label(
            summary_container, 
            text="Analysis Summary", 
            style='Title.TLabel'
        )
        self.summary_title.pack(pady=10)
        
        # Summary stats frame
        stats_frame = ttk.Frame(summary_container)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Left column - text stats
        text_stats = ttk.LabelFrame(stats_frame, text="Statistics")
        text_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Stats content
        self.stats_content = ttk.Frame(text_stats)
        self.stats_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right column - summary chart
        chart_frame = ttk.LabelFrame(stats_frame, text="Overall Sentiment")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Frame to hold the chart
        self.summary_chart_frame = ttk.Frame(chart_frame)
        self.summary_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top products section
        top_frame = ttk.LabelFrame(summary_container, text="Top Products")
        top_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Split into positive and negative columns
        top_pos_frame = ttk.LabelFrame(top_frame, text="Most Positive")
        top_pos_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        top_neg_frame = ttk.LabelFrame(top_frame, text="Most Negative")
        top_neg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeviews for top products
        # Positive
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

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def setup_preview_area(self):
        """Set up the preview area for CSV data."""
        # Container for preview table
        preview_container = ttk.Frame(self.preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Preview label
        self.preview_label = ttk.Label(preview_container, text="No file loaded", style='Title.TLabel')
        self.preview_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Frame for preview table with scrollbars
        table_frame = ttk.Frame(preview_container)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create preview table
        columns = ("Review Title", "Review", "Star", "Product")
        self.preview_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=10
        )
        
        # Configure scrollbars
        y_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        x_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Configure default columns
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=200, minwidth=100)
        
        # Place the components using grid for better control
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

    def create_table(self):
        """Create the results table."""
        # Container for results table
        table_container = ttk.Frame(self.table_frame, relief='groove', borderwidth=1)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL)
        x_scrollbar = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
        
        # Create the table
        self.tree = ttk.Treeview(
            table_container, 
            columns=("Product", "Positive", "Negative", "Neutral", "Total"),
            show="headings",
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set
        )
        
        # Configure scrollbars
        y_scrollbar.config(command=self.tree.yview)
        x_scrollbar.config(command=self.tree.xview)
        
        # Configure columns
        self.tree.heading("Product", text="Product", command=lambda: self.sort_treeview(self.tree, "Product", False))
        self.tree.heading("Positive", text="Positive (%)", command=lambda: self.sort_treeview(self.tree, "Positive", True))
        self.tree.heading("Negative", text="Negative (%)", command=lambda: self.sort_treeview(self.tree, "Negative", True))
        self.tree.heading("Neutral", text="Neutral (%)", command=lambda: self.sort_treeview(self.tree, "Neutral", True))
        self.tree.heading("Total", text="Total Reviews", command=lambda: self.sort_treeview(self.tree, "Total", True))
        
        # Set column widths
        self.tree.column("Product", width=250, minwidth=150)
        self.tree.column("Positive", width=100, minwidth=80)
        self.tree.column("Negative", width=100, minwidth=80)
        self.tree.column("Neutral", width=100, minwidth=80)
        self.tree.column("Total", width=100, minwidth=80)
        
        # Place the components using grid
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)
        
        # Add search functionality
        search_frame = ttk.Frame(self.table_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_table)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)

    def filter_table(self, *args):
        """Filter the table based on search text."""
        search_text = self.search_var.get().lower()
        
        # Clear the table
        self.tree.delete(*self.tree.get_children())
        
        # Add matching rows
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
        """Sort treeview when column header is clicked."""
        # Get all items in the treeview
        item_list = [(treeview.set(k, col), k) for k in treeview.get_children('')]
        
        # Remember reverse state
        if not hasattr(self, 'sort_reverse'):
            self.sort_reverse = {}
        if col not in self.sort_reverse:
            self.sort_reverse[col] = False
            
        # Sort based on column type
        if numeric:
            # Extract numeric value from format like "42.5%"
            item_list.sort(key=lambda x: float(x[0].rstrip('%')) if x[0].rstrip('%') else 0, 
                           reverse=self.sort_reverse[col])
        else:
            item_list.sort(reverse=self.sort_reverse[col])
            
        # Toggle sort direction for next click
        self.sort_reverse[col] = not self.sort_reverse[col]
        
        # Rearrange items in sorted positions
        for index, (val, k) in enumerate(item_list):
            treeview.move(k, '', index)
            
        # Change column header to show sort direction
        for c in treeview["columns"]:
            if c != col:
                treeview.heading(c, text=treeview.heading(c, "text").rstrip(" ↑↓"))
        
        direction = " ↓" if self.sort_reverse[col] else " ↑"
        treeview.heading(col, text=treeview.heading(col, "text").rstrip(" ↑↓") + direction)

    def setup_chart_area(self):
        """Set up the scrollable chart area."""
        # Create container frame
        chart_container = ttk.Frame(self.chart_frame)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(chart_container, highlightthickness=0)
        y_scrollbar = ttk.Scrollbar(chart_container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        # Frame to hold all charts
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=y_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add mousewheel scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mousewheel events for canvas scrolling."""
        # Different handling based on platform
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/macOS
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def configure_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        
        # Configure basic styles
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TLabel', font=('Segoe UI', 9))
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Treeview', font=('Segoe UI', 10))
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    def apply_theme(self):
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


    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()
        
        # Recreate charts if results exist
        if self.results:
            self.display_results(self.results)

    def open_file(self):
        """Open a CSV file and display a preview."""
        # Show file dialog
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )

        if not filename:
            print("No file selected.")
            return

        try:
            # Clear previous results
            self.clear_results()
            self.preview_tree.delete(*self.preview_tree.get_children())
            
            # Check file exists and is readable
            if not os.path.isfile(filename):
                raise FileNotFoundError("The selected file does not exist.")
            
            # Use pandas for efficient CSV reading
            try:
                # Read just the header and first 10 rows for preview
                preview_df = pd.read_csv(filename, nrows=10, encoding='utf-8', on_bad_lines='skip')
                
                if len(preview_df.columns) < 4:
                    raise ValueError("The CSV file must have at least 4 columns.")
                    
                # Set up preview tree columns
                self.preview_tree["columns"] = list(preview_df.columns)
                for col in preview_df.columns:
                    self.preview_tree.heading(col, text=col)
                    self.preview_tree.column(col, width=150, minwidth=80)
                
                # Add preview rows
                for _, row in preview_df.iterrows():
                    self.preview_tree.insert("", "end", values=list(row))
                    
            except Exception as e:
                # Fall back to basic CSV reader if pandas fails
                with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader, None)
                    
                    if not header or len(header) < 4:
                        raise ValueError("The CSV file must have at least 4 columns.")
                    
                    # Set up preview tree columns
                    self.preview_tree["columns"] = header
                    for col in header:
                        self.preview_tree.heading(col, text=col)
                        self.preview_tree.column(col, width=150, minwidth=80)
                    
                    # Add up to 10 rows for preview
                    for i, row in enumerate(csv_reader):
                        if i >= 10:
                            break
                        self.preview_tree.insert("", "end", values=row)

            # Get file size
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB

            # Update UI
            file_name = os.path.basename(filename)
            self.status_label.config(text=f"Loaded: {file_name}")
            self.file_info_label.config(text=f"Size: {file_size:.2f} MB")
            self.preview_label.config(text=f"Preview of {file_name}")
            self.btn_analyze.config(state=tk.NORMAL)
            self.filename = filename
            
            # Select preview tab
            self.notebook.select(0)
            
            # Check for cached results
            cache_file = self.get_cache_filename(filename)
            if os.path.exists(cache_file):
                # Get cache file modification time
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

    def get_cache_filename(self, csv_filename):
        """Generate a cache filename based on the CSV filename and modification time."""
        base_name = os.path.basename(csv_filename)
        mod_time = os.path.getmtime(csv_filename)
        # Use simple hash of filename and mod time
        cache_name = f"{base_name.replace('.', '_')}_{int(mod_time)}.json"
        return os.path.join(CACHE_DIR, cache_name)

    def load_cache(self, cache_file):
        """Load analysis results from cache."""
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
        """Start sentiment analysis on the loaded file."""
        if self.analysis_running:
            messagebox.showinfo("Analysis in Progress", "Analysis is already running!")
            return
            
        print("Starting analysis...")
        self.analysis_running = True
        self.btn_analyze.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.status_label.config(text="Analyzing...")
        
        # Update max workers from slider
        self.max_workers = self.thread_var.get()
        
        # Start analysis in a separate thread
        threading.Thread(target=self.perform_analysis, daemon=True).start()

    def perform_analysis(self):
        """Perform sentiment analysis on the product reviews."""
        try:
            if not self.filename or not os.path.exists(self.filename):
                raise FileNotFoundError("The selected file does not exist or was moved.")
                
            print("Processing file...")
            products = self.process_file()
            print(f"Products loaded: {len(products)}")
            
            if not products:
                raise ValueError("No valid products found in the file.")
                
            # Calculate sentiment scores
            self.results = self.calculate_sentiments(products)
            
            # Cache results
            try:
                cache_file = self.get_cache_filename(self.filename)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not cache results: {e}")
                
            # Update UI in the main thread
            self.root.after(0, self.display_results, self.results)
            self.root.after(0, lambda: self.btn_export.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_label.config(text="Analysis complete!"))
            self.root.after(0, lambda: self.notebook.select(1))  # Switch to table view
            
        except Exception as e:
            print(f"Analysis error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Analysis failed"))
            
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.btn_open.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL if self.filename else tk.DISABLED))
            self.analysis_running = False

    def process_file(self):
        """Read the CSV file in chunks and process the data."""
        products = {}
        total_rows = 0
        
        try:
            # Use pandas for efficient reading and processing
            self.update_progress(5)
            self.status_label.config(text="Reading file...")
            
            # For large files, use chunking
            file_size = os.path.getsize(self.filename) / (1024 * 1024)  # Size in MB
            
            if file_size > 100:  # For files larger than 100MB
                # Process in chunks
                chunk_size = 100_000  # Adjust based on available memory
                
                # Count total rows for progress tracking
                with open(self.filename, 'r', encoding='utf-8', errors='replace') as f:
                    total_rows = sum(1 for _ in f) - 1  # Subtract header
                
                # Process chunks
                chunks_processed = 0
                for chunk in pd.read_csv(
                    self.filename, 
                    chunksize=chunk_size,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    low_memory=True
                ):
                    # Check expected columns
                    if 'review' not in chunk.columns and len(chunk.columns) >= 2:
                        # Assume second column is the review text
                        review_col = chunk.columns[1]
                    else:
                        review_col = 'review'
                        
                    if 'product' not in chunk.columns and len(chunk.columns) >= 4:
                        # Assume fourth column is the product name
                        product_col = chunk.columns[3]
                    else:
                        product_col = 'product'
                    
                    # Process this chunk
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
                    
                    # Update progress
                    chunks_processed += 1
                    progress = 5 + (chunks_processed * chunk_size / total_rows) * 35
                    self.update_progress(min(40, progress))
            else:
                # For smaller files, load all at once
                df = pd.read_csv(self.filename, encoding='utf-8', on_bad_lines='skip')
                
                # Check expected columns
                if 'review' not in df.columns and len(df.columns) >= 2:
                    # Assume second column is the review text
                    review_col = df.columns[1]
                else:
                    review_col = 'review'
                    
                if 'product' not in df.columns and len(df.columns) >= 4:
                    # Assume fourth column is the product name
                    product_col = df.columns[3]
                else:
                    product_col = 'product'
                
                # Group by product and aggregate reviews
                for product_name, group in df.groupby(product_col):
                    reviews = group[review_col].dropna().tolist()
                    # Filter out empty reviews and ensure they're strings
                    reviews = [str(r).strip() for r in reviews if str(r).strip() and len(str(r).strip()) > 5]
                    if reviews:
                        products[product_name] = reviews
                
                self.update_progress(40)
            
            # Filter products with too few reviews
            min_reviews = 1
            filtered_products = {k: v for k, v in products.items() if len(v) >= min_reviews}
            
            # Print stats
            total_reviews = sum(len(v) for v in filtered_products.values())
            print(f"Processed {len(filtered_products)} products with {total_reviews} reviews")
            
            return filtered_products
            
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def calculate_sentiments(self, products):
        """Calculate sentiment scores for each product in parallel."""
        results = {}
        total_products = len(products)
        processed = 0
        
        # Set up thread pool executor 
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Update status
            self.root.after(0, lambda: self.status_label.config(text=f"Analyzing sentiment with {self.max_workers} threads..."))
            
            # Submit tasks
            future_to_product = {
                executor.submit(self._analyze_product, product, reviews): product
                for product, reviews in products.items()
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_product):
                product = future_to_product[future]
                try:
                    results[product] = future.result()
                    processed += 1
                    progress = 40 + (processed / total_products * 60)
                    self.update_progress(progress)
                except Exception as e:
                    print(f"Error analyzing {product}: {e}")
                    # Add a placeholder result
                    results[product] = {
                        'pos': 0.0,
                        'neg': 0.0,
                        'neu': 0.0,
                        'total': 0,
                        'error': str(e)
                    }
        
        return results

    def _analyze_product(self, product, reviews):
        """Analyze sentiment for a single product's reviews."""
        counts = {'pos': 0, 'neg': 0, 'neu': 0, 'total': len(reviews)}
        
        # Process reviews in batches for better performance
        batch_size = 100
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            for review in batch:
                # Get sentiment scores
                try:
                    # Use cached sentiment analysis
                    score = get_sentiment_score(review)
                    if score['compound'] > 0.05:
                        counts['pos'] += 1
                    elif score['compound'] < -0.05:
                        counts['neg'] += 1
                    else:
                        counts['neu'] += 1
                except Exception:
                    # Skip this review if there's an error
                    counts['total'] -= 1
        
        # Calculate percentages
        total = counts['total']
        if total <= 0:
            return {
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'total': 0,
                'compound': 0.0
            }
            
        # Calculate compound sentiment score (weighted average)
        compound = (counts['pos'] - counts['neg']) / total
        
        return {
            'pos': round((counts['pos'] / total * 100), 2) if total > 0 else 0.0,
            'neg': round((counts['neg'] / total * 100), 2) if total > 0 else 0.0,
            'neu': round((counts['neu'] / total * 100), 2) if total > 0 else 0.0,
            'total': total,
            'compound': round(compound, 4)
        }

    def update_progress(self, value):
        """Update the progress bar on the main thread."""
        self.root.after(0, lambda: self.progress.config(value=value))

    def display_results(self, results):
        """Display the sentiment results on the GUI."""
        # Clear existing data
        self.tree.delete(*self.tree.get_children())
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Clear summary widgets
        for widget in self.stats_content.winfo_children():
            widget.destroy()
        for widget in self.summary_chart_frame.winfo_children():
            widget.destroy()
        self.top_pos_tree.delete(*self.top_pos_tree.get_children())
        self.top_neg_tree.delete(*self.top_neg_tree.get_children())

        # Sort results by product name
        sorted_items = sorted(results.items(), key=lambda x: x[0])

        # Update table view
        for product, data in sorted_items:
            self.tree.insert("", "end", values=(
                product, 
                f"{data['pos']}%", 
                f"{data['neg']}%", 
                f"{data['neu']}%",
                data['total']
            ))

        # Create summary statistics
        self.create_summary_statistics(results)

        # Update chart view in smaller batches
        self._create_charts_batch(sorted_items)

    def create_summary_statistics(self, results):
        """Create and display summary statistics."""
        if not results:
            return
            
        # Calculate overall sentiment
        total_reviews = sum(data['total'] for data in results.values())
        weighted_pos = sum(data['pos'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        weighted_neg = sum(data['neg'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        weighted_neu = sum(data['neu'] * data['total'] for data in results.values()) / total_reviews if total_reviews > 0 else 0
        
        # Display text statistics
        stats_list = [
            ("Total Products", f"{len(results)}"),
            ("Total Reviews", f"{total_reviews:,}"),
            ("Average Reviews per Product", f"{total_reviews / len(results):.1f}" if len(results) > 0 else "0"),
            ("Overall Positive", f"{weighted_pos:.1f}%"),
            ("Overall Negative", f"{weighted_neg:.1f}%"),
            ("Overall Neutral", f"{weighted_neu:.1f}%"),
        ]
        
        # Create stat labels
        row = 0
        for label, value in stats_list:
            ttk.Label(self.stats_content, text=label + ":", font=('Segoe UI', 10)).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(self.stats_content, text=value, font=('Segoe UI', 10, 'bold')).grid(row=row, column=1, sticky="e", padx=5, pady=2)
            row += 1
            
        # Create summary chart
        theme = THEMES[self.current_theme]
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=theme['chart_bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(theme['chart_bg'])
        
        # Plot data
        categories = ['Positive', 'Negative', 'Neutral']
        values = [weighted_pos, weighted_neg, weighted_neu]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=categories, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': theme['chart_bg'], 'linewidth': 1}
        )
        
        # Set text colors
        for text in texts:
            text.set_color(theme['text'])
        for autotext in autotexts:
            autotext.set_color(theme['chart_bg'])
            autotext.set_fontweight('bold')
            
        ax.set_title('Overall Sentiment Distribution', color=theme['text'])
        fig.tight_layout()
        
        # Add chart to frame
        chart_canvas = FigureCanvasTkAgg(fig, self.summary_chart_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Populate top products lists
        # Sort by positive sentiment
        top_positive = sorted(results.items(), key=lambda x: x[1]['pos'], reverse=True)[:10]
        for product, data in top_positive:
            self.top_pos_tree.insert("", "end", values=(product, f"{data['pos']}%"))
        
        # Sort by negative sentiment
        top_negative = sorted(results.items(), key=lambda x: x[1]['neg'], reverse=True)[:10]
        for product, data in top_negative:
            self.top_neg_tree.insert("", "end", values=(product, f"{data['neg']}%"))

    def _create_charts_batch(self, items):
        theme = THEMES[self.current_theme]
        items = [(product, data) for product, data in items if data['total'] > 0]
        items = sorted(items, key=lambda x: x[1]['total'], reverse=True)[:50]
        batch_size = 5

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            for product, data in batch:
                card_frame = ttk.Frame(self.scrollable_frame, relief='groove', borderwidth=2, padding=24)
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
        """Save analysis results to a file."""
        if not self.results:
            messagebox.showinfo("No Results", "No analysis results to export.")
            return
        
        # Ask user for file location
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("JSON Files", "*.json")]
        )
        
        if not filename:
            return
        
        try:
            # Create a dataframe from results
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
            
            # Export based on file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                df.to_csv(filename, index=False)
            elif ext == '.xlsx':
                df.to_excel(filename, index=False)
            elif ext == '.json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2)
            else:
                # Default to CSV
                df.to_csv(filename, index=False)
                
            messagebox.showinfo("Export Successful", f"Results saved to {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save results: {str(e)}")

    def clear_results(self):
        """Clear all current analysis results."""
        self.results = {}
        self.tree.delete(*self.tree.get_children())
        
        # Clear all charts
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Clear summary
        for widget in self.stats_content.winfo_children():
            widget.destroy()
        for widget in self.summary_chart_frame.winfo_children():
            widget.destroy()
        self.top_pos_tree.delete(*self.top_pos_tree.get_children())
        self.top_neg_tree.delete(*self.top_neg_tree.get_children())
            
        self.btn_export.config(state=tk.DISABLED)
        self.status_label.config(text="Results cleared")

    def clear_cache(self):
        """Clear the cached analysis results."""
        try:
            # Count files
            file_count = len([name for name in os.listdir(CACHE_DIR) if os.path.isfile(os.path.join(CACHE_DIR, name))])
            
            if file_count == 0:
                messagebox.showinfo("Cache Empty", "No cached results to clear.")
                return
                
            # Confirm before deleting
            confirm = messagebox.askyesno("Clear Cache", f"Delete {file_count} cached result files?")
            if not confirm:
                return
                
            # Delete cache files
            deleted = 0
            for filename in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                        deleted += 1
                    except:
                        pass
                        
            messagebox.showinfo("Cache Cleared", f"Deleted {deleted} cache files.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")

    def show_about(self):
        """Show about dialog."""
        about_text = """Sentiment Analyzer Pro v2.0
        
A professional tool for analyzing sentiment in product reviews.

Features:
• Fast CSV processing with pandas
• Multi-threaded sentiment analysis
• Interactive data visualization
• Light and dark themes
• Result caching for improved performance

© 2025 Sentiment Analysis Inc."""

        messagebox.showinfo("About", about_text)

    def on_close(self):
        """Handle window close event."""
        if self.analysis_running:
            if not messagebox.askyesno("Quit", "Analysis is still running. Are you sure you want to quit?"):
                return
                
        self.root.destroy()