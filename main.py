import tkinter as tk
import traceback
import sys
import os
import platform
import nltk
import pandas as pd
import matplotlib
from app_utils import SentimentAnalysisApp, CACHE_DIR

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = {
        'tkinter': tk,
        'pandas': pd,
        'matplotlib': matplotlib,
        'nltk': nltk
    }
    
    missing = []
    for package, module in required_packages.items():
        if module is None:
            missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them using pip: pip install " + " ".join(missing))
        return False
    return True

def setup_environment():
    """Set up the application environment."""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('vader_lexicon')
    
    # Set platform-specific settings
    if platform.system() == "Windows":
        # For Windows: improve DPI awareness
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    elif platform.system() == "Darwin":  # macOS
        # For macOS: set appearance
        os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Silence deprecation warnings

def main():
    """Main function to start the application."""
    try:
        # Check dependencies
        if not check_dependencies():
            print("Exiting due to missing dependencies.")
            input("Press Enter to exit...")
            sys.exit(1)
        
        # Setup environment
        setup_environment()
        
        # Create main window
        root = tk.Tk()
        root.title("Sentiment Analyzer Pro")
        
        # Set window icon (if available)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            try:
                root.iconbitmap(icon_path)
            except:
                pass  # Silently fail if icon can't be loaded
        
        # Handle command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Sentiment Analyzer Pro')
        parser.add_argument('--file', '-f', help='CSV file to analyze')
        parser.add_argument('--theme', '-t', choices=['light', 'dark'], default='light', 
                            help='Application theme (light or dark)')
        args = parser.parse_args()
        
        # Initialize app
        app = SentimentAnalysisApp(root)
        
        # Apply theme if specified
        if args.theme == 'dark':
            app.current_theme = 'dark'
            app.apply_theme()
        
        # Open file if specified
        if args.file and os.path.exists(args.file):
            root.after(500, lambda: app.open_specific_file(args.file))
        
        # Start main event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print(traceback.format_exc())
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()