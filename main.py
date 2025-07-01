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
    """Checks if all required dependencies are installed."""
    required_packages = {
        'tkinter': tk,
        'pandas': pd,
        'matplotlib': matplotlib,
        'nltk': nltk
    }

    missing = []
    for package, module in required_packages.items():
        try:
            # Check if the module is correctly loaded/accessible
            # Using hasattr and checking module type is a robust way for already imported modules
            if not hasattr(sys.modules[package], '__file__'):  # Checks if it's a real loaded module
                missing.append(package)
        except KeyError:  # Module not found in sys.modules
            missing.append(package)
        except Exception:  # Catch any other unexpected errors
            missing.append(package)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them using pip: pip install " + " ".join(missing))
        return False
    return True


def setup_environment():
    """Sets up the application environment."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        print("Downloading required NLTK resources...")
        # Add quiet=True to prevent excessive console output during download if app is GUI focused
        nltk.download('vader_lexicon', quiet=True)

    if platform.system() == "Windows":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            # Log the error but don't stop execution
            print(f"Warning: Could not set DPI awareness on Windows: {e}")
            pass
    elif platform.system() == "Darwin":  # macOS
        # For macOS: silence Tkinter deprecation warnings
        os.environ['TK_SILENCE_DEPRECATION'] = '1'


def main():
    """Main function to start the application."""
    try:
        if not check_dependencies():
            print("Exiting due to missing dependencies.")
            # Keep input() for console visibility in case the app is run directly
            input("Press Enter to exit...")
            sys.exit(1)

        setup_environment()

        root = tk.Tk()
        root.title("Sentiment Analyzer Pro")

        # Imposta la finestra massimizzata (ma non in fullscreen) per mostrare i pulsanti di sistema
        root.state('zoomed')  # Su Windows e la maggior parte dei desktop manager
        # root.attributes('-zoomed', True)  # Su alcune versioni di Linux (commenta se non serve)
        # root.attributes('-fullscreen', False)  # Assicurati che fullscreen sia disattivato

        # Set window icon (if available)
        # Ensure icon.ico is in the same directory as main.py or provide a full path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            try:
                root.iconbitmap(icon_path)
            except tk.TclError as e:
                print(f"Warning: Could not load window icon: {e}")
                pass  # Silently fail if icon can't be loaded (e.g., wrong format)

        # Handle command line arguments (only --file remains)
        import argparse
        parser = argparse.ArgumentParser(description='Sentiment Analyzer Pro')
        parser.add_argument('--file', '-f', help='CSV file to analyze')
        args = parser.parse_args()

        app = SentimentAnalysisApp(root)

        # Removed theme application logic, as dark mode is no longer an option.

        # Open file if specified via command line
        if args.file and os.path.exists(args.file):
            # Schedule open_file to run after the Tkinter main loop starts
            # This avoids issues with Tkinter operations before the window is fully initialized
            root.after(500, lambda: app.open_file(args.file))

        # Start main event loop
        root.mainloop()

    except Exception as e:
        # Catch and print any unexpected errors during application startup or runtime
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())  # Print full traceback for debugging
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()