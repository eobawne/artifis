from app import App
from cli.parser import parse_arguments
import sys


def main():
    """
    Main entry point for the Artifis application.

    This function initializes the application and runs it in either
    Command Line Interface (CLI) mode or Graphical User Interface (GUI)
    mode based on the presence of command line arguments.
    """
    app = App()
    # check if any command line args were passed
    if len(sys.argv) > 1:
        args = parse_arguments(sys.argv[1:])
        app.run_cli(args)
    else:
        # no args were passed, run in GUI mode
        app.run_gui()


if __name__ == "__main__":
    # ensures main() is called only when the script is executed directly
    main()
