import os
import subprocess
import sys


def main():
    # Define the command to run your tests. Here we're using unittest.
    test_command = ["python", "-m", "unittest"]  # , "-vv"]

    print(f"Running tests with command: '{' '.join(test_command)}'")
    result = subprocess.run(test_command, capture_output=True, text=True)

    # Print the output of the test command
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Exit with the return code from the test command
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
