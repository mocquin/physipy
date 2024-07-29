import os
import sys


def main():
    print("Checking for debug statements...")
    has_debug = False
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    for line in f:
                        if (
                            "pdb.set_trace()" in line
                            and file != "check_debug_statement.py"
                        ):  # 'print(' in line or
                            has_debug = True
                            print(
                                f"Debug statement found in {file_path}: {line.strip()}"
                            )
    if has_debug:
        sys.exit(1)
    print("No debug statements found.")


if __name__ == "__main__":
    main()
