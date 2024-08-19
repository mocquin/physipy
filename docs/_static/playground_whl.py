import os

import requests

# Parameters
OUTPUT_JS = "docs/_static/"
PLAYGROUND_WHEELS = [
    "https://files.pythonhosted.org/packages/7b/93/831b4c5b4355210827b3de34f539297e1833c39a68c26a8b454d8cf9f5ed/numpy-2.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    'https://files.pythonhosted.org/packages/f7/3f/01c8b82017c199075f8f788d0d906b9ffbbc5a47dc9918a945e13d5a2bda/pygments-2.18.0-py3-none-any.whl',
]
PACKAGES = ["physipy"]

DEFAULT_COMMANDS = [""]
CONFIG = """\
var colorNotebook = {{
    "playgroundWheels": {},
    "notebookWheels": [],
    "defaultPlayground": "{}"
}}
"""


if __name__ == "__main__":

    if os.path.exists(OUTPUT_JS + "playground-config.js"):
        os.remove(OUTPUT_JS + "playground-config.js")

    # Scrape whl file
    PACKAGE_WHEELS = []
    for package in PACKAGES:
        response = requests.get(f"https://pypi.org/pypi/{package}/json")
        package_wheel = [
            url["url"] for url in response.json()["urls"] if url["url"].endswith("whl")
        ]
        if len(package_wheel) > 1:
            package_wheel = sorted([url for url in package_wheel if "macos" in url])[
                -1:
            ]
        PACKAGE_WHEELS.extend(package_wheel)
    print("Fetched whls:", PACKAGE_WHEELS)

    # Create the config that specifies which wheels need to be used
    config = (
        CONFIG.format(
            str(PLAYGROUND_WHEELS + PACKAGE_WHEELS), "\n".join(DEFAULT_COMMANDS)
        )
        .replace("\r", "")
        .encode("utf-8")
    )
    with open(OUTPUT_JS + "playground-config.js", "wb") as f:
        f.write(config)
