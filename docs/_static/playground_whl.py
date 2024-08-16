import os

import requests

# Parameters
OUTPUT_JS = "docs/_static/"
PLAYGROUND_WHEELS = [
    "https://files.pythonhosted.org/packages/3a/d0/edc009c27b406c4f9cbc79274d6e46d634d139075492ad055e3d68445925/numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
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
