import json

import requests


packages = [
    (
        "fractal-tasks-core",
        "https://fractal-analytics-platform.github.io/fractal-tasks-core",
        "https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/main/fractal_tasks_core/__FRACTAL_MANIFEST__.json",  # noqa
    ),
    (
        "scMultiplex",
        "https://github.com/fmi-basel/gliberal-scMultipleX",
        "https://raw.githubusercontent.com/fmi-basel/gliberal-scMultipleX/main/src/scmultiplex/__FRACTAL_MANIFEST__.json",  # noqa
    ),
    (
        "fractal-faim-hcs",
        "https://github.com/jluethi/fractal-faim-hcs",
        "https://raw.githubusercontent.com/jluethi/fractal-faim-hcs/main/src/fractal_faim_hcs/__FRACTAL_MANIFEST__.json",  # noqa
    ),
    (
        "abbott",
        "https://github.com/MaksHess/abbott",
        "https://raw.githubusercontent.com/MaksHess/abbott/main/src/abbott/__FRACTAL_MANIFEST__.json",  # noqa
    ),
]

with open("_all.md", "w") as md:
    for package in packages:
        package_name, homepage_url, manifest_url = package[:]
        print(package_name)
        r = requests.get(manifest_url)
        if not r.status_code == 200:
            raise ValueError(
                f"Something wrong with the request to {manifest_url}"
            )
        manifest = json.loads(r.content.decode("utf-8"))
        print(list(manifest.keys()))
        task_list = manifest["task_list"]
        md.write(f"Package [{package_name}]({homepage_url}):\n\n")
        for task in task_list:
            name = task["name"]
            docs_link = task.get("docs_link")
            if package_name == "fractal-tasks-core":
                md.write(f"* [{name}]({docs_link})\n")
            else:
                md.write(f"* {name}\n")
        md.write("\n\n")
