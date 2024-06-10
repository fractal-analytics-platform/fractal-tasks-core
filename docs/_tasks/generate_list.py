import json
from pathlib import Path

import requests


pkgs = dict()
pkgs["fractal-tasks-core"] = dict(
    homepage_url="https://fractal-analytics-platform.github.io/fractal-tasks-core",  # noqa
    manifest_url="https://raw.githubusercontent.com/fractal-analytics-platform/fractal-tasks-core/main/fractal_tasks_core/__FRACTAL_MANIFEST__.json",  # noqa
)

pkgs["scMultiplex"] = dict(
    homepage_url="https://github.com/fmi-basel/gliberal-scMultipleX",
    manifest_url="https://raw.githubusercontent.com/fmi-basel/gliberal-scMultipleX/main/src/scmultiplex/__FRACTAL_MANIFEST__.json",  # noqa
)
pkgs["fractal-faim-ipa"] = dict(
    homepage_url="https://github.com/jluethi/fractal-faim-ipa",
    manifest_url="https://raw.githubusercontent.com/jluethi/fractal-faim-ipa/main/src/fractal_faim_ipa/__FRACTAL_MANIFEST__.json",  # noqa
)
pkgs["abbott"] = dict(
    homepage_url="https://github.com/MaksHess/abbott",
    manifest_url="https://raw.githubusercontent.com/MaksHess/abbott/main/src/abbott/__FRACTAL_MANIFEST__.json",  # noqa
)
pkgs["fractal-helper-tasks"] = dict(
    homepage_url="https://github.com/jluethi/fractal-helper-tasks",
    manifest_url="https://raw.githubusercontent.com/jluethi/fractal-helper-tasks/main/src/fractal_helper_tasks/__FRACTAL_MANIFEST__.json",  # noqa
)
pkgs["APx_fractal_task_collection"] = dict(
    homepage_url="https://github.com/Apricot-Therapeutics/APx_fractal_task_collection",  # noqa
    manifest_url="https://raw.githubusercontent.com/Apricot-Therapeutics/APx_fractal_task_collection/main/src/apx_fractal_task_collection/__FRACTAL_MANIFEST__.json",  # noqa
)


script_path = __file__
script_dir = Path(script_path).parent
markdown_file = script_dir / "_all.md"
print(f"Writing output to {markdown_file}")

with markdown_file.open("w") as md:
    for package_name, package in pkgs.items():
        homepage_url = package["homepage_url"]
        manifest_url = package["manifest_url"]
        description = package.get("description", None)
        print(package_name)
        r = requests.get(manifest_url)
        if not r.status_code == 200:
            raise ValueError(
                f"Something wrong with the request to {manifest_url}"
            )
        manifest = json.loads(r.content.decode("utf-8"))
        task_list = manifest["task_list"]
        md.write(f"## `{package_name}`\n")
        md.write(f"Package: `{package_name}`\n\n")
        md.write(f"Home page: {homepage_url}\n\n")
        if description is not None:
            md.write(f"Description: {description}\n\n")
        md.write("Tasks:\n\n")
        for task in task_list:
            name = task["name"]
            docs_link = task.get("docs_link")
            if package_name == "fractal-tasks-core":
                md.write(f"* [{name}]({docs_link})\n")
            else:
                md.write(f"* {name}\n")
        md.write("\n\n")
