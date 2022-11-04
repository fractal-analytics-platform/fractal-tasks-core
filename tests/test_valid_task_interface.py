import json
import subprocess
from pathlib import Path
from shlex import split as shlex_split
from subprocess import PIPE

import pytest
from devtools import debug
import sys

import fractal_tasks_core


def validate_command(cmd: str):
    """
    Call command and return stdout, stderr, retcode
    """
    debug(cmd)
    result = subprocess.run(  # nosec
        shlex_split(cmd),
        stdout=PIPE,
        stderr=PIPE,
    )
    # This must always fail, since tmp_file_args includes invalid arguments
    assert result.returncode == 1
    stderr = result.stderr.decode()
    debug(stderr)

    # Tasks may fail if they require optional dependency `cellpose`. In that
    # case we manually skip the assertions
    if "MissingOptionalDependencyError" in stderr:
        from warnings import warn
        warn(
            "Skipping checks because of missing optional dependency.\n"
            f"{stderr}"
        )
        return

    # Valid stderr includes pydantic.error_wrappers.ValidationError (type
    # match between model and function, but tmp_file_args has wrong arguments)
    assert "pydantic.error_wrappers.ValidationError" in stderr
    # Invalid stderr includes ValueError
    assert "ValueError" not in stderr
    # print(result.stderr.decode())


module_dir = Path(fractal_tasks_core.__file__).parent
with (module_dir / "__FRACTAL_MANIFEST__.json").open("r") as fin:
    manifest_dict = json.load(fin)


@pytest.mark.parametrize("task", manifest_dict["task_list"])
def test_task_interface(task, tmp_path):

    tmp_file_args = str(tmp_path / "args.json")
    tmp_file_metadiff = str(tmp_path / "metadiff.json")
    with open(tmp_file_args, "w") as fout:
        args = dict(wrong_arg_1=123, wrong_arg_2=[1, 2, 3])
        json.dump(args, fout, indent=4)

    executable = task["executable"]
    task_path = f"{str(module_dir)}/{executable}.py"
    cmd = (
        f"{sys.executable} {task_path} "
        f"-j {tmp_file_args} "
        f"--metadata-out {tmp_file_metadiff}"
    )
    validate_command(cmd)
