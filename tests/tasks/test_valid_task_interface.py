import json
import subprocess
from pathlib import Path
from shlex import split as shlex_split

import pytest
from devtools import debug

import fractal_tasks_core


def validate_command(cmd: str):
    """
    Call command and return stdout, stderr, retcode
    """
    debug(cmd)
    result = subprocess.run(  # nosec
        shlex_split(cmd),
        capture_output=True,
        encoding="utf-8",
    )
    # This must always fail, since tmp_file_args includes invalid arguments
    assert result.returncode == 1
    stderr = result.stderr
    debug(stderr)
    # stderr must include ValidationError
    assert "ValidationError" in stderr
    # stderr cannot include ValueError
    assert "ValueError" not in stderr


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

    for key in ["executable_non_parallel", "executable_parallel"]:
        value = task.get(key, None)
        if value is None:
            continue
        task_path = (module_dir / value).as_posix()
        cmd = (
            f"python {task_path} "
            f"--args-json {tmp_file_args} "
            f"--out-json {tmp_file_metadiff}"
        )
        validate_command(cmd)
