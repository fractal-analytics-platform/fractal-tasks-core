"""
Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
University of Zurich

Original authors:
Tommaso Comparin <tommaso.comparin@exact-lab.it>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

logger = logging.getLogger(__name__)


def dummy_fail(
    *,
    input_paths: Iterable[Path],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    component: Optional[Any] = None,
) -> Dict:
    logger.info("START of dummy_fail task (from within task)")
    raise Exception("This is the traceback of dummy_fail.")
    logger.info("END of dummy_fail task (from within task)")
    metadata_update = {}
    return metadata_update
