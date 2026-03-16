from dataclasses import dataclass


@dataclass
class HCSZarrUrl:
    base: str
    plate: str
    row: str
    column: str
    image_path: str

    @property
    def plate_url(self) -> str:
        return f"{self.base}/{self.plate}"

    @property
    def well_url(self) -> str:
        return f"{self.plate_url}/{self.row}/{self.column}"

    @property
    def zarr_url(self) -> str:
        return f"{self.well_url}/{self.image_path}"


def _parse_hcs_zarr_url(zarr_urls: list[str]) -> list[HCSZarrUrl]:
    hcs_urls = []
    for zarr_url in zarr_urls:
        parts = zarr_url.rstrip("/").split("/")
        if len(parts) < 5:
            raise ValueError(
                f"Invalid zarr_url: {zarr_url}. "
                "The zarr_url of an image in a plate should be of the form "
                "`/path/to/plate_name/row/column/image_path`. "
                "The zarr_url given is too short to be valid."
            )
        *base, plate_name, row, column, image_path = parts
        base_dir = "/".join(base)
        hcs_urls.append(
            HCSZarrUrl(
                base=base_dir,
                plate=plate_name,
                row=row,
                column=column,
                image_path=image_path,
            )
        )
    return hcs_urls


def group_by_plate(zarr_urls: list[str]) -> dict[str, list[HCSZarrUrl]]:
    """
    Group a list of zarr_urls by plate.

    Args:
        zarr_urls: list of zarr_urls, each containing the path to an OME-Zarr image.

    """
    hcs_url = _parse_hcs_zarr_url(zarr_urls)
    # Group by plate
    plates: dict[str, list[HCSZarrUrl]] = {}
    for url in hcs_url:
        plate_url = url.plate_url
        if plate_url not in plates:
            plates[plate_url] = []
        plates[plate_url].append(url)
    return plates


def group_by_well(zarr_urls: list[str]) -> dict[str, list[HCSZarrUrl]]:
    """
    Group a list of zarr_urls by well.

    Args:
        zarr_urls: list of zarr_urls, each containing the path to an OME-Zarr image.
    """
    hcs_url = _parse_hcs_zarr_url(zarr_urls)
    # Group by well
    wells: dict[str, list[HCSZarrUrl]] = {}
    for url in hcs_url:
        well_url = url.well_url
        if well_url not in wells:
            wells[well_url] = []
        wells[well_url].append(url)
    return wells


def split_well_path_image_path(zarr_url: str) -> tuple[str, str]:
    """
    Split a zarr_url into the well path and the image path.

    Args:
        zarr_url: zarr_url of the form `/path/to/plate_name/row/column/image_path`.
    Returns:
        well_path: path to the well, of the form `/path/to/plate_name/row/column`.
        image_path: path to the image within the well, of the form `image_path`.
    """
    parts = zarr_url.rstrip("/").split("/")
    if len(parts) < 5:
        raise ValueError(
            f"Invalid zarr_url: {zarr_url}. "
            "The zarr_url of an image in a plate should be of the form "
            "`/path/to/plate_name/row/column/image_path`. "
            "The zarr_url given is too short to be valid."
        )
    *well_parts, image_path = parts
    well_path = "/".join(well_parts)
    return well_path, image_path


def format_template_name(template: str, **kwargs: str) -> str:
    """Format a name from a template string and keyword arguments.

    The template may contain zero or more placeholders in ``{key}`` form.
    If no placeholders are present the template is returned verbatim,
    ignoring the supplied kwargs.

    Args:
        template: A format string such as ``"{image_name}_{method}"``.
        **kwargs: Values to substitute into the template. Allowed placeholder
            names are the keys of kwargs.

    Returns:
        The formatted name.

    Raises:
        ValueError: If the template references a placeholder that is not one
            of the supplied kwargs.
    """
    allowed = ", ".join(f"'{k}'" for k in kwargs)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Template format error: {e} is not a valid placeholder. "
            f"Allowed placeholders are: {allowed}."
        ) from e
