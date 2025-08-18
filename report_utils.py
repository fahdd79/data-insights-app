"""
report_utils.py

Helpers for exporting Matplotlib figures into HTML reports.

"""

from __future__ import annotations

import io
import base64
from typing import Optional

# Optional type hint for better editor support (no runtime dependency required)
try:
    from matplotlib.figure import Figure  # type: ignore
except Exception:  # pragma: no cover
    Figure = object  # fallback type if Matplotlib isn't available at import time


def fig_to_base64(
    fig: "Figure",
    *,
    dpi: Optional[int] = None,
    transparent: bool = False,
    close_fig: bool = False,
) -> str:
    """
    Convert a Matplotlib figure to a base64‐encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure to serialize.
    dpi : int, optional
        Override dots-per-inch when saving. If None, Matplotlib's default is used.
    transparent : bool, default False
        If True, save with a transparent background (useful for dark themes).
    close_fig : bool, default False
        If True, closes the figure after encoding to free memory.

    Returns
    -------
    str
        A UTF-8 base64 string representing the PNG image.

    Notes
    -----
    - We use bbox_inches="tight" to trim extra whitespace around the figure.
    - Keep `close_fig=True` in long sessions or loops to prevent figure buildup.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=dpi,
        transparent=transparent,
    )
    buf.seek(0)
    img_bytes = buf.read()
    if close_fig:
        try:
            import matplotlib.pyplot as plt  # local import to avoid hard dep at module import
            plt.close(fig)
        except Exception:
            # If matplotlib isn't present or close fails, silently continue.
            pass
    return base64.b64encode(img_bytes).decode("utf-8")


def img_b64_html(b64: str, *, alt: str = "Figure", width: int = 700) -> str:
    """
    Build an HTML <img> tag for a base64 PNG string.

    Parameters
    ----------
    b64 : str
        Base64 string (PNG) as returned by fig_to_base64.
    alt : str, default "Figure"
        Alt text for the image tag.
    width : int, default 700
        Pixel width to render the image in the HTML report.

    Returns
    -------
    str
        An HTML snippet you can append to your report_parts.
    """
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" width="{width}"/>'