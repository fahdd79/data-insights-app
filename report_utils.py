import io
import base64

def fig_to_base64(fig):
    """
    Convert a Matplotlib figure to a base64‐encoded PNG string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode()
