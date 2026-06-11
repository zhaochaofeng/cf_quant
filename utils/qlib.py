
import qlib
from config import PROVIDER_URI
from .qlib_ops import PTTM

def init_qlib(provider_uri=PROVIDER_URI, custom_ops=None):
    """初始化qlib """
    if custom_ops is None:
        custom_ops = [PTTM]
    qlib.init(
        provider_uri=provider_uri,
        custom_ops=custom_ops,
    )


