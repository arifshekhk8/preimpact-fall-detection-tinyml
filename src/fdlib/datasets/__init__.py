"""Raw dataset readers. Each returns a list of `Trial`, already decimated to 50 Hz
and converted to g and deg/s, with a namespaced subject ID attached.

Subject IDs are namespaced (`kfall:SA06`, not `SA06`) because SisFall and KFall both
use SA06 for different people. Un-namespaced IDs would silently merge two subjects
into one group and defeat the subject-grouped split.
"""

from . import fallalld, kfall, sisfall, umafall

LOADERS = {
    "sisfall": sisfall.load,
    "kfall": kfall.load,
    "fallalld": fallalld.load,
    "umafall": umafall.load,
}

__all__ = ["sisfall", "kfall", "fallalld", "umafall", "LOADERS"]
