"""Locate mounted Kaggle datasets by content, not by path.

Kaggle's mount layout is not stable. Observed across kernels in this project, in the
same week, with identical `dataset_sources`:

    /kaggle/input/datasets/<owner>/<slug>/...     (nested)
    /kaggle/input/<slug>/...                      (flat)
    /kaggle/input/notebooks/<owner>/<kernel>/...  (kernel output as input)

A notebook that hardcodes either form works until it does not, and the failure is
quiet: the loader finds no files, returns an empty list, and the experiment reports
zero trials rather than crashing. E5 lost a GPU session to exactly that.

So datasets are found by a file only they contain. Slower by one directory walk,
immune to the layout changing underneath.
"""

from __future__ import annotations

from pathlib import Path

INPUT = Path("/kaggle/input")


def find_by_glob(pattern: str, root: Path = INPUT, depth_from_hit: int = 0) -> Path | None:
    """Return the directory containing the first match of `pattern`.

    `depth_from_hit` walks up that many extra parents, for signature files that sit
    inside a subdirectory of the root you actually want.
    """
    if not root.exists():
        return None
    for hit in sorted(root.rglob(pattern)):
        p = hit.parent
        for _ in range(depth_from_hit):
            p = p.parent
        return p
    return None


def require(name: str, path: Path | None) -> Path:
    """Fail loudly and usefully when a dataset is not mounted."""
    if path is not None:
        return path
    listing = [str(p) for p in sorted(INPUT.rglob("*"))
               if p.is_dir() and len(p.relative_to(INPUT).parts) <= 3][:40]
    raise SystemExit(
        f"required dataset not mounted: {name}\n"
        "mounted under /kaggle/input:\n  " + "\n  ".join(listing) +
        "\nAdd it to dataset_sources in this notebook's kernel-metadata.json."
    )


# --------------------------------------------------------------- per-dataset roots
# Each is identified by a file that only that dataset contains.

def sisfall_root() -> Path | None:
    """SisFall: per-trial .txt files named like D01_SA01_R01.txt, under SA*/ dirs."""
    hit = find_by_glob("**/D01_SA01_R01.txt")
    return hit.parent if hit else None  # the dir holding the SA01/ subject folders


def kfall_root() -> Path | None:
    """KFall: sensor_data/ and label_data/ siblings."""
    hit = find_by_glob("**/sensor_data/S*/S*.csv", depth_from_hit=2)
    if hit:
        return hit
    lab = find_by_glob("**/label_data/*_label.xlsx", depth_from_hit=1)
    return lab


def fallalld_root() -> Path | None:
    """FallAllD: the FallAllD.pkl pandas pickle."""
    return find_by_glob("**/FallAllD.pkl")


def umafall_root() -> Path | None:
    """UMAFall: per-trial CSVs named UMAFall_Subject_NN_*.csv."""
    return find_by_glob("**/UMAFall_Subject_*.csv")


def fdlib_root() -> Path | None:
    """The directory to put on sys.path so `import fdlib` works."""
    hit = find_by_glob("**/fdlib/__init__.py")
    return hit.parent if hit else None


def corpus_path(task: str) -> Path | None:
    """The cached window corpus emitted by nb01, wherever it was mounted."""
    for p in sorted(INPUT.rglob(f"windows_{task}.npz")):
        return p
    return None


def report() -> dict[str, str]:
    """What resolved to where -- printed by every notebook for the record."""
    return {
        "sisfall": str(sisfall_root()),
        "kfall": str(kfall_root()),
        "fallalld": str(fallalld_root()),
        "umafall": str(umafall_root()),
        "fdlib": str(fdlib_root()),
    }
