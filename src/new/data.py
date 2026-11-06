"""PhysioNet respiration recordings: download, load, and preprocess (Appendix I.3)."""

import hashlib
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import signal as spsignal
import wfdb


def record_files(records: list[str], data_dir: Path) -> list[Path]:
    """Return the header and signal file paths of the given records."""
    return [data_dir / f"{record}{suffix}" for record in records for suffix in (".hea", ".dat")]


def download_records(database: str, records: list[str], data_dir: Path) -> None:
    """Download any records not already in ``data_dir``, without using the network otherwise.

    Files stay in a subfolder until the download succeeds, so a failed download leaves
    no partial file in ``data_dir``. A rerun resumes it.
    """
    missing = [r for r in records if not all(path.exists() for path in record_files([r], data_dir))]
    if missing:
        incoming = data_dir / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database(database, dl_dir=str(incoming), records=missing, annotators=None)
        for path in incoming.iterdir():
            path.replace(data_dir / path.name)


def download_archive(url: str, sha256: str, data_dir: Path) -> Path:
    """Download and unpack a dataset archive once, checking it against ``sha256``.

    Returns the archive path. The archive is unpacked next to it.
    """
    archive = data_dir / Path(url).name
    if not archive.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{archive} does not match the expected SHA-256. Delete it and rerun.")
    with zipfile.ZipFile(archive) as contents:
        if not (data_dir / contents.namelist()[0]).exists():
            contents.extractall(data_dir)
    return archive


def load_cpap_flow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return times and total flow, the sum of the two Venturi flows, of a CPAP recording."""
    table = pd.read_csv(path)
    return table.iloc[:, 0].to_numpy(), (table.iloc[:, 3] + table.iloc[:, 4]).to_numpy()


def _load_respiration(record: str, data_dir: Path, seconds: float) -> tuple[np.ndarray, float]:
    """Return the first ``seconds`` of the respiration channel and its sampling rate."""
    path = str(data_dir / record)
    fs = float(wfdb.rdheader(path).fs)
    samples, fields = wfdb.rdsamp(path, sampto=int(np.ceil(seconds * fs)))
    channel = next(i for i, name in enumerate(fields["sig_name"]) if "RESP" in name.upper())
    return samples[:, channel].astype(float), fs


def preprocess(
    times: np.ndarray,
    values: np.ndarray,
    fs: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend, optionally bandpass filter, standardize, and downsample a recording.

    Everything is applied once to the whole interval given, before any window is
    selected. Returns times and values on a uniform grid at ``downsample_hz``.
    """
    values = np.array(values, dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.median(values[~missing])
    values = spsignal.detrend(values, type="linear")
    if "bandpass_hz" in config:
        sos = spsignal.butter(4, config["bandpass_hz"], btype="bandpass", fs=fs, output="sos")
        values = spsignal.sosfiltfilt(sos, values)
    values = (values - values.mean()) / values.std()
    step = 1.0 / config["downsample_hz"]
    grid = np.arange(times[0], times[-1] + 0.5 * step, step)
    grid = grid[grid <= times[-1] + 1e-12]
    return grid, np.interp(grid, times, values)


def load_series(record: str, data_dir: Path, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the preprocessed first ``use_seconds`` of a WFDB respiration record."""
    values, fs = _load_respiration(record, data_dir, config["use_seconds"])
    return preprocess(np.arange(values.size) / fs, values, fs, config)


def select_window(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one window, with its times shifted to start at zero."""
    inside = (times >= start - 1e-12) & (times <= start + length + 1e-12)
    return times[inside] - start, values[inside]
