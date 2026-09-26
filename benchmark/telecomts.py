from __future__ import annotations

import gzip
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmark.dataset import DatasetBundle, FEATURES, NODES, SLICES, build_5gc_graph


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "AliMaatouk/TelecomTS"
PAPER_CACHE_SHA256 = "f410765a0cdd4e4802967de4304e5d138455524389e83e132e682e71f9c509b6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    kwargs = {"mode": "rt", "encoding": "utf-8"}
    if path.suffix == ".gz":
        with opener(path, **kwargs) as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with opener(path, **kwargs) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_rows(path: Path, rows: List[dict]) -> None:
    """Write only the fields used by the benchmark, deterministically."""
    def _lines():
        for row in rows:
            yield json.dumps(
                {"KPIs": row["KPIs"], "labels": row["labels"]},
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"

    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                for line in _lines():
                    gz.write(line.encode("utf-8"))
    else:
        with path.open("w", encoding="utf-8") as stream:
            for line in _lines():
                stream.write(line)


def fetch_telecomts_rows(
    cache_path: Path,
    samples: int = 800,
    split: str = "train",
    page_size: int = 100,
    sampling: str = "even",
    allow_download: bool = True,
    expected_sha256: str | None = None,
    transport: str = "rows_api",   # <-- thêm
) -> List[dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    if cache_path.exists():
        if expected_sha256 is not None:
            actual = sha256_file(cache_path)
            if actual != expected_sha256:
                raise ValueError(
                    f"TelecomTS cache checksum mismatch: expected "
                    f"{expected_sha256}, got {actual}"
                )
        rows = _read_rows(cache_path)
        if len(rows) >= samples:
            return rows[:samples]
    if not allow_download:
        raise FileNotFoundError(
            f"Need {samples} TelecomTS rows at {cache_path}; "
            "download is disabled."
        )

    if transport == "datasets":            # <-- thêm nhánh mới
        rows = _fetch_via_datasets_lib(samples, split, sampling)
        if len(rows) < samples:
            raise RuntimeError(f"Downloaded only {len(rows)} of {samples} requested rows")
        _write_rows(cache_path, rows[:samples])
        if expected_sha256 is not None:
            actual = sha256_file(cache_path)
            if actual != expected_sha256:
                raise ValueError(f"Checksum mismatch: expected {expected_sha256}, got {actual}")
        return rows[:samples]

    def read_page(offset: int, length: int) -> tuple[List[dict], int]:
        params = urllib.parse.urlencode(
            {
                "dataset": DATASET_ID,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        with urllib.request.urlopen(f"{HF_ROWS_URL}?{params}", timeout=90) as response:
            payload = json.load(response)
        return [item["row"] for item in payload.get("rows", [])], int(payload.get("num_rows_total", 0))

    if sampling == "even":
        first_batch, total = read_page(0, 1)
        if not total:
            return first_batch[:samples]
        n_pages = int(np.ceil(samples / page_size))
        max_offset = max(total - page_size, 0)
        offsets = np.linspace(0, max_offset, n_pages, dtype=int)
        rows = []
        for offset in offsets:
            batch, _ = read_page(int(offset), min(page_size, samples - len(rows)))
            if not batch:
                continue
            rows.extend(batch)
            if len(rows) >= samples:
                break
    else:
        offset = len(rows)
        while len(rows) < samples:
            length = min(page_size, samples - len(rows))
            batch, _ = read_page(offset, length)
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)

    if len(rows) < samples:
        raise RuntimeError(f"Downloaded only {len(rows)} of {samples} requested rows")
    _write_rows(cache_path, rows[:samples])
    if expected_sha256 is not None:
        actual = sha256_file(cache_path)
        if actual != expected_sha256:
            raise ValueError(
                "The upstream dataset no longer matches the paper snapshot. "
                f"Expected {expected_sha256}, got {actual}. Use the tracked "
                "data/telecomts_800_even.jsonl.gz artifact."
            )
    return rows[:samples]


def _arr(kpis: Dict[str, list], name: str) -> np.ndarray:
    return np.asarray(kpis[name], dtype=float)


def _norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.5)


def _safe_rate(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _make_features_from_kpis(kpis: Dict[str, list], input_len: int) -> np.ndarray:
    t = input_len
    n_nodes, n_features = len(NODES), len(FEATURES)
    x = np.zeros((t, n_nodes, n_features), dtype=float)
    idx = {name: i for i, name in enumerate(NODES)}
    f = {name: i for i, name in enumerate(FEATURES)}

    rsrp = _arr(kpis, "RSRP")[:t]
    dl_bler = _arr(kpis, "DL_BLER")[:t]
    ul_bler = _arr(kpis, "UL_BLER")[:t]
    dl_mcs = _arr(kpis, "DL_MCS")[:t]
    ul_mcs = _arr(kpis, "UL_MCS")[:t]
    ul_snr = _arr(kpis, "UL_SNR")[:t]
    tx = _safe_rate(_arr(kpis, "TX_Bytes")[:t])
    rx = _safe_rate(_arr(kpis, "RX_Bytes")[:t])
    buffer = _safe_rate(_arr(kpis, "Estimated_UL_Buffer")[:t])
    prb_dl = _arr(kpis, "PRB_Utilization_DL")[:t]
    prb_ul = _arr(kpis, "PRB_Utilization_UL")[:t]
    pkt_ul = _safe_rate(_arr(kpis, "UL_NumberOfPackets")[:t])
    pkt_dl = _safe_rate(_arr(kpis, "DL_NumberOfPackets")[:t])

    radio_badness = _norm(-rsrp, 65.0, 115.0) + _norm(10.0 - ul_snr, 0.0, 18.0)
    dl_load = _norm(rx, np.percentile(rx, 5), np.percentile(rx, 95) + 1.0)
    ul_load = _norm(tx + buffer, np.percentile(tx + buffer, 5), np.percentile(tx + buffer, 95) + 1.0)
    dl_util = np.clip(prb_dl / 100.0, 0.0, 1.5)
    ul_util = np.clip(prb_ul / 100.0, 0.0, 1.5)

    nf_map = {
        "AMF": 0.35 * (pkt_ul + pkt_dl) + 0.20 * (dl_load + ul_load),
        "SMF": 0.45 * (pkt_ul + pkt_dl) + 0.35 * (dl_util + ul_util),
        "UPF1": 0.75 * dl_load + 0.20 * dl_util,
        "UPF2": 0.75 * ul_load + 0.20 * ul_util,
        "PCF": 0.25 * (pkt_ul + pkt_dl) + 0.30 * (dl_bler + ul_bler),
        "NSSF": 0.18 * (dl_util + ul_util),
        "NRF": 0.12 * (pkt_ul + pkt_dl),
        "UDM": 0.20 * pkt_ul + 0.15 * radio_badness,
    }
    nf_service = {
        "AMF": 4.0,
        "SMF": 4.2,
        "UPF1": 2.2,
        "UPF2": 2.2,
        "PCF": 3.0,
        "NSSF": 2.6,
        "NRF": 3.0,
        "UDM": 3.0,
    }
    for nf, arrival in nf_map.items():
        node = idx[nf]
        service = nf_service[nf] + 0.25 * np.sin(np.linspace(0, 3.14, t) + node)
        util = np.clip(arrival / np.maximum(service, 1e-3), 0.0, 0.995)
        latency = 3.0 + 12.0 * util**2 + 18.0 * (dl_bler + ul_bler) + 3.0 * radio_badness
        x[:, node, f["cpu"]] = np.clip(0.12 + 0.72 * util, 0.0, 1.0)
        x[:, node, f["memory"]] = np.clip(0.18 + 0.45 * util + 0.08 * radio_badness, 0.0, 1.0)
        x[:, node, f["request_rate"]] = arrival
        x[:, node, f["sessions"]] = pkt_ul + pkt_dl
        x[:, node, f["throughput"]] = tx + rx
        x[:, node, f["latency"]] = latency
        x[:, node, f["loss"]] = np.clip(dl_bler + ul_bler, 0.0, 1.0)
        x[:, node, f["utilization"]] = util
        x[:, node, f["arrival_rate"]] = arrival
        x[:, node, f["service_rate"]] = service

    slice_latency = np.stack(
        [
            8.0 + 35.0 * dl_bler + 8.0 * dl_util + 7.0 * radio_badness + 5.0 * _norm(28.0 - dl_mcs, 0.0, 28.0),
            4.0 + 45.0 * ul_bler + 11.0 * ul_util + 13.0 * _norm(14.0 - ul_snr, 0.0, 22.0) + 4.0 * _norm(buffer, 0.0, np.percentile(buffer, 95) + 1.0),
            12.0 + 18.0 * (dl_bler + ul_bler) + 9.0 * ul_util + 5.0 * _norm(pkt_ul + pkt_dl, 0.0, np.percentile(pkt_ul + pkt_dl, 95) + 1.0),
        ],
        axis=1,
    )
    sla = np.array([45.0, 12.0, 80.0])
    for s, name in enumerate(SLICES):
        node = idx[name]
        x[:, node, f["cpu"]] = np.mean(x[:, [idx["AMF"], idx["SMF"]], f["cpu"]], axis=1)
        x[:, node, f["memory"]] = np.mean(x[:, [idx["AMF"], idx["SMF"]], f["memory"]], axis=1)
        x[:, node, f["request_rate"]] = [dl_load, ul_load, pkt_ul + pkt_dl][s]
        x[:, node, f["sessions"]] = pkt_ul + pkt_dl
        x[:, node, f["throughput"]] = [rx, tx, tx + rx][s]
        x[:, node, f["latency"]] = slice_latency[:, s]
        x[:, node, f["loss"]] = [dl_bler, ul_bler, 0.5 * (dl_bler + ul_bler)][s]
        x[:, node, f["utilization"]] = slice_latency[:, s] / sla[s]
        x[:, node, f["arrival_rate"]] = [dl_load, ul_load, pkt_ul + pkt_dl][s]
        x[:, node, f["service_rate"]] = sla[s]
        x[:, node, f["sla_latency"]] = sla[s]

    for s, name in enumerate(["UE_eMBB", "UE_URLLC", "UE_mMTC"]):
        x[:, idx[name], :] = x[:, idx[SLICES[s]], :]
    return x


def _target_from_kpis(kpis: Dict[str, list], labels: Dict[str, str], input_len: int) -> tuple[np.ndarray, np.ndarray]:
    tail = slice(input_len, None)
    dl_bler = _arr(kpis, "DL_BLER")[tail]
    ul_bler = _arr(kpis, "UL_BLER")[tail]
    dl_mcs = _arr(kpis, "DL_MCS")[tail]
    ul_snr = _arr(kpis, "UL_SNR")[tail]
    buffer = _safe_rate(_arr(kpis, "Estimated_UL_Buffer")[tail])
    prb_dl = _arr(kpis, "PRB_Utilization_DL")[tail] / 100.0
    prb_ul = _arr(kpis, "PRB_Utilization_UL")[tail] / 100.0
    latency = np.array(
        [
            np.mean(8.0 + 35.0 * dl_bler + 8.0 * prb_dl + 5.0 * _norm(28.0 - dl_mcs, 0.0, 28.0)),
            np.mean(4.0 + 45.0 * ul_bler + 11.0 * prb_ul + 13.0 * _norm(14.0 - ul_snr, 0.0, 22.0) + 4.0 * _norm(buffer, 0.0, np.percentile(buffer, 95) + 1.0)),
            np.mean(12.0 + 18.0 * (dl_bler + ul_bler) + 9.0 * prb_ul),
        ]
    )
    anomaly = labels.get("anomaly_present", "No") == "Yes"
    congestion = labels.get("congestion", "No") == "Yes"
    violation = np.array([congestion or anomaly, anomaly, congestion or anomaly], dtype=float)
    return latency, violation


def load_telecomts_bundle(
    cache_path: Path,
    samples: int = 800,
    input_len: int = 96,
    sampling: str = "even",
    allow_download: bool = True,
    expected_sha256: str | None = None,
    transport: str = "rows_api",   # <-- thêm
) -> DatasetBundle:
    rows = fetch_telecomts_rows(
        cache_path=cache_path,
        samples=samples,
        sampling=sampling,
        allow_download=allow_download,
        expected_sha256=expected_sha256,
        transport=transport,
    )
    xs, y_lat, y_vio = [], [], []
    for row in rows:
        kpis = row["KPIs"]
        labels = row["labels"]
        if min(len(values) for values in kpis.values()) <= input_len:
            continue
        xs.append(_make_features_from_kpis(kpis, input_len=input_len))
        lat, vio = _target_from_kpis(kpis, labels, input_len=input_len)
        y_lat.append(lat)
        y_vio.append(vio)
    metadata = {
        "dataset": DATASET_ID,
        "license": "MIT",
        "source": "https://huggingface.co/datasets/AliMaatouk/TelecomTS",
        "nodes": NODES,
        "features": FEATURES,
        "slices": SLICES,
        "input_len": input_len,
        "target": "last 32 KPI steps mapped to 3 slice QoE-risk latency proxies; violation labels from anomaly/congestion flags",
        "sampling": sampling,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
    }
    return DatasetBundle(
        x=np.asarray(xs, dtype=float),
        y_latency=np.asarray(y_lat, dtype=float),
        y_violation=np.asarray(y_vio, dtype=float),
        graph=build_5gc_graph(),
        metadata=metadata,
    )

def _fetch_via_datasets_lib(
    samples: int,
    split: str,
    sampling: str,
) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài package 'datasets': pip install datasets huggingface_hub"
        ) from exc

    ds = load_dataset(DATASET_ID, split=split)
    total = len(ds)
    if sampling == "even":
        n = min(samples, total)
        idx = np.linspace(0, total - 1, n, dtype=int).tolist()
    else:
        idx = list(range(min(samples, total)))
    subset = ds.select(idx)
    return [{"KPIs": r["KPIs"], "labels": r["labels"]} for r in subset]