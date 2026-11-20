import os
import sys

import numpy as np
import scipy.io
import scipy.signal as ssg
from sklearn.model_selection import train_test_split


QTDB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qtdb_raw")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 1301
SAMPLING_RATE = 250


MIT_TO_AAMI = {
    "N": 0, ".": 0,
    "L": 1,
    "R": 2,
    "V": 3,
    "A": 4, "a": 4,
    "S": 5, "J": 5, "e": 5, "j": 5, "n": 5, "F": 5,
    "/": 5, "f": 5, "Q": 5, "~": 5, "|": 5, "!": 5,
    "+": 5, "t": 5, "u": 5, "?": 5, "s": 5, "T": 5,
    "x": 5, "r": 5, "B": 5, "E": 5, "P": 5, "p": 5,
    "D": 5, "Z": 5, "Y": 5, "U": 5,
}


QTDB_RECORDS = [
    "sel100", "sel102", "sel103", "sel104", "sel114", "sel116", "sel117", "sel123",
    "sel1413", "sel1416", "sel1418", "sel1419", "sel1421", "sel1425", "sel1429",
    "sel1432", "sel1433", "sel1438", "sel1441", "sel1452", "sel1454", "sel1460",
    "sel1461", "sel1462", "sel1463", "sel1464", "sel1467", "sel1468", "sel1469",
    "sel1470", "sel1471", "sel1475", "sel1476", "sel1483", "sel1485", "sel1486",
    "sel1487", "sel1488", "sel1494", "sel1495", "sel1496", "sel1497", "sel1499",
    "sel1501", "sel1502", "sel1504", "sel1505", "sel1507", "sel1508", "sel1509",
    "sel1510", "sel1511", "sel1512", "sel1513", "sel1514", "sel1515", "sel1516",
    "sel1517", "sel1518", "sel1519", "sel1520", "sel1521", "sel1522", "sel1523",
    "sel1524", "sel1525", "sel1526", "sel1527", "sel1528", "sel1529", "sel1530",
    "sel1531", "sel1532", "sel1533", "sel1534", "sel1535", "sel1536", "sel1537",
    "sel1538", "sel1539", "sel1540", "sel1541", "sel1542", "sel1543", "sel1544",
    "sel1545", "sel1546", "sel1547", "sel1548", "sel1549", "sel1550", "sel1551",
    "sel1552", "sel1553", "sel1554", "sel1555", "sel1556", "sel1557", "sel1558",
    "sel1559", "sel1564", "sel1565", "sel1566", "sel16265", "sel16272", "sel16273",
    "sel16420", "sel16483", "sel16539", "sel16773", "sel16786", "sel16795",
]

TRAIN_RECORDS = [
    "sel100", "sel103", "sel104", "sel114", "sel116", "sel117", "sel123",
    "sel1413", "sel1416", "sel1418", "sel1419", "sel1421", "sel1425", "sel1429",
    "sel1432", "sel1433", "sel1438", "sel1441", "sel1452", "sel1454",
    "sel1461", "sel1462", "sel1463", "sel1464", "sel1467",
    "sel1470", "sel1471", "sel1475", "sel1476", "sel1483", "sel1485", "sel1486",
    "sel1487", "sel1488", "sel1494", "sel1495", "sel1496", "sel1497", "sel1499",
    "sel1501", "sel1502", "sel1504", "sel1507", "sel1508", "sel1509",
    "sel1510", "sel1511", "sel1512", "sel1514", "sel1515", "sel1516",
    "sel1517", "sel1518", "sel1520", "sel1521", "sel1522", "sel1523",
    "sel1524", "sel1525", "sel1526", "sel1527", "sel1528", "sel1529", "sel1530",
    "sel1531", "sel1532", "sel1533", "sel1535", "sel1536", "sel1537",
    "sel1538", "sel1539", "sel1540", "sel1541", "sel1542", "sel1543", "sel1544",
    "sel1545", "sel1546", "sel1547", "sel1548", "sel1549", "sel1550", "sel1551",
    "sel1552", "sel1553", "sel1554", "sel1555", "sel1556", "sel1557", "sel1558",
    "sel1559", "sel1564", "sel1565", "sel1566", "sel16265", "sel16272", "sel16273",
    "sel16420", "sel16483", "sel16539", "sel16773", "sel16786", "sel16795",
]

TEST_RECORDS = [
    "sel102", "sel1460", "sel1468", "sel1469", "sel1505",
    "sel1513", "sel1519", "sel1534", "sel16483",
]


def download_qtdb(force=False):
    import wfdb
    os.makedirs(QTDB_DIR, exist_ok=True)

    for name in QTDB_RECORDS:
        local_dir = os.path.join(QTDB_DIR, name)
        dat_path = os.path.join(QTDB_DIR, name + ".dat")
        if os.path.exists(dat_path) and not force:
            print(f"  {name}: already exists, skipping")
            continue
        print(f"  Downloading {name}...")
        try:
            wfdb.dl_database("qtdb", os.path.dirname(dat_path), records=[name])
        except Exception as e:
            print(f"  Failed to download {name}: {e}")


def annotate_timesteps(num_samples, ann_sample, ann_symbol):
    labels = np.zeros(num_samples, dtype=np.uint8)
    class_vals = np.full(num_samples, -1, dtype=np.int8)

    prev_idx = 0
    prev_class = 0

    for sample_idx, symbol in zip(ann_sample, ann_symbol):
        cls = MIT_TO_AAMI.get(symbol, 5)
        if sample_idx >= num_samples:
            break
        class_vals[prev_idx:sample_idx] = prev_class
        class_vals[sample_idx] = cls
        prev_idx = sample_idx + 1
        prev_class = cls

    class_vals[prev_idx:] = prev_class

    labels = np.maximum(class_vals, 0).astype(np.uint8)
    return labels


def convert_seq(x, threshold=0.03):
    l = len(x)
    x = ssg.savgol_filter(x, 5, 3)
    X = np.zeros((l, 2))
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] >= threshold:
            X[i, 0] = 1
        elif x[i] - x[i + 1] >= threshold:
            X[i, 1] = 1
    return X


def segment_record(p_signal, timestep_labels, seq_len=1301):
    N = len(p_signal)
    segments_x = []
    segments_y = []

    for start in range(0, N - seq_len + 1, seq_len // 2):
        end = start + seq_len
        x_seg = np.zeros((seq_len, 4), dtype=np.int16)
        for ch in range(2):
            seg = convert_seq(p_signal[start:end, ch], threshold=0.03)
            x_seg[:, ch * 2:ch * 2 + 2] = seg

        y_seg = timestep_labels[start:end]

        seg_labels = np.eye(6, dtype=np.uint8)[y_seg]

        segments_x.append(x_seg)
        segments_y.append(seg_labels)

    if not segments_x:
        return np.empty((0, seq_len, 4), dtype=np.int16), np.empty((0, seq_len, 6), dtype=np.uint8)

    return np.stack(segments_x), np.stack(segments_y)


def process_record(record_name, download_dir, seq_len=1301):
    import wfdb
    path = os.path.join(download_dir, record_name)
    if not os.path.exists(path + ".dat") and not os.path.exists(os.path.join(download_dir, record_name + ".dat")):
        print(f"  {record_name}: data not found, downloading...")
        try:
            wfdb.dl_database("qtdb", download_dir, records=[record_name])
        except Exception as e:
            print(f"  Failed: {e}")
            return None, None

    try:
        sig_path = path if os.path.exists(path + ".dat") else os.path.join(download_dir, record_name)
        record = wfdb.rdrecord(sig_path)
        annotation = wfdb.rdann(sig_path, "atr")
    except Exception as e:
        print(f"  Error reading {record_name}: {e}")
        return None, None

    p_signal = record.p_signal
    ann_sample = annotation.sample
    ann_symbol = annotation.symbol

    timestep_labels = annotate_timesteps(len(p_signal), ann_sample, ann_symbol)

    segments_x, segments_y = segment_record(p_signal, timestep_labels, seq_len)

    print(f"  {record_name}: {len(segments_x)} segments from {len(p_signal)} samples")
    return segments_x, segments_y


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate QTDB dataset .mat files")
    parser.add_argument("--download", action="store_true", help="Download QTDB records from PhysioNet")
    parser.add_argument("--qtdb-dir", type=str, default=QTDB_DIR, help="Directory with raw QTDB recordings")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for .mat files")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length per segment")
    args = parser.parse_args()

    download_dir = args.qtdb_dir
    output_dir = args.output_dir
    seq_len_val = args.seq_len

    if args.download:
        print("Downloading QTDB records from PhysioNet...")
        download_qtdb(force=False)

    all_train_x = []
    all_train_y = []
    all_test_x = []
    all_test_y = []

    all_records = set(TRAIN_RECORDS + TEST_RECORDS)
    UNMATCHED = [r for r in all_records if r not in TRAIN_RECORDS and r not in TEST_RECORDS]
    if UNMATCHED:
        print(f"Warning: records not in train or test: {UNMATCHED}")

    for rec in all_records:
        x_seg, y_seg = process_record(rec, download_dir, seq_len=seq_len_val)
        if x_seg is None:
            continue
        if rec in TRAIN_RECORDS:
            all_train_x.append(x_seg)
            all_train_y.append(y_seg)
        if rec in TEST_RECORDS:
            all_test_x.append(x_seg)
            all_test_y.append(y_seg)

    if not all_train_x and not all_test_x:
        print("No data found. Use --download to fetch QTDB records from PhysioNet.")
        print(f"Place raw recordings in: {download_dir}")
        print("Or verify files exist.")
        sys.exit(1)

    train_x = np.concatenate(all_train_x, axis=0) if all_train_x else np.empty((0, seq_len_val, 4), dtype=np.int16)
    train_y = np.concatenate(all_train_y, axis=0) if all_train_y else np.empty((0, seq_len_val, 6), dtype=np.uint8)
    test_x = np.concatenate(all_test_x, axis=0) if all_test_x else np.empty((0, seq_len_val, 4), dtype=np.int16)
    test_y = np.concatenate(all_test_y, axis=0) if all_test_y else np.empty((0, seq_len_val, 6), dtype=np.uint8)

    max_i_train = np.full((len(train_x), 1), seq_len_val, dtype=np.uint16)
    max_i_test = np.full((len(test_x), 1), seq_len_val, dtype=np.uint16)
    t_train = np.tile(np.arange(seq_len_val, dtype=np.uint16), (len(train_x), 1))
    t_test = np.tile(np.arange(seq_len_val, dtype=np.uint16), (len(test_x), 1))

    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)

    train_path = os.path.join(output_dir, "data", "QTDB_train.mat")
    test_path = os.path.join(output_dir, "data", "QTDB_test.mat")

    scipy.io.savemat(train_path, {
        "x": train_x, "y": train_y,
        "t": t_train, "max_i": max_i_train,
    })
    scipy.io.savemat(test_path, {
        "x": test_x, "y": test_y,
        "t": t_test, "max_i": max_i_test,
    })

    print(f"\nSaved {len(train_x)} training samples to {train_path}")
    print(f"Saved {len(test_x)} test samples to {test_path}")
    print(f"Train class distribution: {np.bincount(train_y.argmax(axis=-1).flatten())}")
    print(f"Test class distribution:  {np.bincount(test_y.argmax(axis=-1).flatten())}")


if __name__ == "__main__":
    main()
