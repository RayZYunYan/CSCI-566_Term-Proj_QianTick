import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "news"

def collect_files(source: str):
    csv_paths, json_paths = [], []
    for teammate_dir in sorted(RAW_NEWS_DIR.iterdir()):
        if (not teammate_dir.is_dir()):
            continue
        for f in teammate_dir.iterdir():
            if (source not in f.name or f.name.endswith("_stats.txt")):
                continue
            if (f.suffix == ".csv"):
                csv_paths.append(f)
            elif (f.suffix == ".json"):
                json_paths.append(f)
    return csv_paths, json_paths

def merge_csvs(paths: list[Path], dedup_cols: list[str] | None = None):
    frames, stats = [], []
    for p in paths:
        df = pd.read_csv(p)
        # Get the data path and the number of columns.
        stats.append((p.relative_to(PROJECT_ROOT).as_posix(), len(df)))
        frames.append(df)
    if (not frames):
        return pd.DataFrame(), stats
    merged = pd.concat(frames, ignore_index=True)
    if (dedup_cols):
        merged.drop_duplicates(subset=dedup_cols, inplace=True)
        merged.reset_index(drop=True, inplace=True)
    return merged, stats

def merge_jsons(paths: list[Path], dedup_key: list[str] | None = None):
    records, stats = [], []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        # Get the data path and the number of columns.
        stats.append((p.relative_to(PROJECT_ROOT).as_posix(), len(data)))
        records.extend(data)
    if (not records):
        return [], stats
    if (dedup_key):
        seen, unique = set(), []
        for rec in records:
            tag = tuple(rec.get(k) for k in dedup_key)
            if (tag not in seen):
                seen.add(tag)
                unique.append(rec)
        records = unique
    return records, stats

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_json(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def merge_source(source: str, csv_dedup_cols: list[str], json_dedup_keys: list[str]):
    report = []
    report.append(f"## Source: {source.upper()}")

    csv_paths, json_paths = collect_files(source)
    if (not csv_paths and not json_paths):
        return report

    # Record CSV files
    merged_csv, csv_stats = merge_csvs(csv_paths, dedup_cols=csv_dedup_cols)
    csv_total_before = sum(n for _, n in csv_stats)
    report.append("\n- [CSV files]")
    for fname, n in csv_stats:
        report.append(f"    {fname:<65s}  rows = {n:>7,}")
    report.append("")
    report.append(f"Total before deduplicate: rows = {csv_total_before}")
    report.append(f"Total after deduplicate: rows = {len(merged_csv)}")
    report.append(f"Duplicates removed {csv_total_before - len(merged_csv)}")

    # Record JSON files
    merged_json, json_stats = merge_jsons(json_paths, dedup_key=json_dedup_keys)
    json_total_before = sum(n for _, n in json_stats)
    report.append("\n- [JSON files]")
    for fname, n in json_stats:
        report.append(f"    {fname:<65s}  rows = {n:>7,}")
    report.append("")
    report.append(f"Total before deduplicate: rows = {json_total_before}")
    report.append(f"Total after deduplicate: rows = {len(merged_json)}")
    report.append(f"Duplicates removed {json_total_before - len(merged_json)}")
    # Save both csv and json files to the data path
    csv_out = OUTPUT_DIR / f"{source}_merged.csv"
    json_out = OUTPUT_DIR / f"{source}_merged.json"
    save_csv(merged_csv, csv_out)
    save_json(merged_json, json_out)

    report.append("")
    report.append(f"### Output:\n")
    report.append(f"{csv_out.relative_to(PROJECT_ROOT).as_posix()}: rows = {len(merged_csv)}")
    report.append(f"{json_out.relative_to(PROJECT_ROOT).as_posix()}: rows = {len(merged_json)}")
    report.append("")
    return report

def main():
    lines = []
    lines.append(f"# QianTick News Data Merge Record\n")
    lines.extend(merge_source(source="finnhub", csv_dedup_cols=["symbol", "datetime", "headline"], json_dedup_keys=["symbol", "datetime", "headline"]))
    lines.extend(merge_source(source="gdelt", csv_dedup_cols=["datetime", "headline", "url"], json_dedup_keys=["datetime", "headline", "url"]))
    record_path = OUTPUT_DIR / "record.txt"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()