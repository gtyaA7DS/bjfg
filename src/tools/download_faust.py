#!/usr/bin/env python3
"""Robust downloader for the FAUST Google Drive folder.

The script first fetches a file manifest from the Drive folder, stores it on
disk, then downloads each file individually with resume + retry support.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import bs4
import gdown
import requests


DEFAULT_URL = "https://drive.google.com/drive/folders/1T5reNd6GqRfQRyhw8lmhQwCVWLcCOZVN?usp=sharing"
DEFAULT_OUTPUT = "/root/data/FAUST"
DEFAULT_MANIFEST_NAME = ".faust_drive_manifest.json"
DEFAULT_STATE_NAME = ".faust_download_state.json"

FILE_RE = re.compile(r"/file/d/([A-Za-z0-9_-]+)")
FOLDER_RE = re.compile(r"/drive/folders/([A-Za-z0-9_-]+)")


def parse_args():
    p = argparse.ArgumentParser("Download FAUST from Google Drive with retries")
    p.add_argument("--url", type=str, default=DEFAULT_URL, help="Google Drive folder URL")
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT, help="Target directory")
    p.add_argument("--manifest_path", type=str, default="", help="Optional manifest JSON path")
    p.add_argument("--state_path", type=str, default="", help="Optional state JSON path")
    p.add_argument("--manifest_retries", type=int, default=8, help="Retries for fetching the Drive folder manifest")
    p.add_argument("--file_retries", type=int, default=8, help="Retries per file download")
    p.add_argument("--retry_delay", type=float, default=8.0, help="Initial retry delay in seconds")
    p.add_argument("--retry_backoff", type=float, default=1.6, help="Backoff multiplier between retries")
    p.add_argument("--sleep_between_files", type=float, default=0.0, help="Optional pause between files")
    p.add_argument("--allow_partial_manifest", action="store_true", default=False, help="Permit falling back to a cached manifest even if it appears incomplete")
    p.add_argument("--resume", action="store_true", default=True, help="Resume partial downloads")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--use_cookies", action="store_true", default=True, help="Use cookies for gdown")
    p.add_argument("--no_cookies", dest="use_cookies", action="store_false")
    p.add_argument("--skip_manifest_refresh", action="store_true", default=False, help="Use the saved manifest if available")
    p.add_argument("--list_only", action="store_true", default=False, help="Only fetch and print the manifest")
    p.add_argument("--max_files", type=int, default=0, help="Only process the first N missing files, useful for testing")
    p.add_argument("--force_redownload", action="store_true", default=False, help="Re-download files even if they already exist")
    return p.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    print(msg, flush=True)


def sleep_with_backoff(attempt, base_delay, backoff):
    delay = float(base_delay) * (float(backoff) ** max(0, attempt - 1))
    time.sleep(delay)


def normalize_manifest_items(items):
    out = []
    for item in items or []:
        if isinstance(item, dict):
            out.append(
                {
                    "id": item.get("id", ""),
                    "path": item.get("path", ""),
                    "local_path": item.get("local_path", ""),
                }
            )
            continue
        out.append(
            {
                "id": getattr(item, "id", ""),
                "path": getattr(item, "path", ""),
                "local_path": getattr(item, "local_path", ""),
            }
        )
    out.sort(key=lambda x: x["path"])
    return out


def extract_folder_id(url_or_id):
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", str(url_or_id).strip()):
        return str(url_or_id).strip()
    parsed = urlparse(str(url_or_id))
    match = FOLDER_RE.search(parsed.path)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", parsed.query)
    if match:
        return match.group(1)
    raise ValueError(f"Unable to extract Google Drive folder id from: {url_or_id}")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
        f.write("\n")
    tmp.replace(path)


def _fetch_embedded_html(session, folder_id, retries, retry_delay, retry_backoff):
    last_error = None
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    for attempt in range(1, int(retries) + 1):
        try:
            log(f"[manifest] folder {folder_id} | attempt {attempt}/{retries}")
            res = session.get(url, timeout=60)
            res.raise_for_status()
            if "<title>" not in res.text or "drive.google.com/file/d/" not in res.text and "drive.google.com/drive/folders/" not in res.text:
                raise RuntimeError("embedded folder page did not contain expected file or folder links")
            return res.text
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            log(f"[manifest] failed: {type(exc).__name__}: {exc}")
            if attempt < int(retries):
                sleep_with_backoff(attempt, retry_delay, retry_backoff)
    raise RuntimeError(f"failed to fetch embedded folder page for {folder_id}: {last_error}")


def _join_rel(parent_rel, name):
    if not parent_rel:
        return name
    return f"{parent_rel}/{name}"


def _walk_embedded_folder(session, folder_id, parent_rel, seen_folders, retries, retry_delay, retry_backoff):
    if folder_id in seen_folders:
        return []
    seen_folders.add(folder_id)
    html = _fetch_embedded_html(session, folder_id, retries, retry_delay, retry_backoff)
    soup = bs4.BeautifulSoup(html, features="html.parser")
    items = []
    for anchor in soup.select("a"):
        href = anchor.get("href") or ""
        name = anchor.get_text(" ", strip=True)
        if not href or not name:
            continue

        folder_match = FOLDER_RE.search(href)
        if folder_match:
            child_folder_id = folder_match.group(1)
            items.extend(
                _walk_embedded_folder(
                    session=session,
                    folder_id=child_folder_id,
                    parent_rel=_join_rel(parent_rel, name),
                    seen_folders=seen_folders,
                    retries=retries,
                    retry_delay=retry_delay,
                    retry_backoff=retry_backoff,
                )
            )
            continue

        file_match = FILE_RE.search(href)
        if file_match:
            rel_path = _join_rel(parent_rel, name)
            items.append({"id": file_match.group(1), "path": rel_path})
    return items


def fetch_manifest(url, output_root, retries, retry_delay, retry_backoff, use_cookies):
    folder_id = extract_folder_id(url)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )
    items = _walk_embedded_folder(
        session=session,
        folder_id=folder_id,
        parent_rel="",
        seen_folders=set(),
        retries=retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
    )
    if not items:
        raise RuntimeError("empty manifest returned from Google Drive embeddedfolderview")
    normalized = []
    for item in normalize_manifest_items(items):
        item["local_path"] = str(Path(output_root) / item["path"])
        normalized.append(item)
    return normalized


def materialize_manifest(args, output_root, manifest_path):
    manifest_data = None
    if not args.skip_manifest_refresh:
        try:
            files = fetch_manifest(
                url=args.url,
                output_root=output_root,
                retries=args.manifest_retries,
                retry_delay=args.retry_delay,
                retry_backoff=args.retry_backoff,
                use_cookies=args.use_cookies,
            )
            manifest_data = {
                "url": args.url,
                "retrieved_at": utc_now(),
                "output_root": str(output_root),
                "files": files,
            }
            save_json(manifest_path, manifest_data)
            log(f"[manifest] saved to {manifest_path}")
        except Exception as exc:
            if Path(manifest_path).exists():
                log(f"[manifest] refresh failed, falling back to saved manifest: {exc}")
            else:
                raise

    if manifest_data is None:
        if not Path(manifest_path).exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest_data = load_json(manifest_path)
        log(f"[manifest] loaded cached manifest from {manifest_path}")
    return manifest_data


def expected_scan_count_from_meshes(output_root):
    meshes_path = Path(output_root) / "meshes.txt"
    if not meshes_path.is_file():
        return None
    names = [line.strip() for line in meshes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(names) or None


def validate_manifest_or_raise(output_root, manifest_data, allow_partial_manifest=False):
    manifest_items = manifest_data.get("files", [])
    scan_items = [item for item in manifest_items if str(item.get("path", "")).startswith("scans/")]
    expected_scan_count = expected_scan_count_from_meshes(output_root)
    if expected_scan_count is None:
        return
    if len(scan_items) < expected_scan_count and not allow_partial_manifest:
        raise RuntimeError(
            "FAUST manifest appears incomplete: "
            f"found {len(scan_items)} scan files in the Drive manifest, "
            f"but meshes.txt lists {expected_scan_count} scans. "
            "This is usually caused by gdown's Google Drive folder limit of 50 files per folder."
        )


def output_path_for_item(output_root, item):
    return Path(output_root) / item["path"]


def summarize_local_files(output_root, manifest_items):
    existing = 0
    missing = []
    for item in manifest_items:
        path = output_path_for_item(output_root, item)
        if path.is_file() and path.stat().st_size > 0:
            existing += 1
        else:
            missing.append(item)
    return existing, missing


def download_one(item, output_root, retries, retry_delay, retry_backoff, use_cookies, resume, force_redownload):
    dest = output_path_for_item(output_root, item)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force_redownload:
        if dest.exists():
            dest.unlink()
        for partial in dest.parent.glob(dest.name + "*.part"):
            partial.unlink()
    last_error = None
    for attempt in range(1, int(retries) + 1):
        try:
            log(f"[file] {item['path']} | attempt {attempt}/{retries}")
            result = gdown.download(
                id=item["id"],
                output=str(dest),
                quiet=False,
                use_cookies=use_cookies,
                resume=resume,
            )
            if result is None:
                raise RuntimeError("gdown returned None")
            if not dest.is_file() or dest.stat().st_size <= 0:
                raise RuntimeError("download finished but target file is missing or empty")
            return {"status": "ok", "path": item["path"], "size": dest.stat().st_size}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            log(f"[file] failed: {type(exc).__name__}: {exc}")
            if attempt < int(retries):
                sleep_with_backoff(attempt, retry_delay, retry_backoff)
    return {"status": "error", "path": item["path"], "error": f"{type(last_error).__name__}: {last_error}"}


def write_state(state_path, manifest_data, output_root, completed, failed):
    state = {
        "updated_at": utc_now(),
        "url": manifest_data.get("url", ""),
        "output_root": str(output_root),
        "manifest_retrieved_at": manifest_data.get("retrieved_at", ""),
        "total_files": len(manifest_data.get("files", [])),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed": completed,
        "failed": failed,
    }
    save_json(state_path, state)


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else output_root / DEFAULT_MANIFEST_NAME
    state_path = Path(args.state_path).resolve() if args.state_path else output_root / DEFAULT_STATE_NAME

    manifest_data = materialize_manifest(args, output_root, manifest_path)
    validate_manifest_or_raise(
        output_root=output_root,
        manifest_data=manifest_data,
        allow_partial_manifest=args.allow_partial_manifest,
    )
    manifest_items = manifest_data.get("files", [])
    if not manifest_items:
        raise RuntimeError("manifest contains no files")

    existing_count, missing_items = summarize_local_files(output_root, manifest_items)
    log(f"[summary] manifest files: {len(manifest_items)}")
    log(f"[summary] already complete: {existing_count}")
    log(f"[summary] still missing: {len(missing_items)}")

    if args.list_only:
        for item in manifest_items:
            log(item["path"])
        return

    queue = manifest_items if args.force_redownload else missing_items
    if args.max_files > 0:
        queue = queue[: int(args.max_files)]
    log(f"[summary] queued for download: {len(queue)}")

    completed = []
    failed = []
    for index, item in enumerate(queue, start=1):
        log(f"[progress] {index}/{len(queue)} -> {item['path']}")
        result = download_one(
            item=item,
            output_root=output_root,
            retries=args.file_retries,
            retry_delay=args.retry_delay,
            retry_backoff=args.retry_backoff,
            use_cookies=args.use_cookies,
            resume=args.resume,
            force_redownload=args.force_redownload,
        )
        if result["status"] == "ok":
            completed.append(result)
        else:
            failed.append(result)
        write_state(state_path, manifest_data, output_root, completed, failed)
        if args.sleep_between_files > 0:
            time.sleep(float(args.sleep_between_files))

    final_existing, final_missing = summarize_local_files(output_root, manifest_items)
    log(f"[done] output_root={output_root}")
    log(f"[done] manifest={manifest_path}")
    log(f"[done] state={state_path}")
    log(f"[done] complete files now: {final_existing}/{len(manifest_items)}")
    log(f"[done] missing files now: {len(final_missing)}")
    if failed:
        log(f"[done] failed in this run: {len(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
