#!/usr/bin/env python3
"""
OpenSky -> model CSV extractor for AI-Project

Outputs CSV with columns required by prepare_sequences.py:
- flight_id (string): aircraft identifier; uses icao24
- timestamp (int, unix seconds): measurement time
- lat (float, degrees)
- lon (float, degrees)
- alt (float, meters): prefers geo_altitude; falls back to baro_altitude
"""
import argparse
import csv
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://opensky-network.org/api"

STATE_FIELDS = [
    "icao24","callsign","origin_country","time_position","last_contact",
    "longitude","latitude","baro_altitude","on_ground","velocity",
    "true_track","vertical_rate","sensors","geo_altitude","squawk",
    "spi","position_source",
]

def retry_request(method, url, params=None, headers=None, max_retries=5, backoff_factor=1.5, timeout=30.0, auth=None):
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, params=params, headers=headers, timeout=timeout, auth=auth)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff_factor ** attempt)
            else:
                resp.raise_for_status()
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_factor ** attempt)
    raise RuntimeError("Failed after retries")

def fetch_states(params: Dict[str, Any], auth: Optional[tuple]) -> Dict[str, Any]:
    resp = retry_request("GET", f"{BASE_URL}/states/all", params=params, auth=auth)
    return resp.json()

def to_model_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not payload or payload.get("states") is None:
        return rows
    snapshot_time = payload.get("time", None)
    for s in payload["states"]:
        if not isinstance(s, list):
            continue
        if len(s) < len(STATE_FIELDS):
            s = s + [None] * (len(STATE_FIELDS) - len(s))

        icao24 = s[0]
        lat = s[6]
        lon = s[5]
        geo_alt = s[13]
        baro_alt = s[7]
        time_position = s[3]
        last_contact = s[4]

        ts = time_position if isinstance(time_position, (int, float)) else last_contact
        if ts is None and snapshot_time is not None:
            ts = snapshot_time

        if icao24 is None or lat is None or lon is None or ts is None:
            continue

        alt = geo_alt if isinstance(geo_alt, (int, float)) else baro_alt
        if alt is None:
            continue

        try:
            row = {
                "flight_id": str(icao24),
                "timestamp": int(ts),
                "lat": float(lat),
                "lon": float(lon),
                "alt": float(alt),
            }
        except (ValueError, TypeError):
            continue

        if row["alt"] > 0:
            rows.append(row)

    rows.sort(key=lambda r: (r["flight_id"], r["timestamp"]))
    deduped: List[Dict[str, Any]] = []
    last_key = None
    for r in rows:
        key = (r["flight_id"], r["timestamp"]) 
        if key != last_key:
            deduped.append(r)
            last_key = key
        else:
            deduped[-1] = r
    return deduped

def validate_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be in [-90, 90]")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be in [-180, 180]")
    if min_lat > max_lat:
        raise ValueError("min_lat must be <= max_lat")
    if min_lon > max_lon:
        raise ValueError("min_lon must be <= max_lon")

def save_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["flight_id", "timestamp", "lat", "lon", "alt"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote CSV: {out_path} ({len(rows)} rows)")

def parse_args():
    ap = argparse.ArgumentParser(description="OpenSky -> model CSV extractor")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_live = sub.add_parser("live", help="Fetch live snapshot")
    p_live.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    p_live.add_argument("--out", type=str, required=True, help="Output CSV path")

    p_hist = sub.add_parser("history", help="Fetch historical states between begin and end (unix seconds)")
    p_hist.add_argument("--begin", type=int, required=True, help="Begin timestamp (unix seconds)")
    p_hist.add_argument("--end", type=int, required=True, help="End timestamp (unix seconds)")
    p_hist.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    p_hist.add_argument("--icao24", type=str, help="Filter by aircraft icao24 hex")
    p_hist.add_argument("--out", type=str, required=True, help="Output CSV path")

    ap.add_argument("--auth", nargs=2, metavar=("USERNAME", "PASSWORD"), help="Optional OpenSky credentials")
    return ap.parse_args()

def main():
    args = parse_args()

    params: Dict[str, Any] = {}
    auth_tuple = tuple(args.auth) if args.auth else None

    if args.mode == "live":
        if args.bbox:
            min_lat, min_lon, max_lat, max_lon = args.bbox
            validate_bbox(min_lat, min_lon, max_lat, max_lon)
            params.update({"lamin": min_lat, "lomin": min_lon, "lamax": max_lat, "lomax": max_lon})
    else:
        if args.end <= args.begin:
            print("Error: --end must be greater than --begin", file=sys.stderr)
            sys.exit(2)
        params.update({"begin": args.begin, "end": args.end})
        if args.bbox:
            min_lat, min_lon, max_lat, max_lon = args.bbox
            validate_bbox(min_lat, min_lon, max_lat, max_lon)
            params.update({"lamin": min_lat, "lomin": min_lon, "lamax": max_lat, "lomax": max_lon})
        if getattr(args, "icao24", None):
            params["icao24"] = args.icao24

    payload = fetch_states(params, auth_tuple)
    rows = to_model_rows(payload)
    save_csv(rows, args.out)

if __name__ == "__main__":
    main()
