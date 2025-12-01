import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from geo import geodetic_to_enu

def within_region(lat, lon, region):
    lon_min, lon_max, lat_min, lat_max = region
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

def compute_features(e, n, u, ts, dt):
    # velocities from finite differences
    de = np.diff(e, prepend=e[0])
    dn = np.diff(n, prepend=n[0])
    du = np.diff(u, prepend=u[0])
    vx = de / dt
    vy = dn / dt
    vz = du / dt
    speed = np.sqrt(vx**2 + vy**2)
    track = np.arctan2(vy, vx)
    cos_track = np.cos(track)
    sin_track = np.sin(track)
    tod = (ts % 86400) / 86400.0 * 2 * np.pi  # time of day angle
    sin_tod = np.sin(tod)
    cos_tod = np.cos(tod)
    X = np.stack([vx, vy, vz, speed, cos_track, sin_track, sin_tod, cos_tod], axis=-1)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True)
    ap.add_argument("--output_npz", type=str, required=True)
    ap.add_argument("--input_len", type=int, default=40)
    ap.add_argument("--pred_len", type=int, default=20)
    ap.add_argument("--resample_sec", type=int, default=30)
    ap.add_argument("--region", type=str, default="113.5,115.5,21.5,23.0",
                    help="lon_min,lon_max,lat_min,lat_max")
    ap.add_argument("--min_points", type=int, default=120)
    args = ap.parse_args()

    lon_min, lon_max, lat_min, lat_max = [float(x) for x in args.region.split(",")]
    region = (lon_min, lon_max, lat_min, lat_max)

    df = pd.read_csv(args.input_csv)
    assert {"flight_id", "timestamp", "lat", "lon", "alt"}.issubset(df.columns)

    # Filter region and positive altitude
    df = df[within_region(df["lat"].values, df["lon"].values, region)]
    df = df[df["alt"].values > 0]
    df = df.sort_values(["flight_id", "timestamp"])

    X_list, Y_list, flights = [], [], []
    dt = args.resample_sec

    for fid, g in tqdm(df.groupby("flight_id"), desc="Flights"):
        if len(g) < args.min_points:
            continue
        g = g.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        ts = g["timestamp"].values.astype(np.int64)
        lat_vals = g["lat"].values.astype(float)
        lon_vals = g["lon"].values.astype(float)
        alt_vals = g["alt"].values.astype(float)

        # Build uniform time grid
        t0, t1 = int(ts.min()), int(ts.max())
        ts_grid = np.arange(t0, t1 + dt, dt, dtype=int)
        if len(ts_grid) < args.input_len + args.pred_len + 1:
            continue
        # Linear interpolation to grid
        lat = np.interp(ts_grid, ts, lat_vals)
        lon = np.interp(ts_grid, ts, lon_vals)
        alt = np.interp(ts_grid, ts, alt_vals)

        # Local ENU around first point
        ref_lat, ref_lon, ref_alt = float(lat[0]), float(lon[0]), float(alt[0])
        e, n, u = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

        # Features
        X = compute_features(e, n, u, ts_grid, dt)  # [T, 8]

        H, K = args.input_len, args.pred_len
        T = len(ts_grid)
        max_start = T - (H + K)
        if max_start <= 0:
            continue
        for s in range(max_start):
            X_hist = X[s:s+H]
            # future offsets relative to last history point
            e0, n0, u0 = e[s+H-1], n[s+H-1], u[s+H-1]
            e_fut = e[s+H:s+H+K] - e0
            n_fut = n[s+H:s+H+K] - n0
            u_fut = u[s+H:s+H+K] - u0
            Y_fut = np.stack([e_fut, n_fut, u_fut], axis=-1)

            X_list.append(X_hist.astype(np.float32))
            Y_list.append(Y_fut.astype(np.float32))
            flights.append(fid)

    X_arr = np.stack(X_list, axis=0) if X_list else np.empty((0, args.input_len, 8), dtype=np.float32)
    Y_arr = np.stack(Y_list, axis=0) if Y_list else np.empty((0, args.pred_len, 3), dtype=np.float32)
    flights = np.array(flights)
    np.savez_compressed(args.output_npz, X=X_arr, Y=Y_arr, flights=flights)
    print(f"Saved windows: X {X_arr.shape}, Y {Y_arr.shape} to {args.output_npz}")

if __name__ == "__main__":
    main()