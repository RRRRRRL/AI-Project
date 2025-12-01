import argparse
import numpy as np
import pandas as pd

# Center near VHHH: 22.3089N, 113.9146E
VHHH_LAT = 22.3089
VHHH_LON = 113.9146

def simulate_flight(flight_id: str, n_steps: int, dt: int = 30,
                    start_lat=VHHH_LAT, start_lon=VHHH_LON):
    # Simple kinematic model with gentle turns and climb/descent
    # Speeds ~ [150, 250] m/s; turn rate small; vertical speed [-5, 5] m/s
    rng = np.random.default_rng(abs(hash(flight_id)) % (2**32))
    speed = rng.uniform(150, 250)  # m/s
    heading = rng.uniform(0, 2*np.pi)
    vz = rng.uniform(-3, 5)  # m/s
    turn_rate = rng.normal(0.0, 0.002)  # rad/s
    lat = start_lat + rng.uniform(-0.2, 0.2)
    lon = start_lon + rng.uniform(-0.2, 0.2)
    alt = rng.uniform(1000, 10000)  # meters

    # Convert to local ENU for motion, then back to lat/lon for output
    from geo import geodetic_to_enu, enu_to_geodetic
    ref_lat, ref_lon, ref_alt = lat, lon, alt
    e, n, u = 0.0, 0.0, 0.0

    rows = []
    t0 = rng.integers(1_700_000_000, 1_800_000_000)  # synthetic UNIX time
    for k in range(n_steps):
        # Update velocities
        heading += turn_rate * dt + rng.normal(0.0, 0.001)
        vx = speed * np.cos(heading) + rng.normal(0, 0.5)
        vy = speed * np.sin(heading) + rng.normal(0, 0.5)
        vz_k = vz + rng.normal(0, 0.2)
        # Update position
        e += vx * dt
        n += vy * dt
        u += vz_k * dt
        lat_k, lon_k, alt_k = enu_to_geodetic(e, n, u, ref_lat, ref_lon, ref_alt)
        ts = int(t0 + k * dt)
        rows.append((flight_id, ts, lat_k, lon_k, alt_k))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_flights", type=int, default=600)
    ap.add_argument("--min_len", type=int, default=160)
    ap.add_argument("--max_len", type=int, default=260)
    ap.add_argument("--dt", type=int, default=30)
    ap.add_argument("--out_csv", type=str, default="data/raw/hkg_synth.csv")
    args = ap.parse_args()

    all_rows = []
    for i in range(args.num_flights):
        fid = f"HKG{str(i).zfill(5)}"
        n_steps = int(np.random.randint(args.min_len, args.max_len + 1))
        rows = simulate_flight(fid, n_steps, dt=args.dt)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows, columns=["flight_id", "timestamp", "lat", "lon", "alt"])
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()