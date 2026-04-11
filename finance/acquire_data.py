"""
Phase 0: Data Lake Acquisition

Downloads and stores all raw data for GPU-accelerated strategy training:
  1. Market OHLCV at multiple timescales (daily 5yr, hourly 2yr, 5min 60d)
  2. Macro/economic indicators from FRED
  3. Astronomical/lunar/seasonal features
  4. Weather data (NYC as Wall Street proxy)

All data saved to finance/data/ as parquet files.

Usage:
    python finance/acquire_data.py                # download everything
    python finance/acquire_data.py --market       # market data only
    python finance/acquire_data.py --macro        # macro data only
    python finance/acquire_data.py --astro        # astronomical only
    python finance/acquire_data.py --weather      # weather only
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from math import pi, sin, cos, asin, sqrt

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Core ETF basket (same as our spectral analysis)
SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'DIA',           # broad market
    'XLK', 'XLF', 'XLE', 'XLV',           # sectors
    'XLI', 'XLY', 'XLP', 'XLU',           # sectors
    'TLT', 'HYG', 'GLD', 'SLV', 'USO',   # bonds/commodities
]

# Extended universe for richer cross-asset signal
EXTRA_SYMBOLS = [
    'VIX',                                  # volatility index (via ^VIX)
    'BTC-USD', 'ETH-USD',                  # crypto
    'EURUSD=X', 'JPYUSD=X',               # FX
    'CL=F', 'GC=F', 'SI=F',              # futures (oil, gold, silver)
    '^GSPC', '^DJI', '^IXIC', '^RUT',    # indices
    '^TNX', '^TYX',                        # treasury yields
]

START_DATE = '2021-01-01'


# ─── Market Data ──────────────────────────────────────────────

def download_market_data():
    """Download OHLCV at multiple timescales."""
    import yfinance as yf

    all_symbols = SYMBOLS + EXTRA_SYMBOLS

    # 1. Daily — 5 years
    print("\n[1/3] Downloading DAILY data (5 years)...")
    daily = yf.download(all_symbols, start=START_DATE, interval='1d', progress=True)
    if not daily.empty:
        daily.to_parquet(DATA_DIR / 'market_daily.parquet')
        print(f"  Saved: {daily.shape[0]} days × {daily.shape[1]} columns")
    else:
        print("  WARNING: daily download returned empty")

    # 2. Hourly — 2 years (730 days max)
    print("\n[2/3] Downloading HOURLY data (2 years)...")
    hourly = yf.download(all_symbols, period='730d', interval='1h', progress=True)
    if not hourly.empty:
        hourly.to_parquet(DATA_DIR / 'market_hourly.parquet')
        print(f"  Saved: {hourly.shape[0]} bars × {hourly.shape[1]} columns")
    else:
        print("  WARNING: hourly download returned empty")

    # 3. 5-minute — chunked downloads (yfinance max 60d per request)
    #    Go back to match hourly coverage (~3 years)
    print("\n[3/3] Downloading 5-MINUTE data (chunked, ~3 years)...")
    chunk_days = 55  # stride (with 60d window = 5d overlap)
    end_date = datetime.now()
    # Go back to match hourly start
    fivemin_start = datetime(2023, 5, 12)

    all_chunks = []
    chunk_start = fivemin_start
    chunk_num = 0

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=60), end_date)
        chunk_num += 1
        start_str = chunk_start.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        print(f"  Chunk {chunk_num}: {start_str} → {end_str} ...", end=' ', flush=True)

        try:
            chunk = yf.download(all_symbols, start=start_str, end=end_str,
                              interval='5m', progress=False)
            if not chunk.empty:
                all_chunks.append(chunk)
                print(f"{chunk.shape[0]} bars")
            else:
                print("empty")
        except Exception as e:
            print(f"FAILED ({e})")

        chunk_start += timedelta(days=chunk_days)

    if all_chunks:
        fivemin = pd.concat(all_chunks)
        # Remove duplicate timestamps from overlapping chunks
        fivemin = fivemin[~fivemin.index.duplicated(keep='first')]
        fivemin = fivemin.sort_index()
        fivemin.to_parquet(DATA_DIR / 'market_5min.parquet')
        print(f"  Saved: {fivemin.shape[0]} bars × {fivemin.shape[1]} columns")
        print(f"  Range: {fivemin.index[0]} → {fivemin.index[-1]}")
    else:
        print("  WARNING: all 5min chunks returned empty")

    print("\nMarket data complete.")


# ─── Macro / Economic ────────────────────────────────────────

FRED_SERIES = {
    # Interest rates
    'DFF':       'fed_funds_rate',
    'DGS2':      'treasury_2yr',
    'DGS10':     'treasury_10yr',
    'DGS30':     'treasury_30yr',
    'T10Y2Y':    'yield_curve_10y2y',
    'T10Y3M':    'yield_curve_10y3m',

    # Inflation
    'CPIAUCSL':  'cpi',
    'CPILFESL':  'core_cpi',
    'PCEPI':     'pce_price_index',
    'T5YIE':     'breakeven_inflation_5yr',
    'T10YIE':    'breakeven_inflation_10yr',

    # Labor
    'UNRATE':    'unemployment_rate',
    'PAYEMS':    'nonfarm_payrolls',
    'ICSA':      'initial_claims',
    'CCSA':      'continuing_claims',

    # Growth
    'GDP':       'gdp',
    'GDPC1':     'real_gdp',

    # Money supply
    'M2SL':      'money_supply_m2',
    'WALCL':     'fed_balance_sheet',

    # Financial conditions
    'BAMLH0A0HYM2': 'high_yield_spread',
    'TEDRATE':      'ted_spread',
    'VIXCLS':       'vix_close',

    # Consumer
    'UMCSENT':   'michigan_sentiment',
    'RSXFS':     'retail_sales',

    # Housing
    'HOUST':     'housing_starts',
    'MORTGAGE30US': 'mortgage_30yr',
}


def download_macro_data():
    """Download macro indicators from FRED."""
    try:
        from fredapi import Fred
    except ImportError:
        print("ERROR: pip install fredapi")
        return

    api_key = os.environ.get('FRED_API_KEY', '')
    if not api_key:
        # FRED allows limited access without key, but let's try
        # If no key, we'll use pandas-datareader as fallback
        print("  No FRED_API_KEY set. Attempting without key...")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  Then: export FRED_API_KEY=your_key_here")

        # Fallback: download what we can from yfinance
        _download_macro_fallback()
        return

    fred = Fred(api_key=api_key)
    frames = {}

    print(f"\nDownloading {len(FRED_SERIES)} FRED series...")
    for series_id, name in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=START_DATE)
            frames[name] = data
            print(f"  {name}: {len(data)} observations")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if frames:
        macro = pd.DataFrame(frames)
        macro.index.name = 'date'
        macro.to_parquet(DATA_DIR / 'macro_fred.parquet')
        print(f"\nSaved: {macro.shape[0]} rows × {macro.shape[1]} columns")
    else:
        print("\nWARNING: no FRED data downloaded")


def _download_macro_fallback():
    """Fallback: get VIX and treasury yields from yfinance."""
    import yfinance as yf

    print("\n  Fallback: downloading VIX + treasury proxies from yfinance...")
    macro_tickers = ['^VIX', '^TNX', '^TYX', '^FVX', '^IRX']
    data = yf.download(macro_tickers, start=START_DATE, interval='1d', progress=False)
    if not data.empty:
        data.to_parquet(DATA_DIR / 'macro_yfinance.parquet')
        print(f"  Saved: {data.shape[0]} days × {data.shape[1]} columns")
    else:
        print("  WARNING: fallback macro download empty")


# ─── Astronomical / Lunar / Seasonal ─────────────────────────

def compute_astro_features():
    """Compute astronomical features for every day in our range."""
    import ephem

    print("\nComputing astronomical features...")

    dates = pd.date_range(START_DATE, datetime.now().strftime('%Y-%m-%d'), freq='D')
    records = []

    for dt in dates:
        d = ephem.Date(dt.strftime('%Y/%m/%d'))

        # Moon phase (0=new, 0.5=full, 1=new again)
        moon = ephem.Moon(d)
        moon_phase = moon.phase / 100.0  # normalize to [0, 1]

        # Moon illumination
        moon_illum = moon.phase / 100.0

        # Sun declination (seasonality proxy)
        sun = ephem.Sun(d)
        sun_dec = float(sun.dec) * 180 / pi  # degrees

        # Day length at NYC latitude (40.7N)
        obs = ephem.Observer()
        obs.lat = '40.7128'
        obs.lon = '-74.0060'
        obs.date = d
        try:
            sunrise = obs.next_rising(ephem.Sun())
            sunset = obs.next_setting(ephem.Sun())
            day_length_hrs = (sunset - sunrise) * 24
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            day_length_hrs = 12.0

        # Planetary positions (ecliptic longitude, degrees)
        mercury = ephem.Mercury(d)
        venus = ephem.Venus(d)
        mars = ephem.Mars(d)
        jupiter = ephem.Jupiter(d)
        saturn = ephem.Saturn(d)

        # Mercury retrograde detection (simplified: when elongation is decreasing
        # and planet is east of sun, it's about to go retrograde)
        # More accurate: check if ecliptic longitude is decreasing
        d_next = ephem.Date(d + 1)
        merc_next = ephem.Mercury(d_next)
        merc_retro = 1.0 if float(merc_next.hlong) < float(mercury.hlong) else 0.0

        # Venus retrograde
        venus_next = ephem.Venus(d_next)
        venus_retro = 1.0 if float(venus_next.hlong) < float(venus.hlong) else 0.0

        # Seasonal features (cyclical encoding)
        day_of_year = dt.timetuple().tm_yday
        year_frac = day_of_year / 365.25
        season_sin = sin(2 * pi * year_frac)
        season_cos = cos(2 * pi * year_frac)

        # Week/month cyclical
        dow = dt.weekday()  # 0=Mon, 4=Fri
        dow_sin = sin(2 * pi * dow / 5)
        dow_cos = cos(2 * pi * dow / 5)
        month = dt.month
        month_sin = sin(2 * pi * month / 12)
        month_cos = cos(2 * pi * month / 12)

        # Quarter-end effects (last 5 trading days of quarter)
        is_quarter_end = 1.0 if (dt.month in [3, 6, 9, 12] and dt.day >= 25) else 0.0

        # Options expiration (3rd Friday of month)
        # Simplified: check if it's a Friday between 15th-21st
        is_opex = 1.0 if (dt.weekday() == 4 and 15 <= dt.day <= 21) else 0.0

        # FOMC meeting rough schedule (every ~6 weeks, Jan/Mar/May/Jun/Jul/Sep/Nov/Dec)
        # This is approximate; real dates should come from a calendar
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        is_fomc_week = 1.0 if (dt.month in fomc_months and 20 <= dt.day <= 28) else 0.0

        records.append({
            'date': dt,
            'moon_phase': moon_phase,
            'moon_illum': moon_illum,
            'sun_declination': sun_dec,
            'day_length_hrs': day_length_hrs,
            'mercury_longitude': float(mercury.hlong) * 180 / pi,
            'venus_longitude': float(venus.hlong) * 180 / pi,
            'mars_longitude': float(mars.hlong) * 180 / pi,
            'jupiter_longitude': float(jupiter.hlong) * 180 / pi,
            'saturn_longitude': float(saturn.hlong) * 180 / pi,
            'mercury_retrograde': merc_retro,
            'venus_retrograde': venus_retro,
            'season_sin': season_sin,
            'season_cos': season_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'is_quarter_end': is_quarter_end,
            'is_opex': is_opex,
            'is_fomc_week': is_fomc_week,
            'day_of_year': day_of_year,
            'year_fraction': year_frac,
        })

    astro = pd.DataFrame(records)
    astro.set_index('date', inplace=True)
    astro.to_parquet(DATA_DIR / 'astro_features.parquet')
    print(f"  Saved: {astro.shape[0]} days × {astro.shape[1]} features")

    # Summary
    retro_days = astro['mercury_retrograde'].sum()
    print(f"  Mercury retrograde days: {int(retro_days)} ({retro_days/len(astro)*100:.1f}%)")
    print(f"  Moon phase range: [{astro['moon_phase'].min():.2f}, {astro['moon_phase'].max():.2f}]")
    print(f"  OpEx Fridays: {int(astro['is_opex'].sum())}")


# ─── Weather ──────────────────────────────────────────────────

def download_weather_data():
    """Download NYC weather from Meteostat."""
    try:
        from meteostat import Point
        try:
            from meteostat import Daily as MeteoDaily
        except ImportError:
            from meteostat import daily as MeteoDaily

        print("\nDownloading NYC weather (Meteostat)...")

        # NYC / Central Park station
        nyc = Point(40.7128, -74.0060, 10)

        start = datetime.strptime(START_DATE, '%Y-%m-%d')
        end = datetime.now()

        ts = MeteoDaily(nyc, start, end)
        data = ts.fetch()
    except Exception as e:
        print(f"\n  Meteostat failed ({e}), using fallback...")
        data = None

    if data is None or (hasattr(data, 'empty') and data.empty):
        print("  WARNING: Meteostat returned empty data")
        _download_weather_fallback()
        return

    # Rename for clarity
    data = data.rename(columns={
        'tavg': 'temp_avg_c',
        'tmin': 'temp_min_c',
        'tmax': 'temp_max_c',
        'prcp': 'precipitation_mm',
        'snow': 'snow_mm',
        'wdir': 'wind_direction',
        'wspd': 'wind_speed_kmh',
        'wpgt': 'wind_gust_kmh',
        'pres': 'pressure_hpa',
        'tsun': 'sunshine_minutes',
    })

    # Derived features
    if 'temp_max_c' in data.columns and 'temp_min_c' in data.columns:
        data['temp_range_c'] = data['temp_max_c'] - data['temp_min_c']
    if 'precipitation_mm' in data.columns:
        data['is_rainy'] = (data['precipitation_mm'] > 0).astype(float)
    if 'snow_mm' in data.columns:
        data['is_snowy'] = (data['snow_mm'] > 0).astype(float)

    # Extreme weather flags
    if 'temp_max_c' in data.columns:
        data['extreme_heat'] = (data['temp_max_c'] > 35).astype(float)
        data['extreme_cold'] = (data['temp_min_c'] < -10).astype(float)

    data.to_parquet(DATA_DIR / 'weather_nyc.parquet')
    print(f"  Saved: {data.shape[0]} days × {data.shape[1]} features")

    # Summary
    if 'temp_avg_c' in data.columns:
        print(f"  Temp range: {data['temp_avg_c'].min():.1f}°C to {data['temp_avg_c'].max():.1f}°C")
    if 'precipitation_mm' in data.columns:
        rainy = data['is_rainy'].sum()
        print(f"  Rainy days: {int(rainy)} ({rainy/len(data)*100:.1f}%)")


def _download_weather_fallback():
    """Fallback: generate synthetic seasonal weather features."""
    print("  Fallback: generating seasonal temperature proxy...")

    dates = pd.date_range(START_DATE, datetime.now().strftime('%Y-%m-%d'), freq='D')
    records = []
    for dt in dates:
        doy = dt.timetuple().tm_yday
        # Rough NYC temperature model: ~12°C avg, ~15°C amplitude
        temp_avg = 12 + 15 * sin(2 * pi * (doy - 80) / 365.25)
        records.append({
            'date': dt,
            'temp_avg_c': temp_avg + np.random.normal(0, 3),
            'temp_synthetic': True,
        })

    weather = pd.DataFrame(records).set_index('date')
    weather.to_parquet(DATA_DIR / 'weather_nyc.parquet')
    print(f"  Saved synthetic: {weather.shape[0]} days")


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Acquire data for spectral trading model')
    parser.add_argument('--market', action='store_true', help='Market data only')
    parser.add_argument('--macro', action='store_true', help='Macro data only')
    parser.add_argument('--astro', action='store_true', help='Astronomical data only')
    parser.add_argument('--weather', action='store_true', help='Weather data only')
    args = parser.parse_args()

    do_all = not (args.market or args.macro or args.astro or args.weather)

    print("=" * 60)
    print("SPECTRAL ALPHA DATA LAKE — Phase 0 Acquisition")
    print(f"Target: {START_DATE} → present")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)

    if do_all or args.market:
        download_market_data()

    if do_all or args.macro:
        download_macro_data()

    if do_all or args.astro:
        compute_astro_features()

    if do_all or args.weather:
        download_weather_data()

    # Summary
    print("\n" + "=" * 60)
    print("DATA LAKE INVENTORY")
    print("=" * 60)
    total_bytes = 0
    for f in sorted(DATA_DIR.glob('*.parquet')):
        size = f.stat().st_size
        total_bytes += size
        df = pd.read_parquet(f)
        print(f"  {f.name:<30} {size/1024:>8.1f} KB  {df.shape[0]:>6} rows × {df.shape[1]:>3} cols")
    print(f"  {'TOTAL':<30} {total_bytes/1024:>8.1f} KB")


if __name__ == '__main__':
    main()
