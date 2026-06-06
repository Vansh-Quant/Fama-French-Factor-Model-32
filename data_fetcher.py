# data_fetcher.py
"""
Project #032: Ingestion Layer
Downloads, extracts, and cleanly processes the official Kenneth French 3-Factor daily data.
"""

import io
import zipfile
import requests
import pandas as pd

class FamaFrenchFetcher:
    def __init__(self):
        # Official institutional URL for daily historical factors
        self.DATA_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

    def fetch_daily_factors(self) -> pd.DataFrame:
        """Downloads the zipped CSV, extracts text data, and formats a clean risk-factor dataframe."""
        print(f"[INGESTION] Reaching out to Kenneth French Data Library...")
        response = requests.get(self.DATA_URL)
        
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch data stream from target repository. HTTP Status: {response.status_code}")
            
        # Extract the zip file contents completely in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                raw_bytes = f.read()
                # Decode parsing ignoring non-standard header formats
                raw_text = raw_bytes.decode('utf-8', errors='ignore')

        print(f"[INGESTION] Decompressing archive and parsing structural vectors...")
        
        # Locate the clean split point in French's text formatting where the table officially starts
        lines = raw_text.split('\r\n')
        data_start_idx = 0
        for idx, line in enumerate(lines):
            if "Mkt-RF" in line:
                data_start_idx = idx
                break

        # Re-assemble pure data rows, stripping metadata out
        clean_csv_buffer = io.StringIO('\n'.join(lines[data_start_idx:]))
        
        # Load parsing into pandas matrix framework
        df = pd.read_csv(clean_csv_buffer, index_col=0, nrows=25000) # Safety boundary row cap
        
        # Formatting cleanup
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna()
        
        # CRITICAL MATH SCALE: Kenneth French publishes factor returns as raw percentages (e.g., 1.25 means 1.25%)
        # We divide by 100 to scale them cleanly to decimal percentages (e.g., 0.0125) matching trade logs.
        df = df.astype(float) / 100.0
        
        # Explicit column naming verification
        df.columns = ['Mkt_RF', 'SMB', 'HML', 'RF']
        print(f"[INGESTION SUCCESS] Loaded {len(df)} daily historical factor lines cleanly.")
        return df