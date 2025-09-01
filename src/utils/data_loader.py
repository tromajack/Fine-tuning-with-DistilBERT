"""
CFPB Data Loader - Downloads complaint data from CFPB API
"""

import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from src.utils.logging import app_logger
from tqdm import tqdm
from pathlib import Path

from ..config import (
    CFPB_API_BASE_URL, 
    API_TIMEOUT, 
    MAX_API_RETRIES, 
    API_BATCH_SIZE,
    RAW_DATA_DIR
)

class CFPBDataLoader:
    """Class for downloading and processing CFPB complaint data"""
    
    def __init__(self):
        self.base_url = CFPB_API_BASE_URL
        self.timeout = API_TIMEOUT
        self.max_retries = MAX_API_RETRIES
        self.batch_size = API_BATCH_SIZE
        

    def download_complaints_csv(self, 
                            start_date: str,
                            end_date: str,
                            size: int,
                            save_path: Optional[str] = None,
                            max_retries: int = 3) -> str:
        """
        Download CFPB complaints data as CSV
        
        Args:
            start_date: Start date for complaints (YYYY-MM-DD format)
            end_date: End date for complaints (YYYY-MM-DD format)
            size: Maximum number of complaints to download
            save_path: Path to save the CSV file
            
        Returns:
            Path to saved CSV file
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = RAW_DATA_DIR / f"cfpb_complaints_{timestamp}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        app_logger.info(f"Downloading CFPB complaints from {start_date} to {end_date}")
        app_logger.info(f"Requesting {size} complaints, saving to {save_path}")

        params = {
            'date_received_min': start_date,
            'date_received_max': end_date,
            'size': size,
            'format': 'csv',
            'no_aggs': 'true',
            'field': 'all'
        }
        for attempt in range(max_retries):
            app_logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            try:
                headers = {}
                mode = 'wb'
                resume_byte_pos = 0
                if save_path.exists():
                    resume_byte_pos = save_path.stat().st_size
                    if resume_byte_pos > 0:
                        headers['Range'] = f'bytes={resume_byte_pos}-'
                        mode = 'ab'
                        app_logger.info(f"Resuming download from byte {resume_byte_pos}")

                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=True
                )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                progress_bar = tqdm(
                    total=total_size + resume_byte_pos,
                    initial=resume_byte_pos,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading CSV (attempt {attempt + 1})"
                )

                with open(save_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                progress_bar.close()
                if save_path.exists() and save_path.stat().st_size > 0:
                    app_logger.info(f"Successfully downloaded data to {save_path}")
                    return str(save_path)
                else:
                    raise requests.exceptions.RequestException("Downloaded file is empty or missing")

            except requests.exceptions.RequestException as e:
                if 'progress_bar' in locals():
                    progress_bar.close()
                app_logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    app_logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    app_logger.error(f"All {max_retries} download attempts failed")
                    raise
        raise requests.exceptions.RequestException("Download failed after all retry attempts")

    def load_complaints_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load complaints from CSV file and prepare for training
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with complaint data
        """
        app_logger.info(f"Loading complaints from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Basic data validation
            required_columns = ['Product', 'Consumer complaint narrative']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                app_logger.warning(f"Missing columns: {missing_columns}")
                return df

            df.rename(columns={'Consumer complaint narrative': 'text', 'Product': 'label'}, inplace=True)   

            # Filter out rows without narrative text
            initial_count = len(df)
            df = df[['text', 'label']]
            df.dropna(inplace=True)
            df = df[df['text'].str.strip() != '']
            app_logger.info(f"Loaded {len(df)} complaints (filtered from {initial_count})")
            app_logger.info(f"Product categories found: {df['label'].nunique()}")
            
            return df
            
        except Exception as e:
            app_logger.error(f"Error loading CSV: {e}")
            raise
    
    def fetch_live_complaint(self, complaint_id: str = None, size: int = 1) -> Dict:
        """
        Fetch live complaint data from API for prediction
        Used for getting fresh data for real-time predictions
        
        Args:
            complaint_id: Specific complaint ID to fetch
            size: Number of complaints to fetch if no specific ID
            
        Returns:
            Dictionary containing complaint data
        """
        params = {
            'format': 'json',
            'no_aggs': 'true',
            'size': size
        }
        
        if complaint_id:
            params['complaint_id'] = complaint_id
            
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"Error fetching live data: {e}")
            raise
    
    def get_latest_complaints(self, 
                            days_back: int = 7,
                            size: int = 100) -> List[Dict]:
        """
        Get the latest complaints from the past N days
        Useful for real-time prediction scenarios
        
        Args:
            days_back: Number of days to look back
            size: Number of complaints to fetch
            
        Returns:
            List of complaint dictionaries
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        params = {
            'date_received_min': start_str,
            'date_received_max': end_str,
            'format': 'json',
            'no_aggs': 'true',
            'size': size,
            'sort': 'created_date_desc'
        }
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            complaints = []
            if 'hits' in data and 'hits' in data['hits']:
                for hit in data['hits']['hits']:
                    if '_source' in hit:
                        complaints.append(hit['_source'])
            
            app_logger.info(f"Fetched {len(complaints)} recent complaints")
            return complaints
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"Error fetching recent complaints: {e}")
            raise
    
    def validate_api_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            params = {
                'size': 1,
                'format': 'json',
                'no_aggs': 'true'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            app_logger.info("API connection successful")
            return True
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"API connection failed: {e}")
            return False
        

dataloader = CFPBDataLoader()