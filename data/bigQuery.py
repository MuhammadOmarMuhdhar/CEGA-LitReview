import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import numpy as np
import time
import logging
import json
from datetime import datetime, date
import re
import gc
import weakref
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Client:
    def __init__(self, credentials_json, project_id):
        """
        Initialize the BigQuery API with memory management
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing BigQuery API")
        
        self.credentials_json = credentials_json
        self.project_id = project_id
        self.client = self._build_client()
        self.batch_size = 10000
        
        # Track active jobs for cleanup - USE WEAKREFS TO PREVENT REFERENCE CYCLES
        self._active_jobs = weakref.WeakSet()
        
        self.logger.info(f"BigQuery API initialized with batch size: {self.batch_size}")
    
    def _build_client(self):
        """Build and return the BigQuery client"""
        self.logger.info("Building BigQuery client")
        
        credentials = service_account.Credentials.from_service_account_info(
            self.credentials_json,
            scopes=['https://www.googleapis.com/auth/bigquery']
        )
        
        client = bigquery.Client(project=self.project_id, credentials=credentials)
        self.logger.info("BigQuery client built successfully")
        return client
    
    @contextmanager
    def _managed_query_job(self, query, job_config=None):
        """Context manager for query jobs with automatic cleanup"""
        job = None
        try:
            job = self.client.query(query, job_config=job_config)
            self._active_jobs.add(job)  # Track with weak reference
            yield job
        finally:
            # EXPLICIT CLEANUP
            if job:
                try:
                    # Cancel job if still running
                    if hasattr(job, 'cancel') and job.state in ['PENDING', 'RUNNING']:
                        job.cancel()
                except Exception:
                    pass
                
                # Clear job reference
                job = None
            
            # Force garbage collection
            gc.collect()
    
    def _cleanup_jobs(self):
        """Manually cleanup any remaining job references"""
        try:
            # WeakSet automatically removes dead references
            active_count = len(self._active_jobs)
            if active_count > 0:
                self.logger.warning(f"Found {active_count} active jobs during cleanup")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"Error during job cleanup: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self._cleanup_jobs()
            if hasattr(self, 'client'):
                self.client.close()
        except Exception:
            pass
    
    def _is_client_healthy(self):
        """Check if the BigQuery client connection is still healthy"""
        try:
            test_query = "SELECT 1 as test_connection"
            with self._managed_query_job(test_query) as job:
                job.result(timeout=5)
                return True
        except Exception as e:
            self.logger.warning(f"Client health check failed: {e}")
            return False

    def _refresh_client(self):
        """Rebuild the BigQuery client with fresh credentials"""
        self.logger.info("Refreshing BigQuery client connection")
        
        # CLEANUP OLD CLIENT
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass
        
        # Cleanup any remaining jobs
        self._cleanup_jobs()
        
        # Build new client
        self.client = self._build_client()
        return self.client

    def get_healthy_client(self):
        """Get a healthy BigQuery client, refreshing if necessary"""
        if not self._is_client_healthy():
            self.logger.info("Client unhealthy, refreshing connection")
            self._refresh_client()
        return self.client
    
    def execute_query(self, query, use_storage_api=True):
        """Execute query with proper memory management"""
        try:
            # Ensure we have a healthy client
            client = self.get_healthy_client()
            
            job_config = bigquery.QueryJobConfig(use_query_cache=True)
            
            # Use context manager for automatic cleanup
            with self._managed_query_job(query, job_config) as query_job:
                
                if use_storage_api:
                    try:
                        df = query_job.to_dataframe(create_bqstorage_client=True)
                    except Exception as e:
                        self.logger.warning(f"Storage API failed, using standard API: {e}")
                        df = query_job.to_dataframe()
                else:
                    df = query_job.to_dataframe()
                
                # IMPORTANT: Make a copy to break references to the job
                df_copy = df.copy()
                
                # Clear original dataframe reference
                del df
                
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
        finally:
            # Force cleanup
            gc.collect()
    
    def read(self, dataset_id, table_id, query=None, limit=None, use_db_dtypes=True):
        """
        Reads data with proper memory management
        """
        self.logger.info(f"Starting read operation - Dataset: {dataset_id}, Table: {table_id}")
        
        try:
            if query:
                sql_query = query
                self.logger.info("Using custom query")
            else:
                sql_query = f"""
                SELECT *
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                """
                
                if limit:
                    sql_query += f" LIMIT {limit}"
            
            self.logger.info(f"Executing query: {sql_query}")
            
            # Use context manager for automatic cleanup
            with self._managed_query_job(sql_query) as query_job:
                
                try:
                    if use_db_dtypes:
                        df = query_job.to_dataframe()
                    else:
                        raise ValueError("Skipping db-dtypes")
                except (ValueError, ImportError) as e:
                    if "db-dtypes" in str(e) or not use_db_dtypes:
                        self.logger.warning("db-dtypes package not available, using alternative method")
                        
                        # MEMORY-EFFICIENT ROW PROCESSING
                        results = query_job.result()
                        
                        # Process in chunks to avoid memory buildup
                        chunk_size = 10000
                        all_chunks = []
                        
                        current_chunk = []
                        for i, row in enumerate(results):
                            row_dict = {}
                            for key, value in row.items():
                                if hasattr(value, 'isoformat'):
                                    row_dict[key] = value.isoformat()
                                else:
                                    row_dict[key] = value
                            current_chunk.append(row_dict)
                            
                            # Process in chunks
                            if len(current_chunk) >= chunk_size:
                                all_chunks.append(pd.DataFrame(current_chunk))
                                current_chunk = []
                                
                                # Periodic garbage collection
                                if len(all_chunks) % 10 == 0:
                                    gc.collect()
                        
                        # Add remaining rows
                        if current_chunk:
                            all_chunks.append(pd.DataFrame(current_chunk))
                        
                        # Combine chunks efficiently
                        if all_chunks:
                            df = pd.concat(all_chunks, ignore_index=True)
                            # Clear chunk references
                            del all_chunks
                        else:
                            df = pd.DataFrame()
                    else:
                        raise e
                
                # Make a copy to break job references
                df_copy = df.copy()
                del df
                
                self.logger.info(f"Read operation completed - DataFrame shape: {df_copy.shape}")
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error reading data: {str(e)}")
            raise
        finally:
            gc.collect()
    
    def read_fast(self, dataset_id, table_id, query=None):
        """Fast read with memory management"""
        try:
            if query:
                sql_query = query
            else:
                sql_query = f"SELECT * FROM `{self.project_id}.{dataset_id}.{table_id}`"
            
            with self._managed_query_job(sql_query) as job:
                try:
                    df = job.to_dataframe(create_bqstorage_client=True)
                except Exception as e:
                    self.logger.warning(f"Storage API failed, using standard API: {e}")
                    df = job.to_dataframe()
                
                # Return copy to break job references
                return df.copy()
                
        except Exception as e:
            self.logger.error(f"Error in fast read: {str(e)}")
            raise
        finally:
            gc.collect()
    
    # Rest of your methods with similar patterns...
    def _sanitize_cell_value(self, value):
        """Comprehensive sanitization of individual cell values for BigQuery"""
        # Handle None/NaN/null values
        if value is None or pd.isna(value):
            return None
        
        # Handle numpy NaN specifically
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        
        # Handle numpy integers
        if isinstance(value, np.integer):
            return int(value)
        
        # Handle numpy floats
        if isinstance(value, np.floating):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        
        # Handle numpy arrays - convert to string representation
        if isinstance(value, np.ndarray):
            try:
                as_list = value.tolist()
                return json.dumps(as_list, default=str, ensure_ascii=False)
            except:
                return str(value)
        
        # Handle Python lists - convert to JSON string
        if isinstance(value, list):
            try:
                return json.dumps(value, default=str, ensure_ascii=False)
            except:
                return str(value)
        
        # Handle dictionaries - convert to JSON string
        if isinstance(value, dict):
            try:
                return json.dumps(value, default=str, ensure_ascii=False)
            except:
                return str(value)
        
        # Handle datetime objects
        if isinstance(value, (datetime, date)):
            return value
        
        # Handle pandas Timestamp
        if hasattr(value, 'to_pydatetime'):
            try:
                return value.to_pydatetime()
            except:
                return str(value)
        
        # Handle boolean values
        if isinstance(value, bool):
            return value
        
        # Handle complex numbers
        if isinstance(value, complex):
            return f"{value.real}+{value.imag}i"
        
        # Handle bytes
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except:
                return str(value)
        
        # Convert to string and clean for remaining types
        try:
            str_value = str(value)
            # Remove control characters
            str_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str_value)
            return str_value
        except Exception as e:
            self.logger.warning(f"Failed to convert value {type(value)} to string: {e}")
            return None