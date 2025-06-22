import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import numpy as np
import time
import logging
import json
from datetime import datetime, date
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Client:
    def __init__(self, credentials_json, project_id):
        """
        Initialize the BigQuery API
        
        Parameters:
        credentials_json (dict): Service account credentials as dictionary
        project_id (str): Google Cloud Project ID
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing BigQuery API")
        
        self.credentials_json = credentials_json
        self.project_id = project_id
        self.client = self._build_client()
        # Set batch size for BigQuery operations
        self.batch_size = 10000  # BigQuery can handle larger batches than Sheets
        
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
    
    def _is_client_healthy(self):
        """Check if the BigQuery client connection is still healthy"""
        try:
            # Simple query to test connection - should complete in <1 second
            test_query = "SELECT 1 as test_connection"
            query_job = self.client.query(test_query)
            query_job.result(timeout=5)  # 5 second timeout
            return True
        except Exception as e:
            self.logger.warning(f"Client health check failed: {e}")
            return False

    def _refresh_client(self):
        """Rebuild the BigQuery client with fresh credentials"""
        self.logger.info("Refreshing BigQuery client connection")
        self.client = self._build_client()
        return self.client

    def get_healthy_client(self):
        """Get a healthy BigQuery client, refreshing if necessary"""
        if not self._is_client_healthy():
            self.logger.info("Client unhealthy, refreshing connection")
            self._refresh_client()
        return self.client
    
    def _sanitize_cell_value(self, value):
        """
        Comprehensive sanitization of individual cell values for BigQuery
        
        Parameters:
        value: Any value from a DataFrame cell
        
        Returns:
        A value suitable for BigQuery
        """
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
    
    def _sanitize_column_name(self, col_name):
        """
        Sanitize column names for BigQuery compatibility
        
        Parameters:
        col_name: Original column name
        
        Returns:
        BigQuery-compatible column name
        """
        # Convert to string
        clean_name = str(col_name)
        
        # BigQuery column names must start with letter or underscore
        if not clean_name[0].isalpha() and clean_name[0] != '_':
            clean_name = '_' + clean_name
        
        # Replace invalid characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        
        # Remove consecutive underscores
        clean_name = re.sub(r'_{2,}', '_', clean_name)
        
        # Remove trailing underscores
        clean_name = clean_name.rstrip('_')
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = 'column_'
        
        # BigQuery has a 300 character limit for column names
        if len(clean_name) > 300:
            clean_name = clean_name[:297] + "___"
        
        return clean_name
    
    def _validate_and_clean_dataframe(self, df):
        """
        Comprehensive DataFrame validation and cleaning for BigQuery
        
        Parameters:
        df: pandas DataFrame to clean
        
        Returns:
        Cleaned pandas DataFrame
        """
        self.logger.info(f"Starting DataFrame validation and cleaning - Shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Clean column names for BigQuery compatibility
        original_columns = df_clean.columns.tolist()
        clean_columns = [self._sanitize_column_name(col) for col in original_columns]
        
        # Handle duplicate column names
        seen_columns = {}
        final_columns = []
        for col in clean_columns:
            if col in seen_columns:
                seen_columns[col] += 1
                final_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                final_columns.append(col)
        
        df_clean.columns = final_columns
        
        if final_columns != original_columns:
            self.logger.info("Cleaned column names for BigQuery compatibility")
        
        # Apply sanitization to all values
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(self._sanitize_cell_value)
        
        # Log data quality info
        nan_counts = df_clean.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            self.logger.info(f"Found {total_nans} null values across {(nan_counts > 0).sum()} columns")
        
        self.logger.info("DataFrame validation completed")
        return df_clean
    
    def _get_table_reference(self, dataset_id, table_id):
        """Get BigQuery table reference"""
        return self.client.dataset(dataset_id).table(table_id)
    
    def _create_dataset_if_not_exists(self, dataset_id):
        """Create dataset if it doesn't exist"""
        try:
            self.client.get_dataset(dataset_id)
            self.logger.info(f"Dataset {dataset_id} already exists")
        except Exception:
            self.logger.info(f"Creating dataset {dataset_id}")
            dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
            dataset.location = "US"  # You can change this based on your needs
            self.client.create_dataset(dataset, timeout=30)
            self.logger.info(f"Dataset {dataset_id} created successfully")
    
    def append(self, df, dataset_id, table_id, create_dataset=True):
        """
        Appends pandas DataFrame data to a BigQuery table
        
        Parameters:
        df (pandas.DataFrame): DataFrame to append to the table
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID
        create_dataset (bool): Whether to create dataset if it doesn't exist
        
        Returns:
        google.cloud.bigquery.job.LoadJob: The completed load job
        """
        self.logger.info(f"Starting append operation - Dataset: {dataset_id}, Table: {table_id}")
        
        try:
            # Create dataset if needed
            if create_dataset:
                self._create_dataset_if_not_exists(dataset_id)
            
            # Clean the DataFrame
            df_clean = self._validate_and_clean_dataframe(df)
            
            # Get table reference
            table_ref = self._get_table_reference(dataset_id, table_id)
            
            # Configure the load job
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
                autodetect=True  # Auto-detect schema
            )
            
            # Load data to BigQuery
            self.logger.info(f"Loading {len(df_clean)} rows to BigQuery")
            job = self.client.load_table_from_dataframe(
                df_clean, 
                table_ref, 
                job_config=job_config
            )
            
            # Wait for the job to complete
            job.result()
            
            # Get the updated table info
            table = self.client.get_table(table_ref)
            self.logger.info(f"Append completed - Total rows in table: {table.num_rows}")
            
            return job
            
        except Exception as e:
            self.logger.error(f"Error appending data: {str(e)}")
            raise
    
    def replace(self, df, dataset_id, table_id, create_dataset=True):
        """
        Replaces all data in a BigQuery table
        
        Parameters:
        df (pandas.DataFrame): DataFrame to write to the table
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID
        create_dataset (bool): Whether to create dataset if it doesn't exist
        
        Returns:
        google.cloud.bigquery.job.LoadJob: The completed load job
        """
        self.logger.info(f"Starting replace operation - Dataset: {dataset_id}, Table: {table_id}")
        
        try:
            # Create dataset if needed
            if create_dataset:
                self._create_dataset_if_not_exists(dataset_id)
            
            # Clean the DataFrame
            df_clean = self._validate_and_clean_dataframe(df)
            
            # Get table reference
            table_ref = self._get_table_reference(dataset_id, table_id)
            
            # Configure the load job to truncate (replace) existing data
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
                autodetect=True  # Auto-detect schema
            )
            
            # Load data to BigQuery
            self.logger.info(f"Replacing table with {len(df_clean)} rows")
            job = self.client.load_table_from_dataframe(
                df_clean, 
                table_ref, 
                job_config=job_config
            )
            
            # Wait for the job to complete
            job.result()
            
            # Get the updated table info
            table = self.client.get_table(table_ref)
            self.logger.info(f"Replace completed - Total rows in table: {table.num_rows}")
            
            return job
            
        except Exception as e:
            self.logger.error(f"Error replacing data: {str(e)}")
            raise
    
    def read(self, dataset_id, table_id, query=None, limit=None, use_db_dtypes=True):
        """
        Reads data from a BigQuery table and returns a pandas DataFrame
        
        Parameters:
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID
        query (str, optional): Custom SQL query. If provided, dataset_id and table_id are ignored
        limit (int, optional): Maximum number of rows to return
        use_db_dtypes (bool): Whether to use db-dtypes package for type conversion
        
        Returns:
        pandas.DataFrame: DataFrame with the table data
        """
        self.logger.info(f"Starting read operation - Dataset: {dataset_id}, Table: {table_id}")
        
        try:
            if query:
                # Use custom query
                sql_query = query
                self.logger.info("Using custom query")
            else:
                # Build query from dataset and table
                sql_query = f"""
                SELECT *
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                """
                
                if limit:
                    sql_query += f" LIMIT {limit}"
            
            self.logger.info(f"Executing query: {sql_query}")
            
            # Execute query
            query_job = self.client.query(sql_query)
            
            try:
                # Try to use to_dataframe() with db-dtypes
                if use_db_dtypes:
                    df = query_job.to_dataframe()
                else:
                    raise ValueError("Skipping db-dtypes")
            except (ValueError, ImportError) as e:
                if "db-dtypes" in str(e) or not use_db_dtypes:
                    self.logger.warning("db-dtypes package not available, using alternative method")
                    # Alternative: Convert to list of dictionaries first, then to DataFrame
                    results = query_job.result()
                    rows = []
                    for row in results:
                        # Convert Row to dictionary, handling special types
                        row_dict = {}
                        for key, value in row.items():
                            # Convert datetime objects to strings if needed
                            if hasattr(value, 'isoformat'):
                                row_dict[key] = value.isoformat()
                            else:
                                row_dict[key] = value
                        rows.append(row_dict)
                    
                    df = pd.DataFrame(rows)
                else:
                    raise e
            
            self.logger.info(f"Read operation completed - DataFrame shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading data: {str(e)}")
            raise
    
    def execute_query(self, query, use_storage_api=True):
        """Execute query with connection health check"""
        try:
            # Ensure we have a healthy client
            client = self.get_healthy_client()
            
            job_config = bigquery.QueryJobConfig(use_query_cache=True)
            query_job = client.query(query, job_config=job_config)
            
            if use_storage_api:
                try:
                    df = query_job.to_dataframe(create_bqstorage_client=True)
                except Exception as e:
                    self.logger.warning(f"Storage API failed, using standard API: {e}")
                    df = query_job.to_dataframe()
            else:
                df = query_job.to_dataframe()
                
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    def list_tables(self, dataset_id):
        """
        List all tables in a dataset
        
        Parameters:
        dataset_id (str): BigQuery dataset ID
        
        Returns:
        list: List of table IDs
        """
        try:
            dataset_ref = self.client.dataset(dataset_id)
            tables = list(self.client.list_tables(dataset_ref))
            table_ids = [table.table_id for table in tables]
            
            self.logger.info(f"Found {len(table_ids)} tables in dataset {dataset_id}")
            return table_ids
            
        except Exception as e:
            self.logger.error(f"Error listing tables: {str(e)}")
            raise

    def read_fast(self, dataset_id, table_id, query=None):
        """
        Fast read using BigQuery Storage API only
        
        Parameters:
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID  
        query (str, optional): Custom SQL query
        
        Returns:
        pandas.DataFrame: DataFrame with the table data
        """
        try:
            if query:
                sql_query = query
            else:
                sql_query = f"SELECT * FROM `{self.project_id}.{dataset_id}.{table_id}`"
            
            job = self.client.query(sql_query)
            return job.to_dataframe(create_bqstorage_client=True)
            
        except Exception as e:
            # Fallback to standard API if Storage API fails
            self.logger.warning(f"Storage API failed, using standard API: {e}")
            job = self.client.query(sql_query)
            return job.to_dataframe()