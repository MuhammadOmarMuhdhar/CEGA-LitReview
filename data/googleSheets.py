import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
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

class API:
    def __init__(self, credentials_json):
        """
        Initialize the GoogleSheeAPI
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Google Sheets API")
        
        self.credentials_json = credentials_json
        self.service = self._build_service()
        # Set batch size (Google Sheets has a limit of 10MB per request)
        self.batch_size = 1000  
        
        self.logger.info(f"API initialized with batch size: {self.batch_size}")
    
    def _build_service(self):
        """Build and return the Google Sheets API service"""
        self.logger.info("Building Google Sheets service")
        
        credentials = service_account.Credentials.from_service_account_info(
            self.credentials_json,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        service = build('sheets', 'v4', credentials=credentials)
        self.logger.info("Google Sheets service built successfully")
        return service
    
    def _sanitize_cell_value(self, value):
        """
        Comprehensive sanitization of individual cell values for Google Sheets API
        
        Parameters:
        value: Any value from a DataFrame cell
        
        Returns:
        A JSON-serializable value suitable for Google Sheets API
        """
        # Handle None/NaN/null values
        if value is None or pd.isna(value):
            return ""
        
        # Handle numpy NaN specifically
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return ""
        
        # Handle numpy integers
        if isinstance(value, np.integer):
            return int(value)
        
        # Handle numpy floats
        if isinstance(value, np.floating):
            if np.isnan(value) or np.isinf(value):
                return ""
            return float(value)
        
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            try:
                # Convert to list first, then sanitize each element
                as_list = value.tolist()
                return self._sanitize_list_or_array(as_list)
            except:
                return str(value)
        
        # Handle Python lists
        if isinstance(value, list):
            return self._sanitize_list_or_array(value)
        
        # Handle dictionaries
        if isinstance(value, dict):
            try:
                # Convert dict to a readable string representation
                return json.dumps(value, default=str, ensure_ascii=False)
            except:
                return str(value)
        
        # Handle datetime objects
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        
        # Handle pandas Timestamp
        if hasattr(value, 'isoformat'):
            try:
                return value.isoformat()
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
        
        # Convert to string and clean
        try:
            str_value = str(value)
            
            # Remove control characters that might break JSON
            str_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str_value)
            
            # Handle very long strings (Google Sheets has a 50,000 character limit per cell)
            if len(str_value) > 50000:
                str_value = str_value[:49997] + "..."
                self.logger.warning(f"Truncated long string value (original length: {len(str(value))})")
            
            return str_value
            
        except Exception as e:
            self.logger.warning(f"Failed to convert value {type(value)} to string: {e}")
            return ""
    
    def _sanitize_list_or_array(self, lst):
        """
        Convert lists/arrays to comma-separated string, handling nested structures
        
        Parameters:
        lst: List or array to convert
        
        Returns:
        String representation of the list
        """
        try:
            # Recursively sanitize each element
            sanitized_items = []
            for item in lst:
                if isinstance(item, (list, np.ndarray)):
                    # Handle nested lists/arrays
                    sanitized_items.append(f"[{self._sanitize_list_or_array(item)}]")
                else:
                    sanitized_item = self._sanitize_cell_value(item)
                    sanitized_items.append(str(sanitized_item))
            
            result = ", ".join(sanitized_items)
            
            # Limit length to prevent Google Sheets cell limit issues
            if len(result) > 50000:
                result = result[:49997] + "..."
                self.logger.warning("Truncated long list representation")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to convert list to string: {e}")
            return str(lst)
    
    def _validate_and_clean_dataframe(self, df):
        """
        Comprehensive DataFrame validation and cleaning
        
        Parameters:
        df: pandas DataFrame to clean
        
        Returns:
        Cleaned pandas DataFrame
        """
        self.logger.info(f"Starting DataFrame validation and cleaning - Shape: {df.shape}")
        
        df_clean = df.copy()
        
        # Clean column names
        original_columns = df_clean.columns.tolist()
        clean_columns = []
        
        for col in original_columns:
            # Convert column name to string and clean
            clean_col = str(col)
            # Remove control characters
            clean_col = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_col)
            # Limit length
            if len(clean_col) > 100:
                clean_col = clean_col[:97] + "..."
            clean_columns.append(clean_col)
        
        df_clean.columns = clean_columns
        
        if clean_columns != original_columns:
            self.logger.info("Cleaned column names")
        
        # Check for problematic data types and log warnings
        problematic_dtypes = []
        for col in df_clean.columns:
            dtype = df_clean[col].dtype
            if dtype == 'object':
                # Check what types of objects we have
                sample_types = df_clean[col].dropna().apply(type).value_counts()
                if len(sample_types) > 1:
                    problematic_dtypes.append((col, sample_types))
        
        if problematic_dtypes:
            self.logger.warning(f"Found columns with mixed data types: {[col for col, _ in problematic_dtypes]}")
        
        # Count and log data quality issues
        nan_counts = df_clean.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            self.logger.info(f"Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
            top_nan_cols = nan_counts[nan_counts > 0].head()
            self.logger.debug(f"Columns with most NaN values: {top_nan_cols.to_dict()}")
        
        self.logger.info("DataFrame validation completed")
        return df_clean
    
    def _prepare_data(self, df, include_headers):
        """
        Prepare DataFrame for Google Sheets API with comprehensive cleaning
        
        Parameters:
        df: pandas DataFrame to prepare
        include_headers: bool, whether to include column headers
        
        Returns:
        List of lists ready for Google Sheets API
        """
        self.logger.info(f"Preparing data - DataFrame shape: {df.shape}, include_headers: {include_headers}")
        
        # First, validate and clean the DataFrame
        df_clean = self._validate_and_clean_dataframe(df)
        
        # Apply comprehensive sanitization to all values
        self.logger.info("Applying comprehensive data sanitization...")
        
        # Process data row by row to handle large datasets efficiently
        processed_rows = []
        
        # Add headers if requested
        if include_headers:
            header_row = [self._sanitize_cell_value(col) for col in df_clean.columns]
            processed_rows.append(header_row)
        
        # Process data rows
        total_rows = len(df_clean)
        batch_size_for_processing = 1000  # Process in smaller batches to avoid memory issues
        
        for start_idx in range(0, total_rows, batch_size_for_processing):
            end_idx = min(start_idx + batch_size_for_processing, total_rows)
            batch_df = df_clean.iloc[start_idx:end_idx]
            
            self.logger.debug(f"Processing rows {start_idx} to {end_idx-1}")
            
            for _, row in batch_df.iterrows():
                processed_row = [self._sanitize_cell_value(cell) for cell in row]
                processed_rows.append(processed_row)
        
        # Final validation - ensure all data is JSON serializable
        self.logger.info("Performing final JSON serialization validation...")
        
        try:
            # Test serialize a sample of the data
            sample_size = min(10, len(processed_rows))
            test_data = processed_rows[:sample_size]
            json.dumps(test_data, default=str)
            self.logger.info("JSON serialization validation passed")
        except Exception as e:
            self.logger.error(f"JSON serialization validation failed: {e}")
            raise ValueError(f"Data still contains non-serializable values after cleaning: {e}")
        
        self.logger.info(f"Data preparation completed - Total rows: {len(processed_rows)}")
        return processed_rows
    
    def _batch_update(self, values, spreadsheet_id, sheet_name, start_row=1):
        """Execute batched update operations with enhanced error handling"""
        self.logger.info(f"Starting batch update - Sheet: {sheet_name}, Start row: {start_row}, Rows to update: {len(values)}")
        
        try:
            num_rows = len(values)
            num_cols = len(values[0]) if values else 0
            
            # Calculate the range
            end_row = start_row + num_rows - 1
            range_to_update = f'{sheet_name}!A{start_row}:{self._col_num_to_letter(num_cols)}{end_row}'
            
            self.logger.debug(f"Update range: {range_to_update}")
            
            # Additional validation before sending
            if not values:
                self.logger.warning("No values to update")
                return None
            
            body = {
                'values': values
            }
            
            # Validate the body can be serialized
            try:
                json.dumps(body, default=str)
            except Exception as e:
                self.logger.error(f"Body serialization failed: {e}")
                # Log problematic data
                for i, row in enumerate(values[:3]):  # Log first 3 rows
                    for j, cell in enumerate(row):
                        try:
                            json.dumps(cell)
                        except:
                            self.logger.error(f"Problematic cell at row {i}, col {j}: {type(cell)} = {repr(cell)}")
                raise
            
            result = self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_to_update,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            cells_updated = result.get('updatedCells', 0)
            self.logger.info(f"Batch update completed - Cells updated: {cells_updated}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch update: {str(e)}")
            # Enhanced error logging
            if "Invalid JSON payload" in str(e):
                self.logger.error("JSON payload error detected. This usually means data contains non-serializable values.")
                # Log sample of the data being sent
                sample_data = values[:2] if len(values) > 0 else []
                self.logger.error(f"Sample data: {sample_data}")
            return None
    
    def _col_num_to_letter(self, col_num):
        """Convert column number to letter (1->A, 27->AA, etc.)"""
        string = ""
        while col_num > 0:
            col_num, remainder = divmod(col_num - 1, 26)
            string = chr(65 + remainder) + string
        return string
    
    def append(self, df, spreadsheet_id, sheet_name='Sheet1', include_headers=False):
        """
        Appends pandas DataFrame data to a Google Sheet with batching and comprehensive cleaning
        
        Parameters:
        df (pandas.DataFrame): DataFrame to append to the sheet
        spreadsheet_id (str): The ID of the spreadsheet
        sheet_name (str, optional): Name of the sheet. Defaults to 'Sheet1'.
        include_headers (bool, optional): Whether to include column headers. Defaults to False.
        
        Returns:
        list: Results of all batch append operations
        """
        self.logger.info(f"Starting append operation - Spreadsheet: {spreadsheet_id}, Sheet: {sheet_name}")
        
        try:
            converted_data = self._prepare_data(df, include_headers)
            
            # Process data in batches
            results = []
            total_rows = len(converted_data)
            self.logger.info(f"Total rows to append: {total_rows}")
            
            for i in range(0, total_rows, self.batch_size):
                batch_num = i // self.batch_size + 1
                batch_data = converted_data[i:i + self.batch_size]
                
                self.logger.info(f"Processing batch {batch_num} - Rows {i} to {min(i + self.batch_size, total_rows)}")
                
                # Create the request body for this batch
                body = {
                    'values': batch_data
                }
                
                # Execute the append request for this batch
                result = self.service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_name,
                    valueInputOption='USER_ENTERED',
                    insertDataOption='INSERT_ROWS',
                    body=body
                ).execute()
                
                rows_appended = result.get('updates', {}).get('updatedRows', 0)
                self.logger.info(f"Batch {batch_num}: Rows appended: {rows_appended}")
                results.append(result)
                
                # Add a small delay to avoid hitting rate limits
                if i + self.batch_size < total_rows:
                    self.logger.debug("Adding delay to avoid rate limits")
                    time.sleep(0.5)
            
            self.logger.info(f"Append operation completed - Total rows appended: {total_rows}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error appending data: {str(e)}")
            return None
    
    def replace(self, df, spreadsheet_id, sheet_name='Sheet1', include_headers=True):
        """
        Replaces all data in a Google Sheet with batching support and comprehensive cleaning
        
        Parameters:
        df (pandas.DataFrame): DataFrame to write to the sheet
        spreadsheet_id (str): The ID of the spreadsheet
        sheet_name (str, optional): Name of the sheet. Defaults to 'Sheet1'.
        include_headers (bool, optional): Whether to include column headers. Defaults to True.
        
        Returns:
        list: Results of all batch update operations
        """
        self.logger.info(f"Starting replace operation - Spreadsheet: {spreadsheet_id}, Sheet: {sheet_name}")
        
        try:
            # First, check if the sheet exists
            self.logger.info("Checking if sheet exists")
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            
            sheet_exists = False
            sheet_id = None
            
            for sheet in spreadsheet.get('sheets', []):
                if sheet['properties']['title'] == sheet_name:
                    sheet_exists = True
                    sheet_id = sheet['properties']['sheetId']
                    break
            
            # If sheet doesn't exist, create it
            if not sheet_exists:
                self.logger.info(f"Sheet '{sheet_name}' not found, creating new sheet")
                
                create_sheet_body = {
                    'requests': [{
                        'addSheet': {
                            'properties': {
                                'title': sheet_name
                            }
                        }
                    }]
                }
                
                batch_update_result = self.service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body=create_sheet_body
                ).execute()
                
                self.logger.info(f"Created new sheet: {sheet_name}")
                
                # Get the newly created sheet ID
                for reply in batch_update_result.get('replies', []):
                    if 'addSheet' in reply:
                        sheet_id = reply['addSheet']['properties']['sheetId']
                        break
            else:
                self.logger.info(f"Sheet '{sheet_name}' exists")
            
            # Clear all content from the sheet
            self.logger.info("Clearing existing content from sheet")
            clear_request = self.service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}',
                body={}
            )
            clear_request.execute()
            self.logger.info("Sheet cleared")
            
            # Prepare data with comprehensive cleaning
            converted_data = self._prepare_data(df, include_headers)
            
            # Process data in batches
            results = []
            total_rows = len(converted_data)
            start_row = 1
            self.logger.info(f"Total rows to replace: {total_rows}")
            
            for i in range(0, total_rows, self.batch_size):
                batch_num = i // self.batch_size + 1
                batch_data = converted_data[i:i + self.batch_size]
                
                self.logger.info(f"Processing batch {batch_num} - Rows {i} to {min(i + self.batch_size, total_rows)}")
                
                # Execute batch update
                result = self._batch_update(
                    batch_data, 
                    spreadsheet_id, 
                    sheet_name, 
                    start_row
                )
                
                if result:
                    cells_updated = result.get('updatedCells', 0)
                    self.logger.info(f"Batch {batch_num}: Cells updated: {cells_updated}")
                    results.append(result)
                    start_row += len(batch_data)
                else:
                    self.logger.error(f"Batch {batch_num} failed")
                
                # Add a small delay to avoid hitting rate limits
                if i + self.batch_size < total_rows:
                    self.logger.debug("Adding delay to avoid rate limits")
                    time.sleep(0.5)
            
            self.logger.info(f"Replace operation completed - Total rows replaced: {total_rows}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")
            return None
        
    def read(self, spreadsheet_id, sheet_range='Sheet1', header_row=0):
        """
        Reads data from a Google Sheet and returns a pandas DataFrame
        
        Parameters:
        spreadsheet_id (str): The ID of the spreadsheet
        sheet_range (str): Range to read (e.g., 'Sheet1!A1:F10' or just 'Sheet1')
        header_row (int): Which row to use as the header (0-based index)
        
        Returns:
        pandas.DataFrame: DataFrame with the sheet data
        """
        self.logger.info(f"Starting read operation - Spreadsheet: {spreadsheet_id}, Range: {sheet_range}, Header row: {header_row}")
        
        try:
            # Execute the request to get values
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=sheet_range
            ).execute()
            
            # Extract values from the result
            values = result.get('values', [])
            
            if not values:
                self.logger.warning("No data found in sheet")
                return pd.DataFrame()
            
            self.logger.info(f"Data retrieved - Rows: {len(values)}, Columns: {len(values[0]) if values else 0}")
                
            # Convert to DataFrame
            if header_row >= 0 and header_row < len(values):
                # Use specified row as header
                headers = values[header_row]
                data = values[header_row + 1:]
                df = pd.DataFrame(data, columns=headers)
                self.logger.info(f"Created DataFrame with headers from row {header_row}")
            else:
                # No headers
                df = pd.DataFrame(values)
                self.logger.info("Created DataFrame without headers")
            
            self.logger.info(f"Read operation completed - DataFrame shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading data: {str(e)}")
            return pd.DataFrame()