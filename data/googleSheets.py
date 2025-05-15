import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import numpy as np

class API:
    def __init__(self, credentials_json):
        """
        Initialize the GoogleSheeAPI
        """
        self.credentials_json = credentials_json
        self.service = self._build_service()
    
    def _build_service(self):
        """Build and return the Google Sheets API service"""
        credentials = service_account.Credentials.from_service_account_info(
            self.credentials_json,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        return build('sheets', 'v4', credentials=credentials)
    
    def _prepare_data(self, df, include_headers):
        """Prepare DataFrame for Google Sheets API"""
        df_copy = df.copy()
        
        # Convert all list columns to strings
        for col in df_copy.columns:
            if df_copy[col].apply(lambda x: isinstance(x, list)).any():
                df_copy[col] = df_copy[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
        
        # Convert DataFrame to a list of lists
        if include_headers:
            data = [df_copy.columns.tolist()]  # Headers
            data.extend(df_copy.values.tolist())  # Data rows
        else:
            data = df_copy.values.tolist()  # Only data rows, no headers
        
        # Convert data to Python native types to ensure it's JSON serializable
        converted_data = []
        for row in data:
            converted_row = []
            for cell in row:
                # Convert numpy types to Python native types
                if isinstance(cell, np.integer):
                    cell = int(cell)
                elif isinstance(cell, np.floating):
                    cell = float(cell)
                elif isinstance(cell, np.ndarray):
                    cell = cell.tolist()
                # Convert any remaining lists to strings
                elif isinstance(cell, list):
                    cell = ', '.join(map(str, cell))
                converted_row.append(cell)
            converted_data.append(converted_row)
        
        return converted_data
    
    def append(self, df, spreadsheet_id, sheet_name='Sheet1', include_headers=False):
        """
        Appends pandas DataFrame data to a Google Sheet
        
        Parameters:
        df (pandas.DataFrame): DataFrame to append to the sheet
        sheet_name (str, optional): Name of the sheet. Defaults to 'Sheet1'.
        include_headers (bool, optional): Whether to include column headers. Defaults to False.
        
        Returns:
        dict: Result of the append operation or None if an error occurred
        """
        try:
            converted_data = self._prepare_data(df, include_headers)
            
            # Create the request body
            body = {
                'values': converted_data
            }
            range_to_append = f'{sheet_name}'
            
            # Execute the append request
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_to_append,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            print(f"Rows appended: {result.get('updates').get('updatedRows')}")
            return result
            
        except Exception as e:
            print(f"Error appending data: {str(e)}")
            return None
    
    def replace(self, df, spreadsheet_id, sheet_name='Sheet1', include_headers=True):
        """
        Replaces all data in a Google Sheet with the provided pandas DataFrame.
        Creates the sheet if it doesn't exist.
        
        Parameters:
        df (pandas.DataFrame): DataFrame to write to the sheet
        spreadsheet_id (str): The ID of the spreadsheet
        sheet_name (str, optional): Name of the sheet. Defaults to 'Sheet1'.
        include_headers (bool, optional): Whether to include column headers. Defaults to True.
        
        Returns:
        dict: Result of the update operation or None if an error occurred
        """
        try:
            # First, check if the sheet exists
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
                
                print(f"Created new sheet: {sheet_name}")
                
                # Get the newly created sheet ID
                for reply in batch_update_result.get('replies', []):
                    if 'addSheet' in reply:
                        sheet_id = reply['addSheet']['properties']['sheetId']
                        break
            
            converted_data = self._prepare_data(df, include_headers)
            
            # Clear all content from the sheet
            clear_request = self.service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}',
                body={}
            )
            clear_request.execute()
            
            # Create the update request body
            body = {
                'values': converted_data
            }
            
            range_to_update = f'{sheet_name}!A1'
            
            # Execute the update request
            result = self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_to_update,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            print(f"Cells updated: {result.get('updatedCells')}")
            return result
            
        except Exception as e:
            print(f"Error updating data: {str(e)}")
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
        try:
            # Execute the request to get values
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=sheet_range
            ).execute()
            
            # Extract values from the result
            values = result.get('values', [])
            
            if not values:
                print("No data found.")
                return pd.DataFrame()
                
            # Convert to DataFrame
            if header_row >= 0 and header_row < len(values):
                # Use specified row as header
                headers = values[header_row]
                data = values[header_row + 1:]
                df = pd.DataFrame(data, columns=headers)
            else:
                # No headers
                df = pd.DataFrame(values)
                
            return df
            
        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return pd.DataFrame()