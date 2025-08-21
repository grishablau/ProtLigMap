import sys
import pandas as pd
import requests
from loguru import logger
from src.fetch_preproc_data.setup_db import add_column_to_table
import mysql.connector
from src.utils.file_system import update_settings
import asyncio
import aiohttp


def filter_invalid_rows(df):
    """Filters out rows where 'kinase_name' or 'protein_sequence' is invalid in a pandas DataFrame."""

    # List of invalid protein sequence values
    invalid_protein_sequences = [
        "Assay data not found",
        "No target ID found",
        "Target data not found",
        "No numeric target ID found",
        "Target component data not found",
        "No protein sequence found"
    ]

    # Apply filter conditions
    filtered_df = df[
        ~df["kinase_name"].isin([None, "Unknown"]) &
        ~df["protein_sequence"].isin(invalid_protein_sequences)
        ]

    return filtered_df


async def fetch_assay_data(session, assay_id, kinase_map, assay_url, target_url, target_component_url):
    """Fetches the assay data and target data for a given assay ID."""

    assay_url = f"{assay_url}{assay_id}.json"
    retries = 5  # Number of retries
    delay = 2  # Initial delay in seconds

    for attempt in range(retries):
        try:
            async with session.get(assay_url, timeout=30) as response:
                if response.status == 200:
                    assay_data = await response.json()
                    target_id = assay_data.get("target_chembl_id", None)

                    if target_id:
                        # Fetch the target data
                        target_url = f"{target_url}{target_id}.json"
                        async with session.get(target_url, timeout=30) as target_response:
                            if target_response.status == 200:
                                target_data = await target_response.json()

                                # Fetch the kinase name from target data (pref_name)
                                kinase_map[assay_id] = target_data.get("pref_name", "Unknown")

                                # Extract target_numeric_id
                                try:
                                    target_numeric_id = target_data['target_components'][0]['component_id']
                                except (KeyError, IndexError) as e:
                                    target_numeric_id = None
                                    logger.error(f"Error extracting target_numeric_id for assay {assay_id}: {e}")
                                if target_numeric_id:
                                    # Now query the target component endpoint for protein sequence
                                    target_component_url = f"{target_component_url}{target_numeric_id}.json"
                                    async with session.get(target_component_url, timeout=30) as component_response:
                                        if component_response.status == 200:
                                            component_data = await component_response.json()

                                            # Extract the protein sequence from the component data
                                            protein_sequence = component_data.get('sequence',
                                                                                  "No protein sequence found")

                                            # Update kinase_map with both kinase name and protein sequence
                                            kinase_map[assay_id] = {
                                                "kinase_name": kinase_map[assay_id],
                                                "protein_sequence": protein_sequence
                                            }
                                        else:
                                            kinase_map[assay_id] = {"kinase_name": kinase_map[assay_id],
                                                                    "protein_sequence": "Target component data not found"}
                                else:
                                    logger.info(f"assay_url: {assay_url}")
                                    logger.info(f"target_data: {target_data}")
                                    kinase_map[assay_id] = {"kinase_name": kinase_map[assay_id],
                                                            "protein_sequence": "No numeric target ID found"}
                            else:
                                kinase_map[assay_id] = {"kinase_name": kinase_map[assay_id],
                                                        "protein_sequence": "Target data not found"}
                    else:
                        kinase_map[assay_id] = {"kinase_name": "Unknown", "protein_sequence": "No target ID found"}
                else:
                    kinase_map[assay_id] = {"kinase_name": "Unknown", "protein_sequence": "Assay data not found"}

                return  # Exit on successful request
        except asyncio.TimeoutError:
            logger.error(f"Timeout occurred for assay ID {assay_id}. Retrying...")
        except aiohttp.ClientError as e:
            logger.error(f"Request failed for assay ID {assay_id}: {str(e)}. Retrying...")

        await asyncio.sleep(delay)  # Non-blocking delay
        delay *= 2  # Exponential backoff


# Fetching data from ChEMBL API asynchronously
async def get_kinase_info_bulk(assay_ids, assay_url, target_url, target_component_url):
    """Fetches kinase names for multiple assay IDs in bulk using async calls."""
    kinase_map = {}
    async with aiohttp.ClientSession() as session:
        tasks = []

        for assay_id in assay_ids:
            tasks.append(fetch_assay_data(session, assay_id, kinase_map, assay_url, target_url, target_component_url))

        # Run all tasks asynchronously
        await asyncio.gather(*tasks)

    return kinase_map


def fetch_and_process_data(input_dict):
    if 'settings_result_save' not in input_dict:
        input_dict['settings_result_save'] = False
    if 'settings_result_return' not in input_dict:
        input_dict['settings_result_return'] = True

    input_dict = setup_settings(input_dict)
    """Fetches bioactivity data from ChEMBL and cleans the data before saving it"""

    (fetch_columns_names, max_records, activity_url, std_type_filter_by, min_ligand_per_kinase,
     num_kinase_with_min_ligands, limit, offset, assay_url, target_url, save_data_path, target_component_url) = (
        input_dict[k] for k in
        ("fetch_columns",
         "max_records",
         "activity_url",
         "std_type_filter_by",
         "min_ligand_per_kinase",
         "num_kinase_with_min_ligands",
         "limit", "offset",
         "assay_url",
         "target_url",
         "save_data_path",
         "target_component_url"
         ))
    fetch_columns_names = fetch_columns_names.split()
    std_type_filter_by = std_type_filter_by.split()

    top_kinase_ligands_amount = [0] * num_kinase_with_min_ligands

    # Initialise dataframe
    all_data = pd.DataFrame(columns=fetch_columns_names)

    while not (len(all_data) >= max_records):
        while not ((len(all_data) >= max_records) and all(
                value >= min_ligand_per_kinase for value in top_kinase_ligands_amount)):
            activity_url_final = activity_url.format(limit=limit, offset=offset)
            response = requests.get(activity_url_final)
            if response.status_code != 200:
                logger.error(f"Failed to fetch data (status code: {response.status_code})")
                break

            data = response.json()
            activities = data.get("activities", [])

            if not activities:
                break  # Stop when no more records are available

            df = pd.DataFrame(activities)[fetch_columns_names]
            filtered_df = df[df['standard_type'].isin(std_type_filter_by)]
            if not filtered_df.empty:
                all_data = pd.concat([all_data, filtered_df], ignore_index=True)
            most_common_assays = all_data['assay_chembl_id'].value_counts().head(2)
            top_kinase_ligands_amount = most_common_assays[0], most_common_assays[1]
            logger.info(f"Most common assays: {most_common_assays}")
            offset += limit
            logger.success(f"  Downloaded {len(all_data)} records...", end="\r")

            columns_to_check = [col for col in all_data.columns if col not in ['kinase_name', 'protein_sequence']]
            all_data = all_data.dropna(subset=columns_to_check)

            # The resulting filtered_data will have NaN values dropped, but 'kinase_name' and 'protein_sequence' columns will not be affected.

        # Get kinase names in bulk asynchronously
        assay_ids = list(set(all_data["assay_chembl_id"]))
        # Refactored (using await instead)

        loop = asyncio.get_event_loop()  # Get the existing event loop
        kinase_map = loop.run_until_complete(
            get_kinase_info_bulk(assay_ids, assay_url, target_url, target_component_url))
        # Add kinase names to DataFrame

        all_data["kinase_name"] = all_data["assay_chembl_id"].map(
            lambda x: kinase_map.get(x, {}).get("kinase_name", "Unknown"))
        all_data["protein_sequence"] = all_data["assay_chembl_id"].map(
            lambda x: kinase_map.get(x, {}).get("protein_sequence", "Unknown"))

        # Clean table from nan values

        all_data = filter_invalid_rows(all_data)

    if 'result_save' not in input_dict:
        input_dict['result_save'] = False
    else:
        if input_dict['result_save']:
            all_data.to_parquet(save_data_path, engine='pyarrow')
            logger.success(f"Data has been processed and saved to {save_data_path}")
    if 'result_return' not in input_dict or not input_dict['result_return']:
        return all_data


def create_database(input_dict):
    database_name, MYSQL_CONFIG = (input_dict[k] for k in ("database_name", "MYSQL_CONFIG"))
    try:

        # Connect to MySQL server without specifying a database
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        # Check if the database already exists
        cursor.execute(f"SHOW DATABASES LIKE '{database_name}'")
        result = cursor.fetchone()

        if result is None:
            # Create the database if it does not exist
            cursor.execute(f"CREATE DATABASE {database_name}")
            print(f"Database '{database_name}' created successfully!")
        else:
            print(f"Database '{database_name}' already exists.")

        # Close the cursor and connection
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error: {err}")


def setup_settings(input_dict):
    updated_settings = update_settings(input_dict)
    if 'settings_result_save' not in input_dict:
        input_dict['settings_result_save'] = True
    if input_dict['settings_result_save']:
        save_path = None
        if 'save_settings_path' in input_dict:
            if input_dict['save_settings_path'] is not None:
                save_path = input_dict['save_settings_path']
        if save_path is None:
            save_path = input_dict['input_file_path']
        with open(save_path, 'w') as python_file:
            python_file.write(f"processed_settings = {updated_settings}")
        logger.success(f"Settings have been processed and saved to {save_path}")
    elif input_dict['settings_result_save'] is not False:
        logger.error(f"Must input True or False value into 'settings_result_save' variable")
    if 'settings_result_return' not in input_dict:
        input_dict['settings_result_save'] = False
    if input_dict['settings_result_return']:
        return updated_settings
    elif input_dict['settings_result_return'] is not False:
        logger.error(f"Must input True or False value into 'settings_result_return' variable")


def add_column_params(input_dict):
    if input_dict['add_column_params']:
        add_column_to_table(input_dict['add_column_params'])
