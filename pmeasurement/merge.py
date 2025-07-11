import pandas as pd
import os

"""sacar de byma data el historico de Titulos Publicos y Letras para hacer el merger"""

# Path to the Excel file
excel_path = os.path.join(os.path.dirname(__file__), 'historicalsBD.xlsx')

# Read all sheets
all_sheets = pd.read_excel(excel_path, sheet_name=None)

# Use GD30D (or first sheet) for master date column
master_asset = 'GD30D' if 'GD30D' in all_sheets else list(all_sheets.keys())[0]
master_df = all_sheets[master_asset][['FECHA']].copy()
master_df['FECHA'] = pd.to_datetime(master_df['FECHA'])

# Prepare the merged DataFrame
merged_df = master_df.copy()

missing_fecha_cierre_assets = []
for asset, df in all_sheets.items():
    if 'FECHA' in df.columns and 'CIERRE' in df.columns:
        temp = df[['FECHA', 'CIERRE']].copy()
        temp['FECHA'] = pd.to_datetime(temp['FECHA'])
        temp = temp.rename(columns={'CIERRE': asset})
        merged_df = pd.merge(merged_df, temp, on='FECHA', how='left')
    else:
        print(f"Sheet '{asset}' missing FECHA or CIERRE columns, adding column with nulls.")
        merged_df[asset] = pd.NA
        missing_fecha_cierre_assets.append(asset)

# Save to new Excel file
output_path = os.path.join(os.path.dirname(__file__), 'historicalsBD_merged.xlsx')

# --- Add missing asset data from letras.xlsx ---
letras_path = os.path.join(os.path.dirname(__file__), 'letras.xlsx')
if os.path.exists(letras_path):
    letras_df = pd.read_excel(letras_path)
    # Only search for assets missing FECHA or CIERRE columns
    for asset in missing_fecha_cierre_assets:
        asset_rows = letras_df[letras_df['SIMBOLO'] == asset]
        if not asset_rows.empty:
            temp = asset_rows[['FECHA', 'CIERRE']].copy()
            # Ensure both FECHA columns are datetime for correct merge
            temp['FECHA'] = pd.to_datetime(temp['FECHA'])
            merged_df['FECHA'] = pd.to_datetime(merged_df['FECHA'])
            temp = temp.rename(columns={'CIERRE': asset})
            merged_df = pd.merge(merged_df, temp, on='FECHA', how='left', suffixes=('', '_letras'))
            # If both columns exist (from previous nulls), fill nulls with letras data
            if asset + '_letras' in merged_df.columns:
                merged_df[asset] = merged_df[asset].combine_first(merged_df[asset + '_letras'])
                merged_df.drop(columns=[asset + '_letras'], inplace=True)
        else:
            print(f"Asset '{asset}' not found in letras.xlsx.")

# Drop columns (except FECHA) that have all null values
cols_to_drop = [col for col in merged_df.columns if col != 'FECHA' and merged_df[col].isna().all()]
if cols_to_drop:
    print(f"Dropping columns with all nulls: {cols_to_drop}")
    merged_df.drop(columns=cols_to_drop, inplace=True)

merged_df.to_excel(output_path, index=False)
