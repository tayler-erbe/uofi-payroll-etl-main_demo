import os
import io
import pandas as pd


# ============================================================
# 1. AUTHENTICATE TO BOX
# ============================================================

from boxsdk import Client
from boxsdk.auth import DeveloperTokenAuth


developer_token = os.environ["BOX_DEVELOPER_TOKEN"]

auth = DeveloperTokenAuth(developer_token=developer_token)
client = Client(auth)

me = client.user().get()
print(f"Connected to Box as: {me.name} ({me.login})")


# ============================================================
# 2. RECURSIVE FILE LISTING
# ============================================================
def list_files_recursive(folder, parent_path=""):
    file_info = []

    for item in folder.get_items():
        full_path = os.path.join(parent_path, item.name)

        if item.type == "folder":
            file_info.extend(list_files_recursive(client.folder(item.id), full_path))

        elif item.type == "file":
            metadata = client.file(item.id).get()
            file_info.append({
                "file_name": item.name,
                "file_path": full_path,
                "extension": os.path.splitext(item.name)[1].lower(),
                "file_id": metadata.id,
                "size_bytes": metadata.size,
                "modified_at": metadata.modified_at,
                "created_at": metadata.created_at
            })

    return file_info


# ============================================================
# 3. LOAD PRIMARY CPA/PUA FILES FROM FOLDER 351817182894
# ============================================================
primary_folder_id = "351817182894"
primary_folder = client.folder(primary_folder_id)

print("\n--- Scanning Primary CPA/PUA Folder ---")
primary_files = list_files_recursive(primary_folder)
df_primary = pd.DataFrame(primary_files)

print(df_primary[["file_name", "extension", "file_id"]])

# Correct detection for PUA / CPA
pua_row = df_primary[
    df_primary["file_name"].str.contains("PUA", case=False, na=False)
    & df_primary["extension"].isin([".xlsx", ".xls"])
]

cpa_row = df_primary[
    df_primary["file_name"].str.contains(r"(^|\s|_)CPA(\s|_|$)", case=False, na=False)
    & df_primary["extension"].isin([".xlsx", ".xls"])
]

pua_df = None
cpa_df = None

# --- Load PUA Excel ---
if not pua_row.empty:
    pua_id = pua_row.iloc[0]["file_id"]
    print("\nPUA file found:", pua_row.iloc[0]["file_name"])
    pua_df = pd.read_excel(io.BytesIO(client.file(pua_id).content()))
else:
    print("\nWARNING: No valid PUA Excel file found.")

# --- Load CPA Excel ---
if not cpa_row.empty:
    cpa_id = cpa_row.iloc[0]["file_id"]
    print("CPA file found:", cpa_row.iloc[0]["file_name"])
    cpa_df = pd.read_excel(io.BytesIO(client.file(cpa_id).content()))
else:
    print("WARNING: No valid CPA Excel file found.")


# ============================================================
# 4. LOAD LOOKUP FILES FROM FOLDER 351829926330
# ============================================================
lookup_folder_id = "351829926330"
lookup_folder = client.folder(lookup_folder_id)

print("\n--- Scanning Lookup Folder ---")
lookup_files = list_files_recursive(lookup_folder)
df_lookup = pd.DataFrame(lookup_files)

print(df_lookup[["file_name", "extension", "file_id"]])


# ============================================================
# 5. LOAD 5 INDIVIDUAL LOOKUP CSVs
# ============================================================
lookup_targets = {
    "Overtime_E_Class.csv": "df_overtime_eclass",
    "Feeder_List.csv": "df_feeder_list",
    "TS_Dept.csv": "df_ts_dept",
    "TS_Org.csv": "df_ts_org",
    "TE_M.csv": "df_te_m"
}

lookup_dfs = {}

for csv_name, var_name in lookup_targets.items():
    row = df_lookup[df_lookup["file_name"].str.lower() == csv_name.lower()]

    if not row.empty:
        file_id = row.iloc[0]["file_id"]
        print(f"Loaded lookup file: {csv_name}")
        df_loaded = pd.read_csv(io.BytesIO(client.file(file_id).content()))
        lookup_dfs[var_name] = df_loaded
    else:
        print(f"WARNING: Lookup file missing: {csv_name}")

# Unpack DataFrames
df_overtime_eclass = lookup_dfs.get("df_overtime_eclass")
df_feeder_list = lookup_dfs.get("df_feeder_list")
df_ts_dept = lookup_dfs.get("df_ts_dept")
df_ts_org = lookup_dfs.get("df_ts_org")
df_te_m = lookup_dfs.get("df_te_m")


# ============================================================
# 6. LOAD PUA YTD FILE
# ============================================================
pua_ytd_row = df_lookup[df_lookup["file_name"].str.contains("YTD", case=False, na=False)]
df_pua_ytd = None

if not pua_ytd_row.empty:
    pua_ytd_id = pua_ytd_row.iloc[0]["file_id"]
    print("\nPUA YTD file found:", pua_ytd_row.iloc[0]["file_name"])
    df_pua_ytd = pd.read_excel(io.BytesIO(client.file(pua_ytd_id).content()))
else:
    print("\nWARNING: No PUA YTD file found.")


# ============================================================
# 7. LOAD BW + MN CERTIFICATION CSV FILES
# ============================================================
cert_bw_row = df_lookup[df_lookup["file_name"].str.contains("BW", case=False, na=False)]
cert_mn_row = df_lookup[df_lookup["file_name"].str.contains("MN", case=False, na=False)]

df_cert_bw = None
df_cert_mn = None

if not cert_bw_row.empty:
    file_id = cert_bw_row.iloc[0]["file_id"]
    print("\nLoaded BW certification CSV")
    df_cert_bw = pd.read_csv(io.BytesIO(client.file(file_id).content()))

if not cert_mn_row.empty:
    file_id = cert_mn_row.iloc[0]["file_id"]
    print("Loaded MN certification CSV")
    df_cert_mn = pd.read_csv(io.BytesIO(client.file(file_id).content()))


# ============================================================
# 8. FINAL SUMMARY
# ============================================================
print("\n========== LOAD SUMMARY ==========")
print("PUA:", "Loaded" if pua_df is not None else "Missing")
print("CPA:", "Loaded" if cpa_df is not None else "Missing")
print("PUA YTD:", "Loaded" if df_pua_ytd is not None else "Missing")

print("\nLookup Tables:")
print("Overtime E-Class:", "Loaded" if df_overtime_eclass is not None else "Missing")
print("Feeder List:", "Loaded" if df_feeder_list is not None else "Missing")
print("TS Dept:", "Loaded" if df_ts_dept is not None else "Missing")
print("TS Org:", "Loaded" if df_ts_org is not None else "Missing")
print("TE M:", "Loaded" if df_te_m is not None else "Missing")

print("\nCertification CSVs:")
print("BW Cert:", "Loaded" if df_cert_bw is not None else "Missing")
print("MN Cert:", "Loaded" if df_cert_mn is not None else "Missing")
print("==================================\n")

# ====== PUA ETL → PreTAM output ======
import os
import re
import pandas as pd
from collections import OrderedDict

# ----------------------------
# Utilities
# ----------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def strip_decimal_str(series: pd.Series) -> pd.Series:
    """Remove trailing '.0' from numeric-looking codes represented as strings."""
    s = series.astype(str).str.strip()
    return s.str.replace(r'\.0$', '', regex=True)

def mode_map(series: pd.Series) -> str:
    m = series.mode()
    return m.iat[0] if not m.empty else series.iloc[0]

def ensure_string(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

def safe_merge_left(df_left, df_right, **kwargs):
    """Left merge that keeps left row count; useful in sanity checks."""
    before = len(df_left)
    merged = df_left.merge(df_right, how='left', **kwargs)
    after = len(merged)
    if before != after:
        print(f"[warn] merge rowcount changed: {before} → {after}")
    return merged

# # ----------------------------
# # Normalize all column headers
# # ----------------------------
# for _df in [df_cpa_2025, df_pua_2025, df_pua_ytd, df_cert_bw, df_cert_mn,
#             df_overtime_eclass, df_feeder_list, df_ts_dept, df_ts_org, df_te_m]:
#     clean_column_names(_df)

# ----------------------------
# PUA working copy
# ----------------------------
df = pua_df .copy()

# --- Derived fields & cleaning ---
# TS-Org Code = TS COA + '-' + TS ORG
ensure_string(df, ['TS COA', 'TS ORG'])
df['TS-Org Code'] = df['TS COA'] + '-' + df['TS ORG']
# Ensure DEPT Code is string without '.0'
df['DEPT Code'] = strip_decimal_str(df['DEPT Code'])
# TS-Org Department Code = TS COA + '-' + DEPT Code
df['TS-Org Department Code'] = df['TS COA'] + '-' + df['DEPT Code']
# Department Name (keep original)
ensure_string(df, ['Department Name'])
# ECLS DESC → E-Class (for intermediate use; PreTAM final uses “E-Class” as description column)
df['E-Class'] = df['ECLS DESC'].astype(str).str.strip()

# Pay Event parts
ensure_string(df, ['Year', 'Pay ID', 'Pay #', 'Seq #'])
df['Pay Event'] = df['Year'] + df['Pay ID'] + df['Pay #'] + df['Seq #']

# Job Number = POSN + '-' + SUFF (strip '.0')
df['POSN'] = strip_decimal_str(df['POSN'])
df['SUFF'] = strip_decimal_str(df['SUFF'])
df['Job Number'] = df['POSN'] + '-' + df['SUFF']

# College = College Code + '-' + College Name
ensure_string(df, ['College Code', 'College Name'])
df['College'] = df['College Code'] + '-' + df['College Name']

# Normalize adjustment reason column names (handle variants)
df.rename(columns={
    'ADj Reason Code': 'ADJ Reason Code',
    'ADJ Reason DESC': 'ADJ Reason DESC',
    'Adj Reason Code': 'ADJ Reason Code',
    'Adj Reason': 'ADJ Reason DESC'
}, inplace=True)

# --- Lookups ---
# TS-Org Title from TS-Org Code
org_keep = df_ts_org[['TS-Org Code', 'TS-Org Title']].drop_duplicates()
df = safe_merge_left(df, org_keep, on='TS-Org Code')

# TS-Org Dept Title from TS-Org Dept Code; fallback to Department Name when missing
dept_keep = df_ts_dept[['TS-Org Dept Code', 'TS-Org Dept Title']].drop_duplicates()
df = safe_merge_left(
    df,
    dept_keep,
    left_on='TS-Org Department Code',
    right_on='TS-Org Dept Code'
)
df['TS-Org Dept Title'] = df['TS-Org Dept Title'].astype('string')
df['TS-Org Dept Title'] = df['TS-Org Dept Title'].fillna(df['Department Name'])

# Overtime via ECLS mapping
ot_keep = df_overtime_eclass[['Job Eclass', 'Overtime FLSA']].drop_duplicates()
df = safe_merge_left(df, ot_keep, left_on='ECLS', right_on='Job Eclass')
df.rename(columns={'Overtime FLSA': 'Overtime'}, inplace=True)
df.drop(columns=['Job Eclass'], errors='ignore', inplace=True)

# Time Entry via TE M (mode per code)
te = (
    df_te_m[['TE M', 'Time Entry Method']]
      .dropna(subset=['TE M', 'Time Entry Method'])
      .assign(**{
          'TE M': lambda d: d['TE M'].astype(str).str.strip(),
          'Time Entry Method': lambda d: d['Time Entry Method'].astype(str).str.strip()
      })
)
te_map = te.groupby('TE M')['Time Entry Method'].agg(mode_map).to_dict()
df['TE M'] = df['TE M'].astype(str).str.strip()
df['Time Entry'] = df.get('Time Entry', pd.Series(pd.NA, index=df.index, dtype='string'))
df['Time Entry'] = df['Time Entry'].astype('string').str.strip()
mapped_time_entry = df['TE M'].map(te_map).astype('string')
df['Time Entry'] = df['Time Entry'].where(df['Time Entry'].notna() & (df['Time Entry'] != ''), mapped_time_entry)

# --- Final column selection (keep sources needed to build PreTAM layout) ---
source_fields = [
    'UIN', 'Pay ID', 'Year', 'Pay #', 'Seq #', 'Job Number',
    'College Code', 'College Name', 'College',
    'TS COA', 'TS ORG', 'TS-Org Code', 'TS-Org Title',
    'DEPT Code', 'TS-Org Dept Code', 'TS-Org Department Code', 'TS-Org Dept Title',
    'ECLS', 'ECLS DESC', 'E-Class', 'TE M', 'Time Entry', 'Overtime',
    'Earn Code', 'DESCRIPTION', 'ADJ Reason Code', 'ADJ Reason DESC',
    'Calc Date', 'Pay Event', 'POSN', 'SUFF'
]
source_fields = [c for c in source_fields if c in df.columns]
df_fin = df[source_fields].copy()

# Deduplicate (business key)
for k in ['UIN', 'Pay Event', 'Job Number']:
    if k not in df_fin.columns:
        print(f"[warn] missing key for dedupe: {k}")
df_fin = df_fin.drop_duplicates(subset=[c for c in ['UIN', 'Pay Event', 'Job Number'] if c in df_fin.columns])

# Types
# Strings (leave Calc Date as datetime)
string_cols = [c for c in df_fin.columns if c != 'Calc Date']
ensure_string(df_fin, string_cols)
if 'Calc Date' in df_fin.columns:
    df_fin['Calc Date'] = pd.to_datetime(df_fin['Calc Date'], errors='coerce')

# Fill missing Adjustment Reason as INT/Internal
if 'ADJ Reason Code' in df_fin.columns:
    mask = df_fin['ADJ Reason Code'].isna() | (df_fin['ADJ Reason Code'].astype(str).str.strip().isin(['', 'nan', 'NaN']))
    df_fin.loc[mask, 'ADJ Reason Code'] = 'INT'
    if 'ADJ Reason DESC' in df_fin.columns:
        df_fin.loc[mask, 'ADJ Reason DESC'] = 'Internal'

# ----------------------------
# Build PreTAM-style output (names & order)
# ----------------------------
col_map = OrderedDict([
    ("UIN",                         "UIN"),
    ("Pay ID",                      "Pay ID"),
    ("Year",                        "Year"),
    ("Pay #",                       "Pay #"),
    ("Seq #",                       "Seq #"),
    ("Job Number",                  "Job Number"),
    ("College Code",                "College Code"),
    ("College Name",                "College Name"),
    ("College",                     "College"),
    ("TS COA",                      "TS COA"),
    ("TS Org",                      "TS ORG"),                 # PreTAM header w/ space
    ("TS-Org Code",                 "TS-Org Code"),
    ("TS-Org Title",                "TS-Org Title"),
    ("Dept Code",                   "DEPT Code"),              # PreTAM header “Dept Code”
    ("TS-Org Dept Code",            "TS-Org Dept Code"),
    ("TS-Org Dept Title",           "TS-Org Dept Title"),
    ("E-Class Code",                "ECLS"),                   # code
    ("E-Class",                     "ECLS DESC"),              # description
    ("TE M",                        "TE M"),
    ("Time Entry",                  "Time Entry"),
    ("Overtime",                    "Overtime"),
    ("Earn Code",                   "Earn Code"),
    ("Earn Code Description",       "DESCRIPTION"),
    ("Adjustment Reason Code",      "ADJ Reason Code"),
    ("Adjustment Reason",           "ADJ Reason DESC"),
    ("Calc Date",                   "Calc Date"),
])

out_cols = {}
missing_sources = []
for out_name, src in col_map.items():
    if src in df_fin.columns:
        out_cols[out_name] = df_fin[src]
    else:
        out_cols[out_name] = pd.Series([None] * len(df_fin), index=df_fin.index)
        missing_sources.append((out_name, src))

pua_out = pd.DataFrame(out_cols)

# Ensure final dtypes (strings except Calc Date)
for c in pua_out.columns:
    if c != "Calc Date":
        pua_out[c] = pua_out[c].astype("string").str.strip()
if "Calc Date" in pua_out.columns:
    pua_out["Calc Date"] = pd.to_datetime(pua_out["Calc Date"], errors="coerce")

# ----------------------------
# Harmonize headers + Save
# ----------------------------
import os
import pandas as pd

# 1) Rename to Preetam's header names BEFORE save
rename_map = {
    "TS Org": "TS ORG",
    "Adjustment Reason": "Adjustment Reason Description",
}
pua_out.rename(columns=rename_map, inplace=True)

# 2) (Optional) Reorder to exactly match Preetam's column order if available
preetam_path = os.path.join(BASE_PATH, "2025_PUA_Data_Preetam.xlsx")
if os.path.exists(preetam_path):
    preetam_cols = pd.read_excel(preetam_path, nrows=0).columns.tolist()
    # Add any missing columns as blank, then reorder
    added_blank = []
    for col in preetam_cols:
        if col not in pua_out.columns:
            pua_out[col] = pd.NA
            added_blank.append(col)
    pua_out = pua_out[preetam_cols]
else:
    preetam_cols = None
    added_blank = []

# # 3) Save once
# out_path = os.path.join(
#     BASE_PATH,
#     r"C:\Users\terbe\Documents\Final Work ETL Process\2025_PUA_Data_Tayler_11182025.xlsx"
# )
# pua_out.to_excel(out_path, index=False)

# # 4) Sanity prints
# print("PUA rows:", len(pua_out))
# print("Saved:", out_path)
# if added_blank:
#     print("\n[info] Added blank columns to match Preetam order:")
#     for c in added_blank:
#         print(f"  - {c}")

import io
from datetime import datetime

# ============================================================
# Generate date string for filenames
# ============================================================
date_str = datetime.now().strftime("%m%d%Y")   # MMDDYYYY

# Box folder where output should go
box_folder_id = "351818509913"

# Filenames with automatically injected date string
csv_filename  = f"2025_PUA_Data_csv_{date_str}.csv"
xlsx_filename = f"2025_PUA_Data_excel_{date_str}.xlsx"

# ============================================================
# Save CSV directly to Box
# ============================================================
csv_stream = io.BytesIO()
pua_out.to_csv(csv_stream, index=False, encoding="utf-8")
csv_stream.seek(0)

uploaded_csv = client.folder(box_folder_id).upload_stream(
    csv_stream,
    csv_filename
)

print("\nCSV uploaded successfully!")
print(f"  File: {uploaded_csv.name}")
print(f"  ID:   {uploaded_csv.id}")

# ============================================================
# Save Excel directly to Box
# ============================================================
xlsx_stream = io.BytesIO()
pua_out.to_excel(xlsx_stream, index=False, engine="openpyxl")
xlsx_stream.seek(0)

uploaded_xlsx = client.folder(box_folder_id).upload_stream(
    xlsx_stream,
    xlsx_filename
)

print("\nExcel uploaded successfully!")
print(f"  File: {uploaded_xlsx.name}")
print(f"  ID:   {uploaded_xlsx.id}")

# ============================================================
# Summary
# ============================================================
print("\n--- SAVE SUMMARY ---")
print("Rows saved:", len(pua_out))
print("Date stamp:", date_str)
print("Uploaded to Box folder:", box_folder_id)

if added_blank:
    print("\n[info] Added blank columns:")
    for c in added_blank:
        print("  -", c)

import os
import pandas as pd
from datetime import datetime

# Standardize column names (remove leading/trailing spaces)
df_cert_bw.columns = df_cert_bw.columns.str.strip()
df_cert_mn.columns = df_cert_mn.columns.str.strip()

# Combine both CPA files into one unified DataFrame
df_cpa_combined = pd.concat([df_cert_bw, df_cert_mn], ignore_index=True)

# Step 1: Parse date column
df_cpa_combined['TRAN_CREATE_DT'] = pd.to_datetime(df_cpa_combined['TRAN_CREATE_DT'], errors='coerce')
# Step 2: Inspect actual data range
min_date = df_cpa_combined['TRAN_CREATE_DT'].min()
max_date = df_cpa_combined['TRAN_CREATE_DT'].max()
# Step 3: Define fiscal year window based on *calendar year*, not current month
today = datetime.today()
current_year = today.year
# Always use July 1 (previous year) to June 30 (current year)
fy_start = datetime(current_year - 1, 7, 1)
fy_end = datetime(current_year, 6, 30)

# print(f"Filtering data for fiscal year: {fy_start.date()} to {fy_end.date()}")

# Step 4: Validate oldest record
if min_date < fy_start.replace(year=fy_start.year - 1):
    raise ValueError(
        f"⚠️ ERROR: Oldest transaction date ({min_date.date()}) is more than one fiscal year before the current fiscal year. "
        "This may indicate outdated or unexpected data."
    )

# Step 5: Apply filter
df_cpa_fy = df_cpa_combined[
    (df_cpa_combined['TRAN_CREATE_DT'] >= fy_start) &
    (df_cpa_combined['TRAN_CREATE_DT'] <= fy_end)
]

# print(f"✅ Filtered CPA data: {len(df_cpa_fy)} rows remain in fiscal year window.")
# Define expected raw columns from the documentation
expected_columns = [
    'UIN', 'PAY_YEAR', 'PAY_ID', 'PAY_NBR', 'PAY_SEQ', 'TRAN_ID', 'TRAN_COMPNT', 'ADJ_REASON',
    'TRAN_CREATE_DT', 'TRAN_CLOSED_DT', 'JOB', 'JOB_TITLE', 'JOB_TS_COAS', 'JOB_TS_ORGN',
    'JOB_ECLS', 'COLLEGE', 'OWNING_UIN', 'LAST_NAME', 'FIRST_NAME', 'UI_ENTERPRISE_ID',
    'EMAIL_ADDR', 'HRLY_RATE', 'RT_LEAVE_DT', 'RT_ENTER_DT', 'RT_CREATE_DT', 'LVL', 'ROLE',
    'ACTION', 'ROUTED_BY_UIN', 'RETURNED_FLAG', 'TRAN_ROUTE_DT', 'ELAPSED_WORK_TIME',
    'ROUTE_STOP_TIME', 'ELAPSED_TRAN_TIME'
]

# Strip spaces from current df columns just in case
actual_columns = df_cpa_fy.columns.str.strip().tolist()

# Check if columns match exactly (same order and values)
if actual_columns == expected_columns:
    print("✅ Column check passed: Columns in df_cpa_fy match expected raw columns.")
else:
    print("❌ Column mismatch detected:")
    
    # Show which columns are missing or extra
    missing = set(expected_columns) - set(actual_columns)
    extra   = set(actual_columns) - set(expected_columns)

    if missing:
        print(f"  - Missing columns: {sorted(missing)}")
    if extra:
        print(f"  - Unexpected/extra columns: {sorted(extra)}")

    # Optionally, show side-by-side mismatch if lengths are equal but order is wrong
    if len(actual_columns) == len(expected_columns):
        print("\n🔍 Comparing column order:")
        for i, (act, exp) in enumerate(zip(actual_columns, expected_columns)):
            if act != exp:
                print(f"  Position {i}: expected '{exp}', found '{act}'")

                
# 1. Create 'TS-Org Code' = JOB_TS_COAS + "-" + JOB_TS_ORGN
df_cpa_fy['TS-Org Code'] = df_cpa_fy['JOB_TS_COAS'].astype(str).str.strip() + '-' + df_cpa_fy['JOB_TS_ORGN'].astype(str).str.strip()

# 2. Create 'Dept TS-Org' = first 5 characters of TS-Org Code
df_cpa_fy['Dept TS-Org'] = df_cpa_fy['TS-Org Code'].str[:5]

# Define format patterns
pattern_ts_org_code = r'^\d-\d{6}$'
pattern_dept_ts_org = r'^\d-\d{3}$'

# Check each column for non-matching values
invalid_ts_org_code = df_cpa_fy[~df_cpa_fy['TS-Org Code'].str.match(pattern_ts_org_code)]
invalid_dept_ts_org = df_cpa_fy[~df_cpa_fy['Dept TS-Org'].str.match(pattern_dept_ts_org)]

# Report results
if len(invalid_ts_org_code) == 0 and len(invalid_dept_ts_org) == 0:
    print("✅ All derived fields match expected format.")
else:
    if len(invalid_ts_org_code) > 0:
        print(f"❌ TS-Org Code format issue in {len(invalid_ts_org_code)} rows.")
        print(invalid_ts_org_code[['JOB_TS_COAS', 'JOB_TS_ORGN', 'TS-Org Code']].head())

    if len(invalid_dept_ts_org) > 0:
        print(f"❌ Dept TS-Org format issue in {len(invalid_dept_ts_org)} rows.")
        print(invalid_dept_ts_org[['TS-Org Code', 'Dept TS-Org']].head())


df_cpa_fy = df_cpa_fy.apply(
    lambda col: col.astype(str).str.strip() if col.dtype == "object" else col
)

# Standardize key columns
df_cpa_fy['JOB_ECLS'] = df_cpa_fy['JOB_ECLS'].astype(str).str.strip()
df_cpa_fy['PAY_ID']   = df_cpa_fy['PAY_ID'].astype(str).str.strip()
df_cpa_fy['UIN Job'] = df_cpa_fy['UIN'].astype(str).str.strip() + '-' + df_cpa_fy['JOB'].astype(str).str.strip()

df_overtime_eclass['Job Eclass'] = df_overtime_eclass['Job Eclass'].astype(str).str.strip()
df_overtime_eclass['Pay ID']     = df_overtime_eclass['Pay ID'].astype(str).str.strip()


# === Merge TS-Org Name ===
df_cpa_fy = df_cpa_fy.merge(
    df_ts_org[['TS-Org Code', 'TS-Org Title']].drop_duplicates(),
    on='TS-Org Code',
    how='left'
)
df_cpa_fy = df_cpa_fy.rename(columns={'TS-Org Title': 'TS-Org Name'})

# === Merge TS-Org Department Name ===
df_cpa_fy = df_cpa_fy.merge(
    df_ts_dept[['TS-Org Dept Code', 'TS-Org Dept Title']].drop_duplicates(),
    left_on='Dept TS-Org',
    right_on='TS-Org Dept Code',
    how='left'
)
df_cpa_fy = df_cpa_fy.rename(columns={'TS-Org Dept Title': 'TS-Org Department Name'})

# ===== TE M → Time Entry (canonical creation: do ONCE, at the end) =====

# Build a robust TE M -> Time Entry Method mapping by mode per code
temp_te = (
    df_te_m.loc[:, ["TE M", "Time Entry Method"]]
           .dropna(subset=["TE M", "Time Entry Method"])
           .assign(**{
               "TE M": lambda d: d["TE M"].astype(str).str.strip(),
               "Time Entry Method": lambda d: d["Time Entry Method"].astype(str).str.strip()
           })
)

# === Merge TE M (Time Entry Method / Type) ===
df_cpa_fy = df_cpa_fy.merge(
    df_te_m[['UIN Job', 'TE M', 'Time Entry Method', 'Time Entry Type']].drop_duplicates(),
    on='UIN Job',
    how='left'
)

te_mapping = (
    temp_te.groupby("TE M")["Time Entry Method"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
           .to_dict()
)

# Standardize TE M; create Time Entry once and only once here
df_cpa_fy['TE M'] = df_cpa_fy['TE M'].astype(str).str.strip()
existing_time_entry = (
    df_cpa_fy['Time Entry'].astype('string').str.strip()
    if 'Time Entry' in df_cpa_fy.columns else pd.Series(pd.NA, index=df_cpa_fy.index, dtype='string')
)

mapped_time_entry = pd.Series(df_cpa_fy['TE M'].map(te_mapping), index=df_cpa_fy.index, dtype='string')
df_cpa_fy['Time Entry'] = existing_time_entry.where(
    existing_time_entry.notna() & (existing_time_entry != ''),
    mapped_time_entry
)

# === Merge Overtime FLSA and E-Class Description ===
df_cpa_fy = df_cpa_fy.merge(
    df_overtime_eclass[['Job Eclass', 'Pay ID', 'Overtime FLSA', 'Job Detail E-Class Long Desc']].drop_duplicates(),
    left_on=['JOB_ECLS', 'PAY_ID'],
    right_on=['Job Eclass', 'Pay ID'],
    how='left'
)

# === Rename for clarity ===
df_cpa_fy = df_cpa_fy.rename(columns={
    'Job Detail E-Class Long Desc': 'E-Class Description'
})


df_cpa_fy = df_cpa_fy[df_cpa_fy["ACTION"] == "3 - Apply"]
df_cpa_fy = df_cpa_fy.drop_duplicates()
df_cpa_fy = df_cpa_fy.drop_duplicates(subset=["UIN Job"], keep="first")
# Step 1: Strip leading/trailing whitespace from column names
df_cpa_fy.columns = df_cpa_fy.columns.str.strip()

# Step 2: Coerce all columns into cleaned strings
for col in df_cpa_fy.columns:
    # Convert float-looking strings to integers if possible
    # e.g., "123.0" → "123"
    df_cpa_fy[col] = (
        df_cpa_fy[col]
        .apply(lambda x: int(float(x)) if pd.notna(x) and str(x).strip().replace('.', '', 1).isdigit() and float(x).is_integer() else x)
        .astype(str)               # Ensure string
        .str.strip()               # Strip whitespace
        .str.replace(r'\.0$', '', regex=True)  # Remove lingering .0
    )
    
# Force conversion to datetime
df_cpa_fy['TRAN_CREATE_DT'] = pd.to_datetime(df_cpa_fy['TRAN_CREATE_DT'], errors='coerce')
# Optional: Check for any invalid entries that became NaT
invalid_dates = df_cpa_fy['TRAN_CREATE_DT'].isna().sum()
# Force conversion to datetime
df_cpa_fy['TRAN_CLOSED_DT'] = pd.to_datetime(df_cpa_fy['TRAN_CLOSED_DT'], errors='coerce')
# Optional: Check for any invalid entries that were coerced to NaT
invalid_closed_dates = df_cpa_fy['TRAN_CLOSED_DT'].isna().sum()

df_cpa_fy = df_cpa_fy.drop_duplicates()
# Remove duplicates based on TRAN_ID
df_cpa_fy = df_cpa_fy.drop_duplicates(subset='TRAN_ID')

df_cpa_fy.reset_index()
college_code = []
college_name = []

for val in df_cpa_fy["COLLEGE"].astype(str):
    parts = val.split("-", 1)
    college_code.append(parts[0].strip())
    college_name.append(parts[1].strip() if len(parts) > 1 else None)
df_cpa_fy["College Code"] = college_code
df_cpa_fy["College Name"] = college_name



# Rename columns
df_cpa_fy.rename(columns={
    'UIN':'UIN',
    "PAY_ID": "Pay ID",
    "PAY_YEAR": "Year",
    "PAY_NBR": "Pay #",
    "PAY_SEQ": "Seq #",
    "JOB": "Job Number",
    "College Code": "College Code",
    "College Name": "College Name",
    "COLLEGE": "College",
    "JOB_TS_COAS": "TS COA",
    "JOB_TS_ORGN": "TS Org",
    'TS-Org Code': 'TS-Org Code',
    "TS-Org Name": "TS-Org Title",
    'TS-Org Dept Code':'TS-Org Dept Code',
    'TS-Org Department Name': 'TS-Org Dept Title',
    "JOB_ECLS": "E-Class Code",
    'E-Class Description':"E-Class",
    'TE M':'TE M',
    'Time Entry':'Time Entry',
    "Overtime FLSA":"Overtime"
}, inplace=True)

# Select only the 20 required columns and overwrite df_cpa_final
df_cpa_fy = df_cpa_fy[['UIN', 'Pay ID', 'Year', 'Pay #', 'Seq #', 'Job Number', 'College Code',
       'College Name', 'College', 'TS COA', 'TS Org', 'TS-Org Code',
       'TS-Org Title', 'TS-Org Dept Code', 'TS-Org Dept Title', 'E-Class Code',
       'E-Class', 'TE M', 'Time Entry', 'Overtime']]


df_cpa_fy = df_cpa_fy.loc[:, ~df_cpa_fy.columns.duplicated()]


import io
from datetime import datetime

# ============================================================
# 1. Generate date string
# ============================================================
date_str = datetime.now().strftime("%m%d%Y")   # MMDDYYYY

# Box folder for CPA output
box_folder_id = "351818509913"

# ============================================================
# 2. Build filenames (distinct for CSV and XLSX)
# ============================================================
cpa_csv_filename  = f"2025_CPA_Data_csv_{date_str}.csv"
cpa_xlsx_filename = f"2025_CPA_Data_excel_{date_str}.xlsx"

# ============================================================
# 3. Upload CPA CSV to Box
# ============================================================
csv_stream = io.BytesIO()
df_cpa_fy.to_csv(csv_stream, index=False, encoding="utf-8")
csv_stream.seek(0)

uploaded_csv = client.folder(box_folder_id).upload_stream(
    csv_stream,
    cpa_csv_filename
)

print("\nCPA CSV uploaded successfully!")
print(f"  File: {uploaded_csv.name}")
print(f"  ID:   {uploaded_csv.id}")

# ============================================================
# 4. Upload CPA Excel to Box
# ============================================================
xlsx_stream = io.BytesIO()
df_cpa_fy.to_excel(xlsx_stream, index=False, engine="openpyxl")
xlsx_stream.seek(0)

uploaded_xlsx = client.folder(box_folder_id).upload_stream(
    xlsx_stream,
    cpa_xlsx_filename
)

print("\nCPA Excel uploaded successfully!")
print(f"  File: {uploaded_xlsx.name}")
print(f"  ID:   {uploaded_xlsx.id}")

# ============================================================
# 5. Final summary
# ============================================================
print("\n--- CPA SAVE SUMMARY ---")
print("Rows saved:", len(df_cpa_fy))
print("Date stamp:", date_str)
print("Uploaded to Box folder:", box_folder_id)
