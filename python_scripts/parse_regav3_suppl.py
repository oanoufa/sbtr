import docx
import pandas as pd
import re

from src import config

WORKSPACE_PATH = config.WORKSPACE_PATH

# Load document
doc = docx.Document(f'{WORKSPACE_PATH}/data/input_sequences/regav3_testset/regav3_supplementary.docx')

# Map tables to LANL datasets:
# Table 3 (index 2) -> LANL full length
# Table 4 (index 3) -> LANL RT
# Table 5 (index 4) -> LANL PR/PROT
tables_map = {
    2: 'full',
    3: 'rt',
    4: 'prot'
}

# class,n,TP,FP,FN,TN,sensitivity,sens_CI_lo,sens_CI_hi,specificity,spec_CI_lo,spec_CI_hi,n_excluded_ambiguous
columns = [
    'class', 'n', 'TP', 'FP', 'FN', 'TN', 
    'sensitivity', 'sens_CI_lo', 'sens_CI_hi', 
    'specificity', 'spec_CI_lo', 'spec_CI_hi'
]

for table_idx, dataset_type in tables_map.items():
    table = doc.tables[table_idx]
    current_tool = None
    rows_data = []
    
    for row in table.rows:
        cells = [c.text.strip().replace('\n', ' ') for c in row.cells]
        
        # Detect new tool block header
        if cells[0] == 'Subtype' and cells[1] not in ['', 'n', 'TP']:
            if current_tool and rows_data:
                tool_slug = re.sub(r'[^a-zA-Z0-9]', '', current_tool.lower())
                print(f"Found {tool_slug}", flush=True)
                df = pd.DataFrame(rows_data, columns=columns)
                df.to_csv(f"{WORKSPACE_PATH}/data/input_sequences/regav3_testset/{tool_slug}_{dataset_type}_results.csv", index=False)
                rows_data = []
            current_tool = cells[1].strip()
            continue
            
        # Skip sub-header rows
        if cells[0] in ['Subtype', '', 'Total']:
            continue
            
        # Collect data rows
        if current_tool and cells[0]:
            rows_data.append(cells)
            
    # Export the last tool block in the table
    if current_tool and rows_data:
        tool_slug = re.sub(r'[^a-zA-Z0-9]', '', current_tool.lower())
        df = pd.DataFrame(rows_data, columns=columns)
        df.to_csv(f"{WORKSPACE_PATH}/data/input_sequences/regav3_testset/{tool_slug}_{dataset_type}_results.csv", index=False)