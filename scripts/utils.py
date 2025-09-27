# scripts/utils.py
import pandas as pd
import numpy as np

def print_df_stats(df):
    print(f"""Total nodes: {df.shape[0]}
            train: {df.train_mask.sum()}
            val:  {df.val_mask.sum()}
            test: {df.test_mask.sum()}
    """)

def print_stats(graph_data):
    print(f"""Total nodes: {graph_data.num_nodes}
            train: {graph_data.train_mask.sum()}
            val:  {graph_data.val_mask.sum()}
            test: {graph_data.test_mask.sum()}
    """)

def convert2group(cur_rel, src_col, dst_col):
    """
    Groups a list of model relationships into a dictionary without using explicit loops.

    Args:
        cur_rel: A list of dictionaries.
        src_col: The column name for the source of the relationship (e.g., 'base_model_id').
        dst_col: The column name for the destination of the relationship (e.g., 'model_id').

    Returns:
        A dictionary mapping each source ID to a list of its associated destination IDs.
    """
    if not cur_rel:
        return {}
        
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(cur_rel)
    
    return df.groupby(src_col)[dst_col].apply(list).to_dict()

def encode_onehot(temp_df):
    """
    Efficiently creates a one-hot encoded matrix for multi-label data.
    """
    # Ensure every entry in the column is a list, converting None/NaN to []
    cleaned_labels = temp_df['y_multi_lab'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create coordinate lists for the sparse matrix
    rows = []
    cols = []
    for i, sublist in enumerate(cleaned_labels):
        for item in sublist:
            # Check if the item is not None/NaN and is a valid integer index
            if pd.notna(item) and isinstance(item, (int, float)) and not isinstance(item, bool):
                try:
                    cols.append(int(item))
                    rows.append(i)
                except (ValueError, TypeError):
                    # Skip items that cannot be converted to an int
                    pass
    
    # Determine the size of the one-hot vector
    num_classes = max(cols) + 1 if cols else 0
    
    # --- Vectorized One-Hot Matrix Creation ---
    num_rows = len(temp_df)
    one_hot_matrix = np.zeros((num_rows, num_classes), dtype=int)
    
    # Use NumPy's fast indexing if there are labels to encode
    if rows:
        one_hot_matrix[rows, cols] = 1
    
    # Return the result as a list of numpy arrays
    return list(one_hot_matrix)
    
