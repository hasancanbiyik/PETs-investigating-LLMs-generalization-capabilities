import pandas as pd
import numpy as np
import os
import glob

"""
Category-Level Label Distribution & Error Analysis

This script analyzes:
1. Label distributions by category for each OPET/NOPET split
2. Cross-lingual transfer performance by category
3. Error patterns in predictions

File naming convention: preds_Train-{TRAIN_SPLIT}_Test-{TEST_SPLIT}_Fold{N}.csv
"""

# --- UPDATE THIS PATH ---
EXPERIMENT_RESULTS_DIR = "/path/to/experiment_results"


def detect_columns(df):
    """Auto-detect column names for label, prediction, category, PET, and text"""
    columns = df.columns.tolist()
    print(f"  Available columns: {columns}")
    
    # Possible names for each required column
    label_candidates = ['labels', 'label', 'Label', 'true_label', 'gold_label', 'y_true', 'ground_truth']
    pred_candidates = ['pred_label', 'prediction', 'predicted', 'pred', 'y_pred', 'predicted_label']
    category_candidates = ['category', 'Category', 'cat', 'domain', 'Domain']
    pet_candidates = ['PET', 'pet', 'term', 'Term', 'target', 'target_term']
    text_candidates = ['text', 'Text', 'sentence', 'Sentence', 'context']
    euph_status_candidates = ['euph_status', 'euphemism_status', 'status', 'always_euph']
    
    def find_column(candidates):
        for c in candidates:
            if c in columns:
                return c
        return None
    
    detected = {
        'label': find_column(label_candidates),
        'pred': find_column(pred_candidates),
        'category': find_column(category_candidates),
        'pet': find_column(pet_candidates),
        'text': find_column(text_candidates),
        'euph_status': find_column(euph_status_candidates)
    }
    
    print(f"  Detected columns: {detected}")
    return detected

# Splits we care about for cross-lingual analysis
CROSS_LINGUAL_TRANSFERS = [
    ("EN_OPET", "TR_OPET"),   # English OPETs → Turkish OPETs
    ("EN_OPET", "TR_NOPET"),  # English OPETs → Turkish NOPETs
    ("EN_NOPET", "TR_OPET"),  # English NOPETs → Turkish OPETs
    ("EN_NOPET", "TR_NOPET"), # English NOPETs → Turkish NOPETs
    ("TR_OPET", "EN_OPET"),   # Turkish OPETs → English OPETs
    ("TR_OPET", "EN_NOPET"),  # Turkish OPETs → English NOPETs
    ("TR_NOPET", "EN_OPET"),  # Turkish NOPETs → English OPETs
    ("TR_NOPET", "EN_NOPET"), # Turkish NOPETs → English NOPETs
]

# In-domain for reference
IN_DOMAIN = [
    ("EN_OPET", "EN_OPET"),
    ("EN_NOPET", "EN_NOPET"),
    ("TR_OPET", "TR_OPET"),
    ("TR_NOPET", "TR_NOPET"),
]


def load_fold_results(results_dir, train_split, test_split):
    """Load and combine all fold results for a train/test combination"""
    pattern = f"preds_Train-{train_split}_Test-{test_split}_Fold*.csv"
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        print(f"  ⚠️  No files found for {train_split} → {test_split}")
        return None, None
    
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        fold_num = int(f.split('Fold')[1].split('.')[0])
        df['fold'] = fold_num
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Loaded {len(files)} folds, {len(combined)} total instances")
    
    # Detect columns
    cols = detect_columns(combined)
    
    return combined, cols


def analyze_label_distribution(df, split_name):
    """Analyze label distribution by category"""
    print(f"\n{'='*60}")
    print(f"LABEL DISTRIBUTION: {split_name}")
    print(f"{'='*60}")
    
    # Overall
    overall_euph = df['label'].mean() * 100
    print(f"Overall: {overall_euph:.1f}% euphemistic (n={len(df)})")
    
    # By category
    print(f"\n{'Category':<25} {'N':>6} {'% Euph':>8}")
    print("-" * 45)
    
    results = []
    for cat in sorted(df['category'].unique()):
        cat_df = df[df['category'] == cat]
        n = len(cat_df)
        pct_euph = cat_df['label'].mean() * 100
        print(f"{cat:<25} {n:>6} {pct_euph:>7.1f}%")
        results.append({
            'split': split_name,
            'category': cat,
            'n': n,
            'pct_euphemistic': pct_euph
        })
    
    return pd.DataFrame(results)


def analyze_transfer_by_category(df, cols, train_split, test_split):
    """Analyze transfer performance by category"""
    print(f"\n{'='*60}")
    print(f"TRANSFER PERFORMANCE: {train_split} → {test_split}")
    print(f"{'='*60}")
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Get column names
    label_col = cols['label']
    pred_col = cols['pred']
    cat_col = cols['category']
    
    if not label_col or not pred_col or not cat_col:
        print(f"  ❌ Missing required columns. Detected: {cols}")
        return None
    
    results = []
    
    print(f"\n{'Category':<25} {'N':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} {'%Euph':>7}")
    print("-" * 70)
    
    # Get unique categories, filtering out NaN values
    categories = df[cat_col].dropna().unique()
    categories = sorted([str(c) for c in categories])  # Convert to strings and sort
    
    for cat in categories:
        cat_df = df[df[cat_col] == cat]
        n = len(cat_df)
        
        if n < 5:  # Skip very small categories
            print(f"{cat:<25} {n:>6} {'(too few)':>30}")
            continue
        
        y_true = cat_df[label_col]
        y_pred = cat_df[pred_col]
        
        # Handle edge cases
        if len(y_true.unique()) < 2 or len(y_pred.unique()) < 2:
            f1 = precision = recall = float('nan')
        else:
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
        
        pct_euph_label = y_true.mean() * 100
        
        print(f"{cat:<25} {n:>6} {f1:>7.2f} {precision:>7.2f} {recall:>7.2f} {pct_euph_label:>6.1f}%")
        
        results.append({
            'train': train_split,
            'test': test_split,
            'category': cat,
            'n': n,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'label_pct_euph': pct_euph_label
        })
    
    return pd.DataFrame(results)


def analyze_errors(df, cols, train_split, test_split, top_n=10):
    """Detailed error analysis"""
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {train_split} → {test_split}")
    print(f"{'='*60}")
    
    # Get column names
    label_col = cols['label']
    pred_col = cols['pred']
    cat_col = cols['category']
    pet_col = cols['pet']
    text_col = cols['text']
    euph_status_col = cols['euph_status']
    
    if not label_col or not pred_col:
        print(f"  ❌ Missing required columns. Detected: {cols}")
        return None
    
    # Identify errors
    df['correct'] = df[label_col] == df[pred_col]
    errors = df[~df['correct']]
    
    total = len(df)
    n_errors = len(errors)
    accuracy = (total - n_errors) / total * 100
    
    print(f"Total instances: {total}")
    print(f"Errors: {n_errors} ({100-accuracy:.1f}%)")
    
    # False Negatives (missed euphemisms)
    fn = errors[(errors[label_col] == 1) & (errors[pred_col] == 0)]
    # False Positives (over-detected)
    fp = errors[(errors[label_col] == 0) & (errors[pred_col] == 1)]
    
    print(f"\nFalse Negatives (missed euphemisms): {len(fn)}")
    print(f"False Positives (over-detected): {len(fp)}")
    
    # Errors by PET
    if pet_col:
        print(f"\n--- Top {top_n} PETs with most errors ---")
        pet_errors = errors.groupby(pet_col).size().sort_values(ascending=False).head(top_n)
        for pet, count in pet_errors.items():
            pet_total = len(df[df[pet_col] == pet])
            print(f"  {pet}: {count} errors / {pet_total} instances ({count/pet_total*100:.1f}%)")
    
    # Errors by category
    if cat_col:
        print(f"\n--- Errors by category ---")
        # Filter out NaN categories
        errors_with_cat = errors[errors[cat_col].notna()]
        cat_errors = errors_with_cat.groupby(cat_col).size().sort_values(ascending=False)
        for cat, count in cat_errors.items():
            cat_total = len(df[df[cat_col] == cat])
            print(f"  {cat}: {count} errors / {cat_total} instances ({count/cat_total*100:.1f}%)")
    
    # Errors by euph_status
    if euph_status_col:
        print(f"\n--- Errors by euph_status ---")
        status_errors = errors.groupby(euph_status_col).size()
        for status, count in status_errors.items():
            status_total = len(df[df[euph_status_col] == status])
            print(f"  {status}: {count} errors / {status_total} instances ({count/status_total*100:.1f}%)")
    
    # Sample some FN and FP for qualitative analysis
    print(f"\n--- Sample False Negatives (missed euphemisms) ---")
    if len(fn) > 0:
        for _, row in fn.head(5).iterrows():
            pet_val = row[pet_col] if pet_col else "N/A"
            cat_val = row[cat_col] if cat_col else "N/A"
            text_val = row[text_col][:100] if text_col else "N/A"
            print(f"  PET: {pet_val} | Category: {cat_val}")
            print(f"  Text: {text_val}...")
            print()
    
    print(f"\n--- Sample False Positives (over-detected) ---")
    if len(fp) > 0:
        for _, row in fp.head(5).iterrows():
            pet_val = row[pet_col] if pet_col else "N/A"
            cat_val = row[cat_col] if cat_col else "N/A"
            text_val = row[text_col][:100] if text_col else "N/A"
            print(f"  PET: {pet_val} | Category: {cat_val}")
            print(f"  Text: {text_val}...")
            print()
    
    return {
        'train': train_split,
        'test': test_split,
        'n_total': total,
        'n_errors': n_errors,
        'n_fn': len(fn),
        'n_fp': len(fp),
        'accuracy': accuracy
    }


def verify_paper_claim(results_dir):
    """
    Specifically verify the claim from the paper:
    'Turkish NOPETs in these categories have higher proportions of euphemistic 
    instances (e.g., Employment NOPETs: 82% euphemistic vs OPETs: 44% euphemistic)'
    """
    print("\n" + "="*70)
    print("VERIFICATION: Paper claim about Employment category")
    print("="*70)
    
    # Try to load from cross-lingual transfer files (these should exist)
    test_splits = ['TR_OPET', 'TR_NOPET', 'EN_OPET', 'EN_NOPET']
    
    for test_split in test_splits:
        # Try to find any file that tests on this split
        pattern = f"preds_Train-*_Test-{test_split}_Fold1.csv"
        files = glob.glob(os.path.join(results_dir, pattern))
        
        if not files:
            print(f"  ⚠️  No files found for Test-{test_split}")
            continue
        
        # Load the first matching file
        filepath = files[0]
        df = pd.read_csv(filepath)
        cols = detect_columns(df)
        
        label_col = cols['label']
        cat_col = cols['category']
        
        if not label_col or not cat_col:
            print(f"  ⚠️  Missing columns for {test_split}")
            continue
        
        print(f"\n  {test_split} (from {os.path.basename(filepath)}):")
        
        # Find Employment category (might be named differently)
        emp_cats = [c for c in df[cat_col].unique() if 'employ' in str(c).lower() or 'financ' in str(c).lower()]
        
        if emp_cats:
            for cat in emp_cats:
                cat_df = df[df[cat_col] == cat]
                pct = cat_df[label_col].mean() * 100
                print(f"    {cat}: {pct:.1f}% euphemistic (n={len(cat_df)})")
        else:
            # Show all categories with their distributions
            print(f"    No Employment category found. All categories:")
            categories = df[cat_col].dropna().unique()
            for cat in sorted([str(c) for c in categories]):
                cat_df = df[df[cat_col] == cat]
                pct = cat_df[label_col].mean() * 100
                print(f"      {cat}: {pct:.1f}% euphemistic (n={len(cat_df)})")


def main():
    print("="*70)
    print("CATEGORY-LEVEL ANALYSIS FOR CROSS-LINGUAL EUPHEMISM DETECTION")
    print("="*70)
    
    # First, verify the paper's specific claim
    verify_paper_claim(EXPERIMENT_RESULTS_DIR)
    
    # Analyze key cross-lingual transfers
    print("\n\n" + "="*70)
    print("CROSS-LINGUAL TRANSFER ANALYSIS")
    print("="*70)
    
    all_transfer_results = []
    all_error_summaries = []
    
    # Focus on the most interesting transfers for the paper
    key_transfers = [
        ("EN_OPET", "TR_OPET"),   # Does OPET training help OPET testing?
        ("EN_OPET", "TR_NOPET"),  # Does OPET training generalize to NOPETs?
        ("TR_OPET", "EN_OPET"),   # Reverse direction
        ("TR_OPET", "EN_NOPET"),  # Reverse direction
    ]
    
    for train_split, test_split in key_transfers:
        df, cols = load_fold_results(EXPERIMENT_RESULTS_DIR, train_split, test_split)
        
        if df is not None and cols is not None:
            # Category-level transfer performance
            transfer_results = analyze_transfer_by_category(df, cols, train_split, test_split)
            if transfer_results is not None:
                all_transfer_results.append(transfer_results)
            
            # Error analysis
            error_summary = analyze_errors(df, cols, train_split, test_split)
            if error_summary is not None:
                all_error_summaries.append(error_summary)
    
    # Save results
    if all_transfer_results:
        combined_transfer = pd.concat(all_transfer_results, ignore_index=True)
        combined_transfer.to_csv("category_transfer_analysis.csv", index=False)
        print("\n✅ Saved category transfer analysis to category_transfer_analysis.csv")
    
    if all_error_summaries:
        error_df = pd.DataFrame(all_error_summaries)
        error_df.to_csv("error_summary.csv", index=False)
        print("✅ Saved error summary to error_summary.csv")


if __name__ == "__main__":
    main()
