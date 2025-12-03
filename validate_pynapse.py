#!/usr/bin/env python3
"""
Pynapse Output Validation Script
================================
Validates Pynapse output against og2 (original SynapseJ macro) reference output.

Usage:
    uv run python validate_pynapse.py [options]
    uv run python validate_pynapse.py pynapse_dir og2_dir
    uv run python validate_pynapse.py --pynapse-dir path/to/output --og2-dir path/to/reference

Examples:
    uv run python validate_pynapse.py
    uv run python validate_pynapse.py test/output_test test/og2
    uv run python validate_pynapse.py -p test/output_test -o test/og2

Defaults:
    pynapse_dir: test/output_test
    og2_dir: test/og2

Known acceptable differences:
- RawIntDen column may have .000 suffix (numeric values must match)
- IFALog.txt contains timing info (ignored)
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def ok(msg): return f"{Colors.GREEN}✅ PASS{Colors.END}: {msg}"
def fail(msg): return f"{Colors.RED}❌ FAIL{Colors.END}: {msg}"
def warn(msg): return f"{Colors.YELLOW}⚠️  WARN{Colors.END}: {msg}"
def info(msg): return f"{Colors.BLUE}ℹ️  INFO{Colors.END}: {msg}"
def header(msg): return f"\n{Colors.BOLD}{'='*60}\n{msg}\n{'='*60}{Colors.END}"


class ValidationResult:
    def __init__(self, name, passed, details="", warnings=None):
        self.name = name
        self.passed = passed
        self.details = details
        self.warnings = warnings or []
    
    def __str__(self):
        status = ok(self.name) if self.passed else fail(self.name)
        if self.details:
            status += f" - {self.details}"
        for w in self.warnings:
            status += f"\n    {warn(w)}"
        return status


def validate_tsv_linewise(py_path, og_path, name, rawintden_col='RawIntDen'):
    """
    Line-by-line TSV validation for files with irregular column counts.
    Handles CorrResults files that may have embedded tabs in label fields.
    """
    try:
        with open(py_path) as f:
            py_lines = f.readlines()
        with open(og_path) as f:
            og_lines = f.readlines()
    except Exception as e:
        return ValidationResult(name, False, f"Read error: {e}")
    
    if len(py_lines) != len(og_lines):
        return ValidationResult(name, False, f"Line count: {len(py_lines)} vs {len(og_lines)}")
    
    # Get header and find RawIntDen column index
    header = py_lines[0].strip().split('\t')
    rawintden_idx = None
    if rawintden_col in header:
        rawintden_idx = header.index(rawintden_col)
    
    real_diffs = 0
    format_diffs = 0
    warnings = []
    
    for line_num, (py_line, og_line) in enumerate(zip(py_lines[1:], og_lines[1:]), start=2):
        if py_line.strip() == og_line.strip():
            continue
        
        py_cols = py_line.strip().split('\t')
        og_cols = og_line.strip().split('\t')
        
        # Check if only RawIntDen differs by format
        if len(py_cols) == len(og_cols):
            all_same = True
            rawintden_format = False
            
            for i, (p, o) in enumerate(zip(py_cols, og_cols)):
                if p != o:
                    if rawintden_idx is not None and i == rawintden_idx:
                        try:
                            if abs(float(p) - float(o)) < 0.001:
                                rawintden_format = True
                                continue
                        except:
                            pass
                    all_same = False
            
            if rawintden_format and all_same:
                format_diffs += 1
            elif not all_same:
                real_diffs += 1
                if len(warnings) < 3:
                    warnings.append(f"Line {line_num} differs")
        else:
            real_diffs += 1
            if len(warnings) < 3:
                warnings.append(f"Line {line_num}: column count {len(py_cols)} vs {len(og_cols)}")
    
    if real_diffs > 0:
        return ValidationResult(name, False, f"{real_diffs} lines differ", warnings)
    
    details = f"{len(py_lines)-1} rows"
    if format_diffs > 0:
        details += f" (RawIntDen format: {format_diffs})"
    
    return ValidationResult(name, True, details)


def validate_tsv_file(py_path, og_path, name, rawintden_col='RawIntDen'):
    """
    Validate a TSV file, allowing RawIntDen format differences (.000 suffix).
    Returns ValidationResult.
    """
    if not os.path.exists(py_path):
        return ValidationResult(name, False, f"Missing Pynapse file: {py_path}")
    if not os.path.exists(og_path):
        return ValidationResult(name, False, f"Missing og2 file: {og_path}")
    
    try:
        # Try pandas first, fall back to line-by-line for irregular files
        try:
            py_df = pd.read_csv(py_path, sep='\t', dtype=str, keep_default_na=False)
            og_df = pd.read_csv(og_path, sep='\t', dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            # Fall back to line-by-line comparison for irregular files (like CorrResults)
            return validate_tsv_linewise(py_path, og_path, name, rawintden_col)
    except Exception as e:
        return ValidationResult(name, False, f"Read error: {e}")
    
    # Check row counts
    if len(py_df) != len(og_df):
        return ValidationResult(name, False, f"Row count mismatch: {len(py_df)} vs {len(og_df)}")
    
    # Check column counts
    if len(py_df.columns) != len(og_df.columns):
        return ValidationResult(name, False, f"Column count mismatch: {len(py_df.columns)} vs {len(og_df.columns)}")
    
    # Compare columns
    warnings = []
    real_diffs = 0
    rawintden_format_diffs = 0
    
    for col in py_df.columns:
        if col not in og_df.columns:
            return ValidationResult(name, False, f"Missing column in og2: {col}")
        
        py_col = py_df[col]
        og_col = og_df[col]
        
        # Check for differences
        diff_mask = py_col != og_col
        if diff_mask.any():
            diff_count = diff_mask.sum()
            
            if col == rawintden_col:
                # Check if all diffs are just .000 format
                try:
                    py_vals = py_col[diff_mask].astype(float)
                    og_vals = og_col[diff_mask].astype(float)
                    if np.allclose(py_vals, og_vals, rtol=1e-9, atol=1e-9):
                        rawintden_format_diffs = diff_count
                    else:
                        real_diffs += diff_count
                except:
                    real_diffs += diff_count
            else:
                # Try numeric comparison for floating point columns
                try:
                    py_vals = py_col[diff_mask].astype(float)
                    og_vals = og_col[diff_mask].astype(float)
                    if not np.allclose(py_vals, og_vals, rtol=1e-6, atol=1e-9):
                        real_diffs += diff_count
                        # Sample some differences for debugging
                        sample_idx = diff_mask[diff_mask].head(3).index.tolist()
                        for idx in sample_idx:
                            warnings.append(f"Col '{col}' row {idx}: '{py_col[idx]}' vs '{og_col[idx]}'")
                except:
                    # String comparison failed, these are real diffs
                    real_diffs += diff_count
                    sample_idx = diff_mask[diff_mask].head(3).index.tolist()
                    for idx in sample_idx:
                        warnings.append(f"Col '{col}' row {idx}: '{py_col[idx]}' vs '{og_col[idx]}'")
    
    if real_diffs > 0:
        return ValidationResult(name, False, f"{real_diffs} value differences", warnings[:5])
    
    details = f"{len(py_df)} rows"
    if rawintden_format_diffs > 0:
        details += f" (RawIntDen format: {rawintden_format_diffs})"
    
    return ValidationResult(name, True, details)


def validate_tif_file(py_path, og_path, name):
    """
    Validate a TIF file pixel-by-pixel using PIL.
    Returns ValidationResult.
    """
    if not os.path.exists(py_path):
        return ValidationResult(name, False, f"Missing Pynapse file")
    if not os.path.exists(og_path):
        return ValidationResult(name, False, f"Missing og2 file")
    
    try:
        from PIL import Image
    except ImportError:
        return ValidationResult(name, False, "PIL not available - run: uv pip install pillow")
    
    try:
        py_img = Image.open(py_path)
        og_img = Image.open(og_path)
        
        py_frames = getattr(py_img, 'n_frames', 1)
        og_frames = getattr(og_img, 'n_frames', 1)
        
        if py_frames != og_frames:
            return ValidationResult(name, False, f"Frame count mismatch: {py_frames} vs {og_frames}")
        
        total_pixels = 0
        for frame in range(py_frames):
            py_img.seek(frame)
            og_img.seek(frame)
            
            py_arr = np.array(py_img)
            og_arr = np.array(og_img)
            
            if py_arr.shape != og_arr.shape:
                return ValidationResult(name, False, f"Shape mismatch at frame {frame}: {py_arr.shape} vs {og_arr.shape}")
            
            if not np.array_equal(py_arr, og_arr):
                diff_count = np.sum(py_arr != og_arr)
                return ValidationResult(name, False, f"Pixel mismatch at frame {frame}: {diff_count} pixels differ")
            
            total_pixels += py_arr.size
        
        return ValidationResult(name, True, f"{py_frames} frames, {total_pixels:,} pixels")
    
    except Exception as e:
        return ValidationResult(name, False, f"Error: {e}")


def validate_zip_file(py_path, og_path, name):
    """
    Validate a ZIP file by comparing contents.
    Returns ValidationResult.
    """
    if not os.path.exists(py_path):
        return ValidationResult(name, False, f"Missing Pynapse file")
    if not os.path.exists(og_path):
        return ValidationResult(name, False, f"Missing og2 file")
    
    try:
        with zipfile.ZipFile(py_path, 'r') as py_zip, zipfile.ZipFile(og_path, 'r') as og_zip:
            py_names = set(py_zip.namelist())
            og_names = set(og_zip.namelist())
            
            if py_names != og_names:
                missing_in_py = og_names - py_names
                extra_in_py = py_names - og_names
                details = []
                if missing_in_py:
                    details.append(f"Missing in Pynapse: {len(missing_in_py)}")
                if extra_in_py:
                    details.append(f"Extra in Pynapse: {len(extra_in_py)}")
                return ValidationResult(name, False, ", ".join(details))
            
            # Compare file contents
            for fname in py_names:
                py_data = py_zip.read(fname)
                og_data = og_zip.read(fname)
                if py_data != og_data:
                    return ValidationResult(name, False, f"Content mismatch in {fname}")
            
            return ValidationResult(name, True, f"{len(py_names)} files")
    
    except Exception as e:
        return ValidationResult(name, False, f"Error: {e}")


def validate_text_file(py_path, og_path, name, ignore_differences=False):
    """
    Validate a plain text file line by line.
    Returns ValidationResult.
    """
    if not os.path.exists(py_path):
        return ValidationResult(name, False, f"Missing Pynapse file")
    if not os.path.exists(og_path):
        return ValidationResult(name, False, f"Missing og2 file")
    
    if ignore_differences:
        return ValidationResult(name, True, "Ignored (log/timing file)")
    
    try:
        with open(py_path) as f:
            py_lines = f.readlines()
        with open(og_path) as f:
            og_lines = f.readlines()
        
        if len(py_lines) != len(og_lines):
            return ValidationResult(name, False, f"Line count: {len(py_lines)} vs {len(og_lines)}")
        
        diffs = []
        for i, (py, og) in enumerate(zip(py_lines, og_lines)):
            if py.strip() != og.strip():
                diffs.append(i + 1)
        
        if diffs:
            return ValidationResult(name, False, f"{len(diffs)} lines differ (first: {diffs[0]})")
        
        return ValidationResult(name, True, f"{len(py_lines)} lines")
    
    except Exception as e:
        return ValidationResult(name, False, f"Error: {e}")


def run_validation(py_dir, og_dir):
    """
    Run full validation suite.
    """
    results = []
    
    # ========== TSV Aggregate Files ==========
    print(header("TSV AGGREGATE FILES"))
    
    tsv_files = [
        "Syn Pre Results.txt",
        "Syn Post Results.txt", 
        "All Pre Results.txt",
        "All Post Results.txt",
        "CorrResults.txt",
        "CorrResults2.txt",
        "Collated ResultsIF.txt",
    ]
    
    for f in tsv_files:
        result = validate_tsv_file(
            os.path.join(py_dir, f),
            os.path.join(og_dir, f),
            f
        )
        results.append(result)
        print(result)
    
    # ========== Excel Folder Files ==========
    print(header("EXCEL FOLDER FILES"))
    
    excel_dir_py = os.path.join(py_dir, "excel")
    excel_dir_og = os.path.join(og_dir, "excel")
    
    if os.path.isdir(excel_dir_og):
        for f in sorted(os.listdir(excel_dir_og)):
            if f.endswith('.txt'):
                result = validate_tsv_file(
                    os.path.join(excel_dir_py, f),
                    os.path.join(excel_dir_og, f),
                    f"excel/{f}"
                )
                results.append(result)
                print(result)
    else:
        print(warn("excel/ folder not found in og2"))
    
    # ========== TIF Files ==========
    print(header("TIF IMAGE FILES"))
    
    for f in sorted(os.listdir(og_dir)):
        if f.endswith('.tif'):
            result = validate_tif_file(
                os.path.join(py_dir, f),
                os.path.join(og_dir, f),
                f
            )
            results.append(result)
            print(result)
    
    # ========== Merge Folder TIFs ==========
    print(header("MERGE FOLDER TIF FILES"))
    
    merge_dir_py = os.path.join(py_dir, "merge")
    merge_dir_og = os.path.join(og_dir, "merge")
    
    if os.path.isdir(merge_dir_og):
        for f in sorted(os.listdir(merge_dir_og)):
            if f.endswith('.tif'):
                result = validate_tif_file(
                    os.path.join(merge_dir_py, f),
                    os.path.join(merge_dir_og, f),
                    f"merge/{f}"
                )
                results.append(result)
                print(result)
    else:
        print(warn("merge/ folder not found in og2"))
    
    # ========== ZIP Files ==========
    print(header("ROI ZIP FILES"))
    
    for f in sorted(os.listdir(og_dir)):
        if f.endswith('.zip'):
            result = validate_zip_file(
                os.path.join(py_dir, f),
                os.path.join(og_dir, f),
                f
            )
            results.append(result)
            print(result)
    
    # ========== Other Files ==========
    print(header("OTHER FILES"))
    
    # IFALog is timing/log - ignore
    result = validate_text_file(
        os.path.join(py_dir, "IFALog.txt"),
        os.path.join(og_dir, "IFALog.txt"),
        "IFALog.txt",
        ignore_differences=True
    )
    results.append(result)
    print(result)
    
    # ========== Summary ==========
    print(header("VALIDATION SUMMARY"))
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    
    print(f"\n{Colors.BOLD}Total: {total} files validated{Colors.END}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")
        print(f"\n{Colors.RED}Failed files:{Colors.END}")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.details}")
        return 1
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL VALIDATIONS PASSED!{Colors.END}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate Pynapse output against og2 (original SynapseJ) reference output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                                    # Use defaults
    %(prog)s test/output_test test/og2          # Positional arguments  
    %(prog)s -p test/output_test -o test/og2    # Named arguments
    %(prog)s --pynapse-dir output --og2-dir ref # Full named arguments
        """
    )
    
    # Default directories
    script_dir = Path(__file__).parent
    default_py = script_dir / "test" / "output_test"
    default_og = script_dir / "test" / "og2"
    
    parser.add_argument(
        "pynapse_dir", 
        nargs="?",
        default=str(default_py),
        help=f"Directory containing Pynapse output (default: {default_py})"
    )
    parser.add_argument(
        "og2_dir",
        nargs="?", 
        default=str(default_og),
        help=f"Directory containing og2 reference output (default: {default_og})"
    )
    parser.add_argument(
        "-p", "--pynapse-dir",
        dest="pynapse_dir_named",
        help="Directory containing Pynapse output (overrides positional)"
    )
    parser.add_argument(
        "-o", "--og2-dir",
        dest="og2_dir_named",
        help="Directory containing og2 reference output (overrides positional)"
    )
    
    args = parser.parse_args()
    
    # Named arguments override positional
    py_dir = args.pynapse_dir_named if args.pynapse_dir_named else args.pynapse_dir
    og_dir = args.og2_dir_named if args.og2_dir_named else args.og2_dir
    
    print(f"{Colors.BOLD}Pynapse Output Validation{Colors.END}")
    print(f"Pynapse output: {py_dir}")
    print(f"og2 reference:  {og_dir}")
    
    if not os.path.isdir(py_dir):
        print(fail(f"Pynapse directory not found: {py_dir}"))
        return 1
    
    if not os.path.isdir(og_dir):
        print(fail(f"og2 directory not found: {og_dir}"))
        return 1
    
    return run_validation(py_dir, og_dir)


if __name__ == "__main__":
    sys.exit(main())
