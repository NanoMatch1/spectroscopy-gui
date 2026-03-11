from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


def read_two_block_optical_constants(filepath: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a text file containing two whitespace-delimited datasets separated by a blank line.

    Expected format:
        Block 1 header: Frequency <real-column-name>
        ...
        <blank line>
        Block 2 header: Frequency <imag-column-name>
        ...

    Returns
    -------
    df1, df2 : pd.DataFrame
        Two dataframes, each containing a 'Frequency' column and one data column.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")

    # Split on blank lines, ignoring accidental extra whitespace-only blocks
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        raise ValueError("Could not find two data blocks separated by a blank line.")

    df1 = pd.read_csv(io.StringIO(blocks[0]), sep=r"\s+", engine="python")
    df2 = pd.read_csv(io.StringIO(blocks[1]), sep=r"\s+", engine="python")

    if "Frequency" not in df1.columns or "Frequency" not in df2.columns:
        raise ValueError("Both datasets must contain a 'Frequency' column.")

    if len(df1.columns) != 2 or len(df2.columns) != 2:
        raise ValueError("Each dataset must contain exactly two columns.")

    return df1, df2


def interpolate_onto_union_axis(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Interpolate two datasets onto the union of their frequency axes.

    Outside each dataset's original range, values are set to NaN.
    """
    # Identify the non-frequency column names
    col1 = [c for c in df1.columns if c != "Frequency"][0]
    col2 = [c for c in df2.columns if c != "Frequency"][0]

    # Ensure sorting by frequency
    df1 = df1.sort_values("Frequency").drop_duplicates(subset="Frequency")
    df2 = df2.sort_values("Frequency").drop_duplicates(subset="Frequency")

    f1 = df1["Frequency"].to_numpy(dtype=float)
    y1 = df1[col1].to_numpy(dtype=float)

    f2 = df2["Frequency"].to_numpy(dtype=float)
    y2 = df2[col2].to_numpy(dtype=float)

    # Common axis = union of both original axes
    f_common = np.union1d(f1, f2)

    # Interpolate
    y1_interp = np.interp(f_common, f1, y1)
    y2_interp = np.interp(f_common, f2, y2)

    # Mask out extrapolated regions so they become NaN
    y1_interp[(f_common < f1.min()) | (f_common > f1.max())] = np.nan
    y2_interp[(f_common < f2.min()) | (f_common > f2.max())] = np.nan

    return pd.DataFrame({
        "Frequency": f_common,
        col1: y1_interp,
        col2: y2_interp,
    })


def export_three_column_file(
    df: pd.DataFrame,
    output_path: str | Path,
    fmt: str = "%.8E",
) -> None:
    """
    Export the aligned dataframe as a whitespace-delimited text file
    with header and three columns in the same general style as the input.
    """
    output_path = Path(output_path)

    # Use tab separation for clean column formatting while keeping it plain text
    df.to_csv(
        output_path,
        sep="\t",
        index=False,
        float_format=fmt,
        na_rep="NaN",
    )


def align_and_export(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Full pipeline:
    - read two-block file
    - interpolate onto common axis
    - export aligned three-column file
    """
    df1, df2 = read_two_block_optical_constants(input_path)
    aligned = interpolate_onto_union_axis(df1, df2)
    export_three_column_file(aligned, output_path)
    return aligned


if __name__ == "__main__":
    this_dir = Path(__file__).parent
    input_file = this_dir / "vasilis_optical-constants.txt"
    output_file = this_dir / "vasilis_optical-constants_aligned.txt"

    aligned_df = align_and_export(input_file, output_file)

    print(aligned_df.head())
    print(f"\nSaved aligned file to: {output_file}")