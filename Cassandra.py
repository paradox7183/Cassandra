#!/usr/bin/env python

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional
print (r"""
  /$$$$$$                                                         /$$                   
 /$$__  $$                                                       | $$                   
| $$  \__/  /$$$$$$   /$$$$$$$ /$$$$$$$  /$$$$$$  /$$$$$$$   /$$$$$$$  /$$$$$$  /$$$$$$ 
| $$       |____  $$ /$$_____//$$_____/ |____  $$| $$__  $$ /$$__  $$ /$$__  $$|____  $$
| $$        /$$$$$$$|  $$$$$$|  $$$$$$   /$$$$$$$| $$  \ $$| $$  | $$| $$  \__/ /$$$$$$$
| $$    $$ /$$__  $$ \____  $$\____  $$ /$$__  $$| $$  | $$| $$  | $$| $$      /$$__  $$
|  $$$$$$/|  $$$$$$$ /$$$$$$$//$$$$$$$/|  $$$$$$$| $$  | $$|  $$$$$$$| $$     |  $$$$$$$
 \______/  \_______/|_______/|_______/  \_______/|__/  |__/ \_______/|__/      \_______/ 0.0.1
Definitive Exploratory Data Analysis Script                                                                           
       """)

try:
    import missingno as msno
except ImportError:
    msno = None

try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None


try:
    import dask.dataframe as dd
    dask_installed = True
except ImportError:
    dask_installed = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sns.set(style="whitegrid")

def load_data(filepath: Union[str, Path], chunksize: Optional[int] = None) -> pd.DataFrame:

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    ext = path.suffix.lower()
    if ext == '.csv':
        if chunksize is not None:
            logger.info("Using pandas chunking with chunksize=%s", chunksize)
            df = pd.concat(pd.read_csv(path, chunksize=chunksize))
        else:
            if dask_installed:
                logger.info("Using Dask for loading CSV")
                ddf = dd.read_csv(path)
                df = ddf.compute()
            else:
                logger.info("Dask not installed; falling back to pandas")
                df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    elif ext == '.json':
        df = pd.read_json(path)
    else:
        raise ValueError("Unsupported file type.")

    logger.info(f"Loaded data with shape {df.shape}")
    return df

def overview(df: pd.DataFrame) -> None:
   
    logger.info("Displaying basic dataset information...")
    print("Dataset Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print("\nData Info:")
    df.info()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print("\nMissing Values per Column (%):\n", missing_pct.round(2))

def descriptive_stats(df: pd.DataFrame) -> None:
  
    logger.info("Calculating descriptive statistics...")
    print("\nDescriptive Statistics (Numerical):")
    print(df.describe())
    
    categorical_cols = df.select_dtypes(include='object').columns
    if categorical_cols.size > 0:
        print("\nDescriptive Statistics (Categorical):")
        print(df[categorical_cols].describe())
    else:
        print("\nNo categorical columns found.")
    
    # Skewness and Kurtosis for numerical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.size > 0:
        print("\nSkewness of Numerical Columns:")
        print(df[numeric_cols].skew())
        print("\nKurtosis of Numerical Columns:")
        print(df[numeric_cols].kurt())

def duplicate_and_value_counts(df: pd.DataFrame) -> None:
   
    logger.info("Analyzing duplicates and categorical values...")
    print("\nNumber of Duplicate Rows:", df.duplicated().sum())
    
    categorical_cols = df.select_dtypes(include='object').columns
    if categorical_cols.size > 0:
        for col in categorical_cols:
            print(f"\nValue Counts for '{col}':")
            print(df[col].value_counts())
    else:
        print("\nNo categorical columns for value counts analysis.")

def correlation_and_heatmap(df: pd.DataFrame, save_plots: Optional[Path] = None) -> None:
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.size > 1:
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        if save_plots:
            plot_file = save_plots / "correlation_heatmap.png"
            plt.savefig(plot_file)
            logger.info(f"Saved correlation heatmap to {plot_file}")
        else:
            plt.show()
        plt.close()
    else:
        print("\nNot enough numerical columns for correlation analysis.")

def additional_visualizations(df: pd.DataFrame, save_plots: Optional[Path] = None) -> None:
   
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Distribution histograms with KDE
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        if save_plots:
            plot_file = save_plots / f"{col}_distribution.png"
            plt.savefig(plot_file)
            logger.info(f"Saved distribution plot for {col} to {plot_file}")
        else:
            plt.show()
        plt.close()
    
    # Box plots for outlier detection
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot for {col}")
        if save_plots:
            plot_file = save_plots / f"{col}_boxplot.png"
            plt.savefig(plot_file)
            logger.info(f"Saved boxplot for {col} to {plot_file}")
        else:
            plt.show()
        plt.close()
    
    # Pairplot if dataset is small enough
    if numeric_cols.size >= 2 and df.shape[0] < 5000:
        pairplot_fig = sns.pairplot(df[numeric_cols])
        pairplot_fig.fig.suptitle("Pairplot of Numerical Features", y=1.02)
        if save_plots:
            plot_file = save_plots / "pairplot.png"
            pairplot_fig.savefig(plot_file)
            logger.info(f"Saved pairplot to {plot_file}")
        else:
            plt.show()
        plt.close()
    
    # Missing values visualization using missingno (if available)
    if msno is not None:
        msno.matrix(df)
        plt.title("Missing Values Matrix")
        if save_plots:
            plot_file = save_plots / "missing_values_matrix.png"
            plt.savefig(plot_file)
            logger.info(f"Saved missing values matrix to {plot_file}")
        else:
            plt.show()
        plt.close()
    else:
        logger.info("Missingno is not installed. Skipping missing values visualization.")

def generate_automated_report(df: pd.DataFrame, output_file: str = "EDA_report.html") -> None:

    if ProfileReport is not None:
        report = ProfileReport(df, title="Automated EDA Report", explorative=True)
        report.to_file(output_file)
        logger.info(f"Automated EDA report saved to {output_file}")
    else:
        logger.info("pandas_profiling is not installed. Skipping automated report generation.")

def run_tests() -> None:

    logger.info("Running tests...")
    try:
        # Example test: create a sample DataFrame and check its shape
        df_test = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        assert df_test.shape == (3, 2)
        logger.info("Test passed: DataFrame shape is correct.")
    except AssertionError:
        logger.error("Test failed: DataFrame shape is not as expected.")
    # Add more tests as needed

def main(args) -> None:
    # Run tests if requested
    if args.run_tests:
        run_tests()
        return

   
    save_plots_dir = None
    if args.save_plots:
        save_plots_dir = Path(args.save_plots)
        save_plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots will be saved to: {save_plots_dir}")
    
    try:
        df = load_data(args.filepath, chunksize=args.chunksize)
    except Exception as e:
        logger.error(e)
        return

    # Selective execution: if any of these flags are set, only run those analyses.
    selective = args.run_overview or args.run_stats or args.run_plots
    if selective:
        if args.run_overview:
            overview(df)
        if args.run_stats:
            descriptive_stats(df)
            duplicate_and_value_counts(df)
        if args.run_plots:
            correlation_and_heatmap(df, save_plots=save_plots_dir)
            additional_visualizations(df, save_plots=save_plots_dir)
    else:
        # Default: run all analyses
        overview(df)
        descriptive_stats(df)
        duplicate_and_value_counts(df)
        correlation_and_heatmap(df, save_plots=save_plots_dir)
        additional_visualizations(df, save_plots=save_plots_dir)
    
    # Generate automated report if the --run-report flag is provided
    if args.run_report:
        generate_automated_report(df, output_file=args.run_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Definitive EDA Script")
    parser.add_argument("filepath", type=str, help="Path to the dataset file (CSV, Excel, JSON, etc.)")
    # Original flag is preserved as --run-report
    parser.add_argument("--run-report", type=str, default=None,
                        help="Output file path for the automated EDA report (HTML). Optional.")
    
    # New flags added without altering the existing ones
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size for reading large CSV files")
    parser.add_argument("--save-plots", type=str, default=None, help="Directory to save plots instead of displaying them")
    parser.add_argument("--run-overview", action="store_true", help="Run dataset overview analysis")
    parser.add_argument("--run-stats", action="store_true", help="Run descriptive statistics and duplicate analysis")
    parser.add_argument("--run-plots", action="store_true", help="Run correlation and additional visualizations")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    
    args = parser.parse_args()
    main(args)
