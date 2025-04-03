# 📊 Cassandra EDAS ( Exploratory Data Analysis Script)

> A command-line Python tool for comprehensive, modular, and customizable exploratory data analysis (EDA).  
> Supports CSV, Excel, and JSON files, and can generate plots and automated reports.

---

## Features

-  Load data from `.csv`, `.xlsx`, `.xls`, `.json`
-  Dataset overview (shape, columns, head/tail, missing values)
-  Descriptive stats for numerical and categorical columns
-  Skewness and kurtosis
-  Duplicate row detection
-  Value counts for categorical variables
-  Correlation heatmap
-  Distribution histograms & boxplots
-  Pairplots (if dataset is small enough)
-  Missing values visualization (using `missingno`, optional)
-  HTML-based automated report (using `ydata-profiling`, optional)
-  Unit tests built-in
-  Optional use of Dask for large CSVs

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Required packages:**

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`

**Optional:**

- `missingno` – for missing value visualization
- `ydata-profiling` – for automated HTML EDA report
- `dask` – for efficient large CSV loading

---

##  Usage

```bash
python exploring.py path/to/your/data.csv
```

---

## Command-Line Options

| Flag                | Description |
|---------------------|-------------|
| `--run-overview`    | Show dataset info (shape, columns, head, missing values) |
| `--run-stats`       | Show descriptive stats, skew/kurtosis, duplicates |
| `--run-plots`       | Generate correlation heatmap and other visualizations |
| `--save-plots DIR`  | Save plots to a directory instead of displaying them |
| `--chunksize N`     | Load CSV in chunks (recommended for large files) |
| `--run-report FILE` | Output an automated HTML EDA report |
| `--run-tests`       | Run internal tests and exit |

---

##  Examples

### 1. Run full EDA:
```bash
python exploring.py data/myfile.csv
```

### 2. Only show overview:
```bash
python exploring.py data/myfile.csv --run-overview
```

### 3. Run overview + plots and save them:
```bash
python exploring.py data/myfile.csv --run-overview --run-plots --save-plots outputs/
```

### 4. Generate an automated HTML report:
```bash
python exploring.py data/myfile.csv --run-report eda_report.html
```

### 5. Handle large CSVs with chunking:
```bash
python exploring.py data/large_dataset.csv --chunksize 100000
```

### 6. Run internal test suite:
```bash
python exploring.py dummy.csv --run-tests
```

---

## Output

If `--save-plots` is enabled, it will save:

- `correlation_heatmap.png`
- `[feature]_distribution.png`
- `[feature]_boxplot.png`
- `pairplot.png` (for small datasets)
- `missing_values_matrix.png` (if `missingno` is installed)

---

## Tips

- Install `dask` for better performance with large datasets.
- Use `--run-report` to quickly generate a portable HTML summary.
- Combine flags for modular EDA control.

---

## Troubleshooting

- `ModuleNotFoundError`: Install missing packages (`pip install package-name`)
- `FileNotFoundError`: Check your input file path
- `ValueError: Unsupported file type`: Use `.csv`, `.xlsx`, `.xls`, or `.json`

---

## Contribution

Feel free to fork, improve, or extend the script. Pull requests are welcome. Also, feel free to ask me if you can pull in the MASTER branch

---
