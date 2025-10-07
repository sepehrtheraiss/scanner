# Scanner

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Last Commit](https://img.shields.io/github/last-commit/sepehrtheraiss/scanner)](https://github.com/sepehrtheraiss/scanner)

A lightweight **stock data analysis tool** written in Python.  
Scanner fetches historical stock data, detects trends, and generates visualizations for quick insights.

---

## ✨ Features

- 📈 Fetch historical market data via [yfinance](https://pypi.org/project/yfinance/)  
- 🔎 Analyze price trends and detect market patterns  
- 🖼️ Visualize results with charts and overlays  
- ⚡ Command-line interface for fast, reproducible workflows  

---

## 📂 Project Structure

```
scanner/
├── fetch.py     # Fetch stock data (via yfinance/requests)
├── trends.py    # Compute trends and indicators
├── plot.py      # Plot charts with matplotlib
├── main.py      # CLI entry point
└── examples/    # Example outputs (CSV, charts)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/sepehrtheraiss/scanner.git
cd scanner
pip install pandas matplotlib yfinance requests
```

---

## 🚀 Usage

Run directly with Python:

```bash
python main.py --ticker AAPL --start 2023-07-21 --end 2024-05-17 --interval 1d
```

### Arguments
- `--ticker` : Stock ticker symbol (e.g. `AAPL`)  
- `--start` : Start date (`YYYY-MM-DD`)  
- `--end` : End date (`YYYY-MM-DD`)  
- `--interval` : Data interval (`1d`, `1h`, etc.)  

---

## 📊 Example

```bash
python main.py --ticker TSLA --start 2024-01-01 --end 2024-05-01 --interval 1d
```

Outputs:
- `TSLA_2024-01-01_2024-05-01_1d.csv` — raw price data  
- `TSLA_trend.png` — chart with trends highlighted  

---

## 🛠️ Development

Pull requests are welcome! To contribute:
```bash
# clone & create virtual environment
git clone https://github.com/sepehrtheraiss/scanner.git
cd scanner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # (optional once added)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute.

---
