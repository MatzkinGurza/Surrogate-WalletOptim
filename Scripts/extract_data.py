import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from Utils.extraction_module import StockDataFetcher, StockDataWarehouse

with open(ROOT_DIR / "Params" / "paths.json", "r") as f:
    paths = json.load(f)
with open(ROOT_DIR / "Params" / "stocks.json", "r") as f:
    stocks_cfg = json.load(f)

fetcher = StockDataFetcher(
    start_date=stocks_cfg["start_date"],
    end_date=stocks_cfg["end_date"],
    interval=stocks_cfg["interval"],
)
warehouse = StockDataWarehouse(fetcher)

failed = warehouse.add_stocks(stocks_cfg["tickers"], auto_adjust=stocks_cfg["auto_adjust"])
if failed:
    print(f"Falha ao obter dados para: {failed}")

warehouse_path = ROOT_DIR / paths["stocks_warehouse_file"]
warehouse.save(str(warehouse_path))
print(f"Dados de {len(warehouse.list_stocks())} ativos salvos em {warehouse_path}")
