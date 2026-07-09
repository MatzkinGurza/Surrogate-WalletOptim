import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from Utils.extraction_module import StockDataFetcher, StockDataWarehouse
from Utils.plot_module import StockTimeSeriesPlotter

with open(ROOT_DIR / "Params" / "paths.json", "r") as f:
    paths = json.load(f)
with open(ROOT_DIR / "Params" / "configs.json", "r") as f:
    configs = json.load(f)
with open(ROOT_DIR / "Params" / "stocks.json", "r") as f:
    stocks_cfg = json.load(f)

plots_cfg = configs["plots"]
regenerate_plots = plots_cfg["regenerate_plots"]

if regenerate_plots:
    print("regenerate_plots=true: os plots serao regerados.")

    fetcher = StockDataFetcher(
        start_date=stocks_cfg["start_date"],
        end_date=stocks_cfg["end_date"],
        interval=stocks_cfg["interval"],
    )
    warehouse = StockDataWarehouse(fetcher)
    warehouse.load(str(ROOT_DIR / paths["stocks_warehouse_file"]))

    plotter = StockTimeSeriesPlotter(warehouse)
    fig = plotter.plot_price_series(
        normalize=plots_cfg["price_series_normalized"],
        log_scale=plots_cfg["price_series_log_scale"],
    )

    output_path = ROOT_DIR / paths["plots_dir"] / f"price_series.{plots_cfg['output_format']}"
    plotter.save_figure(fig, str(output_path))
    print(f"Plot salvo em {output_path}")
else:
    print("regenerate_plots=false: os plots nao serao regerados.")
