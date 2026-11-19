"""Train the price model and write models/price_model.ubj + models/metadata.json.

Usage:  python train.py
"""

from housefinder import data, model


def main() -> None:
    market = data.load_market()
    booster, meta = model.train(market)
    model.save(booster, meta)
    m = meta["metrics"]
    print(f"Trained on {meta['n_train']} sales, tested on {meta['n_test']} unseen sales")
    for name in ("cv", "holdout", "legacy_holdout"):
        s = m[name]
        print(f"  {name:15s} R2={s['r2']:.4f}  MAE=${s['mae']:,.0f}  MAPE={s['mape']:.2f}%")
    print("  interval coverage:", {k: round(v, 3) for k, v in m["coverage"].items()})


if __name__ == "__main__":
    main()
