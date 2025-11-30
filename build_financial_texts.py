# build_financial_texts.py
#
# Create clean, sentence-based financial snapshot files
# for a small set of public companies using yfinance.
# Output: financial_statements/TICKER.txt

import os
import yfinance as yf

# ---- COMPANIES TO INCLUDE ----
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "NFLX", "NVDA", "JPM", "XOM"
]

os.makedirs("financial_statements", exist_ok=True)


def fmt_dollar(x):
    """Format large dollar values in human form."""
    if x is None:
        return "unknown"
    try:
        x = float(x)
    except Exception:
        return "unknown"
    if x >= 1e12:
        return f"{x / 1e12:.2f} trillion USD"
    if x >= 1e9:
        return f"{x / 1e9:.2f} billion USD"
    if x >= 1e6:
        return f"{x / 1e6:.2f} million USD"
    return f"{x:,.0f} USD"


def fmt_pct(x):
    """Format fraction -> percent string."""
    if x is None:
        return "unknown"
    try:
        x = float(x)
    except Exception:
        return "unknown"
    return f"{x * 100:.1f}%"


def main():
    for t in TICKERS:
        print(f"Building text for {t}...")
        stock = yf.Ticker(t)
        info = stock.info or {}

        name = info.get("longName") or t
        sector = info.get("sector", "an unknown sector")
        industry = info.get("industry", "an unknown industry")
        country = info.get("country", "an unknown country")

        market_cap = fmt_dollar(info.get("marketCap"))
        total_revenue = fmt_dollar(info.get("totalRevenue"))
        profit_margin = fmt_pct(info.get("profitMargins"))
        op_margin = fmt_pct(info.get("operatingMargins"))
        gross_margin = fmt_pct(info.get("grossMargins"))

        employees = info.get("fullTimeEmployees")
        pe = info.get("trailingPE")
        summary = info.get("longBusinessSummary")

        lines = []

        # Basic identity
        lines.append(
            f"{name} ({t}) is a company in the {sector} sector and "
            f"{industry} industry based in {country}."
        )

        # Key headline metrics
        lines.append(
            f"As of the latest data snapshot used in this project, the market cap of "
            f"{name} was {market_cap}."
        )
        lines.append(
            f"The company generated total revenue of {total_revenue} over the "
            f"trailing twelve months."
        )
        lines.append(
            f"Its profit margin was {profit_margin}, operating margin was "
            f"{op_margin}, and gross margin was {gross_margin}."
        )

        # Extra descriptive metrics
        if employees:
            lines.append(f"{name} employs roughly {employees:,} people.")
        if pe:
            try:
                lines.append(
                    f"The trailing price-to-earnings (P/E) ratio was about {float(pe):.1f}."
                )
            except Exception:
                pass

        # Optional business summary
        if summary:
            lines.append(f"Business summary: {summary}")

        out_path = os.path.join("financial_statements", f"{t}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
