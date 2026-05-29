def calculate_5yr_cagr_from_fmp(df, field):
    df = df.sort_values("date")
    start = df[field].iloc[0]
    end = df[field].iloc[-1]
    n_years = len(df) - 1

    if start <= 0 or end <= 0 or n_years == 0:
        return None

    return (end / start) ** (1 / n_years) - 1


def calculate_pe_fair_value(eps, sector_pe):
    if eps and sector_pe:
        return eps * sector_pe
    return None


def calculate_ddm(dividend, required_return, current_price=None, dividend_growth=0.05):
    """
    Gordon Growth Model: P = D1 / (r - g).

    Only meaningful for companies where dividends represent a significant portion
    of shareholder returns. Requires dividend yield >= 1.5% — below that threshold
    the stock's value is driven by growth/buybacks, not dividends, and DDM produces
    nonsensical results (e.g. AAPL at 0.5% yield gives a ~$8 fair value).
    """
    if not dividend or dividend <= 0:
        return None
    if required_return <= dividend_growth:
        return None

    # Yield gate: DDM is theoretically invalid for low-yield stocks
    if current_price and current_price > 0:
        yield_ = dividend / current_price
        if yield_ < 0.015:   # < 1.5% yield → model not applicable
            return None
    else:
        # No price available: fall back to a minimum dividend floor
        if dividend < 1.0:
            return None

    try:
        return dividend * (1 + dividend_growth) / (required_return - dividend_growth)
    except ZeroDivisionError:
        return None


def calculate_dcf(fcf, shares_outstanding, required_return, revenue_cagr=None):
    """
    Returns (dcf_price, annual_fcf, total_value, warning_msg).

    revenue_cagr : historical revenue CAGR from financials — drives the short-term
                   growth assumption. When None a conservative 5% default is used.
                   Clamped to [1%, 20%] to avoid compounding extreme outliers.
    """
    if not (fcf and fcf > 0 and shares_outstanding > 0):
        return None, None, None, None

    try:
        annual_fcf = fcf

        # Short-term growth: anchored to actual historical revenue trajectory,
        # mean-reverted (future growth is rarely as extreme as the past).
        if revenue_cagr is not None and revenue_cagr > 0:
            short_term_growth = max(0.01, min(0.20, revenue_cagr * 0.80))
        else:
            short_term_growth = 0.05   # conservative default when no history available

        # Terminal growth: long-run nominal GDP (2.5% is the standard DCF convention).
        terminal_growth = 0.025

        forecast_years = 5
        discount_rate  = required_return

        fcf_list = [
            annual_fcf * (1 + short_term_growth) ** y / (1 + discount_rate) ** y
            for y in range(1, forecast_years + 1)
        ]

        final_fcf = annual_fcf * (1 + short_term_growth) ** forecast_years
        terminal_value = final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        discounted_terminal_value = terminal_value / (1 + discount_rate) ** forecast_years

        total_value = sum(fcf_list) + discounted_terminal_value
        dcf_price = total_value / shares_outstanding

        return dcf_price, annual_fcf, total_value, None

    except Exception as e:
        return None, None, None, f"DCF calculation failed: {e}"


def calculate_lynch_fair_value(eps, growth_rate):
    """Returns (lynch_price, caption, warning_msg)."""
    if growth_rate < 0.05:
        return None, None, "⚠️ Peter Lynch Fair Value: Not applicable — firm does not meet growth criteria (CAGR < 5%)."
    elif growth_rate < 0.10:
        price = eps * growth_rate * 100
        caption = f"⚠️ Moderate-growth stock. PEG=1 applied conservatively (CAGR: {growth_rate:.2%})."
        return price, caption, None
    else:
        clamped_growth = min(growth_rate, 0.25)
        price = eps * clamped_growth * 100
        caption = f"✅ Based on 5-year EBITDA CAGR: {clamped_growth:.2%}"
        return price, caption, None
