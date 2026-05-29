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


def calculate_ddm(dividend, required_return, dividend_growth=0.05):
    if dividend > 1 and required_return > dividend_growth:
        try:
            return dividend * (1 + dividend_growth) / (required_return - dividend_growth)
        except ZeroDivisionError:
            pass
    return None


def calculate_dcf(fcf, shares_outstanding, required_return):
    """Returns (dcf_price, annual_fcf, total_value, warning_msg)."""
    if not (fcf and fcf > 0 and shares_outstanding > 0):
        return None, None, None, None

    try:
        annual_fcf = fcf

        if annual_fcf > 20_000_000_000:
            short_term_growth, terminal_growth = 0.05, 0.02
        elif annual_fcf > 5_000_000_000:
            short_term_growth, terminal_growth = 0.06, 0.025
        else:
            short_term_growth, terminal_growth = 0.10, 0.03

        forecast_years = 5
        discount_rate = required_return

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
