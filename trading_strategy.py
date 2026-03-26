"""
Estrategia Swing Trading: MACD + RSI + ATR Stop
------------------------------------------------
Horizonte objetivo: 1–4 semanas por operacion.

Señales LONG:
  ENTRADA : MACD cruza señal hacia arriba  AND  RSI entre 40-65  AND  precio > EMA50
  SALIDA  : MACD cruza señal hacia abajo   OR   stop loss (1.5x ATR)  OR  take profit (3x ATR)

Señales SHORT (opcional, activable):
  ENTRADA : MACD cruza señal hacia abajo   AND  RSI entre 35-60  AND  precio < EMA50
  SALIDA  : MACD cruza señal hacia arriba  OR   stop loss (1.5x ATR)  OR  take profit (3x ATR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Optional, Literal


# ──────────────────────────────────────────────
# Indicadores Técnicos
# ──────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Retorna (macd_line, signal_line, histogram)."""
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — mide la volatilidad real del precio."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ──────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    entry_price: float
    side:        Literal["long", "short"] = "long"
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float] = None
    exit_reason: str = ""

    @property
    def hold_days(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.side == "long":
            return (self.exit_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def pnl_pct(self) -> Optional[float]:
        return self.pnl * 100 if self.pnl is not None else None


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if (t.pnl or 0) > 0)
        return wins / len(self.closed_trades) * 100

    @property
    def avg_hold_days(self) -> float:
        days = [t.hold_days for t in self.closed_trades if t.hold_days is not None]
        return sum(days) / len(days) if days else 0.0

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pnl_pct for t in self.closed_trades if (t.pnl or 0) > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.closed_trades if (t.pnl or 0) <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(t.pnl_pct for t in self.closed_trades if (t.pnl or 0) > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.closed_trades if (t.pnl or 0) <= 0))
        return gross_win / gross_loss if gross_loss else float("inf")

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd   = (self.equity_curve - peak) / peak
        return dd.min() * 100

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    def summary(self) -> str:
        exit_reasons: dict = {}
        for t in self.closed_trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        lines = [
            "=" * 52,
            "         RESUMEN DEL BACKTESTING (SWING)",
            "=" * 52,
            f"  Operaciones totales  : {self.total_trades}",
            f"  Win rate             : {self.win_rate:.1f}%",
            f"  Retorno total        : {self.total_return:.2f}%",
            f"  Max drawdown         : {self.max_drawdown:.2f}%",
            f"  Sharpe ratio         : {self.sharpe_ratio:.2f}",
            f"  Profit factor        : {self.profit_factor:.2f}",
            f"  Ganancia media       : +{self.avg_win_pct:.2f}%",
            f"  Perdida media        : {self.avg_loss_pct:.2f}%",
            f"  Hold promedio (dias) : {self.avg_hold_days:.1f}",
            "  Salidas por razon    :",
        ]
        for reason, count in sorted(exit_reasons.items()):
            lines.append(f"    {reason:<20}: {count}")
        lines.append("=" * 52)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Generador de datos simulados (OHLCV)
# ──────────────────────────────────────────────

def generate_price_data(
    n_days: int = 750,
    start_price: float = 100.0,
    volatility: float = 0.014,
    seed: int = 7,
) -> pd.DataFrame:
    """
    ~3 años de datos con ciclos swing realistas (30-90 dias por ciclo).
    """
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-03-26"), periods=n_days)

    drift_array = np.empty(n_days)
    i, bullish = 0, True
    while i < n_days:
        cycle_len = np.random.randint(30, 90)      # ciclos cortos para swing
        drift = 0.0008 if bullish else -0.0005
        end   = min(i + cycle_len, n_days)
        drift_array[i:end] = drift
        bullish = not bullish
        i = end

    returns = np.random.normal(drift_array, volatility, n_days)
    close   = start_price * np.exp(np.cumsum(returns))

    daily_vol = np.abs(np.random.normal(0, volatility * 0.5, n_days))
    high   = close * (1 + daily_vol)
    low    = close * (1 - daily_vol)
    open_  = np.roll(close, 1);  open_[0] = start_price
    base_v = np.random.randint(1_000_000, 4_000_000, n_days)
    volume = (base_v * (1 + np.abs(returns) * 12)).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ──────────────────────────────────────────────
# Estrategia Swing
# ──────────────────────────────────────────────

class SwingMACDStrategy:
    """
    Swing Trading: MACD + RSI + ATR-Stop

    Parámetros
    ----------
    macd_fast      : EMA rápida del MACD (default 12)
    macd_slow      : EMA lenta del MACD  (default 26)
    macd_signal    : EMA de la línea de señal (default 9)
    rsi_period     : período del RSI
    rsi_buy_min/max: rango del RSI para entrar largo (momentum positivo, sin sobrecompra)
    rsi_sell_min/max: rango del RSI para entrar corto
    ema_trend      : EMA para filtro de tendencia general
    atr_period     : período del ATR
    atr_sl_mult    : multiplicador ATR para stop loss
    atr_tp_mult    : multiplicador ATR para take profit
    allow_short    : permitir posiciones cortas
    initial_cash   : capital inicial en USD
    """

    def __init__(
        self,
        macd_fast: int      = 12,
        macd_slow: int      = 26,
        macd_signal: int    = 9,
        rsi_period: int     = 14,
        rsi_buy_min: float  = 40.0,
        rsi_buy_max: float  = 65.0,
        rsi_sell_min: float = 35.0,
        rsi_sell_max: float = 60.0,
        ema_trend: int      = 50,
        atr_period: int     = 14,
        atr_sl_mult: float  = 1.5,
        atr_tp_mult: float  = 3.0,
        allow_short: bool   = True,
        initial_cash: float = 10_000.0,
    ):
        self.macd_fast    = macd_fast
        self.macd_slow    = macd_slow
        self.macd_signal  = macd_signal
        self.rsi_period   = rsi_period
        self.rsi_buy_min  = rsi_buy_min
        self.rsi_buy_max  = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min
        self.rsi_sell_max = rsi_sell_max
        self.ema_trend    = ema_trend
        self.atr_period   = atr_period
        self.atr_sl_mult  = atr_sl_mult
        self.atr_tp_mult  = atr_tp_mult
        self.allow_short  = allow_short
        self.initial_cash = initial_cash

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["macd"], data["macd_sig"], data["macd_hist"] = macd(
            data["close"], self.macd_fast, self.macd_slow, self.macd_signal
        )
        data["rsi"]      = rsi(data["close"], self.rsi_period)
        data["ema_trend"] = ema(data["close"], self.ema_trend)
        data["atr"]       = atr(data["high"], data["low"], data["close"], self.atr_period)

        # Cruce del MACD: +1 cruce alcista, -1 cruce bajista
        cross            = np.sign(data["macd"] - data["macd_sig"])
        data["macd_cross"] = cross.diff()
        return data

    def backtest(self, df: pd.DataFrame):
        data = self.compute_indicators(df)
        data["signal"] = 0

        cash          = self.initial_cash
        result        = BacktestResult()
        equity_vals   : List[float] = []
        current_trade : Optional[Trade] = None

        # Estado de posición activa
        in_position   = False
        pos_side      : str   = "long"
        pos_shares    : float = 0.0
        pos_entry_val : float = 0.0   # capital comprometido al entrar

        for date, row in data.iterrows():
            price = row["close"]

            if not in_position:
                # ── ENTRADA LONG ────────────────────────────────────────────
                long_signal = (
                    row["macd_cross"] > 0
                    and self.rsi_buy_min <= row["rsi"] <= self.rsi_buy_max
                    and price > row["ema_trend"]
                )
                # ── ENTRADA SHORT ───────────────────────────────────────────
                short_signal = self.allow_short and (
                    row["macd_cross"] < 0
                    and self.rsi_sell_min <= row["rsi"] <= self.rsi_sell_max
                    and price < row["ema_trend"]
                )

                if long_signal or short_signal:
                    pos_side      = "long" if long_signal else "short"
                    atr_v         = row["atr"]
                    sl = price - self.atr_sl_mult * atr_v if pos_side == "long" else price + self.atr_sl_mult * atr_v
                    tp = price + self.atr_tp_mult * atr_v if pos_side == "long" else price - self.atr_tp_mult * atr_v

                    pos_entry_val = cash
                    pos_shares    = cash / price
                    cash          = 0.0
                    in_position   = True

                    current_trade = Trade(
                        entry_date=date, entry_price=price,
                        side=pos_side, stop_loss=sl, take_profit=tp,
                    )
                    result.trades.append(current_trade)
                    data.at[date, "signal"] = 1 if pos_side == "long" else -1

            else:
                t = current_trade
                exit_reason = None

                if pos_side == "long":
                    if price <= t.stop_loss:
                        exit_reason = "stop_loss"
                    elif price >= t.take_profit:
                        exit_reason = "take_profit"
                    elif row["macd_cross"] < 0:
                        exit_reason = "macd_cross"
                else:  # short
                    if price >= t.stop_loss:
                        exit_reason = "stop_loss"
                    elif price <= t.take_profit:
                        exit_reason = "take_profit"
                    elif row["macd_cross"] > 0:
                        exit_reason = "macd_cross"

                if exit_reason:
                    # PnL real aplicado al capital comprometido
                    pnl_ratio = (price - t.entry_price) / t.entry_price
                    if pos_side == "short":
                        pnl_ratio = -pnl_ratio
                    cash        = pos_entry_val * (1 + pnl_ratio)
                    in_position = False
                    pos_shares  = 0.0

                    t.exit_date   = date
                    t.exit_price  = price
                    t.exit_reason = exit_reason
                    current_trade = None
                    data.at[date, "signal"] = -1 if pos_side == "long" else 1

            # Equity mark-to-market
            if in_position:
                pnl_ratio = (price - current_trade.entry_price) / current_trade.entry_price
                if pos_side == "short":
                    pnl_ratio = -pnl_ratio
                equity_vals.append(pos_entry_val * (1 + pnl_ratio))
            else:
                equity_vals.append(cash)

        # Cerrar posición abierta al fin del período
        if in_position and current_trade is not None:
            last = data["close"].iloc[-1]
            pnl_ratio = (last - current_trade.entry_price) / current_trade.entry_price
            if pos_side == "short":
                pnl_ratio = -pnl_ratio
            cash = pos_entry_val * (1 + pnl_ratio)
            current_trade.exit_date   = data.index[-1]
            current_trade.exit_price  = last
            current_trade.exit_reason = "fin_periodo"

        result.equity_curve = pd.Series(equity_vals, index=data.index)
        return result, data


# ──────────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────────

def plot_results(data: pd.DataFrame, result: BacktestResult, ticker: str = "ACTIVO"):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"Swing Trading: MACD + RSI + ATR-Stop — {ticker}\n"
        f"Win Rate: {result.win_rate:.1f}%  |  "
        f"Retorno: {result.total_return:.2f}%  |  "
        f"Sharpe: {result.sharpe_ratio:.2f}  |  "
        f"Trades: {result.total_trades}  |  "
        f"Hold prom: {result.avg_hold_days:.0f} dias",
        fontsize=11,
    )
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.42)

    # ── Panel 1: Precio + EMA50 + señales
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data["close"],     label="Precio",  color="black",  lw=1.0)
    ax1.plot(data.index, data["ema_trend"], label="EMA 50",  color="orange", lw=1.2, ls="--")

    for t in result.closed_trades:
        color = "green" if (t.pnl or 0) > 0 else "red"
        ax1.axvspan(t.entry_date, t.exit_date, alpha=0.08, color=color)

    longs  = data[(data["signal"] == 1)]
    shorts = data[(data["signal"] == -1)]
    ax1.scatter(longs.index,  longs["close"],  marker="^", color="green", s=90, zorder=5, label="Entrada Long")
    ax1.scatter(shorts.index, shorts["close"], marker="v", color="red",   s=90, zorder=5, label="Entrada Short / Salida")

    ax1.set_ylabel("Precio (USD)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # ── Panel 2: MACD
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(data.index, data["macd"],     label="MACD",   color="blue",   lw=1)
    ax2.plot(data.index, data["macd_sig"], label="Señal",  color="orange", lw=1)
    colors = ["green" if v >= 0 else "red" for v in data["macd_hist"]]
    ax2.bar(data.index, data["macd_hist"], color=colors, alpha=0.5, width=0.8, label="Histograma")
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_ylabel("MACD")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(alpha=0.25)

    # ── Panel 3: RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(data.index, data["rsi"], color="purple", lw=1)
    ax3.axhline(65, color="red",   ls="--", lw=0.8, label="RSI 65")
    ax3.axhline(40, color="green", ls="--", lw=0.8, label="RSI 40")
    ax3.fill_between(data.index, 40, 65, alpha=0.07, color="blue", label="Zona entrada")
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(alpha=0.25)

    # ── Panel 4: Curva de capital
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(result.equity_curve.index, result.equity_curve, color="darkgreen", lw=1.2)
    ax4.set_ylabel("Capital (USD)")
    ax4.set_xlabel("Fecha")
    ax4.grid(alpha=0.25)

    plt.savefig("trading_strategy_results.png", dpi=150, bbox_inches="tight")
    print("Grafico guardado en: trading_strategy_results.png")
    plt.show()


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Generando datos de precio simulados (~3 años)...")
    df = generate_price_data(n_days=750)

    strategy = SwingMACDStrategy(
        macd_fast     = 12,
        macd_slow     = 26,
        macd_signal   = 9,
        rsi_period    = 14,
        rsi_buy_min   = 40.0,   # RSI mínimo para entrar largo
        rsi_buy_max   = 65.0,   # RSI máximo (evita sobrecompra)
        rsi_sell_min  = 35.0,   # RSI mínimo para entrar corto
        rsi_sell_max  = 60.0,   # RSI máximo para entrar corto
        ema_trend     = 50,     # filtro de tendencia
        atr_period    = 14,
        atr_sl_mult   = 1.5,    # stop loss = 1.5x ATR
        atr_tp_mult   = 3.0,    # take profit = 3.0x ATR  (ratio 1:2)
        allow_short   = True,
        initial_cash  = 10_000.0,
    )

    print("Ejecutando backtesting...")
    result, data = strategy.backtest(df)

    print(result.summary())

    print(f"\n{'Entrada':<12} {'Side':<6} {'P.Entrada':>10} {'Salida':<12} "
          f"{'P.Salida':>10} {'PnL%':>8} {'Dias':>5} {'Razon'}")
    print("-" * 74)
    for t in result.closed_trades:
        print(
            f"{str(t.entry_date.date()):<12} "
            f"{t.side:<6} "
            f"{t.entry_price:>10.2f} "
            f"{str(t.exit_date.date()):<12} "
            f"{t.exit_price:>10.2f} "
            f"{t.pnl_pct:>+8.2f}% "
            f"{t.hold_days:>5} "
            f"{t.exit_reason}"
        )

    plot_results(data, result, ticker="SIM-750")
