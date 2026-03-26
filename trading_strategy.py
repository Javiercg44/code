"""
Estrategia de Trading: Golden Cross 50/200 + ADX + Trailing Stop
-----------------------------------------------------------------
Filosofía: pocas operaciones, alta calidad, horizonte mínimo de 3 meses.

Señales:
  COMPRA  : SMA50 cruza SMA200 hacia arriba  AND  ADX > 25  AND  volumen > media 20d
  SALIDA  : SMA50 cruza SMA200 hacia abajo   OR   trailing stop activado
            (ambas condiciones respetan un mínimo de 63 días hábiles en posición)

Indicadores adicionales en el gráfico: RSI, ADX, curva de capital.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Indicadores Técnicos
# ──────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — mide la fortaleza de la tendencia (0-100)."""
    up   = high.diff()
    down = -low.diff()
    dm_plus  = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr    = tr.rolling(period).mean()
    di_pos = 100 * pd.Series(dm_plus,  index=close.index).rolling(period).mean() / atr
    di_neg = 100 * pd.Series(dm_minus, index=close.index).rolling(period).mean() / atr
    dx     = (100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan))
    return dx.rolling(period).mean()


# ──────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    entry_price: float
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float] = None
    exit_reason: str = ""
    side: str = "long"

    @property
    def hold_days(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price

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
        lines = [
            "=" * 52,
            "         RESUMEN DEL BACKTESTING",
            "=" * 52,
            f"  Operaciones totales  : {self.total_trades}",
            f"  Win rate             : {self.win_rate:.1f}%",
            f"  Retorno total        : {self.total_return:.2f}%",
            f"  Max drawdown         : {self.max_drawdown:.2f}%",
            f"  Sharpe ratio         : {self.sharpe_ratio:.2f}",
            f"  Profit factor        : {self.profit_factor:.2f}",
            f"  Ganancia media       : +{self.avg_win_pct:.2f}%",
            f"  Pérdida media        : {self.avg_loss_pct:.2f}%",
            f"  Hold promedio (días) : {self.avg_hold_days:.0f}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Generador de datos simulados (OHLCV)
# ──────────────────────────────────────────────

def generate_price_data(
    n_days: int = 2500,
    start_price: float = 100.0,
    volatility: float = 0.013,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera OHLCV con ciclos alcistas/bajistas alternados de ~6-12 meses
    para que la estrategia de largo plazo tenga señales realistas.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-03-26"), periods=n_days)

    # Ciclos de mercado: cada ciclo dura entre 120 y 300 días hábiles
    drift_array = np.empty(n_days)
    i = 0
    bullish = True
    while i < n_days:
        cycle_len = np.random.randint(120, 300)
        drift = 0.0006 if bullish else -0.0004
        end = min(i + cycle_len, n_days)
        drift_array[i:end] = drift
        bullish = not bullish
        i = end

    returns = np.random.normal(drift_array, volatility, n_days)
    close   = start_price * np.exp(np.cumsum(returns))

    daily_vol = np.abs(np.random.normal(0, volatility * 0.5, n_days))
    high   = close * (1 + daily_vol)
    low    = close * (1 - daily_vol)
    open_  = np.roll(close, 1)
    open_[0] = start_price

    base_vol = np.random.randint(1_000_000, 3_000_000, n_days)
    volume   = (base_vol * (1 + np.abs(returns) * 15)).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ──────────────────────────────────────────────
# Estrategia
# ──────────────────────────────────────────────

class GoldenCrossADXStrategy:
    """
    Estrategia de largo plazo: Golden Cross + ADX + Trailing Stop

    Parámetros
    ----------
    fast_ma        : período SMA rápida (50 por defecto)
    slow_ma        : período SMA lenta  (200 por defecto)
    adx_period     : período del ADX
    adx_threshold  : ADX mínimo para confirmar tendencia (25 = tendencia fuerte)
    vol_period     : período de la media de volumen para filtro
    trailing_pct   : trailing stop en % desde el máximo desde la entrada
    min_hold_days  : días hábiles mínimos antes de poder salir (≈3 meses = 63)
    initial_cash   : capital inicial en USD
    """

    def __init__(
        self,
        fast_ma: int        = 50,
        slow_ma: int        = 200,
        adx_period: int     = 14,
        adx_threshold: float = 25.0,
        vol_period: int     = 20,
        trailing_pct: float = 0.15,
        min_hold_days: int  = 63,
        initial_cash: float = 10_000.0,
    ):
        self.fast_ma       = fast_ma
        self.slow_ma       = slow_ma
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold
        self.vol_period    = vol_period
        self.trailing_pct  = trailing_pct
        self.min_hold_days = min_hold_days
        self.initial_cash  = initial_cash

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["sma_fast"] = sma(data["close"], self.fast_ma)
        data["sma_slow"] = sma(data["close"], self.slow_ma)
        data["adx"]      = adx(data["high"], data["low"], data["close"], self.adx_period)
        data["rsi"]      = rsi(data["close"])
        data["vol_avg"]  = data["volume"].rolling(self.vol_period).mean()

        cross = np.sign(data["sma_fast"] - data["sma_slow"])
        data["cross_signal"] = cross.diff()
        return data

    def backtest(self, df: pd.DataFrame):
        data = self.compute_indicators(df)

        cash   = self.initial_cash
        shares = 0.0
        result = BacktestResult()
        equity_values: List[float] = []

        current_trade: Optional[Trade] = None
        peak_price: float = 0.0
        hold_count: int   = 0       # días hábiles en posición

        # columna para marcar señales en el gráfico
        data["signal"] = 0

        for date, row in data.iterrows():
            price = row["close"]

            if shares == 0:
                # ── Condición de ENTRADA ───────────────────────────────────
                golden_cross    = row["cross_signal"] > 0
                trend_confirmed = row["adx"] > self.adx_threshold
                high_volume     = row["volume"] > row["vol_avg"]

                if golden_cross and trend_confirmed and high_volume:
                    shares = cash / price
                    cash   = 0.0
                    peak_price    = price
                    hold_count    = 0
                    current_trade = Trade(entry_date=date, entry_price=price)
                    result.trades.append(current_trade)
                    data.at[date, "signal"] = 1

            else:
                hold_count += 1
                peak_price  = max(peak_price, price)

                # ── Condición de SALIDA ────────────────────────────────────
                death_cross   = row["cross_signal"] < 0
                trailing_stop = price < peak_price * (1 - self.trailing_pct)

                can_exit = hold_count >= self.min_hold_days

                if can_exit and (death_cross or trailing_stop):
                    reason = "death_cross" if death_cross else "trailing_stop"
                    cash   = shares * price
                    shares = 0.0
                    if current_trade is not None:
                        current_trade.exit_date   = date
                        current_trade.exit_price  = price
                        current_trade.exit_reason = reason
                        current_trade = None
                    data.at[date, "signal"] = -1

            equity = cash + shares * price
            equity_values.append(equity)

        # Cerrar posición abierta al cierre del período
        if shares > 0 and current_trade is not None:
            last_price = data["close"].iloc[-1]
            cash = shares * last_price
            current_trade.exit_date   = data.index[-1]
            current_trade.exit_price  = last_price
            current_trade.exit_reason = "fin_periodo"

        result.equity_curve = pd.Series(equity_values, index=data.index)
        return result, data


# ──────────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────────

def plot_results(data: pd.DataFrame, result: BacktestResult, ticker: str = "ACTIVO"):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"Golden Cross 50/200 + ADX + Trailing Stop — {ticker}\n"
        f"Win Rate: {result.win_rate:.1f}%  |  "
        f"Retorno: {result.total_return:.2f}%  |  "
        f"Sharpe: {result.sharpe_ratio:.2f}  |  "
        f"Hold promedio: {result.avg_hold_days:.0f} días",
        fontsize=12,
    )
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.4)

    # ── Panel 1: Precio + SMAs + señales
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data["close"],    label="Precio cierre", color="black",  lw=1.0, alpha=0.85)
    ax1.plot(data.index, data["sma_fast"], label=f"SMA {50}",      color="blue",   lw=1.2)
    ax1.plot(data.index, data["sma_slow"], label=f"SMA {200}",     color="orange", lw=1.4)

    # Marcar zonas de posición abierta
    for t in result.closed_trades:
        ax1.axvspan(t.entry_date, t.exit_date, alpha=0.07,
                    color="green" if (t.pnl or 0) > 0 else "red")

    buys  = data[data["signal"] == 1]
    sells = data[data["signal"] == -1]
    ax1.scatter(buys.index,  buys["close"],  marker="^", color="green", s=100, zorder=5, label="Compra")
    ax1.scatter(sells.index, sells["close"], marker="v", color="red",   s=100, zorder=5, label="Venta")

    ax1.set_ylabel("Precio (USD)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # ── Panel 2: ADX
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(data.index, data["adx"], color="darkcyan", lw=1)
    ax2.axhline(25, color="red", ls="--", lw=0.8, label="Umbral 25")
    ax2.fill_between(data.index, data["adx"], 25,
                     where=data["adx"] > 25, alpha=0.2, color="darkcyan")
    ax2.set_ylabel("ADX")
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    # ── Panel 3: RSI
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(data.index, data["rsi"], color="purple", lw=1)
    ax3.axhline(70, color="red",   ls="--", lw=0.8)
    ax3.axhline(30, color="green", ls="--", lw=0.8)
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
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
    print("Generando datos de precio simulados (2500 dias habiles, ~10 años)...")
    df = generate_price_data(n_days=2500)

    strategy = GoldenCrossADXStrategy(
        fast_ma        = 30,
        slow_ma        = 100,
        adx_period     = 14,
        adx_threshold  = 20.0,   # tendencia moderada-fuerte
        vol_period     = 20,
        trailing_pct   = 0.18,   # trailing stop del 18%
        min_hold_days  = 63,     # mínimo ~3 meses (63 días hábiles)
        initial_cash   = 10_000.0,
    )

    print("Ejecutando backtesting...")
    result, data = strategy.backtest(df)

    print(result.summary())

    print("\nDetalle de todas las operaciones:")
    print(f"{'Entrada':<12} {'P.Entrada':>10} {'Salida':<12} {'P.Salida':>10} "
          f"{'PnL%':>8} {'Días':>6} {'Razón':<14}")
    print("-" * 70)
    for t in result.closed_trades:
        print(
            f"{str(t.entry_date.date()):<12} "
            f"{t.entry_price:>10.2f} "
            f"{str(t.exit_date.date()):<12} "
            f"{t.exit_price:>10.2f} "
            f"{t.pnl_pct:>+8.2f}% "
            f"{t.hold_days:>6} "
            f"{t.exit_reason:<14}"
        )

    plot_results(data, result, ticker="SIM-1500")
