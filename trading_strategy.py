"""
Estrategia de Trading: Cruce de Medias Móviles + RSI + Bandas de Bollinger
--------------------------------------------------------------------------
Señales:
  - COMPRA  : MA corta cruza MA larga hacia arriba  AND  RSI < 70  AND  precio cerca banda inferior
  - VENTA   : MA corta cruza MA larga hacia abajo   OR   RSI > 70
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


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ──────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    side: str = "long"

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
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return dd.min() * 100

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_returns = self.equity_curve.pct_change().dropna()
        if daily_returns.std() == 0:
            return 0.0
        return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    def summary(self) -> str:
        lines = [
            "=" * 48,
            "        RESUMEN DEL BACKTESTING",
            "=" * 48,
            f"  Operaciones totales : {self.total_trades}",
            f"  Win rate            : {self.win_rate:.1f}%",
            f"  Retorno total       : {self.total_return:.2f}%",
            f"  Max drawdown        : {self.max_drawdown:.2f}%",
            f"  Sharpe ratio        : {self.sharpe_ratio:.2f}",
            "=" * 48,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Generador de datos simulados (OHLCV)
# ──────────────────────────────────────────────

def generate_price_data(
    n_days: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.015,
    drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-03-26"), periods=n_days)
    returns = np.random.normal(drift, volatility, n_days)
    close = start_price * np.exp(np.cumsum(returns))

    daily_vol = np.abs(np.random.normal(0, volatility * 0.5, n_days))
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = np.random.randint(1_000_000, 5_000_000, n_days)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ──────────────────────────────────────────────
# Estrategia
# ──────────────────────────────────────────────

class MACrossRSIStrategy:
    """
    Estrategia: Cruce de Medias Móviles + RSI + Bandas de Bollinger

    Parámetros
    ----------
    fast_period   : período de la MA rápida
    slow_period   : período de la MA lenta
    rsi_period    : período del RSI
    rsi_overbought: nivel de sobrecompra del RSI
    bb_period     : período de las Bandas de Bollinger
    bb_std        : desviaciones estándar para las bandas
    initial_cash  : capital inicial en USD
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        initial_cash: float = 10_000.0,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.initial_cash = initial_cash

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["ma_fast"] = sma(data["close"], self.fast_period)
        data["ma_slow"] = sma(data["close"], self.slow_period)
        data["rsi"] = rsi(data["close"], self.rsi_period)
        data["bb_upper"], data["bb_mid"], data["bb_lower"] = bollinger_bands(
            data["close"], self.bb_period, self.bb_std
        )
        # Cruce: +1 MA rápida sobre lenta, -1 por debajo
        data["ma_cross"] = np.sign(data["ma_fast"] - data["ma_slow"])
        data["cross_signal"] = data["ma_cross"].diff()
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["signal"] = 0

        buy = (
            (data["cross_signal"] > 0)             # cruce alcista
            & (data["rsi"] < self.rsi_overbought)  # RSI no sobrecomprado
        )
        sell = (
            (data["cross_signal"] < 0)             # cruce bajista
            | (data["rsi"] > self.rsi_overbought)  # RSI sobrecomprado
        )

        data.loc[buy, "signal"] = 1
        data.loc[sell, "signal"] = -1
        return data

    def backtest(self, df: pd.DataFrame) -> BacktestResult:
        data = self.compute_indicators(df)
        data = self.generate_signals(data)

        cash = self.initial_cash
        shares = 0.0
        result = BacktestResult()
        equity_values = []
        current_trade: Optional[Trade] = None

        for date, row in data.iterrows():
            price = row["close"]

            if row["signal"] == 1 and shares == 0:
                shares = cash / price
                cash = 0.0
                current_trade = Trade(entry_date=date, entry_price=price)
                result.trades.append(current_trade)

            elif row["signal"] == -1 and shares > 0:
                cash = shares * price
                shares = 0.0
                if current_trade is not None:
                    current_trade.exit_date = date
                    current_trade.exit_price = price
                    current_trade = None

            equity = cash + shares * price
            equity_values.append(equity)

        # Cerrar posición abierta al final
        if shares > 0 and current_trade is not None:
            last_price = data["close"].iloc[-1]
            cash = shares * last_price
            current_trade.exit_date = data.index[-1]
            current_trade.exit_price = last_price

        result.equity_curve = pd.Series(equity_values, index=data.index)
        return result, data


# ──────────────────────────────────────────────
# Visualización
# ──────────────────────────────────────────────

def plot_results(data: pd.DataFrame, result: BacktestResult, ticker: str = "ACTIVO"):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Estrategia MA-Crossover + RSI + Bollinger — {ticker}", fontsize=14)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)

    # ── Panel 1: Precio + indicadores
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data["close"], label="Precio cierre", color="black", lw=1.2)
    ax1.plot(data.index, data["ma_fast"], label=f"MA rápida ({data['ma_fast'].name})", color="blue", lw=1)
    ax1.plot(data.index, data["ma_slow"], label=f"MA lenta ({data['ma_slow'].name})", color="orange", lw=1)
    ax1.fill_between(data.index, data["bb_lower"], data["bb_upper"], alpha=0.1, color="purple", label="Bandas Bollinger")

    buys  = data[data["signal"] == 1]
    sells = data[data["signal"] == -1]
    ax1.scatter(buys.index,  buys["close"],  marker="^", color="green", s=80, zorder=5, label="Compra")
    ax1.scatter(sells.index, sells["close"], marker="v", color="red",   s=80, zorder=5, label="Venta")

    ax1.set_ylabel("Precio (USD)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    # ── Panel 2: RSI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(data.index, data["rsi"], color="purple", lw=1)
    ax2.axhline(70, color="red",   ls="--", lw=0.8, label="Sobrecompra 70")
    ax2.axhline(30, color="green", ls="--", lw=0.8, label="Sobreventa 30")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    # ── Panel 3: Curva de capital
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(result.equity_curve.index, result.equity_curve, color="darkgreen", lw=1.2)
    ax3.set_ylabel("Capital (USD)")
    ax3.set_xlabel("Fecha")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("trading_strategy_results.png", dpi=150, bbox_inches="tight")
    print("Gráfico guardado en: trading_strategy_results.png")
    plt.show()


# ──────────────────────────────────────────────
# Punto de entrada
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Generar datos simulados (reemplaza con datos reales via yfinance, ccxt, etc.)
    print("Generando datos de precio simulados...")
    df = generate_price_data(n_days=500)

    # 2. Configurar y ejecutar estrategia
    strategy = MACrossRSIStrategy(
        fast_period=5,
        slow_period=20,
        rsi_period=14,
        rsi_overbought=70.0,
        bb_period=20,
        bb_std=2.0,
        initial_cash=10_000.0,
    )

    print("Ejecutando backtesting...")
    result, data = strategy.backtest(df)

    # 3. Mostrar resumen
    print(result.summary())

    # Detalle de las últimas 5 operaciones
    print("\nÚltimas 5 operaciones:")
    print(f"{'Entrada':<12} {'P.Entrada':>10} {'Salida':<12} {'P.Salida':>10} {'PnL%':>8}")
    print("-" * 56)
    for t in result.closed_trades[-5:]:
        print(
            f"{str(t.entry_date.date()):<12} "
            f"{t.entry_price:>10.2f} "
            f"{str(t.exit_date.date()):<12} "
            f"{t.exit_price:>10.2f} "
            f"{t.pnl_pct:>+8.2f}%"
        )

    # 4. Graficar resultados
    plot_results(data, result, ticker="SIM-500")
