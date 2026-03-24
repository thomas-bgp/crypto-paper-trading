"""
Regime Detector v2 — HMM with anti-whipsaw fixes.
Fix 1: Minimum regime duration (5 candles)
Fix 2: Temperature scaling + hysteresis (kills overconfidence)
Fix 3: Continuous allocation blending (no binary switches)
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MinDurationFilter:
    """Fix 1: Only accept regime change after N consecutive confirmations."""

    def __init__(self, min_duration: int = 5):
        self.min_duration = min_duration
        self.current = None
        self.candidate = None
        self.count = 0

    def update(self, raw: str) -> str:
        if self.current is None:
            self.current = raw
            self.candidate = raw
            self.count = self.min_duration
            return self.current

        if raw == self.current:
            self.candidate = raw
            self.count = self.min_duration
        elif raw == self.candidate:
            self.count += 1
            if self.count >= self.min_duration:
                self.current = self.candidate
        else:
            self.candidate = raw
            self.count = 1

        return self.current


class HysteresisFilter:
    """Fix 2: Require probability margin to switch regime."""

    def __init__(self, threshold: float = 0.15, min_prob: float = 0.55):
        self.threshold = threshold
        self.min_prob = min_prob
        self.current_idx = None

    def update(self, state_probs: np.ndarray) -> int:
        best = int(np.argmax(state_probs))
        best_prob = state_probs[best]

        if self.current_idx is None:
            self.current_idx = best
            return self.current_idx

        current_prob = state_probs[self.current_idx]
        if (best != self.current_idx
                and best_prob >= self.min_prob
                and best_prob > current_prob + self.threshold):
            self.current_idx = best

        return self.current_idx


def temperature_scale(log_probs: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """Fix 2b: Soften HMM probabilities to prevent confidence=1.0 collapse."""
    scaled = log_probs / temperature
    exp_p = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_p / exp_p.sum(axis=1, keepdims=True)


class RegimeDetector:
    """
    Composite regime detector v2 with anti-whipsaw.
    """

    def __init__(self, n_states: int = 3, lookback: int = 500):
        self.n_states = n_states
        self.lookback = lookback
        self.hmm = None
        self.scaler = StandardScaler()
        self.state_map = {}
        self.state_map_inv = {}  # name -> idx

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        out['rvol'] = out['log_ret'].rolling(14).std()
        vol_ma = df['volume'].rolling(20).mean()
        out['vol_ratio'] = np.where(vol_ma > 0, np.log(df['volume'] / vol_ma), 0)
        out.replace([np.inf, -np.inf], 0, inplace=True)
        out.dropna(inplace=True)
        return out

    def fit(self, df: pd.DataFrame):
        feats = self._build_features(df)
        X = feats.tail(self.lookback).values
        if len(X) < 50:
            return self
        X_scaled = self.scaler.fit_transform(X)

        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type='full',
            n_iter=200,
            tol=1e-4,
            random_state=42
        )
        self.hmm.fit(X_scaled)

        means = [self.hmm.means_[i][0] for i in range(self.n_states)]
        bull_idx = int(np.argmax(means))
        bear_idx = int(np.argmin(means))
        side_idx = [i for i in range(self.n_states) if i not in (bull_idx, bear_idx)][0]
        self.state_map = {bull_idx: 'BULL', bear_idx: 'BEAR', side_idx: 'SIDEWAYS'}
        self.state_map_inv = {v: k for k, v in self.state_map.items()}
        return self

    def get_state_probs(self, single_feat_scaled: np.ndarray) -> np.ndarray:
        """Get softened state probabilities for a single observation."""
        raw_proba = self.hmm.predict_proba(single_feat_scaled)[0]
        # Temperature scaling to prevent confidence=1.0
        log_p = np.log(raw_proba + 1e-10).reshape(1, -1)
        soft_proba = temperature_scale(log_p, temperature=2.0)[0]
        return soft_proba


def rolling_regime_detection(df: pd.DataFrame, df_funding: pd.DataFrame = None,
                             df_fng: pd.DataFrame = None,
                             refit_every: int = 42,
                             lookback: int = 500) -> pd.DataFrame:
    """
    Walk-forward regime detection v2 with all fixes:
    - Temperature scaling (no more confidence=1.0)
    - Hysteresis (need probability margin to switch)
    - Minimum duration (5 candles = 20h confirmation)
    - Funding rate + FNG overrides
    """
    detector = RegimeDetector(lookback=lookback)
    hysteresis = HysteresisFilter(threshold=0.15, min_prob=0.55)
    min_dur = MinDurationFilter(min_duration=5)
    all_regimes = []

    start_idx = lookback
    feats_df = detector._build_features(df)

    for i in range(start_idx, len(feats_df)):
        # Re-fit HMM periodically
        if (i - start_idx) % refit_every == 0:
            train_slice = df.iloc[max(0, i - lookback):i]
            if len(train_slice) >= lookback:
                detector.fit(train_slice)

        if detector.hmm is None:
            all_regimes.append({'date': feats_df.index[i], 'regime': 'SIDEWAYS',
                                'confidence': 0.5, 'fundingRate': 0.0, 'fng': 50,
                                'mom_weight': 0.30})
            continue

        # Get softened probabilities (Fix 2b)
        single_feat = detector.scaler.transform(feats_df.values[i:i+1])
        soft_proba = detector.get_state_probs(single_feat)
        confidence = float(soft_proba.max())

        # Hysteresis filter (Fix 2)
        hyst_state_idx = hysteresis.update(soft_proba)
        hyst_regime = detector.state_map.get(hyst_state_idx, 'SIDEWAYS')

        # Get funding rate
        ts = feats_df.index[i]
        fr_val = 0.0
        if df_funding is not None and not df_funding.empty:
            mask = df_funding.index <= ts
            if mask.any():
                fr_val = df_funding.loc[mask, 'fundingRate'].iloc[-1]

        # Get FNG
        fng_val = 50
        if df_fng is not None and not df_fng.empty:
            mask = df_fng.index <= ts
            if mask.any():
                fng_val = df_fng.loc[mask, 'fng'].iloc[-1]

        # Apply FR/FNG overrides on hysteresis output
        overridden = _apply_overrides(hyst_regime, fr_val, fng_val, confidence)

        # Minimum duration filter (Fix 1)
        final_regime = min_dur.update(overridden)

        # Continuous allocation blending (Fix 3)
        # Instead of binary allocation, blend based on state probabilities
        bull_idx = detector.state_map_inv.get('BULL', 0)
        bear_idx = detector.state_map_inv.get('BEAR', 1)
        side_idx = detector.state_map_inv.get('SIDEWAYS', 2)

        p_bull = soft_proba[bull_idx]
        p_bear = soft_proba[bear_idx]
        p_side = soft_proba[side_idx]

        # Momentum weight = weighted blend of regime allocations
        # BULL=0.80, SIDEWAYS=0.40, BEAR=0.15 (floor)
        mom_weight = p_bull * 0.80 + p_side * 0.40 + p_bear * 0.15

        # Override: if final regime is hard BEAR (confirmed), cap at 0.20
        if final_regime == 'BEAR' and confidence > 0.70:
            mom_weight = min(mom_weight, 0.20)

        # Clamp
        mom_weight = float(np.clip(mom_weight, 0.15, 0.85))

        all_regimes.append({
            'date': ts,
            'regime': final_regime,
            'confidence': confidence,
            'fundingRate': fr_val,
            'fng': fng_val,
            'mom_weight': mom_weight,
        })

    return pd.DataFrame(all_regimes).set_index('date')


def _apply_overrides(base: str, fr: float, fng: float, conf: float) -> str:
    if conf < 0.45:
        return 'SIDEWAYS'

    # Funding rate overrides (softer thresholds)
    if fr > 0.0005 and base == 'BULL':
        return 'SIDEWAYS'
    if fr < -0.0002 and base == 'BULL':
        return 'SIDEWAYS'

    # FNG extreme
    if fng < 12:
        return 'BEAR'
    if fng > 88 and base == 'BULL':
        return 'SIDEWAYS'

    return base
