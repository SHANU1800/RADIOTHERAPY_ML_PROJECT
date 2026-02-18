"""
Label definitions for ML tasks (classification and regression).

Classification A: Breath-hold / stable window vs free-breathing.
  - From Balloon Valve Status: 4 = inflated = breath-hold; 1 = deflated = free-breathing.
  - For window-level labels we use the majority or max status in the window.

Classification B: Gating OK vs not OK.
  - From Gating Mode: "Automated" = OK; "Manual Overide" / "Manual Override" = not OK.
  - Other values (blank, "-") treated as not OK or dropped.

Regression: Next-step volume or short-horizon stability (e.g. volume std in next window).
  - Target: next value of Volume (liters) or next-window mean/std.
"""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# Balloon Valve: 1=deflated, 2=ready inhale, 3=ready exhale, 4=inflated, 5=fault
BREATH_HOLD_BALLOON = 4  # inflated
FREE_BREATHING_BALLOON = 1  # deflated

# Gating Mode
GATING_OK = "Automated"
GATING_NOT_OK = ("Manual Overide", "Manual Override", "Manual override")


def balloon_to_breath_hold_label(balloon_status: pd.Series) -> pd.Series:
    """
    Map Balloon Valve Status to binary: 1 = breath-hold (inflated), 0 = free-breathing (deflated).
    Other statuses (2, 3, 5) mapped to 0 for simplicity; can be extended to multi-class.
    """
    if pd is None:
        raise ImportError("pandas required")
    numeric = pd.to_numeric(balloon_status, errors="coerce")
    return (numeric == BREATH_HOLD_BALLOON).astype(int)


def gating_mode_to_ok_label(gating_mode: pd.Series) -> pd.Series:
    """
    Map Gating Mode to binary: 1 = OK (Automated), 0 = not OK (manual or other).
    """
    if pd is None:
        raise ImportError("pandas required")
    s = gating_mode.astype(str).str.strip()
    return (s.str.lower() == "automated").astype(int)


def window_label_breath_hold(balloon_in_window: pd.Series) -> int:
    """
    Single window: label 1 if majority or any inflated (4), else 0.
    balloon_in_window: series of Balloon Valve Status values in the window.
    """
    if pd is None:
        raise ImportError("pandas required")
    numeric = pd.to_numeric(balloon_in_window, errors="coerce")
    if (numeric == BREATH_HOLD_BALLOON).any():
        return 1
    return 0


def window_label_gating_ok(gating_in_window: pd.Series) -> int:
    """
    Single window: label 1 if any "Automated", else 0.
    """
    if pd is None:
        raise ImportError("pandas required")
    s = gating_in_window.astype(str).str.strip().str.lower()
    if (s == "automated").any():
        return 1
    return 0
