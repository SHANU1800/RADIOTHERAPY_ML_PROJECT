"""
Dataset Explorer page.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.utils.data_helpers import get_patient_list, get_file_list, load_summary_stats
from frontend.utils.error_handling import user_facing_message
from frontend.utils.visualization import (
    plot_breathing_curve,
    plot_bar_counts,
    plot_volume_histogram,
    export_fig_png,
)
from frontend.utils.icons import icon_html
from src.load_data import load_patient_file, get_patient_session_ini
from src.features import build_windows
import config as cfg


@st.cache_data(ttl=300)
def load_patient_data(patient_id: str, file_id: str, file_path_str: str):
    """Cache loaded patient data. Pass file_path as string for stable hashing."""
    from pathlib import Path
    return load_patient_file(Path(file_path_str), patient_id=patient_id, file_id=file_id)


def show():
    st.markdown(f'<p class="page-header">{icon_html("search", 28)} Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Browse and visualize existing patient breathing curve data.</p>', unsafe_allow_html=True)

    # Patient selector with search
    patients = get_patient_list()
    if len(patients) == 0:
        st.warning("No patients found in dataset directory.")
        return

    sel1, sel2 = st.columns([1, 1])
    with sel1:
        search_term = st.text_input(
            "Search patients",
            placeholder="Type to filter...",
            key="patient_search",
        )
        if search_term:
            filtered_patients = [p for p in patients if search_term.lower() in p.lower()]
        else:
            filtered_patients = patients
        if not filtered_patients:
            st.warning(f"No patients matching '{search_term}'")
            filtered_patients = patients
        patient_id = st.selectbox("Select Patient", filtered_patients, key="patient_select")

    with sel2:
        files = get_file_list(patient_id)
        if len(files) == 0:
            st.warning(f"No curve files found for patient: {patient_id}")
            return
        file_options = [f"{fid} ({path.name})" for fid, path in files]
        file_idx = st.selectbox("Select File", range(len(file_options)), format_func=lambda i: file_options[i])

    if file_idx is None:
        return

    file_id, file_path = files[file_idx]

    with st.spinner("Loading file..."):
        try:
            df = load_patient_data(patient_id, file_id, str(file_path))

            if len(df) == 0:
                st.error(user_facing_message(ValueError("Empty file or parsing error")))
                return

            duration = df["Session Time"].max() - df["Session Time"].min()
            vol = df["Volume (liters)"]

            # Compact stats row
            st.markdown(
                '<div class="stat-row">'
                f'<span class="stat-pill">{duration:.1f}s <small>duration</small></span>'
                f'<span class="stat-pill">{len(df):,} <small>rows</small></span>'
                f'<span class="stat-pill">{vol.min():.3f}L <small>min vol</small></span>'
                f'<span class="stat-pill">{vol.max():.3f}L <small>max vol</small></span>'
                f'<span class="stat-pill">{vol.mean():.3f}L <small>mean vol</small></span>'
                f'<span class="stat-pill">{vol.std():.3f}L <small>std vol</small></span>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Tabbed exploration
            tab_viz, tab_signals, tab_stats, tab_windows = st.tabs([
                "Breathing Curve", "Signal Channels", "Statistics & Status", "Window Breakdown",
            ])

            # --- TAB: Breathing Curve ---
            with tab_viz:
                show_balloon = st.checkbox("Show Balloon Valve Status", value=True, key="exp_balloon")
                show_rolling_mean = st.checkbox("Show rolling mean", value=True, key="exp_rolling")

                fig = plot_breathing_curve(
                    df,
                    show_balloon=show_balloon,
                    show_rolling_mean=show_rolling_mean,
                    subtitle=f"{file_path.name} | Patient: {patient_id}",
                )
                st.plotly_chart(fig, use_container_width=True, key="explorer_curve")

                dl1, dl2 = st.columns(2)
                with dl1:
                    png_curve = export_fig_png(fig)
                    if png_curve:
                        st.download_button("Download curve (PNG)", data=png_curve, file_name="breathing_curve.png", mime="image/png", key="dl_explorer_curve_png")
                with dl2:
                    curve_csv = df[["Session Time", "Volume (liters)"]].head(15000).to_csv(index=False)
                    st.download_button("Download curve data (CSV)", data=curve_csv, file_name=f"curve_{patient_id}_{file_id}.csv", mime="text/csv", key="dl_explorer_curve_csv")

            # --- TAB: Signal Channels ---
            with tab_signals:
                st.caption("Multi-channel signal decomposition for this file.")
                try:
                    from frontend.utils.visualization import plot_signal_analysis, plot_frequency_spectrum
                    fig_sig = plot_signal_analysis(df, sample_rate_hz=cfg.SAMPLE_RATE_HZ)
                    st.plotly_chart(fig_sig, use_container_width=True, key="explorer_signal")

                    st.markdown("#### Frequency Spectrum")
                    vol_arr = pd.to_numeric(df["Volume (liters)"], errors="coerce").fillna(0).values
                    if len(vol_arr) > 10000:
                        vol_arr = vol_arr[:10000]
                    fig_spec = plot_frequency_spectrum(vol_arr, sample_rate=cfg.SAMPLE_RATE_HZ)
                    st.plotly_chart(fig_spec, use_container_width=True, key="explorer_spectrum")
                except Exception as e:
                    st.warning(f"Could not plot signal analysis: {e}")

            # --- TAB: Statistics & Status ---
            with tab_stats:
                sc1, sc2 = st.columns(2)
                with sc1:
                    if "Balloon Valve Status" in df.columns:
                        st.markdown("#### Balloon Valve Status")
                        balloon_counts = df["Balloon Valve Status"].astype(str).value_counts()
                        fig_balloon = plot_bar_counts(balloon_counts, "Balloon Valve Status", "Status")
                        st.plotly_chart(fig_balloon, use_container_width=True, key="explorer_balloon")
                with sc2:
                    if "Gating Mode" in df.columns:
                        st.markdown("#### Gating Mode")
                        gating_counts = df["Gating Mode"].astype(str).value_counts()
                        fig_gating = plot_bar_counts(gating_counts, "Gating Mode", "Mode")
                        st.plotly_chart(fig_gating, use_container_width=True, key="explorer_gating")

                st.markdown("#### Volume Distribution")
                fig_hist = plot_volume_histogram(vol, title=f"Volume (L) — {file_path.name}", bins=50)
                st.plotly_chart(fig_hist, use_container_width=True, key="explorer_hist")

                session_blocks = get_patient_session_ini(cfg.DATASET_DIR, patient_id)
                if session_blocks:
                    with st.expander("Session metadata (session.ini)"):
                        for i, block in enumerate(session_blocks):
                            st.caption("Block %d" % (i + 1))
                            st.json(block)

                file_stats = {
                    "patient_id": [patient_id], "file": [file_path.name],
                    "duration_s": [round(duration, 2)], "rows": [len(df)],
                    "volume_min_L": [round(vol.min(), 4)],
                    "volume_max_L": [round(vol.max(), 4)],
                    "volume_mean_L": [round(vol.mean(), 4)],
                }
                if "Balloon Valve Status" in df.columns:
                    bc = df["Balloon Valve Status"].astype(str).value_counts()
                    file_stats["balloon_counts"] = ["; ".join("%s: %d" % (k, v) for k, v in bc.items())]
                file_stats_df = pd.DataFrame(file_stats)
                st.download_button("Download file stats (CSV)", data=file_stats_df.to_csv(index=False),
                                   file_name=f"file_stats_{patient_id}_{file_id}.csv", mime="text/csv", key="dl_explorer_file_stats")

            # --- TAB: Window Breakdown (auto-compute) ---
            with tab_windows:
                st.markdown("#### Ground Truth Window Labels")
                window_sec = st.slider("Window Size (s)", 1.0, 5.0, cfg.WINDOW_SEC, 0.5, key="explorer_window")

                try:
                    windows = build_windows(
                        df,
                        window_sec=window_sec,
                        sample_rate_hz=cfg.SAMPLE_RATE_HZ,
                        min_rows=cfg.MIN_WINDOW_ROWS,
                    )
                except Exception as e:
                    st.error("Error building windows: " + user_facing_message(e))
                    windows = None

                if windows is not None and len(windows) > 0:
                    bh = int((windows["label_breath_hold"] == 1).sum())
                    fb = int((windows["label_breath_hold"] == 0).sum())
                    total_w = len(windows)

                    st.markdown(
                        '<div class="stat-row">'
                        f'<span class="stat-pill">{total_w} <small>total windows</small></span>'
                        f'<span class="stat-pill">{bh} ({100*bh/total_w:.0f}%) <small>breath-hold</small></span>'
                        f'<span class="stat-pill">{fb} ({100*fb/total_w:.0f}%) <small>free-breathing</small></span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    fig_w = plot_breathing_curve(df, predictions=windows.rename(columns={"label_breath_hold": "prediction"}))
                    st.plotly_chart(fig_w, use_container_width=True, key="explorer_windows_curve")

                    display_cols = ["time_start", "time_end", "label_breath_hold", "vol_mean", "vol_std"]
                    avail = [c for c in display_cols if c in windows.columns]
                    st.dataframe(windows[avail].head(500), use_container_width=True)
                    if len(windows) > 500:
                        st.caption(f"Showing first 500 of {len(windows)} windows.")
                elif windows is not None:
                    st.warning("No windows produced (file too short?)")

        except Exception as e:
            st.error("Error loading file: " + user_facing_message(e))
            st.exception(e)

    # Dataset-wide stats
    with st.expander("Dataset-wide statistics"):
        try:
            file_summary, patient_summary, _ = load_summary_stats()
            if len(patient_summary) > 0:
                st.markdown(
                    f'<div class="stat-row">'
                    f'<span class="stat-pill">{len(patient_summary)} <small>patients</small></span>'
                    f'<span class="stat-pill">{len(file_summary)} <small>files</small></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(patient_summary.head(20), use_container_width=True, height=200)
            else:
                st.caption("Run analysis/analyze_dataset.py to generate summaries.")
        except Exception as e:
            st.caption("Could not load summary stats: " + user_facing_message(e))
