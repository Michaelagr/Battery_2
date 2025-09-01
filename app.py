import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
from datetime import datetime

# === EXTRACTED CONSTANTS FROM ORIGINAL FILE ===
EXPORT_REVENUE_FACTOR = 0.8
VOLLLASTSTUNDEN_THRESHOLD = 2500
DEMAND_CHARGE_HIGH = 200
DEMAND_CHARGE_LOW = 20
BATTERY_EFFICIENCY = 1
PV_CAPACITY_KWP = 0
LOW_PRICE_PERCENTILE = 30
HIGH_PRICE_PERCENTILE = 70

DEBUG = 0

# === PERFORMANCE OPTIMIZATION SETTINGS ===
# Set to True to enable automatic data resampling for faster chart rendering
ENABLE_CHART_OPTIMIZATION = True
# Minimum number of data points before resampling kicks in
MIN_POINTS_FOR_RESAMPLING = 10000
# Resampling frequency (e.g., "1H" for hourly, "30T" for 30 minutes)
RESAMPLE_FREQUENCY = "1H"

# === EXTRACTED FUNCTIONS FROM ORIGINAL FILE ===

def read_price_data():
    """Read spot price data and format for analysis"""
    price_file = "data/spot_data_2024.xlsx"
    df_prices = pd.read_excel(price_file)

    # Use the correct column names based on actual data structure
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], utc=True, format='mixed')
    # Use the "Day-ahead Price (EUR/MWh)" column and convert to EUR/kWh
    df_prices['price'] = pd.to_numeric(df_prices['Day-ahead Price (EUR/MWh)'], errors='coerce') / 1000

    return df_prices[['timestamp', 'price']].dropna()


def load_solar_data(pv_total):
    """Load and process solar generation data"""
    MAGIC_YEARLY_PV_MULTIPLIER = 1000
    INTERVAL_HOURS = 0.25

    df_pv = pd.read_csv("data/solar_data_de_small.csv")

    df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], format="%d.%m.%y %H:%M", utc=True)
    df_pv["yearly_production_kw"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER / INTERVAL_HOURS
    df_pv["yearly_production_kwh"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER

    return df_pv[['timestamp', 'yearly_production_kw', 'yearly_production_kwh']]


def read_load_profile(file_path):
    """Read load profile from Excel file"""
    df = pd.read_excel(file_path)

    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format="mixed")
    
    # Try to find the correct column for load in the Excel file
    load_col = None
    for col in ['kWh', 'kwh', 'value', 'value_kwh', 'load']:
        if col in df.columns:
            load_col = col
            break
    if load_col is None:
        raise ValueError("No suitable load column found in Excel file (tried: kWh, kwh, value, value_kwh, load)")

    if load_col == "load":
        df['load_org'] = df[load_col]
        df['total_kwh'] = df['load_org'] * 0.25
    else:
        df['load_org'] = df[load_col] / 0.25
        df['total_kwh'] = df['load_org'] * 0.25

    return df

def battery_simulation_ps(df, battery_capacity, power_rating, threshold_kw, depth_of_discharge, battery_efficiency):
    """
    Peak shaving battery simulation function
    """
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 10%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = 0.25  # 15-minute intervals

    threshold_kw = df["net_load_kw"].max() - threshold_kw

    optimized = []
    charge = []
    discharge = []
    soc_state = []

    for load in df["net_load_kw"]:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            #energy_used = actual_discharge_power * interval_hours / battery_efficiency
            energy_used = actual_discharge_power * interval_hours / 1
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours
            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            charge.append(actual_charge_power)
            discharge.append(0)

        else:
            charge.append(0)
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["ps_grid_load"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df

def rolling_quantile(arr, window, q):
    """Calculate rolling quantile for price analysis"""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        end = min(i + window, len(arr))
        if end > i:
            result[i] = np.percentile(arr[i:end], q)
    return result

def fast_forward_quantile(arr, window, q):
    # build windows looking forward
    padded = np.pad(arr, (0, window-1), constant_values=np.nan)
    windows = sliding_window_view(padded, window)   # shape (n, window)
    return np.nanpercentile(windows, q, axis=1)

def run_battery_analysis(file_path, power_rating=100, capacity=215, pv_capacity=0, low_price_percentile=30, high_price_percentile=70):
    """
    Run the complete battery analysis for a single file.
    This is the exact algorithm extracted from battery_savings_case_4_v3.py
    """
    try:
        # Load data
        df_prices = read_price_data()
        df_pv = load_solar_data(pv_capacity)
        df_load = read_load_profile(file_path)
        
        # Merge all dataframes
        df_merged = pd.merge(df_load, df_prices, on='timestamp', how='left')
        df_merged = pd.merge(df_merged, df_pv, on='timestamp', how='left')
        df_merged['load_pv'] = df_merged['load_org'] - df_merged['yearly_production_kw']
        df_merged['net_load_kw'] = df_merged['load_pv']
        df_merged['net_load_kwh'] = df_merged['load_pv'] * 0.25
        
        # Battery parameters
        battery_power_kw = power_rating
        battery_capacity_kwh = capacity
        interval_hours = 0.25
        depth_of_discharge = 0.1 * battery_capacity_kwh
        
        # === CASE 4: Smart Battery Implementation ===
        df_case_4 = df_merged.copy()

        # Initialize battery columns
        df_case_4['battery_soc_kwh'] = 0.0

        df_case_4['battery_charge_kw'] = 0.0
        df_case_4['battery_charge_pv_kw'] = 0.0
        df_case_4['battery_charge_grid_kw'] = 0.0
        
        df_case_4['battery_discharge_kw'] = 0.0
        df_case_4['battery_discharge_ps_kw'] = 0.0
        df_case_4['battery_discharge_ls_kw'] = 0.0

        df_case_4['load_shift_charge_kw'] = 0.0
        df_case_4['load_shift_discharge_kw'] = 0.0

        df_case_4['grid_import_kw'] = 0.0
        df_case_4['grid_export_kw'] = 0.0
        
        df_case_4 = df_case_4.reset_index(drop=True)
        
        # === STEP 1: Peak Shaving Optimization ===
        # Find optimal peak shaving threshold
        peak_load = df_case_4["net_load_kw"].max()
        total_energy_kwh_case_4 = df_case_4["net_load_kwh"].sum()
        
        # Test different peak reduction values using proper battery simulation
        all_results = []
        min_reduction_ps = min(0.2 * battery_power_kw, 10)
        max_reduction_ps = min(battery_power_kw, peak_load)
        reduction_values_ps = np.arange(min_reduction_ps, max_reduction_ps, 1)
        
        best_peak_reduction = -np.inf
        
        for ps_reduction_value in reduction_values_ps:
            # Create a copy for testing this threshold
            df_test = df_case_4.copy()
            
            # Run battery simulation with this threshold
            df_ps = battery_simulation_ps(
                df_test, battery_capacity_kwh, battery_power_kw, 
                threshold_kw=ps_reduction_value, depth_of_discharge=90, battery_efficiency=1
            )
            
            # Calculate results after peak shaving
            peak_after_ps = df_ps["ps_grid_load"].max()
            peak_reduction_ps = peak_load - peak_after_ps


            # Is this correct or should it be peak_load - peak_after_ps (before: peak_load - ps_reduction_value)?
            threshold_kw = peak_load - peak_reduction_ps
            
            volllaststunden_ps = total_energy_kwh_case_4 / peak_after_ps if peak_after_ps > 0 else 0
            demand_charge = DEMAND_CHARGE_HIGH if volllaststunden_ps >= VOLLLASTSTUNDEN_THRESHOLD else DEMAND_CHARGE_LOW
            #annual_savings_ps = peak_reduction_ps * demand_charge
            
            all_results.append({
                "threshold_kW": threshold_kw,
                "peak_after_ps": peak_after_ps,
             #   "volllaststunden_ps": volllaststunden_ps,
             #   "annual_savings_ps": annual_savings_ps,
                "kW_reduction": peak_reduction_ps
            })
            
            if peak_reduction_ps >= best_peak_reduction:
                best_peak_reduction = peak_reduction_ps
                best_threshold_ps_only = threshold_kw
            else:
                break
        
        # Find best threshold
        best = max(all_results, key=lambda x: x["kW_reduction"])
        best_threshold_ps_only = best["threshold_kW"]


        ######################################################################################
        # === STEP 2: Complete Smart Battery Implementation ===
        battery_soc_kwh = battery_capacity_kwh
        peak_shaving_threshold_kw = best_threshold_ps_only

        min_soc_kwh = depth_of_discharge + battery_capacity_kwh * 0.1

        
        # Look-ahead parameters

        look_ahead_intervals_peak = 96 if battery_capacity_kwh < 431 else 96*2
        look_ahead_intervals_price = 96
        # Use the dynamic percentiles from function parameters
        low_price_perc = low_price_percentile
        high_price_perc = high_price_percentile
        
        # Precompute look-ahead values
        excess_kw = np.clip(df_case_4['net_load_kw'] - peak_shaving_threshold_kw, 0, None)
        df_case_4['future_excess_power_kw'] = (
            pd.Series(excess_kw).rolling(look_ahead_intervals_peak, min_periods=1).max().shift(-look_ahead_intervals_peak + 1).to_numpy())
        df_case_4['future_excess_energy_kwh'] = (
            pd.Series(excess_kw).rolling(look_ahead_intervals_peak, min_periods=1).sum().shift(-look_ahead_intervals_peak + 1).to_numpy() * interval_hours)
        
        s = df_case_4['net_load_kw'].to_numpy()
        fw_max = (pd.Series(s[::-1]).rolling(look_ahead_intervals_peak, min_periods=1).max()[::-1]).to_numpy()
        df_case_4['future_peak_max_kw'] = fw_max
        
        # Price thresholds using dynamic percentiles
        # df_case_4['future_price_low'] = rolling_quantile(df_case_4['price'].to_numpy(), look_ahead_intervals_price, low_price_perc / 100.0)
        # prices = df_case_4['price'].to_numpy()
        # prices_for_high = np.where(prices > 0, prices, 0.01)
        # df_case_4['future_price_high'] = rolling_quantile(prices_for_high, look_ahead_intervals_price, high_price_perc / 100.0)

        # Price thresholds using dynamic percentiles
        prices = df_case_4['price'].to_numpy()
        # Apply the forward-looking rolling quantile by reversing the array
        # df_case_4['future_price_low'] = rolling_quantile(prices[::-1], look_ahead_intervals_price, low_price_perc / 100.0)[::-1]
        # df_case_4['future_price_low'] = rolling_quantile(prices[::-1], look_ahead_intervals_price, low_price_perc)[::-1]

        prices_for_high = np.where(prices > 0, prices, 0.01)
        # Apply the same forward-looking technique for the high price threshold
        # df_case_4['future_price_high'] = rolling_quantile(prices_for_high[::-1], look_ahead_intervals_price, high_price_perc )[::-1]  
       

        #df_case_4['future_price_low'] = fast_forward_quantile(df_case_4['price'].to_numpy(), look_ahead_intervals_price, low_price_perc)
        #df_case_4['future_price_high'] = fast_forward_quantile(prices_for_high, look_ahead_intervals_price, high_price_perc)

        df_case_4['future_price_low'] = rolling_quantile(prices, look_ahead_intervals_price, low_price_perc)
        df_case_4['future_price_high'] = rolling_quantile(prices_for_high, look_ahead_intervals_price, high_price_perc)

        df_case_4 = df_case_4.reset_index(drop=True)
        
        # Prepare numpy arrays for fast computation
        net_load_kwh = df_case_4['net_load_kwh'].to_numpy()
        net_load_kw_np = df_case_4['net_load_kw'].to_numpy()
        price = df_case_4['price'].to_numpy()
        future_price_low = df_case_4['future_price_low'].to_numpy()
        future_price_high = df_case_4['future_price_high'].to_numpy()
        future_peak_max_kw = df_case_4['future_peak_max_kw'].to_numpy()
        future_excess_energy_kwh = df_case_4['future_excess_energy_kwh'].to_numpy()
        
        # Initialize result arrays
        battery_discharge_kw = np.zeros(len(df_case_4))
        battery_discharge_ls_kw = np.zeros(len(df_case_4))
        battery_discharge_ps_kw = np.zeros(len(df_case_4))

        battery_soc_kwh_arr = np.zeros(len(df_case_4))
        grid_import_kw = np.zeros(len(df_case_4))
        grid_export_kw = np.zeros(len(df_case_4))
        battery_charge_kw = np.zeros(len(df_case_4))
        
        total_energy_charged = 0.0
        
        # === MAIN SIMULATION LOOP ===
        for idx in range(len(df_case_4)):
            net_load_kw = net_load_kw_np[idx]
            net_load_kwh_interval = net_load_kwh[idx]
            current_price = price[idx]
            low_price_threshold = future_price_low[idx]
            high_price_threshold = future_price_high[idx]
            future_peak_value = future_peak_max_kw[idx]
            required_energy_for_peak = future_excess_energy_kwh[idx]
            
            # Initialize interval values
            battery_discharge_kwh_interval = 0.0
            battery_discharge_kwh_interval_ls = 0.0
            battery_discharge_kwh_interval_ps = 0.0

            grid_import_kwh_interval = 0.0
            grid_export_kwh_interval = 0.0

            battery_charge_kwh_interval = 0.0
            battery_charge_kwh_interval_pv = 0.0
            battery_charge_kwh_interval_grid = 0.0
            
            battery_power_kwh = battery_power_kw * interval_hours
            
            # Start with base net load
            grid_import_kwh_interval = max(0, net_load_kwh_interval)
            grid_import_kw_interval = grid_import_kwh_interval / interval_hours

            action_taken = 0
            
            # 1. Peak Shaving
            if net_load_kw > peak_shaving_threshold_kw:
                discharge_needed_kwh = (net_load_kw - peak_shaving_threshold_kw) * interval_hours
                max_discharge_possible_kwh = min(battery_power_kwh, (battery_soc_kwh - depth_of_discharge))
                actual_discharge_kwh = min(discharge_needed_kwh, max_discharge_possible_kwh)
                
                if actual_discharge_kwh > 0:
                    battery_discharge_kwh_interval = actual_discharge_kwh
                    battery_discharge_kwh_interval_ps += actual_discharge_kwh
                    battery_soc_kwh -= battery_discharge_kwh_interval
                    grid_import_kwh_interval = max(0, net_load_kwh_interval - battery_discharge_kwh_interval)
                    action_taken = 1
                else:
                    action_taken = 0

            
            # 2. PV Surplus Charging
            elif net_load_kw < 0 and action_taken == 0:
                charge_potential_kwh_interval = abs(net_load_kwh_interval)
                max_charge_possible_kwh_interval = min(battery_power_kwh, battery_capacity_kwh - battery_soc_kwh)
                actual_charge_from_PV_kwh_interval = min(charge_potential_kwh_interval, max_charge_possible_kwh_interval)
                
                if actual_charge_from_PV_kwh_interval > 0:
                    battery_charge_kwh_interval_pv = actual_charge_from_PV_kwh_interval
                    battery_soc_kwh += battery_charge_kwh_interval_pv
                    total_energy_charged += battery_charge_kwh_interval_pv
                    grid_export_kwh_interval = charge_potential_kwh_interval - battery_charge_kwh_interval_pv
                    grid_import_kwh_interval = 0
                    action_taken = 1
                else:
                    action_taken = 0
            

            # 3. Arbitrage and Emergency Charging
            else:  # Threshold > net_load_kwh > 0

                # 3.1 Emergency charging for upcoming peak
                future_peak_detected = future_peak_value > peak_shaving_threshold_kw
                peak_reserve_soc = min(battery_capacity_kwh, min_soc_kwh + required_energy_for_peak)
                
                if future_peak_detected and battery_soc_kwh < peak_reserve_soc:
                    max_charge_without_exceeding_threshold = max(0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
                    charge_amount_kwh = min(
                        max_charge_without_exceeding_threshold,
                        battery_power_kw * interval_hours,
                        battery_capacity_kwh - battery_soc_kwh,
                        peak_reserve_soc - battery_soc_kwh)
                    
                    if charge_amount_kwh > 0:
                        battery_charge_kwh_interval_grid = charge_amount_kwh
                        battery_soc_kwh += charge_amount_kwh * BATTERY_EFFICIENCY
                        total_energy_charged += charge_amount_kwh * BATTERY_EFFICIENCY
                        grid_import_kwh_interval += charge_amount_kwh

                    action_taken = 1 if charge_amount_kwh != 0 else 0
                
                # # Arbitrage charging (negative prices) - can charge in addition to emergency charging if it's cheap
                # if current_price < 0 and battery_soc_kwh < battery_capacity_kwh:
                #     charge_amount_kwh = min(
                #         battery_power_kwh,
                #         battery_capacity_kwh - battery_soc_kwh,
                #         (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours
                #     )
                #     if charge_amount_kwh > 0:
                #         battery_charge_kwh_interval_grid += charge_amount_kwh
                #         battery_soc_kwh += charge_amount_kwh * BATTERY_EFFICIENCY
                #         total_energy_charged += charge_amount_kwh * BATTERY_EFFICIENCY
                #         grid_import_kwh_interval += charge_amount_kwh
                
                # 3.2 Arbitrage charging (low prices)  - can charge in addition to emergency charging if it's cheap
                if current_price <= low_price_threshold and battery_soc_kwh < battery_capacity_kwh:
                    max_charge_without_exceeding_threshold = max(
                        0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
                    
                    charge_amount_kwh = min(
                        max_charge_without_exceeding_threshold,
                        battery_power_kwh,
                        battery_capacity_kwh - battery_soc_kwh
                    )
                    if charge_amount_kwh > 0:
                        battery_charge_kwh_interval_grid += charge_amount_kwh
                        battery_soc_kwh += charge_amount_kwh * BATTERY_EFFICIENCY
                        total_energy_charged += charge_amount_kwh * BATTERY_EFFICIENCY
                        grid_import_kwh_interval += charge_amount_kwh

                    action_taken = 1 if charge_amount_kwh != 0 else 0

                # 3.3 Arbitrage discharging (high prices)
                if (current_price >= high_price_threshold and action_taken == 0 and
                    battery_soc_kwh > peak_reserve_soc and 
                    (battery_charge_kwh_interval_pv + battery_charge_kwh_interval_grid) == 0.0):
                    
                    required_reserve = min(battery_capacity_kwh, peak_reserve_soc)
                    max_discharge_for_arbitrage = max(0.0, battery_soc_kwh - required_reserve)
                    discharge_amount_kwh = min(battery_power_kwh, max_discharge_for_arbitrage)
                    
                    if discharge_amount_kwh > 0:
                        battery_soc_kwh -= discharge_amount_kwh
                        battery_discharge_kwh_interval += discharge_amount_kwh
                        battery_discharge_kwh_interval_ls += discharge_amount_kwh
                        grid_export_kwh_interval += discharge_amount_kwh
                        action_taken = 1
                    else:
                        action_taken = 0

            
            # Store results
            battery_discharge_kw[idx] = battery_discharge_kwh_interval / interval_hours
            battery_discharge_ls_kw[idx] = battery_discharge_kwh_interval_ls / interval_hours
            battery_discharge_ps_kw[idx] = battery_discharge_kwh_interval_ps / interval_hours

            battery_soc_kwh_arr[idx] = battery_soc_kwh
            
            grid_import_kw[idx] = grid_import_kwh_interval / interval_hours
            grid_export_kw[idx] = grid_export_kwh_interval / interval_hours
            battery_charge_kw[idx] = (battery_charge_kwh_interval_grid + battery_charge_kwh_interval_pv) / interval_hours
        
        # Update dataframe with results
        df_case_4['battery_discharge_kw'] = battery_discharge_kw
        df_case_4['battery_discharge_ls_kw'] = battery_discharge_ls_kw
        df_case_4['battery_discharge_ps_kw'] = battery_discharge_ps_kw

        df_case_4['battery_soc_kwh'] = battery_soc_kwh_arr
        
        df_case_4['grid_import_kw'] = grid_import_kw
        df_case_4['grid_export_kw'] = grid_export_kw
        
        df_case_4['battery_charge_kw'] = battery_charge_kw
        
        return df_case_4, power_rating, capacity, peak_shaving_threshold_kw
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, None, None


# === STREAMLIT DASHBOARD ===

def create_load_profile_chart(df, peak_threshold=None):
    """Create the main load profile visualization with battery operations."""
    # Show progress indicator for large datasets
    if len(df) > 10000 and DEBUG ==1:
        st.info(f"🔄 Rendering chart with {len(df):,} data points...")
    
    # Ensure data is sorted by timestamp to avoid horizontal lines
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # OPTIMIZATION: Resample data for faster rendering
    # Get optimization settings from session state or use defaults
    optimization_enabled = st.session_state.get('chart_optimization_enabled', ENABLE_CHART_OPTIMIZATION)
    resample_freq = st.session_state.get('resample_frequency', RESAMPLE_FREQUENCY)
    min_points = st.session_state.get('min_points_for_resampling', MIN_POINTS_FOR_RESAMPLING)
    
    if optimization_enabled and len(df_sorted) > min_points:
        with st.spinner(f"Optimizing chart performance by resampling {len(df_sorted):,} points to {resample_freq} intervals..."):
            try:
                # Handle mixed data types during resampling
                df_indexed = df_sorted.set_index("timestamp")
                
                # Separate numeric and non-numeric columns
                numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
                non_numeric_cols = df_indexed.select_dtypes(exclude=[np.number]).columns
                
                # Resample numeric columns with mean
                df_resampled_numeric = df_indexed[numeric_cols].resample(resample_freq).mean()
                
                # For non-numeric columns, take the first value in each interval
                if len(non_numeric_cols) > 0:
                    df_resampled_non_numeric = df_indexed[non_numeric_cols].resample(resample_freq).first()
                    # Combine the resampled data
                    df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1).reset_index()
                else:
                    # Only numeric columns, no need to concatenate
                    df_resampled = df_resampled_numeric.reset_index()
                
                st.sidebar.success(f"✅ Resampled {len(df_sorted):,} Datenpunkte zu {len(df_resampled):,} (für schnellere Darstellung).")
                df_sorted = df_resampled
                
            except Exception as e:
                st.warning(f"⚠️ Resampling failed: {str(e)}. Using original data resolution.")
                st.info("This usually happens with mixed data types. Consider adjusting resampling frequency.")
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": False}]]
    )
    
    # Original load (consumption) - main load profile
    if df_sorted['yearly_production_kw'].max() > 0:
        fig.add_trace(
            go.Scattergl(  # Use Scattergl for better performance
                x=df_sorted['timestamp'],
                y=df_sorted['load_org'],
                name='Ursprüngliche Last',
                line=dict(color='#9e9d9d', width=1.2),  # Removed spline smoothing
                mode='lines',
                hovertemplate='<b>Ursprüngliche Last</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
            )
        )
    

    
    # Net Grid Load (after PV) - actual grid interaction
    fig.add_trace(
        go.Scattergl(  # Use Scattergl for better performance
            x=df_sorted['timestamp'],
            y=df_sorted['net_load_kw'],
            name='Netto-Netzlast (nach PV)',
            line=dict(color='#A1D99B', width=1.2),  # Removed spline smoothing
            mode='lines',
            hovertemplate='<b>Netto-Netzlast (nach PV)</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
        )
    )
    
    # Net Grid Load (after PV) - actual grid interaction
    fig.add_trace(
        go.Scattergl(  # Use Scattergl for better performance
            x=df_sorted['timestamp'],
            y=df_sorted['grid_import_kw'],
            name='Importierte Netzlast mit Batterie',
            line=dict(color='#2ca02c', width=1.2),  # Removed spline smoothing
            mode='lines',
            hovertemplate='<b>Importierte Netzlast</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
        )
    )

    # PV Generation (negative values to show as generation) - only if PV > 0
    if 'yearly_production_kw' in df_sorted.columns and df_sorted['yearly_production_kw'].max() > 0:
        fig.add_trace(
            go.Scattergl(  # Use Scattergl for better performance
                x=df_sorted['timestamp'],
                y=-df_sorted['yearly_production_kw'],
                name='PV-Erzeugung',
                line=dict(color='#E6550D', width=1.2),  # Removed spline smoothing
                mode='lines',
                hovertemplate='<b>PV-Erzeugung</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
            )
        )

    # Battery Charge (only show when > 0 to avoid flat lines)
    charge_data = df_sorted['battery_charge_kw']
    if charge_data.max() > 0:
        fig.add_trace(
            go.Scattergl(  # Use Scattergl for better performance
                x=df_sorted['timestamp'],
                y=charge_data,
                name='Batterieladung',
                line=dict(color='#3182BD', width=1.0),  # Removed spline smoothing
                mode='lines',
                fill='tozeroy',
                opacity=0.4, 
                connectgaps=False,
                hovertemplate='<b>Batterieladung</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
            )
        )
    
    # Battery Discharge (negative values, only show when > 0)
    discharge_data = df_sorted['battery_discharge_kw']
    if discharge_data.max() > 0:
        fig.add_trace(
            go.Scattergl(  # Use Scattergl for better performance
                x=df_sorted['timestamp'],
                y=-discharge_data,
                name='Batterieentladung',
                line=dict(color='#75d7eb', width=1.0),  # Removed spline smoothing
                fill='tozeroy',
                mode='lines',
                connectgaps=False,
                hovertemplate='<b>Batterieentladung</b><br>%{y:.1f} kW<br>%{x}<extra></extra>'
            )
        )

    # Add cost
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['price'] * 1000,  # Convert back to EUR/MWh for better readability
            name='Strompreis',
            line=dict(color='#E2EC2B', width=1),
            mode='lines',
            hovertemplate='<b>Strompreis</b><br>%{y:.2f} EUR/MWh<br>%{x}<extra></extra>'
        )
    )

    # Add peak shaving threshold as horizontal line
    if peak_threshold is not None:
        fig.add_hline(
            y=peak_threshold,
            line_dash="dash",
            line_color="#E41A1C",
            line_width=2,
            annotation_text=f"Spitzenlast-Schwellenwert: {peak_threshold:.0f} kW",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        )

            # Add peak shaving threshold as horizontal line
    if DEBUG ==1 and peak_threshold is not None:
        fig.add_hline(
            y=peak_threshold,
            line_dash="dash",
            line_color="#E41A1C",
            line_width=2,
            annotation_text=f"Net Load kw: {df_sorted["net_load_kw"].max():.0f} kW",
            annotation_position="bottom left",
            annotation=dict(
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        ) 

            # Add peak shaving threshold as horizontal line
    if (df_sorted["grid_import_kw"].max()-1 > peak_threshold) and peak_threshold is not None:
        fig.add_hline(
            y=peak_threshold,
            line_dash="dash",
            line_color="#E41A1C",
            line_width=2,
            annotation_text=f"Schwellenwert für Peakshaving nicht optimal. Spitzenlast: {df_sorted["grid_import_kw"].max():.0f} kW, Zielwert: {peak_threshold:.0f} kW",
            annotation_position="top left",
            annotation=dict(
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        )


    fig.add_annotation(
        x=df_sorted['timestamp'].iloc[len(df_sorted)//2],
        y=peak_threshold+30,
        text=f"Eingesparte Lastspitze: {df["net_load_kw"].max() - df["grid_import_kw"].max():.0f} kW",
        showarrow=False,
        bgcolor="white",
        bordercolor="red",
        borderwidth=1,
        font=dict(color="red", size=12)
    )
    
    # Update layout for full year view
    fig.update_layout(
        title={'text': 'Lastprofil-Analyse', 'x': 0.5, 'xanchor': 'center', 'y': 0.95},
        xaxis_title='Zeit',
        yaxis_title='Leistung (kW)',
        hovermode='x unified',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.05,  # Move legend higher to avoid title overlap
            xanchor="center", 
            x=0.5,  # Center the legend
            bgcolor="rgba(255,255,255,0.8)",  # Add background for better readability
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        height=650,  # Make bigger to accommodate legend spacing
        template='plotly_white',
        margin=dict(t=100),  # Add top margin for title and legend
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=False),  # Add range slider for easier navigation
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="7T", step="day", stepmode="backward"),
                    dict(count=30, label="30T", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(step="all", label="Alle")
                ])
            )
        )
    )
    
    # Additional performance optimization for very large datasets
    if len(df_sorted) > 50000:
        # Reduce hover points for better performance
        fig.update_traces(
            hoverinfo='skip',
            hovermode=False
        )
        st.info("ℹ️ Hover-Details deaktiviert für bessere Performance bei sehr großen Datensätzen")
    
    return fig


def create_load_profile_chart_2(df, peak_threshold=None, soc_col=None,
                              savings_eur=None, original_peak=None, reduced_peak=None):
    """Create a professional 2-panel visualization for load + battery behavior."""
    # Show progress indicator for large datasets
    if len(df) > 10000:
        st.info(f"🔄 Rendering 2-panel chart with {len(df):,} data points...")

    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # OPTIMIZATION: Resample data for faster rendering
    # Get optimization settings from session state or use defaults
    optimization_enabled = st.session_state.get('chart_optimization_enabled', ENABLE_CHART_OPTIMIZATION)
    resample_freq = st.session_state.get('resample_frequency', RESAMPLE_FREQUENCY)
    min_points = st.session_state.get('min_points_for_resampling', MIN_POINTS_FOR_RESAMPLING)
    
    if optimization_enabled and len(df_sorted) > min_points:
        with st.spinner(f"Optimizing chart performance by resampling {len(df_sorted):,} points to {resample_freq} intervals..."):
            try:
                # Handle mixed data types during resampling
                df_indexed = df_sorted.set_index("timestamp")
                
                # Separate numeric and non-numeric columns
                numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
                non_numeric_cols = df_indexed.select_dtypes(exclude=[np.number]).columns
                
                # Resample numeric columns with mean
                df_resampled_numeric = df_indexed[numeric_cols].resample(resample_freq).mean()
                
                # For non-numeric columns, take the first value in each interval
                if len(non_numeric_cols) > 0:
                    df_resampled_non_numeric = df_indexed[non_numeric_cols].resample(resample_freq).first()
                    # Combine the resampled data
                    df_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1).reset_index()
                else:
                    # Only numeric columns, no need to concatenate
                    df_resampled = df_resampled_numeric.reset_index()
                
                st.success(f"✅ Resampled from {len(df_sorted):,} to {len(df_resampled):,} points for faster rendering")
                df_sorted = df_resampled
                
            except Exception as e:
                st.warning(f"⚠️ Resampling failed: {str(e)}. Using original data resolution.")
                st.info("This usually happens with mixed data types. Consider adjusting resampling frequency.")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=['Lastprofile & Spitzenlast', 'Batterieverhalten']
    )

    # ---------- TOP PANEL: LOAD + PEAK SHAVING ----------
    # Original load
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['load_org'],
            name='Ursprüngliche Last',
            line=dict(color='#4D4D4D', width=1.2),
            mode='lines'
        ),
        row=1, col=1
    )

    # Net load (after PV)
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['net_load_kw'],
            name='Netto-Netzlast (nach PV)',
            line=dict(color='#2ca02c', width=1.5),
            mode='lines'
        ),
        row=1, col=1
    )

    # Grid import
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['grid_import_kw'],
            name='Importierte Netzlast',
            line=dict(color='#A1D99B', width=1.2),
            mode='lines'
        ),
        row=1, col=1
    )

    # PV production (optional)
    if 'yearly_production_kw' in df_sorted.columns and df_sorted['yearly_production_kw'].max() > 0:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=-df_sorted['yearly_production_kw'],
                name='PV-Erzeugung',
                line=dict(color='#E6550D', width=1.2),
                mode='lines'
            ),
            row=1, col=1
        )

    # Peak threshold line
    if peak_threshold is not None:
        fig.add_hline(
            y=peak_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Spitzenlast: {peak_threshold:.0f} kW",
            annotation_position="top right",
            row=1, col=1
        )

        # Highlight avoided peaks
        avoided = df_sorted['net_load_kw'].clip(lower=peak_threshold)
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=avoided,
                name='Vermeidene Spitzenlast',
                fill='tonexty',
                line=dict(color='rgba(228,26,28,0)'),
                fillcolor='rgba(228,26,28,0.2)',
                hoverinfo='skip',
                showlegend=True
            ),
            row=1, col=1
        )

    # ---------- BOTTOM PANEL: BATTERY ----------
    # Battery charge (filled)
    if df_sorted['battery_charge_kw'].max() > 0:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=df_sorted['battery_charge_kw'],
                name='Batterieladung',
                mode='lines',
                line=dict(color='#3182BD'),
                fill='tozeroy',
                opacity=0.4
            ),
            row=2, col=1
        )

    # Battery discharge (filled)
    if df_sorted['battery_discharge_kw'].max() > 0:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=-df_sorted['battery_discharge_kw'],
                name='Batterieentladung',
                mode='lines',
                line=dict(color='#6BAED6'),
                fill='tozeroy',
                opacity=0.4
            ),
            row=2, col=1
        )


    # Optional State of Charge line
    if soc_col and soc_col in df_sorted.columns:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=df_sorted[soc_col],
                name='SoC (%)',
                mode='lines',
                line=dict(color='#9467bd', dash='dot', width=1.2),
                yaxis="y2"
            ),
            row=2, col=1
        )

    # ---------- LAYOUT ----------
    fig.update_layout(
        title={'text': 'Energiefluss-Analyse', 'x': 0.5, 'xanchor': 'center'},
        hovermode='x unified',
        template='plotly_white',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)", borderwidth=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7T", step="day", stepmode="backward"),
                    dict(count=30, label="30T", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(step="all", label="Alle")
                ])
            )
        )
    )



    # Add savings annotation if given
    if savings_eur and original_peak and reduced_peak:
        fig.add_annotation(
            x=df_sorted['timestamp'].iloc[len(df_sorted)//2],
            y=peak_threshold + 40,
            text=(f"Eingesparte Lastspitze: {original_peak - reduced_peak:.0f} kW<br>"
                  f"≈ {savings_eur:,.0f} €/Jahr"),
            showarrow=False,
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            font=dict(color="red", size=12)
        )

    fig.update_yaxes(title_text="Leistung (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Batterie (kW)", row=2, col=1)

    return fig


def create_soc_chart(df):
    """Create the State of Charge (SoC) visualization."""
    # Ensure data is sorted by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['battery_soc_kwh'],
            name='Batterie-Ladezustand',
            line=dict(color='#17becf', width=2),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(23, 190, 207, 0.1)',
            hovertemplate='<b>Batterie-Ladezustand</b><br>%{y:.1f} kWh<br>%{x}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title={'text': 'Batterie-Ladezustand (SoC) - Ganzes Jahr', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Zeit',
        yaxis_title='Energie (kWh)',
        hovermode='x unified',
        height=400,  # Make bigger
        template='plotly_white',
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=True),  # Add range slider for easier navigation
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7T", step="day", stepmode="backward"),
                    dict(count=30, label="30T", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(step="all", label="Alle")
                ])
            )
        )
    )
    
    return fig


def create_price_chart(df):
    """Create the electricity price visualization with arbitrage boundaries."""
    # Ensure data is sorted by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    fig = go.Figure()
    
    # Current electricity price
    fig.add_trace(
        go.Scattergl(
            x=df_sorted['timestamp'],
            y=df_sorted['price'] * 1000,  # Convert back to EUR/MWh for better readability
            name='Strompreis',
            line=dict(color='#2E86AB', width=1),
            mode='lines',
            hovertemplate='<b>Strompreis</b><br>%{y:.2f} EUR/MWh<br>%{x}<extra></extra>'
        )
    )
    
    # High price threshold (discharge boundary) - only if available
    if 'future_price_high' in df_sorted.columns:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=df_sorted['future_price_high'] * 1000,  # Convert to EUR/MWh
                name=f'Hochpreis-Schwellenwert (dyn. Perzentil)',
                line=dict(color='#F18F01', width=1, dash='dash'),
                mode='lines',
                hovertemplate='<b>Hochpreis-Schwellenwert</b><br>%{y:.2f} EUR/MWh<br>Batterie entlädt oberhalb dieses Preises<br>%{x}<extra></extra>'
            )
        )
    
    # Low price threshold (charge boundary) - only if available
    if 'future_price_low' in df_sorted.columns:
        fig.add_trace(
            go.Scattergl(
                x=df_sorted['timestamp'],
                y=df_sorted['future_price_low'] * 1000,  # Convert to EUR/MWh
                name=f'Niedrigpreis-Schwellenwert (dyn. Perzentil)',
                line=dict(color='#C73E1D', width=1, dash='dash'),
                mode='lines',
                hovertemplate='<b>Niedrigpreis-Schwellenwert</b><br>%{y:.2f} EUR/MWh<br>Batterie lädt unterhalb dieses Preises<br>%{x}<extra></extra>'
            )
        )
    
    # Add zero line for reference
    fig.add_hline(
        y=0, 
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text="Nullpreis",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title={'text': 'Strompreise & Arbitrage-Grenzen - Ganzes Jahr', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Zeit',
        yaxis_title='Preis (EUR/MWh)',
        hovermode='x unified',
        height=400,  # Make bigger
        template='plotly_white',
        showlegend=False,  # Remove legend from chart
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7T", step="day", stepmode="backward"),
                    dict(count=30, label="30T", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(step="all", label="Alle")
                ])
            )
        )
    )
    
    return fig


def create_info_box(power_rating, capacity):
    """Create an information box showing battery specifications."""

    return f"""
    <div style="
        background: white;
        border: 2px solid #AEAEAE;
        padding: 20px;
        border-radius: 10px;
        color: #00095B;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h3 style="margin-top: 0; color: #00095B;">⚡ Batterie</h3>
        <div style="font-size: 18px; margin: 10px 0; color: #7582F6;">
            <strong>Nennleistung:</strong> {power_rating:,.0f} kW
        </div>
        <div style="font-size: 18px; margin: 10px 0; color: #7582F6;">
            <strong>Kapazität:</strong> {capacity:,.0f} kWh
        </div>
        <div style="font-size: 14px; margin-top: 15px; color: #1A2FEE;">
            C-Rate: {power_rating/capacity:.1f}C
        </div>
    </div>
    """


def create_load_stats_box(df):
    """Create an information box showing load profile statistics."""
    # Calculate statistics
    total_consumption_no_battery = df['net_load_kw'].sum() * 0.25  # Convert to kWh
    peak_load_no_battery = df['net_load_kw'].max()
    total_grid_consumption_with_battery = df['grid_import_kw'].sum() * 0.25  # Convert to kWh
    peak_load_with_battery = df['grid_import_kw'].max()
    
    # Calculate reductions for summary
    consumption_reduction = total_consumption_no_battery - total_grid_consumption_with_battery
    peak_reduction = peak_load_no_battery - peak_load_with_battery
    peak_reduction_percent = (peak_reduction / peak_load_no_battery * 100) if peak_load_no_battery > 0 else 0
    consumption_reduction_percent = (consumption_reduction / total_consumption_no_battery * 100) if total_consumption_no_battery > 0 else 0
    
    # Create comparison table using single line HTML
    html_content = f'<div style="background: white; border: 2px solid #AEAEAE; padding: 20px; border-radius: 10px; color: #00095B; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);margin: 10px 0;"><h3 style="margin-top: 0; color: #00095B;">📊 Lastprofil</h3><table style="width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; background: #f8f9fa; border-radius: 8px; border: 1px solid #AEAEAE;"><thead><tr style="background: #7582F6; color: #00095B;"><th style="padding: 8px; text-align: left;">Kennzahl</th><th style="padding: 8px; text-align: right;">Ohne Batterie</th><th style="padding: 8px; text-align: right;">Mit Batterie</th></tr></thead><tbody><tr><td style="padding: 6px; color: #00095B;">Gesamtverbrauch</td><td style="padding: 6px; text-align: right; color: #00095B;">{total_consumption_no_battery/1000:,.0f} MWh</td><td style="padding: 6px; text-align: right; color: #00095B;">{total_grid_consumption_with_battery/1000:,.0f} MWh</td></tr><tr><td style="padding: 6px; color: #00095B;">Spitzenlast</td><td style="padding: 6px; text-align: right; color: #00095B;">{peak_load_no_battery:,.1f} kW</td><td style="padding: 6px; text-align: right; color: #00095B;">{peak_load_with_battery:,.1f} kW</td></tr></tbody></table><div style="font-size: 16px; margin: 15px 0; padding: 10px; background: #7582F6; color: #00095B; border-radius: 8px; border: 1px solid #AEAEAE;"><strong>📉 Reduktionen:</strong><br><span style="font-size: 14px;">Verbrauch: {consumption_reduction/1000:,.0f} MWh ({consumption_reduction_percent:.1f}%)</span><br><span style="font-size: 14px;">Spitzenlast: -{peak_reduction:.1f} kW ({peak_reduction_percent:.1f}%)</span></div></div>'
    
    return html_content


def create_cost_box(df):
    """Create an information box showing cost analysis comparison."""
    
    # === CASE 4: With Battery ===
    energy_cost_case_4 = (df['grid_import_kw'] * 0.25 * df['price']).sum()
    energy_revenue_case_4 = (df['grid_export_kw'] * 0.25 * df['price'] * EXPORT_REVENUE_FACTOR).sum()
    
    # Calculate demand charge Case 4
    peak_load_case_4 = df['grid_import_kw'].max()
    total_energy_imported_case_4 = (df['grid_import_kw'] * 0.25).sum()
    volllaststunden_case_4 = total_energy_imported_case_4 / peak_load_case_4 if peak_load_case_4 > 0 else 0
    demand_charge_rate_case_4 = DEMAND_CHARGE_HIGH if volllaststunden_case_4 >= VOLLLASTSTUNDEN_THRESHOLD else DEMAND_CHARGE_LOW
    demand_cost_case_4 = peak_load_case_4 * demand_charge_rate_case_4
    
    net_cost_case_4 = energy_cost_case_4 - energy_revenue_case_4 + demand_cost_case_4
    
    # === CASE 1: Without Battery (Original Load) ===
    # Use original load and net_load_kw (which is load after PV)
    energy_cost_case_1 = ((df['net_load_kw'].clip(lower=0)) * 0.25 * df['price']).sum()  # Only positive net load (consumption)
    
    # Calculate revenue from PV surplus (when net_load_kw < 0)
    pv_surplus = np.where(df['net_load_kw'] < 0, -df['net_load_kw'] * 0.25, 0)  # Only negative net load
    energy_revenue_case_1 = (pv_surplus * df['price'] * EXPORT_REVENUE_FACTOR).sum()
    
    # Calculate demand charge Case 1 - based on original peak load
    peak_load_case_1 = df['net_load_kw'].max()
    total_energy_case_1 = (df['net_load_kw'] * 0.25).sum()
    volllaststunden_case_1 = total_energy_case_1 / peak_load_case_1 if peak_load_case_1 > 0 else 0
    demand_charge_rate_case_1 = DEMAND_CHARGE_HIGH if volllaststunden_case_1 >= VOLLLASTSTUNDEN_THRESHOLD else DEMAND_CHARGE_LOW
    demand_cost_case_1 = peak_load_case_1 * demand_charge_rate_case_1
    net_cost_case_1 = energy_cost_case_1 - energy_revenue_case_1 + demand_cost_case_1
    
    


    # FIX POTENTIAL PEAK DEMAND MISMATCH
    if volllaststunden_case_1 < 2500:
        demand_charge_rate_case_1 = 100
        demand_charge_rate_case_4 = 100

        # Change case 4
        demand_cost_case_4 = peak_load_case_4 * demand_charge_rate_case_4
        net_cost_case_4 = energy_cost_case_4 - energy_revenue_case_4 + demand_cost_case_4

        # Change case 1
        demand_cost_case_1 = peak_load_case_1 * demand_charge_rate_case_1
        net_cost_case_1 = energy_cost_case_1 - energy_revenue_case_1 + demand_cost_case_1


    
    # Calculate savings
    total_savings = net_cost_case_1 - net_cost_case_4
    savings_percentage = (total_savings / net_cost_case_1 * 100) if net_cost_case_1 > 0 else 0
    
    # Create HTML content using a single line to avoid formatting issues
    html_content = f'<div style="background: white; border: 2px solid #AEAEAE; padding: 20px; border-radius: 10px; color: #00095B; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0;"><h3 style="margin-top: 0; color: #00095B;">Kosten-Aufstellung</h3><table style="width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; background: #f8f9fa; border-radius: 8px; border: 1px solid #AEAEAE;"><thead><tr style="background: #7582F6; color: #00095B;"><th style="padding: 8px; text-align: left;">Kostenkomponente</th><th style="padding: 8px; text-align: right;">Ohne Batterie</th><th style="padding: 8px; text-align: right;">Mit Batterie</th></tr></thead><tbody><tr><td style="padding: 6px; color: #00095B;">Kosten Energiebezug</td><td style="padding: 6px; text-align: right; color: #00095B;">{energy_cost_case_1:,.0f} €</td><td style="padding: 6px; text-align: right; color: #00095B;">{energy_cost_case_4:,.0f} €</td></tr><tr><td style="padding: 6px; color: #00095B;">Erlöse Energieeinspeisung</td><td style="padding: 6px; text-align: right; color: #00095B;">{energy_revenue_case_1:,.0f} €</td><td style="padding: 6px; text-align: right; color: #00095B;">{energy_revenue_case_4:,.0f} €</td></tr><tr><td style="padding: 6px; color: #00095B;">Leistungskosten</td><td style="padding: 6px; text-align: right; color: #00095B;">{demand_cost_case_1:,.0f} €</td><td style="padding: 6px; text-align: right; color: #00095B;">{demand_cost_case_4:,.0f} €</td></tr><tr style="background: #7582F6; color: #00095B; font-weight: bold;"><td style="padding: 8px; font-size: 15px;">Netto-Jahreskosten</td><td style="padding: 8px; text-align: right; font-size: 15px;">{net_cost_case_1:,.0f} €</td><td style="padding: 8px; text-align: right; font-size: 15px;">{net_cost_case_4:,.0f} €</td></tr></tbody></table><div style="font-size: 18px; margin: 15px 0; padding: 12px; background: #7582F6; color: #00095B; border-radius: 8px; border: 1px solid #AEAEAE;"><strong>Jährliche Einsparungen: {total_savings:,.0f} €</strong><br><span style="font-size: 14px;">({savings_percentage:.1f}% Reduktion)</span></div><div style="font-size: 12px; margin-top: 10px; color: #00095B;">Spitzenlast: {peak_load_case_1:.0f}→{peak_load_case_4:.0f} kW | Tarif: 100% Spot, Leistungspreis {demand_charge_rate_case_4} €/kW/Jahr</div></div>'
    
    return html_content


def create_financial_value_box(df, power_rating, capacity):
    """Create a financial/customer value information box with ROI, payback, and benefit analysis."""
    
    # === CALCULATE FINANCIAL METRICS ===
    
    # 1. Battery Investment Cost (230€/kWh + 20% installation)
    battery_cost_per_kwh = 220  # €/kWh
    installation_factor = 1.20  # 20% additional costs
    estimated_battery_cost = capacity * battery_cost_per_kwh * installation_factor

    # Ensure there is no peak demand mismatch

    
    # 2. Calculate Annual Savings (from existing cost analysis)
    # === CASE 4: With Battery ===
    energy_cost_case_4 = (df['grid_import_kw'] * 0.25 * df['price']).sum()
    energy_revenue_case_4 = (df['grid_export_kw'] * 0.25 * df['price'] * EXPORT_REVENUE_FACTOR).sum()
    
    # Calculate demand charge Case 4
    peak_load_case_4 = df['grid_import_kw'].max()
    total_energy_imported_case_4 = (df['grid_import_kw'] * 0.25).sum()
    volllaststunden_case_4 = total_energy_imported_case_4 / peak_load_case_4 if peak_load_case_4 > 0 else 0
    demand_charge_rate_case_4 = DEMAND_CHARGE_HIGH if volllaststunden_case_4 >= VOLLLASTSTUNDEN_THRESHOLD else DEMAND_CHARGE_LOW
    demand_cost_case_4 = peak_load_case_4 * demand_charge_rate_case_4
    
    net_cost_case_4 = energy_cost_case_4 - energy_revenue_case_4 + demand_cost_case_4
    
    # === CASE 1: Without Battery ===
    total_energy_imported_case_1 = df['net_load_kw'][df['net_load_kw'] > 0].sum() * 0.25
    energy_cost_case_1 = (df[df['net_load_kw'] > 0]['net_load_kw'] * 0.25 * df[df['net_load_kw'] > 0]['price']).sum()
    pv_surplus = np.where(df['net_load_kw'] < 0, -df['net_load_kw'] * 0.25, 0)
    energy_revenue_case_1 = (pv_surplus * df['price'] * EXPORT_REVENUE_FACTOR).sum()
    
    peak_load_case_1 = df['net_load_kw'].max()
    total_energy_case_1 = (df['net_load_kw'] * 0.25).sum()
    volllaststunden_case_1 = total_energy_case_1 / peak_load_case_1 if peak_load_case_1 > 0 else 0
    demand_charge_rate_case_1 = DEMAND_CHARGE_HIGH if volllaststunden_case_1 >= VOLLLASTSTUNDEN_THRESHOLD else DEMAND_CHARGE_LOW
    demand_cost_case_1 = peak_load_case_1 * demand_charge_rate_case_1
    
    net_cost_case_1 = energy_cost_case_1 - energy_revenue_case_1 + demand_cost_case_1


    # FIX POTENTIAL PEAK DEMAND MISMATCH
    if volllaststunden_case_1 < 2500:
        demand_charge_rate_case_1 = 100
        demand_charge_rate_case_4 = 100

        # Change case 4
        demand_cost_case_4 = peak_load_case_4 * demand_charge_rate_case_4
        net_cost_case_4 = energy_cost_case_4 - energy_revenue_case_4 + demand_cost_case_4

        # Change case 1
        demand_cost_case_1 = peak_load_case_1 * demand_charge_rate_case_1
        net_cost_case_1 = energy_cost_case_1 - energy_revenue_case_1 + demand_cost_case_1

        


    
    # Total annual savings
    annual_savings = net_cost_case_1 - net_cost_case_4
    
    # 3. Peak Shaving Benefit
    peak_reduction_kw = peak_load_case_1 - peak_load_case_4
    peak_shaving_benefit = peak_reduction_kw * demand_charge_rate_case_4
    
    # 4. Arbitrage Benefit (charging at low prices, discharging at high prices)
    # Calculate when battery was used for arbitrage (not peak shaving)
    # Arbitrage benefit = difference in costs when battery actively trades energy
    battery_charge_cost = (df['battery_charge_kw'] * 0.25 * df['price']).sum()
    battery_discharge_revenue = (df['battery_discharge_kw'] * 0.25 * df['price'] * EXPORT_REVENUE_FACTOR).sum()
    arbitrage_benefit = battery_discharge_revenue - battery_charge_cost
    
    # 5. PV Self-Consumption Optimization Benefit
    # Calculate benefit from storing PV surplus in battery instead of exporting at low prices
    # This is the difference between what we would have earned from export vs. what we save by using stored energy
    
    # PV surplus that gets stored in battery (when net_load_kw < 0 and battery charges)
    pv_stored_in_battery = df[df['net_load_kw'] < 0]['battery_charge_kw'].sum() * 0.25  # kWh
    
    # Calculate what we would have earned from exporting this energy
    # Use average export price for the periods when PV was generating surplus
    pv_surplus_periods = df[df['net_load_kw'] < 0]
    if len(pv_surplus_periods) > 0:
        avg_export_price = pv_surplus_periods['price'].mean()
        export_revenue_lost = pv_stored_in_battery * avg_export_price * EXPORT_REVENUE_FACTOR
    else:
        export_revenue_lost = 0
    
    # Calculate what we save by using this stored energy instead of importing from grid
    # Use average import price for the periods when battery discharges
    battery_discharge_periods = df[df['battery_discharge_kw'] > 0]
    if len(battery_discharge_periods) > 0:
        avg_import_price = battery_discharge_periods['price'].mean()
        import_cost_saved = pv_stored_in_battery * avg_import_price
    else:
        import_cost_saved = 0
    
    # Net benefit from PV self-consumption optimization
    pv_self_consumption_benefit = import_cost_saved - export_revenue_lost
    
    # 5. ROI and Payback calculations
    if annual_savings > 0:
        payback_years = estimated_battery_cost / annual_savings
        roi_percent = (annual_savings / estimated_battery_cost) * 100
    else:
        payback_years = float('inf')
        roi_percent = 0
    
    # Return calculated metrics for native Streamlit display instead of HTML
    return {
        'estimated_battery_cost': estimated_battery_cost,
        'battery_cost_per_kwh': battery_cost_per_kwh,
        'installation_factor': installation_factor,
        'annual_savings': annual_savings,
        'roi_percent': roi_percent,
        'payback_years': payback_years,
        'peak_shaving_benefit': peak_shaving_benefit,
        'peak_reduction_kw': peak_reduction_kw,
        'arbitrage_benefit': arbitrage_benefit,
        'pv_self_consumption_benefit': pv_self_consumption_benefit,
        'pv_stored_in_battery': pv_stored_in_battery
    }


def main():
    st.set_page_config(
        page_title="Batterie-Analyse (Kosten & Ersparnis)",
        page_icon="💚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for chart optimization settings
    if 'chart_optimization_enabled' not in st.session_state:
        st.session_state['chart_optimization_enabled'] = ENABLE_CHART_OPTIMIZATION
    if 'resample_frequency' not in st.session_state:
        st.session_state['resample_frequency'] = RESAMPLE_FREQUENCY
    if 'min_points_for_resampling' not in st.session_state:
        st.session_state['min_points_for_resampling'] = MIN_POINTS_FOR_RESAMPLING
    
    st.title("🔋 Batterie-Analyse")
    st.markdown("*Visualisierung der Batterieleistung und Energieflüsse*")
    
    # Sidebar for file selection and parameters
    st.sidebar.header("📁 Konfiguration")
    
    # File selection
    #input_directories = ["input_high_savings","input", "input_test", "input_test_alloheim", "input4to10", "input_test_benseler"]
    input_directories = ["input"]
    available_files = []
    
    for directory in input_directories:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
            for file in files:
                available_files.append(os.path.join(directory, file))
    
    if not available_files:
        st.error("Keine Excel-Dateien in den Eingabeverzeichnissen gefunden!")
        return
    
    selected_file = st.sidebar.selectbox(
        "Lastprofil auswählen:",
        available_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Battery parameters
    st.sidebar.subheader("🔋 Batteriekonfiguration")
    power_rating = st.sidebar.number_input(
        "Batterieleistung (kW):",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    capacity = st.sidebar.number_input(
        "Batteriekapazität (kWh):",
        min_value=50,
        max_value=5000000,
        value=215,
        step=5
    )
    
    pv_capacity = st.sidebar.number_input(
        "PV-Kapazität (kWp):",
        min_value=0,
        max_value=5000000,
        value=0,
        step=10
    )
    
    # Price boundary configuration
    st.sidebar.subheader("📊 Arbitrage-Schwellenwerte")
    low_price_percentile = st.sidebar.number_input(
        "Niedrigpreis-Perzentil (%):",
        min_value=1,
        max_value=50,
        value=30,
        step=1,
        help="Perzentil für den unteren Preisschwellenwert (Batterie lädt unterhalb)"
    )
    
    high_price_percentile = st.sidebar.number_input(
        "Hochpreis-Perzentil (%):",
        min_value=51,
        max_value=99,
        value=70,
        step=1,
        help="Perzentil für den oberen Preisschwellenwert (Batterie entlädt oberhalb)"
    )
    
    # Performance optimization settings
    st.sidebar.subheader("⚡ Performance-Optimierung")
    enable_optimization = st.sidebar.checkbox(
        "Chart-Optimierung aktivieren",
        value=ENABLE_CHART_OPTIMIZATION,
        help="Automatische Datenresampling für schnellere Chart-Darstellung"
    )
    
    if enable_optimization:
        resample_freq = st.sidebar.selectbox(
            "Resampling-Frequenz:",
            options=["15T", "30T", "1H", "2H", "4H"],
            index=2,  # Default to 1H
            help="Zeitintervall für Datenresampling (kleiner = mehr Details, langsamer)"
        )
        
        min_points = 1000

        # min_points = st.sidebar.number_input(
        #     "Min. Datenpunkte für Resampling:",
        #     min_value=1000,
        #     max_value=50000,
        #     value=MIN_POINTS_FOR_RESAMPLING,
        #     step=1000,
        #     help="Anzahl Datenpunkte, ab der Resampling aktiviert wird"
        # )
        
        # Additional safety options
        # st.sidebar.info("💡 **Tipp**: Bei Problemen mit gemischten Datentypen, versuchen Sie 1H oder 2H Resampling")
        
        # Store settings in session state instead of global variables
        st.session_state['chart_optimization_enabled'] = enable_optimization
        st.session_state['resample_frequency'] = resample_freq
        st.session_state['min_points_for_resampling'] = min_points
    else:
        # Store disabled state
        st.session_state['chart_optimization_enabled'] = False
        st.session_state['resample_frequency'] = RESAMPLE_FREQUENCY
        st.session_state['min_points_for_resampling'] = MIN_POINTS_FOR_RESAMPLING
        
        st.sidebar.info("ℹ️ Chart-Optimierung deaktiviert. Charts werden mit voller Auflösung gerendert.")

    # Analysis button
    if st.sidebar.button("🚀 Analyse starten", type="primary"):
        with st.spinner("Batterieanalyse wird durchgeführt..."):
            df_result, actual_power, actual_capacity, threshold = run_battery_analysis(
                selected_file, power_rating, capacity, pv_capacity, low_price_percentile, high_price_percentile
            )
            
            if df_result is not None:
                st.success("✅ Analyse erfolgreich abgeschlossen!")
                
                # Store results in session state
                st.session_state['df_result'] = df_result
                st.session_state['power_rating'] = actual_power
                st.session_state['capacity'] = actual_capacity
                st.session_state['threshold'] = threshold
                st.session_state['file_name'] = os.path.basename(selected_file)
                st.session_state['pv_capacity'] = pv_capacity
                st.session_state['low_price_percentile'] = low_price_percentile
                st.session_state['high_price_percentile'] = high_price_percentile
                
                # Show data summary
                st.sidebar.info(f"✅ {len(df_result):,} Datenpunkte geladen von {df_result['timestamp'].min().strftime('%d.%m.%Y')} bis {df_result['timestamp'].max().strftime('%d.%m.%Y')}")
            else:
                st.error("❌ Analyse fehlgeschlagen. Bitte überprüfen Sie Ihre Datei und versuchen Sie es erneut.")
                return
    
    # Display results if available
    if 'df_result' in st.session_state:
        df = st.session_state['df_result']
        power = st.session_state['power_rating']
        cap = st.session_state['capacity']
        threshold = st.session_state.get('threshold', None)
        filename = st.session_state['file_name']
        
        st.subheader(f"📊 Ergebnisse für: `{filename}`")
        
        # Calculate the values directly here as fallback
        energy_cost_case_4 = (df['grid_import_kw'] * 0.25 * df['price']).sum()
        energy_revenue_case_4 = (df['grid_export_kw'] * 0.25 * df['price'] * EXPORT_REVENUE_FACTOR).sum()
        peak_load_case_4 = df['grid_import_kw'].max()
        
        energy_cost_case_1 = (df['net_load_kw'].clip(lower=0) * 0.25 * df['price']).sum()
        pv_surplus = np.where(df['net_load_kw'] < 0, -df['net_load_kw'] * 0.25, 0)
        energy_revenue_case_1 = (pv_surplus * df['price'] * EXPORT_REVENUE_FACTOR).sum()
        peak_load_case_1 = df['net_load_kw'].max()

        savings_eur = (energy_cost_case_1 + energy_revenue_case_1)  - (energy_cost_case_4 + energy_revenue_case_4)

        # Create layout with columns
        col1, col2 = st.columns([1,4])
        
        with col1:
            # Battery Specifications box
            st.markdown(
                create_info_box(power, cap),
                unsafe_allow_html=True
            )

            st.markdown(
                create_load_stats_box(df),
                unsafe_allow_html=True
            )

#        with col2:

        with col2:
            # Main energy flow chart
            st.plotly_chart(
                create_load_profile_chart(df, threshold),
                use_container_width=True
            )
####################################################
        col_kpis_1, col_kpis_2, col_kpis_3 = st.columns(3)

        with col_kpis_1:
            # Financial Value box using native Streamlit components
            financial_metrics = create_financial_value_box(df, power, cap)
            
            # Create a beautiful styled box that matches other boxes using single-line HTML like the working boxes
            html_content = f'<div style="background: white; border: 2px solid #AEAEAE; padding: 20px; border-radius: 10px; color: #00095B; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0;"><h3 style="margin-top: 0; color: #00095B;">Finanzielle Bewertung</h3><div style= "display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;"><div style="background: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #AEAEAE;"><strong style="color: #00095B;">Investitionskosten</strong><br><span style="font-size: 16px; color: #00095B;">{financial_metrics["estimated_battery_cost"]:,.0f} €</span><br><span style="font-size: 12px; color: #00095B;">{financial_metrics["battery_cost_per_kwh"]}€/kWh + {(financial_metrics["installation_factor"]-1)*100:.0f}% Installation</span></div><div style="background: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #AEAEAE;"><strong style="color: #00095B;">📈 ROI & Amortisation</strong><br><span style="font-size: 16px; color: #00095B;">{financial_metrics["roi_percent"]:.1f}% / Jahr</span><br><span style="font-size: 12px; color: #00095B;">Amortisation: {financial_metrics["payback_years"]:.1f} Jahre</span></div></div><div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #AEAEAE;"><strong style="font-size: 16px; color: #00095B;">Erträgnis-Aufschlüsselung:</strong><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; font-size: 14px;"><div><strong style="color: #00095B;">Peak Shaving:</strong><br><span style="color: #00095B;">{financial_metrics["peak_shaving_benefit"]:,.0f} €/Jahr</span><br><span style="font-size: 12px; color: #00095B;">(-{financial_metrics["peak_reduction_kw"]:.1f} kW Spitzenlast)</span></div><div><strong style="color: #00095B;">Arbitrage-Handel:</strong><br><span style="color: #00095B;">{financial_metrics["arbitrage_benefit"]:,.0f} €/Jahr</span><br><span style="font-size: 12px; color: #00095B;">(Preis-Optimierung)</span></div><div><strong style="color: #00095B;">PV Eigenverbrauch:</strong><br><span style="color: #00095B;">{financial_metrics["pv_self_consumption_benefit"]:,.0f} €/Jahr</span><br><span style="font-size: 12px; color: #00095B;">({financial_metrics["pv_stored_in_battery"]:.0f} kWh gespeichert)</span></div></div></div><div style="background: #7582F6; color: #00095B; padding: 12px; border-radius: 8px; border: 2px solid #AEAEAE;"><strong style="font-size: 18px;"> Gesamt-Ersparnis: {financial_metrics["annual_savings"]:,.0f} €/Jahr</strong></div></div>'
            
            st.markdown(html_content, unsafe_allow_html=True)

        with col_kpis_2:
            # Cost Analysis box
            try:
                st.markdown(
                    create_cost_box(df),
                    unsafe_allow_html=True
                )
            except Exception as e:
                # Fallback to native Streamlit components if HTML fails
                st.error(f"HTML-Darstellung fehlgeschlagen: {e}")
                st.markdown("### 💰 Kostenvergleich")
            
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fall 1 Energiekosten", f"{energy_cost_case_1:,.0f} €")
                    st.metric("Fall 1 Erlöse", f"{energy_revenue_case_1:,.0f} €")
                with col2:
                    st.metric("Fall 4 Energiekosten", f"{energy_cost_case_4:,.0f} €")
                    st.metric("Fall 4 Erlöse", f"{energy_revenue_case_4:,.0f} €")


        with col_kpis_3:            
            # Enhanced KPI Box with Battery and PV metrics
            if not df.empty:
                # Battery calculations
                total_energy_charged = df['battery_charge_kw'].sum() * 0.25  # Convert to kWh
                battery_cycles = total_energy_charged / cap if cap > 0 else 0
                
                # Get PV capacity from session state
                pv_cap_kwp = st.session_state.get('pv_capacity', 0)
                
                # PV calculations (only if PV is present)
                if pv_cap_kwp > 0 and 'yearly_production_kw' in df.columns:
                    # Total PV generation
                    total_pv_generation_kwh = df['yearly_production_kw'].sum() * 0.25
                    
                    # Peak PV generation
                    peak_pv_generation_kw = df['yearly_production_kw'].max()
                    
                    # PV utilization - what percentage of theoretical maximum was actually generated
                    # Theoretical max = capacity * hours in year * capacity factor (assume ~11% for Germany)
                    hours_in_year = len(df) * 0.25  # 15-min intervals
                    theoretical_max_kwh = pv_cap_kwp * hours_in_year
                    pv_capacity_factor = (total_pv_generation_kwh / theoretical_max_kwh * 100) if theoretical_max_kwh > 0 else 0
                    
                    # Self-consumption ratio - how much PV was used directly vs exported
                    # PV surplus exported = when net_load_kw < 0
                    pv_surplus_exported = np.where(df['net_load_kw'] < 0, -df['net_load_kw'] * 0.25, 0).sum()
                    self_consumption_kwh = total_pv_generation_kwh - pv_surplus_exported
                    self_consumption_ratio = (self_consumption_kwh / total_pv_generation_kwh * 100) if total_pv_generation_kwh > 0 else 0
                    
                    # PV-to-battery ratio - how much PV was stored in battery
                    pv_stored_in_battery = 0
                    if 'battery_charge_kw' in df.columns:
                        # Estimate PV charging by looking at periods when net_load < 0 and battery is charging
                        pv_charging_periods = (df['net_load_kw'] < 0) & (df['battery_charge_kw'] > 0)
                        pv_stored_in_battery = df[pv_charging_periods]['battery_charge_kw'].sum() * 0.25
                    
                    battery_storage_ratio = (pv_stored_in_battery / total_pv_generation_kwh * 100) if total_pv_generation_kwh > 0 else 0
                    
                    # Autarkie-Grad (energy independence) - how much of total consumption was covered by PV
                    total_consumption_kwh = df['load_org'].sum() * 0.25
                    autarkie_grad = (self_consumption_kwh / total_consumption_kwh * 100) if total_consumption_kwh > 0 else 0
                    
                    # Full utilization hours (Volllaststunden)
                    pv_full_load_hours = total_pv_generation_kwh / pv_cap_kwp if pv_cap_kwp > 0 else 0
                    
                    # Create enhanced KPI box with both Battery and PV
                    kpi_box = f'<div style="background: white; border: 2px solid #AEAEAE; padding: 15px; border-radius: 10px; margin: 10px 0;"><h4 style="margin-top: 0; color: #00095B; text-align: center;">📊 System Kennzahlen</h4><h5 style="margin: 15px 0 8px 0; color: #00095B; text-align: center;">🔋 Batterie</h5><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center; margin-bottom: 15px;"><div><strong style="color: #00095B;">Ø SoC:</strong><br><span style="color: #00095B;">{df["battery_soc_kwh"].mean():.0f} kWh</span><br><span style="color: #00095B;">({df["battery_soc_kwh"].mean()/cap*100:.0f}%)</span></div><div><strong style="color: #00095B;">SoC Bereich:</strong><br><span style="color: #00095B;">{df["battery_soc_kwh"].min():.0f} - {df["battery_soc_kwh"].max():.0f} kWh</span></div><div><strong style="color: #00095B;">Zyklen/Jahr:</strong><br><span style="color: #00095B;">{battery_cycles:.1f}</span></div></div><h5 style="margin: 15px 0 8px 0; color: #00095B; text-align: center;">☀️ PV-Anlage ({pv_cap_kwp:.0f} kWp)</h5><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center; margin-bottom: 8px;"><div><strong style="color: #00095B;">Jahresertrag:</strong><br><span style="color: #00095B;">{total_pv_generation_kwh/1000:.1f} MWh</span></div><div><strong style="color: #00095B;">Volllaststunden:</strong><br><span style="color: #00095B;">{pv_full_load_hours:.0f} h</span></div><div><strong style="color: #00095B;">Spitzenlast:</strong><br><span style="color: #00095B;">{peak_pv_generation_kw:.0f} kW</span></div></div><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;"><div><strong style="color: #00095B;">Eigenverbrauch:</strong><br><span style="color: #00095B;">{self_consumption_ratio:.0f}%</span></div><div><strong style="color: #00095B;">Autarkie-Grad:</strong><br><span style="color: #00095B;">{autarkie_grad:.0f}%</span></div><div><strong style="color: #00095B;">Batteriespeicherung:</strong><br><span style="color: #00095B;">{battery_storage_ratio:.0f}%</span></div></div></div>'
                else:
                    # No PV system - show only battery metrics
                    kpi_box = f'<div style="background: white; border: 2px solid #AEAEAE; padding: 15px; border-radius: 10px; margin: 10px 0;"><h4 style="margin-top: 0; color: #00095B; text-align: center;">📊 Batterie Kennzahlen</h4><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;"><div><strong style="color: #00095B;">Ø SoC:</strong><br><span style="color: #00095B;">{df["battery_soc_kwh"].mean():.0f} kWh ({df["battery_soc_kwh"].mean()/cap*100:.0f}%)</span></div><div><strong style="color: #00095B;">SoC Bereich:</strong><br><span style="color: #00095B;">{df["battery_soc_kwh"].min():.0f} - {df["battery_soc_kwh"].max():.0f} kWh</span></div><div><strong style="color: #00095B;">Batteriezyklen:</strong><br><span style="color: #00095B;">{battery_cycles:.1f} Zyklen</span></div></div><div style="margin-top: 10px; text-align: center; color: #00095B; font-size: 12px;">ℹ️ Keine PV-Anlage konfiguriert</div></div>'
                
                st.markdown(kpi_box, unsafe_allow_html=True)
                
                # Enhanced Price statistics in a styled box
                if 'price' in df.columns:
                    # Basic price stats
                    avg_price = df['price'].mean() * 1000  # Convert to EUR/MWh
                    min_price = df['price'].min() * 1000
                    max_price = df['price'].max() * 1000
                    negative_hours = (df['price'] < 0).sum() / 4  # Convert 15-min intervals to hours
                    
                    # Calculate average costs with and without battery
                    # Without battery: net_load_kw (includes PV already)
                    avg_cost_without_battery = (df['net_load_kw'] * 0.25 * df['price']).sum() / ((df['net_load_kw'] * 0.25).sum() + 0.0001)  # Avoid division by zero
                    
                    # With battery: grid_import_kw (actual grid consumption with battery)
                    total_grid_consumption = df['grid_import_kw'].sum() * 0.25
                    if total_grid_consumption > 0:
                        avg_cost_with_battery = (df['grid_import_kw'] * 0.25 * df['price']).sum() / total_grid_consumption
                    else:
                        avg_cost_with_battery = 0
                    
                    # Create styled price statistics box
                    price_box = f"""
                    <div style="
                        background: white;
                        border: 2px solid #AEAEAE;
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                    ">
                        <h4 style="margin-top: 0; color: #00095B; text-align: center;">Preis-Statistiken</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center; margin-bottom: 15px;">
                            <div><strong style="color: #00095B;">Ø Preis:</strong><br><span style="color: #00095B;">{avg_price:.0f} €/MWh</span></div>
                            <div><strong style="color: #00095B;">Min/Max Preis:</strong><br><span style="color: #00095B;">{min_price:.0f} / {max_price:.0f} €/MWh</span></div>
                            <div><strong style="color: #00095B;">Negative Preise:</strong><br><span style="color: #00095B;">{negative_hours:.0f} Stunden</span></div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: center; border-top: 1px solid #AEAEAE; padding-top: 10px;">
                            <div><strong style="color: #00095B;">Ø Kosten ohne Batterie:</strong><br><span style="color: #00095B;">{avg_cost_without_battery*1000:.1f} €/MWh</span></div>
                            <div><strong style="color: #00095B;">Ø Kosten mit Batterie:</strong><br><span style="color: #00095B;">{avg_cost_with_battery*1000:.1f} €/MWh</span></div>
                        </div>
                    </div>
                    """
                    st.markdown(price_box, unsafe_allow_html=True)


        #col1_kpi, col2_kpi, col3_kpi = st.columns(3)
        #col1_kpi.metric("Peak vorher", f"{df['load_org'].max():.0f} kW")
        #col2_kpi.metric("Peak nachher", f"{df['grid_import_kw'].max():.0f} kW")
        #col3_kpi.metric("Ersparnis", f"{savings_eur:,.0f} € / Jahr")

        # st.plotly_chart(create_load_profile_chart_2(df, threshold, 
        #                                         savings_eur=savings_eur, 
        #                                         original_peak=peak_load_case_1, 
        #                                         reduced_peak=peak_load_case_4), 
        #                 use_container_width=True)

        # st.plotly_chart(
        #     create_load_profile_chart_2(df, threshold),
        #     use_container_width=True
        # )
        
        # SoC chart in expandable section
        with st.expander("🔋 Batterie-Details anzeigen (SoC-Verlauf)", expanded=False):
            st.plotly_chart(
                create_soc_chart(df),
                use_container_width=True
            )
        
        # Price chart below
        st.plotly_chart(
            create_price_chart(df),
            use_container_width=True
        )
        
        # Add legend for price chart outside the chart
        # Get current percentile values from session state
        low_perc = st.session_state.get('low_price_percentile', 30)
        high_perc = st.session_state.get('high_price_percentile', 70)
        
        st.markdown(f"""
        **Strompreis-Diagramm Legende:**
        - 🔵 **Blaue Linie**: Aktueller Strompreis (EUR/MWh)
        - 🟠 **Orange gestrichelt**: Hochpreis-Schwellenwert ({high_perc}. Perzentil) - Batterie entlädt oberhalb
        - 🔴 **Rot gestrichelt**: Niedrigpreis-Schwellenwert ({low_perc}. Perzentil) - Batterie lädt unterhalb
        - ⚫ **Grau gepunktet**: Nullpreis-Linie - Darunter = negative Preise
        """)
            
        # Data table (optional)
        with st.expander("📋 Rohdaten anzeigen"):
            display_cols = ['timestamp', 'load_org', 'yearly_production_kw', 'net_load_kw', 
                            'battery_charge_kw', 'battery_discharge_kw', 'battery_soc_kwh',
                            'price', 'future_price_low', 'future_price_high']
            available_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[available_cols].head(100), use_container_width=True)

    else:
        # Welcome message
        st.info("👆 Wählen Sie ein Lastprofil aus und klicken Sie auf 'Analyse starten'!")
    
    # Show available files
    st.markdown("### 📁 Verfügbare Dateien:")
    for directory in input_directories:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
            if files:
                st.markdown(f"**{directory}/** ({len(files)} Dateien)")
                for file in files[:5]:  # Show first 5 files
                    st.markdown(f"  - `{file}`")
                if len(files) > 5:
                    st.markdown(f"  - ... und {len(files) - 5} weitere Dateien")


if __name__ == "__main__":
    main()
