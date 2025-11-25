from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Div, WMTSTileSource, HoverTool, Button
from bokeh.layouts import column, row
import pandas as pd
import numpy as np
import math
import joblib
import time
from pymavlink import mavutil
from collections import deque
from typing import Dict, Tuple, Optional
import logging
import warnings

# --- NEW: Import our window feature extractor ---
try:
    from window_feature import extract_window_features
except ImportError:
    logger.error("FATAL: 'window_feature.py' not found. Please add it to this folder.")
    raise

# ========================
# CONFIGURATION
# ========================
class Config:
    CONNECTION_STRING = "udp:127.0.0.1:14551"
    REFRESH_INTERVAL_MS = 50
    MAX_POINTS = 400
    
    # --- Paths for the window-based model ---
    MODEL_PATH = "window_ids_model.pkl"
    FEATURES_PATH = "window_feature_list.pkl"
    WINDOW_SIZE_PATH = "WINDOW_SIZE.pkl"
    
    # --- List of *raw* features the DataManager needs to buffer ---
    RAW_FEATURE_COLS = [
        'att_roll', 'att_pitch', 'att_yaw',
        'pos_lat', 'pos_lon', 'pos_alt_rel',
        'pos_vx', 'pos_vy', 'pos_vz',
        'nav_roll', 'nav_pitch', 'nav_alt_error',
        'sys_voltage_battery', 'sys_load',
        'vib_x', 'vib_y', 'vib_z'
    ]
    
    # --- RE-ADDED: Hybrid Smoothing Logic Thresholds ---
    INTRUSION_THRESHOLD = 0.70  # For the "Sustained" average
    SPIKE_THRESHOLD = 0.80      # For an "Instant" spike
    PREDICTION_WINDOW_SIZE = 3  # Look at the last 3 predictions (for smoothing)
    
    LOG_LEVEL = logging.INFO

# Setup logging
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# UTILITY FUNCTIONS
# ========================
def latlon_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """Convert lat/lon to Web Mercator coordinates."""
    r_major = 6378137.0
    x = r_major * math.radians(lon)
    y = r_major * math.log(math.tan((math.pi / 4) + (math.radians(lat) / 2)))
    return x, y

# ========================
# MODEL HANDLER
# ========================
class IDSModel:
    def __init__(self, model_path: str, features_path: str, window_size_path: str):
        self.pipeline = None
        self.feature_cols = []
        self.window_size = 0 # This is the FEATURE window (e.g., 50)

        try:
            logger.info("📦 Loading Windowed IDS model pipeline...")
            self.pipeline = joblib.load(model_path)
            self.feature_cols = joblib.load(features_path)
            self.window_size = joblib.load(window_size_path)
            
            logger.info(f"✅ Model loaded with {len(self.feature_cols)} features")
            logger.info(f"✅ Feature window size set to {self.window_size}")
            
        except FileNotFoundError:
            logger.error(f"❌ Failed to load model/features: File not found")
            logger.error("Using dummy model. Predictions will be 0.")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("Using dummy model. Predictions will be 0.")

    def predict_window(self, window_df: pd.DataFrame) -> float:
        """
        Predict on a full window (DataFrame) and return RAW probability.
        """
        if not self.pipeline:
            return 0.0
        
        try:
            # 1. Extract windowed features (e.g., att_roll_mean, pos_lat_slope)
            features_dict = extract_window_features(window_df)
            
            # 2. Convert to 1-row DataFrame in the correct column order
            features_df = pd.DataFrame([features_dict], columns=self.feature_cols)

            # 3. Get probability from the pipeline (handles imputer + scaler)
            prob = self.pipeline.predict_proba(features_df)[0, 1]
            
            # --- MODIFIED: Return only the raw probability ---
            return prob

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

# ========================
# MAVLINK HANDLER (Unchanged)
# ========================
class MAVLinkHandler:
    # (This class is unchanged)
    def __init__(self, connection_string: str):
        try:
            logger.info(f"🔌 Connecting to MAVLink on {connection_string}...")
            self.master = mavutil.mavlink_connection(connection_string)
            self.master.wait_heartbeat()
            logger.info(f"✅ Connected to system {self.master.target_system}")
            self.start_time = time.time()
        except Exception as e:
            logger.error(f"❌ MAVLink connection failed: {e}")
            raise
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def recv_message(self) -> Optional[object]:
        try:
            return self.master.recv_match(blocking=False)
        except Exception as e:
            logger.warning(f"Message receive error: {e}")
            return None
    
    def set_mode(self, mode: str) -> bool:
        try:
            mode_id = self.master.mode_mapping().get(mode.upper())
            if mode_id is None: logger.error(f"❌ Unknown mode: {mode}"); return False
            logger.info(f"🔄 Attempting to set mode to {mode} (ID: {mode_id})")
            self.master.mav.set_mode_send(self.master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
            start_time = time.time()
            timeout = 5.0
            while time.time() - start_time < timeout:
                ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
                if ack_msg:
                    if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        logger.info(f"✅ Mode changed to {mode} successfully"); return True
                    else:
                        logger.error(f"❌ Mode change rejected: {ack_msg.result}"); return False
            logger.warning(f"⚠️ Mode change timeout"); return False
        except Exception as e:
            logger.error(f"❌ Error setting mode: {e}"); return False

# ========================
# DATA MANAGER (Unchanged from window_dashboard_simple.py)
# ========================
class DataManager:
    # (This class is unchanged)
    def __init__(self, raw_feature_cols: list, window_size: int):
        self.window_size = window_size
        self.raw_feature_cols = raw_feature_cols
        self.message_buffer = deque(maxlen=self.window_size)
        self.latest_row = {col: np.nan for col in self.raw_feature_cols}
        self.vis_data = {
            "vis_lat": 0.0, "vis_lon": 0.0, "vis_alt": 0.0,
            "vis_yaw_deg": 0.0, "vis_voltage": 0.0, "vis_current": 0.0
        }
        self.stats = {
            "total_messages": 0,
            "intrusion_count": 0,
            "normal_count": 0
        }
        self.data_is_ready = False
    
    def get_vis_data(self, key: str, default=0.0):
        return self.vis_data.get(key, default)

    def update_from_message(self, msg_type: str, msg: object):
        try:
            if msg_type == "ATTITUDE":
                self.latest_row["att_roll"] = getattr(msg, 'roll', np.nan)
                self.latest_row["att_pitch"] = getattr(msg, 'pitch', np.nan)
                self.latest_row["att_yaw"] = getattr(msg, 'yaw', np.nan)
                self.vis_data["vis_yaw_deg"] = math.degrees(self.latest_row.get("att_yaw", 0))
            
            elif msg_type == 'NAV_CONTROLLER_OUTPUT':
                self.latest_row["nav_roll"] = getattr(msg, 'nav_roll', np.nan)
                self.latest_row["nav_pitch"] = getattr(msg, 'nav_pitch', np.nan)
                self.latest_row["nav_alt_error"] = getattr(msg, 'alt_error', np.nan)

            elif msg_type == "SYS_STATUS":
                self.latest_row["sys_load"] = getattr(msg, "load", np.nan)
                self.latest_row["sys_voltage_battery"] = getattr(msg, "voltage_battery", np.nan)
                self.vis_data["vis_voltage"] = self.latest_row.get("sys_voltage_battery", 0) / 1000.0
                self.vis_data["vis_current"] = getattr(msg, "current_battery", 0) / 100.0
            
            elif msg_type == "VIBRATION":
                self.latest_row["vib_x"] = getattr(msg, "vibration_x", np.nan)
                self.latest_row["vib_y"] = getattr(msg, "vibration_y", np.nan)
                self.latest_row["vib_z"] = getattr(msg, "vibration_z", np.nan)
            
            elif msg_type == "GLOBAL_POSITION_INT":
                self.stats["total_messages"] += 1
                lat, lon = msg.lat / 1e7, msg.lon / 1e7
                alt_rel = msg.relative_alt / 1000.0
                
                self.latest_row.update({
                    "pos_lat": lat, "pos_lon": lon, "pos_alt_rel": alt_rel,
                    "pos_vx": msg.vx / 100.0, "pos_vy": msg.vy / 100.0, "pos_vz": msg.vz / 100.0,
                })
                
                self.vis_data.update({"vis_lat": lat, "vis_lon": lon, "vis_alt": alt_rel})

                if lat != 0 or lon != 0:
                    self.message_buffer.append(self.latest_row.copy())
                    self.latest_row = {col: np.nan for col in self.raw_feature_cols}
                    
                    if not self.data_is_ready and len(self.message_buffer) == self.window_size:
                        self.data_is_ready = True
                        logger.info(f"✅ Data buffer is full ({self.window_size} messages). Starting windowed predictions.")
                    
                    return lat, lon
            
        except Exception as e:
            logger.warning(f"Error processing {msg_type}: {e}")
        
        return None

# ========================
# VISUALIZATION (Unchanged from window_dashboard_simple.py)
# ========================
class Dashboard:
    # (This class is unchanged)
    def __init__(self, prevention_callback):
        self.src_alt = ColumnDataSource(data=dict(time=[], alt=[]))
        self.src_yaw = ColumnDataSource(data=dict(time=[], yaw=[]))
        self.src_path = ColumnDataSource(data=dict(x=[], y=[], label=[]))
        self.src_velocity = ColumnDataSource(data=dict(time=[], vx=[], vy=[], vz=[]))
        self.src_battery = ColumnDataSource(data=dict(time=[], voltage=[], current=[]))
        
        self.alert_div = Div(
            text="<div style='padding:20px; background:#2C3E50; color:white; border-radius:8px; font-size:18px;'>"
                 "⏳ Status: Waiting for core sensor data...</div>",
            sizing_mode="stretch_width"
        )
        
        self.stats_div = Div(
            text="<div style='padding:10px; background:#34495E; color:white; border-radius:5px;'>"
                 "📊 Messages: 0 | ✅ Normal: 0 | 🚨 Intrusions: 0</div>",
            sizing_mode="stretch_width"
        )
        
        self.prevention_button = Button(
            label="🛡️ ACTIVATE PREVENTION (Switch to AUTO Mode)",
            button_type="warning", width=400, height=50, sizing_mode="fixed"
        )
        self.prevention_button.on_click(prevention_callback)

        self.p_alt = self._create_plot("Altitude (m)", "Time (s)", "Altitude", self.src_alt, 'alt', "#3498DB", [("Time", "@time{0.2f}s"), ("Alt", "@alt{0.2f}m")])
        self.p_yaw = self._create_plot("Yaw (°)", "Time (s)", "Yaw", self.src_yaw, 'yaw', "#E67E22", [("Time", "@time{0.2f}s"), ("Yaw", "@yaw{0.2f}°")])
        self.p_velocity = self._create_velocity_plot()
        self.p_battery = self._create_battery_plot()
        self.p_map = self._create_map_plot()
        
        self.model_window_size = 0
    
    def _create_plot(self, title, x_label, y_label, source, y_col, color, tooltips):
        p = figure(height=300, title=title, x_axis_label=x_label, y_axis_label=y_label,
                   background_fill_color="#222222", border_fill_color="#222222",
                   sizing_mode="stretch_width", toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset,save")
        p.line(source=source, x='time', y=y_col, line_width=2, color=color, legend_label=title.split(" ")[0])
        p.legend.location = "top_left"
        p.add_tools(HoverTool(tooltips=tooltips))
        return p

    def _create_velocity_plot(self):
        p = self._create_plot("Velocity (m/s)", "Time (s)", "Velocity", self.src_velocity, 'vx', "#1ABC9C", [])
        p.line(source=self.src_velocity, x='time', y='vx', line_width=2, color="#1ABC9C", legend_label="Vx", alpha=0.8)
        p.line(source=self.src_velocity, x='time', y='vy', line_width=2, color="#9B59B6", legend_label="Vy", alpha=0.8)
        p.line(source=self.src_velocity, x='time', y='vz', line_width=2, color="#F39C12", legend_label="Vz", alpha=0.8)
        p.legend.location = "top_right"; p.legend.click_policy = "hide"
        p.add_tools(HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Vx", "@vx{0.2f}m/s"), ("Vy", "@vy{0.2f}m/s"), ("Vz", "@vz{0.2f}m/s")]))
        return p

    def _create_battery_plot(self):
        p = self._create_plot("Battery Status (V/A)", "Time (s)", "Value", self.src_battery, 'voltage', "#16A085", [])
        p.line(source=self.src_battery, x='time', y='voltage', line_width=2, color="#16A085", legend_label="Voltage (V)", alpha=0.9)
        p.line(source=self.src_battery, x='time', y='current', line_width=2, color="#E74C3C", legend_label="Current (A)", alpha=0.9)
        p.legend.location = "top_right"; p.legend.click_policy = "hide"
        p.add_tools(HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Voltage", "@voltage{0.2f}V"), ("Current", "@current{0.2f}A")]))
        return p

    def _create_map_plot(self):
        tile_provider = WMTSTileSource(url="httpss://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png", attribution="© OpenStreetMap contributors © CARTO")
        p = figure(height=500, x_axis_type="mercator", y_axis_type="mercator", title="2D Flight Path (OpenStreetMap)",
                   sizing_mode="stretch_width", toolbar_location="above", tools="pan,wheel_zoom,reset,save")
        p.add_tile(tile_provider)
        p.line(x="x", y="y", line_width=3, color="#E74C3C", alpha=0.8, source=self.src_path)
        p.circle(x="x", y="y", size=5, color="#C0392B", alpha=0.6, source=self.src_path)
        return p
    
    def update_altitude(self, time_val: float, alt: float): self.src_alt.stream({"time": [time_val], "alt": [alt]}, rollover=Config.MAX_POINTS)
    def update_yaw(self, time_val: float, yaw: float): self.src_yaw.stream({"time": [time_val], "yaw": [yaw]}, rollover=Config.MAX_POINTS)
    def update_velocity(self, time_val: float, vx: float, vy: float, vz: float): self.src_velocity.stream({"time": [time_val], "vx": [vx], "vy": [vy], "vz": [vz]}, rollover=Config.MAX_POINTS)
    def update_battery(self, time_val: float, voltage: float, current: float): self.src_battery.stream({"time": [time_val], "voltage": [voltage], "current": [current]}, rollover=Config.MAX_POINTS)
    def update_map(self, lat: float, lon: float, label: int): mx, my = latlon_to_mercator(lat, lon); self.src_path.stream({"x": [mx], "y": [my], "label": [label]}, rollover=Config.MAX_POINTS)
    def update_stats(self, stats: dict): self.stats_div.text = f"<div style='padding:10px; background:#34495E; color:white; border-radius:5px; text-align:center;'>📊 Messages: {stats['total_messages']} | ✅ Normal: {stats['normal_count']} | 🚨 Intrusions: {stats['intrusion_count']}</div>"

    def update_alert(self, label: int, prob: float, buffer_len_override: Optional[int] = None):
        if label == 1:
            color, icon, status = "#E74C3C", "🚨", "INTRUSION DETECTED"
        elif label == 0:
            color, icon, status = "#27AE60", "✅", "Normal Flight"
        else: # label == -1
            buffer_len = buffer_len_override if buffer_len_override is not None else 0
            window_size = self.model_window_size if self.model_window_size > 0 else "..."
            color, icon, status = "#2C3E50", "⏳", f"Buffering Data ({buffer_len}/{window_size})..."
        
        self.alert_div.text = (
            f"<div style='padding:20px; background:{color}; color:white; border-radius:8px; "
            f"font-size:20px; font-weight:bold; text-align:center;'>"
            f"{icon} {status} | Confidence: {prob:.1%}</div>"
        )
    
    def get_layout(self):
        top_row = row(self.p_alt, self.p_yaw, sizing_mode="stretch_width")
        mid_row = row(self.p_velocity, self.p_battery, sizing_mode="stretch_width")
        prevention_row = row(self.prevention_button, sizing_mode="stretch_width")
        
        return column(
            self.alert_div,
            self.stats_div,
            prevention_row,
            top_row,
            mid_row,
            self.p_map,
            sizing_mode="stretch_width"
        )

# ========================
# MAIN APPLICATION (*** MODIFIED ***)
# ========================
class DroneIDSApp:
    def __init__(self):
        self.model = IDSModel(
            Config.MODEL_PATH,
            Config.FEATURES_PATH,
            Config.WINDOW_SIZE_PATH
        )
        self.mavlink = MAVLinkHandler(Config.CONNECTION_STRING)
        self.data_mgr = DataManager(
            raw_feature_cols=Config.RAW_FEATURE_COLS,
            window_size=self.model.window_size
        )
        self.dashboard = Dashboard(self.activate_prevention)
        self.dashboard.model_window_size = self.model.window_size 
        
        # --- RE-ADDED: Hybrid Smoothing Deque ---
        self.prob_history = deque(maxlen=Config.PREDICTION_WINDOW_SIZE)
    
    def activate_prevention(self):
        """Callback for prevention button - switches drone to AUTO mode."""
        logger.info("🛡️ Prevention activated - attempting to switch to AUTO mode")
        success = self.mavlink.set_mode("AUTO")
        if success:
            logger.info("✅ Prevention Active: Drone switched to AUTO mode")
        else:
            logger.error("❌ Prevention Failed: Could not switch to AUTO mode")
    
    def update(self):
        """Main update callback."""
        msg = self.mavlink.recv_message()
        if not msg:
            return
        
        msg_type = msg.get_type()
        now = self.mavlink.get_elapsed_time()
        
        result = self.data_mgr.update_from_message(msg_type, msg)
        
        if msg_type == "ATTITUDE":
            self.dashboard.update_yaw(now, self.data_mgr.get_vis_data("vis_yaw_deg"))
        
        elif msg_type == "SYS_STATUS":
            self.dashboard.update_battery(now, 
                self.data_mgr.get_vis_data("vis_voltage"), 
                self.data_mgr.get_vis_data("vis_current"))
        
        elif msg_type == "GLOBAL_POSITION_INT" and result:
            lat, lon = result
            
            self.dashboard.update_altitude(now, self.data_mgr.get_vis_data("vis_alt"))
            self.dashboard.update_velocity(now, 
                self.data_mgr.latest_row.get("pos_vx", 0), 
                self.data_mgr.latest_row.get("pos_vy", 0),
                self.data_mgr.latest_row.get("pos_vz", 0)
            )
            
            # 1. Check if the buffer is full
            if not self.data_mgr.data_is_ready:
                current_buffer_len = len(self.data_mgr.message_buffer)
                self.dashboard.update_alert(-1, 0.0, buffer_len_override=current_buffer_len)
                self.dashboard.update_map(lat, lon, -1)
                return
            
            # 2. If buffer is full, create window DataFrame
            window_df = pd.DataFrame(
                list(self.data_mgr.message_buffer),
                columns=self.data_mgr.raw_feature_cols
            )
            
            # 3. Get RAW prediction from the window model
            raw_prob = self.model.predict_window(window_df)
            
            # --- RE-ADDED: Hybrid Smoothing Logic ---
            self.prob_history.append(raw_prob)

            final_label = 0
            final_prob = raw_prob # Default to raw

            # Check for instant spike
            if raw_prob > Config.SPIKE_THRESHOLD:
                final_label = 1
                final_prob = raw_prob
                logger.warning(f"Spike detected! Raw Prob: {raw_prob:.2%}")

            # Check for sustained attack
            elif len(self.prob_history) == Config.PREDICTION_WINDOW_SIZE:
                smoothed_prob = sum(self.prob_history) / Config.PREDICTION_WINDOW_SIZE
                
                if smoothed_prob > Config.INTRUSION_THRESHOLD:
                    final_label = 1
                    final_prob = smoothed_prob # Use the smoothed avg as confidence
                    logger.warning(f"Sustained attack! Smoothed Prob: {smoothed_prob:.2%}")
                else:
                    final_label = 0
                    final_prob = smoothed_prob # Show smoothed (low) prob
            
            else:
                # This case happens when the smoothing buffer isn't full yet
                # We default to 'normal' to avoid pre-mature alerts
                final_label = 0
                final_prob = raw_prob
            
            # 4. Update Dashboard
            self.dashboard.update_map(lat, lon, final_label)
            self.dashboard.update_alert(final_label, final_prob)
            
            # 5. Update statistics
            if final_label == 1:
                self.data_mgr.stats["intrusion_count"] += 1
            else:
                self.data_mgr.stats["normal_count"] += 1
            
            self.dashboard.update_stats(self.data_mgr.stats)

# ========================
# INITIALIZE & RUN
# ========================
try:
    logger.info("=======================================")
    logger.info("🚀 STARTING DRONE IDS (HYBRID WINDOWED) DASHBOARD")
    logger.info("Make sure 'window_feature.py' is in this folder.")
    logger.info("=======================================")
    
    app = DroneIDSApp()
    
    curdoc().template = "bootstrap"
    curdoc().add_root(app.dashboard.get_layout())
    curdoc().add_periodic_callback(app.update, Config.REFRESH_INTERVAL_MS)
    curdoc().title = "Drone IDS Live Dashboard (Hybrid Windowed)"

except Exception as e:
    logger.error(f"❌ Fatal error on startup: {e}", exc_info=True)
    error_div = Div(text=f"<h1>Fatal Error</h1><pre>{e}</pre>", sizing_mode="stretch_both")
    curdoc().template = "bootstrap"
    curdoc().add_root(error_div)
    raise

# To run:
# 1. Make sure you have 'window_feature.py' in the same directory.
# 2. Make sure you have 'window_ids_model.pkl', 'window_feature_list.pkl', and 'WINDOW_SIZE.pkl'.
# 3. Run: bokeh serve --show window_dashboard_hybrid.py