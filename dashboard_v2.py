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

# ========================
# CONFIGURATION
# ========================
class Config:
    CONNECTION_STRING = "udp:127.0.0.1:14551"
    REFRESH_INTERVAL_MS = 50
    MAX_POINTS = 400
    # !! Make sure these files exist in the same folder !!
    MODEL_PATH = "gb_ids_model.pkl" 
    FEATURES_PATH = "feature_list.pkl"
    
    # --- UPDATED: More responsive thresholds ---
    INTRUSION_THRESHOLD = 0.70  # For the "Sustained" average
    SPIKE_THRESHOLD = 0.80      # For an "Instant" spike
    
    PREDICTION_WINDOW_SIZE = 3  # Look at the last 3 predictions
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
    def __init__(self, model_path: str, features_path: str):
        self.model = None
        self.feature_cols = []
        try:
            logger.info("📦 Loading IDS model...")
            self.model = joblib.load(model_path)
            self.feature_cols = joblib.load(features_path)
            logger.info(f"✅ Model loaded with {len(self.feature_cols)} features")
        except FileNotFoundError:
            logger.error(f"❌ Failed to load model: File not found at {model_path} or {features_path}")
            logger.error("Using dummy model. Predictions will be 0.")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("Using dummy model. Predictions will be 0.")

    def predict(self, data: Dict[str, float]) -> Tuple[int, float]:
        """Predict intrusion and return label + probability."""
        if not self.model or not self.feature_cols:
            return 0, 0.0
        
        try:
            # The imputer in your pipeline will handle np.nan
            df_data = {col: [data.get(col, np.nan)] for col in self.feature_cols}
            df = pd.DataFrame.from_dict(df_data)
            
            prob = self.model.predict_proba(df)[0, 1]
            # Return raw probability, we will threshold later
            return int(prob > Config.INTRUSION_THRESHOLD), prob 
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0

# ========================
# MAVLINK HANDLER
# ========================
class MAVLinkHandler:
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
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def recv_message(self) -> Optional[object]:
        """Receive MAVLink message (non-blocking)."""
        try:
            return self.master.recv_match(blocking=False)
        except Exception as e:
            logger.warning(f"Message receive error: {e}")
            return None
    
    def set_mode(self, mode: str) -> bool:
        """Set flight mode to specified mode (e.g., 'AUTO', 'GUIDED', 'RTL')."""
        try:
            # Get mode ID from mode mapping
            mode_id = self.master.mode_mapping().get(mode.upper())
            
            if mode_id is None:
                logger.error(f"❌ Unknown mode: {mode}")
                return False
            
            logger.info(f"🔄 Attempting to set mode to {mode} (ID: {mode_id})")
            
            # Send SET_MODE command
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            # Wait for mode change acknowledgment (with timeout)
            start_time = time.time()
            timeout = 5.0  # 5 second timeout
            
            while time.time() - start_time < timeout:
                ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
                if ack_msg:
                    if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        logger.info(f"✅ Mode changed to {mode} successfully")
                        return True
                    else:
                        logger.error(f"❌ Mode change rejected: {ack_msg.result}")
                        return False
            
            logger.warning(f"⚠️ Mode change timeout - command sent but no acknowledgment received")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error setting mode: {e}")
            return False

# ========================
# DATA MANAGER
# ========================
class DataManager:
    def __init__(self, feature_cols: list):
        # Initialize all features to np.nan
        self.latest = {col: np.nan for col in feature_cols}
        
        # Add keys for visualization
        self.latest.update({
            "vis_lat": 0.0, "vis_lon": 0.0, "vis_alt": 0.0,
            "vis_yaw_deg": 0.0, "vis_voltage": 0.0, "vis_current": 0.0
        })
        
        self.stats = {
            "total_messages": 0,
            "intrusion_count": 0,
            "normal_count": 0
        }
        
        self.seen_msg_types = set()
        self.data_is_ready = False
        self.required_types = {"ATTITUDE"}
    
    def update_from_message(self, msg_type: str, msg: object):
        """Update latest data from MAVLink message."""
        try:
            if msg_type == "ATTITUDE":
                # Save raw RADIANS for the model
                self.latest["att_roll"] = getattr(msg, 'roll', np.nan)
                self.latest["att_pitch"] = getattr(msg, 'pitch', np.nan)
                self.latest["att_yaw"] = getattr(msg, 'yaw', np.nan)
                # Save degrees for visualization
                self.latest["vis_yaw_deg"] = math.degrees(self.latest.get("att_yaw", 0))
            
            elif msg_type == "GLOBAL_POSITION_INT":
                lat, lon = msg.lat / 1e7, msg.lon / 1e7
                self.latest.update({
                    "pos_lat": lat,
                    "pos_lon": lon,
                    "pos_alt_rel": msg.relative_alt / 1000.0,
                    "pos_vx": msg.vx / 100.0,
                    "pos_vy": msg.vy / 100.0,
                    "pos_vz": msg.vz / 100.0,
                })
                self.latest.update({"vis_lat": lat, "vis_lon": lon, "vis_alt": self.latest["pos_alt_rel"]})
                
                if lat != 0 or lon != 0:
                    return lat, lon
            
            elif msg_type == 'NAV_CONTROLLER_OUTPUT':
                self.latest["nav_roll"] = getattr(msg, 'nav_roll', np.nan)
                self.latest["nav_pitch"] = getattr(msg, 'nav_pitch', np.nan)
                self.latest["nav_alt_error"] = getattr(msg, 'alt_error', np.nan)

            elif msg_type == "SYS_STATUS":
                self.latest["sys_load"] = getattr(msg, "load", np.nan)
                self.latest["sys_voltage_battery"] = getattr(msg, "voltage_battery", np.nan)
                self.latest["vis_voltage"] = self.latest.get("sys_voltage_battery", 0) / 1000.0
                self.latest["vis_current"] = getattr(msg, "current_battery", 0) / 100.0
            
            elif msg_type == "VIBRATION":
                self.latest["vib_x"] = getattr(msg, "vibration_x", np.nan)
                self.latest["vib_y"] = getattr(msg, "vibration_y", np.nan)
                self.latest["vib_z"] = getattr(msg, "vibration_z", np.nan)
            
            self.stats["total_messages"] += 1

            # Check for data readiness
            if not self.data_is_ready:
                if msg_type in self.required_types:
                    self.seen_msg_types.add(msg_type)
                
                if self.required_types.issubset(self.seen_msg_types):
                    self.data_is_ready = True
                    logger.info("✅ Core sensors are live. Starting IDS predictions.")
                    
        except Exception as e:
            logger.warning(f"Error processing {msg_type}: {e}")
        
        return None

# ========================
# VISUALIZATION
# ========================
class Dashboard:
    def __init__(self, prevention_callback):
        # Data sources
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
        
        # Prevention button
        self.prevention_button = Button(
            label="🛡️ ACTIVATE PREVENTION (Switch to AUTO Mode)",
            button_type="warning",
            width=400,
            height=50,
            sizing_mode="fixed"
        )
        self.prevention_button.on_click(prevention_callback)

        
        # Create figures
        self.p_alt = self._create_altitude_plot()
        self.p_yaw = self._create_yaw_plot()
        self.p_velocity = self._create_velocity_plot()
        self.p_battery = self._create_battery_plot()
        self.p_map = self._create_map_plot()
    
    def _create_plot(self, title, x_label, y_label):
        """Helper function to create a standard plot."""
        return figure(
            height=300,
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            background_fill_color="#222222",
            border_fill_color="#222222",
            sizing_mode="stretch_width",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

    def _create_altitude_plot(self):
        p = self._create_plot("Altitude (m)", "Time (s)", "Altitude")
        p.line(source=self.src_alt, x='time', y='alt', line_width=2, color="#3498DB", legend_label="Altitude")
        p.legend.location = "top_left"
        hover = HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Alt", "@alt{0.2f}m")])
        p.add_tools(hover)
        return p
    
    def _create_yaw_plot(self):
        p = self._create_plot("Yaw (°)", "Time (s)", "Yaw")
        p.line(source=self.src_yaw, x='time', y='yaw', line_width=2, color="#E67E22", legend_label="Yaw")
        p.legend.location = "top_left"
        hover = HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Yaw", "@yaw{0.2f}°")])
        p.add_tools(hover)
        return p
    
    def _create_velocity_plot(self):
        p = self._create_plot("Velocity (m/s)", "Time (s)", "Velocity")
        p.line(source=self.src_velocity, x='time', y='vx', line_width=2, color="#1ABC9C", legend_label="Vx", alpha=0.8)
        p.line(source=self.src_velocity, x='time', y='vy', line_width=2, color="#9B59B6", legend_label="Vy", alpha=0.8)
        p.line(source=self.src_velocity, x='time', y='vz', line_width=2, color="#F39C12", legend_label="Vz", alpha=0.8)
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        hover = HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Vx", "@vx{0.2f}m/s"), ("Vy", "@vy{0.2f}m/s"), ("Vz", "@vz{0.2f}m/s")])
        p.add_tools(hover)
        return p
    
    def _create_battery_plot(self):
        p = self._create_plot("Battery Status (V/A)", "Time (s)", "Value")
        p.line(source=self.src_battery, x='time', y='voltage', line_width=2, color="#16A085", legend_label="Voltage (V)", alpha=0.9)
        p.line(source=self.src_battery, x='time', y='current', line_width=2, color="#E74C3C", legend_label="Current (A)", alpha=0.9)
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        hover = HoverTool(tooltips=[("Time", "@time{0.2f}s"), ("Voltage", "@voltage{0.2f}V"), ("Current", "@current{0.2f}A")])
        p.add_tools(hover)
        return p
    
    def _create_map_plot(self):
        tile_provider = WMTSTileSource(
            url="https://a.basemaps.cartocdn.com/dark_all/{Z}/{X}/{Y}.png",
            attribution="© OpenStreetMap contributors © CARTO"
        )
        p = figure(
            height=500,
            x_axis_type="mercator",
            y_axis_type="mercator",
            title="2D Flight Path (OpenStreetMap)",
            sizing_mode="stretch_width",
            toolbar_location="above",
            tools="pan,wheel_zoom,reset,save"
        )
        p.add_tile(tile_provider)
        p.line(x="x", y="y", line_width=3, color="#E74C3C", alpha=0.8, source=self.src_path)
        p.circle(x="x", y="y", size=5, color="#C0392B", alpha=0.6, source=self.src_path)
        return p
    
    def update_altitude(self, time_val: float, alt: float):
        self.src_alt.stream({"time": [time_val], "alt": [alt]}, rollover=Config.MAX_POINTS)
    
    def update_yaw(self, time_val: float, yaw: float):
        self.src_yaw.stream({"time": [time_val], "yaw": [yaw]}, rollover=Config.MAX_POINTS)
    
    def update_velocity(self, time_val: float, vx: float, vy: float, vz: float):
        self.src_velocity.stream({"time": [time_val], "vx": [vx], "vy": [vy], "vz": [vz]}, rollover=Config.MAX_POINTS)
    
    def update_battery(self, time_val: float, voltage: float, current: float):
        self.src_battery.stream({"time": [time_val], "voltage": [voltage], "current": [current]}, rollover=Config.MAX_POINTS)
    
    def update_map(self, lat: float, lon: float, label: int):
        mx, my = latlon_to_mercator(lat, lon)
        self.src_path.stream({"x": [mx], "y": [my], "label": [label]}, rollover=Config.MAX_POINTS)
    
    def update_alert(self, label: int, prob: float):
        if label == 1:
            color = "#E74C3C"
            icon = "🚨"
            status = "INTRUSION DETECTED"
        elif label == 0:
            color = "#27AE60"
            icon = "✅"
            status = "Normal Flight"
        else:
            color = "#2C3E50"
            icon = "⏳"
            status = "Waiting for core sensor data..."
        
        self.alert_div.text = (
            f"<div style='padding:20px; background:{color}; color:white; border-radius:8px; "
            f"font-size:20px; font-weight:bold; text-align:center;'>"
            f"{icon} {status} | Confidence: {prob:.1%}</div>"
        )
    
    def update_stats(self, stats: dict):
        self.stats_div.text = (
            f"<div style='padding:10px; background:#34495E; color:white; border-radius:5px; text-align:center;'>"
            f"📊 Messages: {stats['total_messages']} | "
            f"✅ Normal: {stats['normal_count']} | "
            f"🚨 Intrusions: {stats['intrusion_count']}</div>"
        )
    
    def get_layout(self):
        top_row = row(self.p_alt, self.p_yaw, sizing_mode="stretch_width")
        mid_row = row(self.p_velocity, self.p_battery, sizing_mode="stretch_width")
        
        # Add prevention controls
        prevention_row = row(
            self.prevention_button,
            sizing_mode="stretch_width",
            css_classes=["prevention-controls"]
        )
        
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
# MAIN APPLICATION
# ========================
class DroneIDSApp:
    def __init__(self):
        self.model = IDSModel(Config.MODEL_PATH, Config.FEATURES_PATH)
        self.mavlink = MAVLinkHandler(Config.CONNECTION_STRING)
        self.data_mgr = DataManager(self.model.feature_cols)
        self.dashboard = Dashboard(self.activate_prevention)
        self.last_prediction_time = 0
        self.prediction_interval = 0.05
        self.prob_history = deque(maxlen=Config.PREDICTION_WINDOW_SIZE)
    
    def activate_prevention(self):
        """Callback for prevention button - switches drone to AUTO mode."""
        logger.info("🛡️ Prevention activated - attempting to switch to AUTO mode")
        
        success = self.mavlink.set_mode("AUTO")
        
        if success:
            message = "Prevention Active: Drone switched to AUTO mode"
            logger.info(f"✅ {message}")
        else:
            message = "Prevention Failed: Could not switch to AUTO mode"
            logger.error(f"❌ {message}")
        
       
    
    def update(self):
        """Main update callback."""
        msg = self.mavlink.recv_message()
        if not msg:
            return
        
        msg_type = msg.get_type()
        now = self.mavlink.get_elapsed_time()
        
        # Update data from message
        result = self.data_mgr.update_from_message(msg_type, msg)
        
        # Update visualizations based on message type
        if msg_type == "ATTITUDE":
            self.dashboard.update_yaw(now, self.data_mgr.latest.get("vis_yaw_deg", 0))
        
        elif msg_type == "GLOBAL_POSITION_INT" and result:
            lat, lon = result
            if lat and lon:
                alt = self.data_mgr.latest.get("vis_alt", 0)
                vx = self.data_mgr.latest.get("pos_vx", 0)
                vy = self.data_mgr.latest.get("pos_vy", 0)
                vz = self.data_mgr.latest.get("pos_vz", 0)
                
                self.dashboard.update_altitude(now, alt)
                self.dashboard.update_velocity(now, vx, vy, vz)
                
                # Run prediction at controlled intervals
                if now - self.last_prediction_time >= self.prediction_interval:
                    
                    if not self.data_mgr.data_is_ready:
                        self.dashboard.update_alert(-1, 0.0)
                        return
                    
                    # Get Raw Prediction
                    raw_label, raw_prob = self.model.predict(self.data_mgr.latest)
                    
                    # Add raw probability to History
                    self.prob_history.append(raw_prob)

                    final_label = 0
                    final_prob = raw_prob

                    # Check for instant spike
                    if raw_prob > Config.SPIKE_THRESHOLD:
                        final_label = 1
                        final_prob = raw_prob

                    # Check for sustained attack
                    elif len(self.prob_history) == Config.PREDICTION_WINDOW_SIZE:
                        smoothed_prob = sum(self.prob_history) / Config.PREDICTION_WINDOW_SIZE
                        
                        if smoothed_prob > Config.INTRUSION_THRESHOLD:
                            final_label = 1
                            final_prob = smoothed_prob
                        else:
                            final_label = 0
                            final_prob = smoothed_prob
                    
                    else:
                        final_label = 0
                        final_prob = raw_prob
                    
                    # Update Dashboard
                    self.dashboard.update_map(lat, lon, final_label)
                    self.dashboard.update_alert(final_label, final_prob)
                    
                    # Update statistics
                    if final_label == 1:
                        self.data_mgr.stats["intrusion_count"] += 1
                    else:
                        self.data_mgr.stats["normal_count"] += 1
                    
                    self.dashboard.update_stats(self.data_mgr.stats)
                    self.last_prediction_time = now
        
        elif msg_type == "SYS_STATUS":
            voltage = self.data_mgr.latest.get("vis_voltage", 0)
            current = self.data_mgr.latest.get("vis_current", 0)
            self.dashboard.update_battery(now, voltage, current)

# ========================
# INITIALIZE & RUN
# ========================
try:
    app = DroneIDSApp()
    
    curdoc().template = "bootstrap"
    curdoc().add_root(app.dashboard.get_layout())
    curdoc().add_periodic_callback(app.update, Config.REFRESH_INTERVAL_MS)
    curdoc().title = "Drone IDS Live Dashboard"
    logger.info("🚀 Dashboard started successfully")

except Exception as e:
    logger.error(f"❌ Fatal error: {e}")
    error_div = Div(text=f"<h1>Fatal Error</h1><pre>{e}</pre>", sizing_mode="stretch_both")
    curdoc().template = "bootstrap"
    curdoc().add_root(error_div)
    raise

# bokeh serve --show dashboard.py
