from pymavlink import mavutil
import csv
import os
import argparse # Import argparse to handle command-line arguments

# --- Setup command-line argument parsing ---
parser = argparse.ArgumentParser(description="Parse a MAVLink .tlog file to a labeled CSV.")
parser.add_argument("tlog_file", help="Path to the input .tlog file")
parser.add_argument("label", type=int, help="Label for this flight (0 for normal, 1 for attack)")
args = parser.parse_args()

# --- Configuration from arguments ---
tlog_file = args.tlog_file
flight_label = args.label

# Create the flight_id from the filename (e.g., "normal_corridor_base.tlog" -> "normal_corridor_base")
flight_id = os.path.splitext(os.path.basename(tlog_file))[0]

# The name of the output CSV file
csv_file = f'{flight_id}_data.csv'

if not os.path.exists(tlog_file):
    print(f"Error: Log file not found at '{tlog_file}'")
    exit()

# --- Main Parsing Logic ---
print(f"Opening tlog file: {tlog_file}")
mlog = mavutil.mavlink_connection(tlog_file)
print(f"Parsing flight with ID: '{flight_id}' and Label: {flight_label}")

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # --- ADD 'flight_id' TO THE HEADER ---
    header = [
        'flight_id', 'label', 'timestamp',
        'att_roll', 'att_pitch', 'att_yaw',
        'pos_lat', 'pos_lon', 'pos_alt_rel', 'pos_vx', 'pos_vy', 'pos_vz',
        'nav_roll', 'nav_pitch', 'nav_alt_error',
        'servo1_raw', 'servo2_raw', 'servo3_raw', 'servo4_raw',
        'sys_voltage_battery', 'sys_load',
        'vib_x', 'vib_y', 'vib_z',
        'gps_sats_visible'
    ]
    writer.writerow(header)

    latest_data = {}

    while True:
        msg = mlog.recv_match()
        if msg is None:
            break

        msg_type = msg.get_type()
        
        # Store latest data from relevant messages
        # ... (same logic as before to update latest_data dictionary) ...
        if msg_type == 'ATTITUDE': latest_data['att_roll'] = msg.roll; latest_data['att_pitch'] = msg.pitch; latest_data['att_yaw'] = msg.yaw
        elif msg_type == 'NAV_CONTROLLER_OUTPUT': latest_data['nav_roll'] = msg.nav_roll; latest_data['nav_pitch'] = msg.nav_pitch; latest_data['nav_alt_error'] = msg.alt_error
        elif msg_type == 'SERVO_OUTPUT_RAW': latest_data['servo1_raw'] = msg.servo1_raw; latest_data['servo2_raw'] = msg.servo2_raw; latest_data['servo3_raw'] = msg.servo3_raw; latest_data['servo4_raw'] = msg.servo4_raw
        elif msg_type == 'SYS_STATUS': latest_data['sys_voltage_battery'] = msg.voltage_battery; latest_data['sys_load'] = msg.load
        elif msg_type == 'VIBRATION': latest_data['vib_x'] = msg.vibration_x; latest_data['vib_y'] = msg.vibration_y; latest_data['vib_z'] = msg.vibration_z
        elif msg_type == 'GPS_RAW_INT': latest_data['gps_sats_visible'] = msg.satellites_visible
        
        # Trigger row write on GLOBAL_POSITION_INT
        elif msg_type == 'GLOBAL_POSITION_INT':
            # --- ADD flight_id and flight_label TO THE ROW ---
            row = [
                flight_id,
                flight_label,
                msg.time_boot_ms,
                latest_data.get('att_roll'),
                latest_data.get('att_pitch'),
                latest_data.get('att_yaw'),
                msg.lat / 1e7,
                msg.lon / 1e7,
                msg.relative_alt / 1000.0,
                msg.vx / 100.0,
                msg.vy / 100.0,
                msg.vz / 100.0,
                latest_data.get('nav_roll'),
                latest_data.get('nav_pitch'),
                latest_data.get('nav_alt_error'),
                latest_data.get('servo1_raw'),
                latest_data.get('servo2_raw'),
                latest_data.get('servo3_raw'),
                latest_data.get('servo4_raw'),
                latest_data.get('sys_voltage_battery'),
                latest_data.get('sys_load'),
                latest_data.get('vib_x'),
                latest_data.get('vib_y'),
                latest_data.get('vib_z'),
                latest_data.get('gps_sats_visible')
            ]
            writer.writerow(row)

print(f"\nParsing complete! Rich, labeled data saved to '{csv_file}'")
