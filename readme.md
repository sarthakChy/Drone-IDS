# environment setup
## MAVLINK
-> sudo apt install -y build-essential ccache g++ gawk git make wget valgrind screen astyle python-is-python3 libtool libxml2-dev libxslt1-dev python3-dev python3-pip python3-setuptools python3-numpy python3-pyparsing python3-psutil xterm xfonts-base python3-matplotlib python3-serial python3-scipy python3-opencv libcsfml-dev libsfml-dev python3-yaml libgtk-3-dev libfreetype6-dev libportmidi-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libtool-bin ppp g++-arm-linux-gnueabihf lcov gcovr

-> python -m venv .venv
-> source .venv/bin/activate
-> pip install -r requirements.txt

## Arduopilot
# Run Normally
-> Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --out=udp:127.0.0.1:14550 

# Run using created plan and storing the tlog with different name.
-> Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --out=udp:127.0.0.1:14550 --mavproxy-args="--mission=sample.plan --logfile=sample.tlog"

# multiple streams
-> Tools/autotest/sim_vehicle.py -v ArduCopter -f quad --mavproxy-args="--mission=corridor_scan.plan --logfile=live_streaming.tlog --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551"

-> make sure the plan is saved inside ardupilot only for easy accessing. 
    
# parsing to csv
-> python3 parse_tlog.py sample.tlog 0 (for normal plans)
-> python3 parse_tlog.py sample.tlog 1 (for attack plans)

# visualize the normal plans
-> run the visualize.py file. change the name of the csv file accordingly.

## QgroundControl
-> sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl libfuse2 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor-dev -y



