sudo apt-get update
# Prereqs to build and install
sudo apt-get install -y --no-install-recommends gcc g++ swig git  
# Some nice unicode fonts
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends fontconfig fonts-unifont 
# Allow graphics: xauth to allow forwarding the X11 window on linux and  xorg-rdp to allow Windows to remote desktop
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends xauth xrdp xorgxrdp xfce4 dbus-x11
sudo apt-get clean 
sudo rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip setuptools
# Prereqs for specific dependencies
# Needed to install CPU only version
pip install -r requirements.txt