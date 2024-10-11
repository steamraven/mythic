sudo apt-get update
sudo apt-get install -y --no-install-recommends gcc g++ swig git  
sudo apt-get install -y --no-install-recommends fontconfig fonts-unifont xauth xrdp xorgxrdp xfce4 dbus-x11
sudo apt-get clean 
sudo rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip setuptools
# Prereqs for specific dependencies
# Needed to install CPU only version
pip install -r requirements.txt