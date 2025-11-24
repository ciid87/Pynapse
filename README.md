#Setup
wsl --install -d archlinux --name fiji_#

# Update system and install 
pacman -Syu --noconfirm btop wget unzip xorg-server xorg-xinit xorg-apps mesa fontconfig freetype2 ttf-dejavu ttf-liberation noto-fonts gnu-free-fonts git

# Create fiji user
useradd -m -G wheel -s /bin/bash fiji

# Set password for fiji user
passwd fiji
pacman -Sy --noconfirm sudo wget unzip
echo "%wheel ALL=(ALL) ALL" >> /etc/sudoers
echo '[user]' > /etc/wsl.conf
echo 'default=fiji' >> /etc/wsl.conf
exit

# In PowerShell
wsl --terminate fiji_#
wsl -d fiji_#



# In fiji_1 as fiji
mkdir -p ~/fiji
cd ~/fiji
wget https://downloads.imagej.net/fiji/latest/fiji-latest-linux64-jdk.zip
unzip fiji-latest-linux64-jdk.zip
cd Fiji/
./fiji --update refresh-update-sites ImageJ
./fiji --update update net.imagej:imagej-updater
cp SynapseJ/*.ijm plugins/

#get the files!
cp -r /mnt/c/Users/CIID0/Downloads/BRE ~/
cd ~/BRE/

#save to /mnt/c/Users/CIID0/Downloads/BRE/R38_$$_out (As, Cd, Cr)

./fiji -Xmx30G --
