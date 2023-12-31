I have written some programs that send an animations in NDI format. Several versions of the studio clock and ticker. The clock reads data from a calendar shared on the web and displays rundown based on this data. Ticker reads any RSS feed and displays it as a scrolling ticker bar. The tools are written in Python. The installer is beyond my will to do free work. However, I wrote instructions on how to compile the ndi-python library and how to install additional Python libraries necessary to run my programs. Enjoy.

_Macos Install_

NDI Tools for MacOS

You will need at last NDI Monitor to check NDI streams generated by tools.

NDI SDK for MacOS


xCode and xCode commandline tools

Install xcode (on systems older than MacOS 11 you have to download xcode from archival page here but I am afraid you will have to authenticate with your Apple ID). You need xCode and xCode Command Line tools installed.

Install homebrew

curl -fsSL -o install.sh https://raw.githubusercontent.com/Homebrew/install/master/install.sh

Install cmake

brew install cmake

Install virtualenv

pip3 install virtualenv 

Make virtualenv for python3 environment

cd ~

virtualenv -p python3 vw


Activate virtual environment

source vw/bin/activate


Install numpy

pip3 install numpy


Install opencv

brew install opencv (it will take some time)


Install opencv Python wrapper

pip3 install opencv-python


Install Pillow library

pip3 install pillow


Install ICS reading library

pip3 install ics


Install Requests library

pip3 install requests


Install matplotlib

pip3 install matplotlib


Install rss-parser library

pip3 install rss-parser


Clone ndi commandline tools

git clone https://github.com/RadioErewan/clk.git --recursive


Clone NDI-python wrapper

git clone https://github.com/buresu/ndi-python.git --recursive


Build NDI-Python wrapper

cd ndi-python/

cmake ./

cmake --build ./ --config Release


Copy to /clk compiled lib

it will be something like NDIlib.cpython-3x-darwin.so


Go to clk folder

cd clk


To run clock type

python3 clock.py


To run ticker type

python3 ticker.py

To change calendar used by application CLOCK share with everyone your google calendar, find it's address and replace it on this line in clock.py file.

icsurl = "https://calendar.google.com/calendar/ical/gfh9vebps3t1fqu4mqvvpfd8ro%40group.calendar.google.com/public/basic.ics"