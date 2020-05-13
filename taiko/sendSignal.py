import numpy as np
import serial
import time

waitTime = 0.1
signalLength = 42*3

song1 = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261}
song2 = {
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428,
  392, 392, 500, 500, 330, 330, 258,
  392, 392, 500, 500, 330, 330, 258,
  428, 428, 392, 392, 440, 440, 392,
  500, 500, 330, 330, 258, 258, 428}
song3 = {
  350, 350, 277, 277, 440, 440, 277,
  349, 349, 488, 488, 294, 294, 350,
  277, 277, 349, 349, 488, 488, 294,
  277, 277, 349, 349, 488, 488, 294,
  350, 350, 277, 277, 440, 440, 392,
  349, 349, 488, 488, 294, 294, 261}

serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)

formatter = lambda x: "%.3f" % x

print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))

for data in song1:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
for data in song2:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
for data in song3:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)

s.close()
print("Signal sended")