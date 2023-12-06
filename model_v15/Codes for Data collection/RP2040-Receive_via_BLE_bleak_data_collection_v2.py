#Date: 30 Nov 2023
#Author: Jatin Kadge

#Receive the data from Tranmission code and over BLE and save it into a csv file

#Instructions
#PLease add a path to a file variable to save the .csv file. Name the file according to the gesture for which data must be collecteed. One type of gesture at a time

#Transmission code name: Basic_data_collection_BLE_RP2040_v2.ino
#Splitting code: split_csv_files_2.py

import asyncio
import numpy as np
from bleak import BleakScanner, BleakClient
import csv
import pandas as pd

file = 'C:/Users/user/Desktop/Monty/IITM/Work Activity/Codes/Python codes/Rp2040_Data5-3.3 secs/.csv'                   #Add path to a csv file //Eg. Left_Swing.csv for store the data for Left Swing gesture only 
f = open(file, 'w')
writer = csv.writer(f)

async def discover_devices():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Device found: {device}")

async def imu_data_notification_handler(sender: int, data: bytearray):
    # Handle the received IMU data
    #print(f"Received IMU Data: {data.decode()}")
    string_data = data.decode()                                      #converting byte data into string and storing it into another variable
    array_data = np.fromstring(string_data, dtype=np.float, sep=',')      #converting string into float array by seperating string at ','
    print(array_data)
    writer.writerow(array_data)


#async def imu_data_notification_handler(sender: int, data: bytearray):
#    # Handle the received IMU data as raw binary
#    print(f"Received IMU Data: {data}")


async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        print(f"Connected to {address}")

        # Enable notifications for the IMU data characteristic
        imu_data_characteristic = "19B10001-E8F2-537E-4F6C-D104768A1214"
        await client.start_notify(imu_data_characteristic, imu_data_notification_handler)

        try:
            while True:
                # Continue processing in the main loop
                await asyncio.sleep(1)
                #pass
        except KeyboardInterrupt:
            pass
        finally:
            # Stop notifications when exiting
            await client.stop_notify(imu_data_characteristic)

if __name__ == "__main__":
    # Replace with the Bluetooth address of your RP2040
    nrf_address = "84:cc:a8:2f:ec:2a"

    # Discover nearby devices (optional)
    # asyncio.run(discover_devices())

    # Run the main loop to receive IMU data
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(nrf_address, loop))

