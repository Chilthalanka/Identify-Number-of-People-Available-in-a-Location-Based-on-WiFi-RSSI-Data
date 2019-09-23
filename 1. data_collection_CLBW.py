# Get data as this format : {"id" : "1","UoM_Wireless1" : "-58","eduroam1" : "-89","UoM_Wireless1" : "-89","eduroam1" : "-57"}
import paho.mqtt.client as mqtt
import json
import csv
import pandas as pd
import os
import ast

msgCounter = 0

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("entc/wifi/IoT/G6/ACCU")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    #print(msg.topic+"\n"+str(msg.payload)+"\n")
    data_store_csv(msg.payload)
    

# Function to store received data in a csv file
def data_store_csv(data1):

    global msgCounter

    # Define result array
    #result = {"id": 6, "WiFire1": -100, "UoM_Wireless1": -100, "UoM_Wireless6": -100, "UoM_Wireless11":-100, "eduroam1": -100, "eduroam6": -100, "eduroam11": -100}
    result = {"Have a Nice Day10":-100, "WiFire1": -100, "CLBW-Mobitel-HUAWEI-CC008":-100, "UoM_Wireless11":-100, "UoM_Wireless1": -100, "UoM_Wireless6": -100, "eduroam1": -100, "eduroam6": -100, "eduroam11": -100}
    # Define csv file path
    filename = "wifi_data_Final_Five_P.csv"

    try:
        # Get data as this format : {"id" : "1","UoM_Wireless1" : "-58","eduroam1" : "-89","UoM_Wireless1" : "-89","eduroam1" : "-57"}

        data = ast.literal_eval(data1.decode("utf-8"))
        print(data)        
        
        
        for key,value in data.items():      ###########   need to modify  #########
            #if key in result and result[key] == -100:
            if (key in result):         ###############     modified    ##########
                result[key] = float(value)
            if (key not in result):         
                result[key] = float(value)
                    
                
        
        result["id"] = int(data["id"])


        # Create one row of csv file
        df = pd.Series(result).to_frame().T

        #print(df)

        # Write to csv file
        df.to_csv(filename,index=False,mode='a',header=(not os.path.exists(filename)))

        #increase message counter
        msgCounter += 1
        print ("Message Counter: " + str(msgCounter) + "\n")
        
    except Exception as e:
        print(e)

#data2="{\"id\" : \"1\",\"UoM_Wireless1\" : -58,\"eduroam1\" : -89,\"UoM_Wireless1\" : -89,\"eduroam1\" : -57}"
#data_store_csv(data2)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("iot.eclipse.org", 1883, 60)     #keepalive= 60: Maximum period in seconds between communications with the broker
client.loop_forever()
