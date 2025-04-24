import requests
import time
import pandas as pd
from flask import Flask, jsonify

WINDOWS_IP = "192.168.0.118"  # Windows Flask Server IP
URL = f"http://{WINDOWS_IP}:5051/kkkk"
data = {}
df =pd.DataFrame(data = data) 
while True:
    try:
        res = requests.get(URL)
        if res.status_code == 200:
            print('-'*50)
            print("[INFO] 상태 데이터:", res.json())
            print('-'*50)
            # data = jsonify(res)
            # print(data)
        else:
            print("[ERROR] 상태 조회 실패", res.status_code)
    except Exception as e:
        print("[EXCEPTION]", e)

    time.sleep(1)
    
   
# URL = f"http://{WINDOWS_IP}:5051/detect"

# while True:
#     try:
#         res = requests.get(URL)
#         if res.status_code == 200:
#             print("[INFO] 상태 데이터:", res.json())
#         else:
#             print("[ERROR] 상태 조회 실패", res.status_code)
#     except Exception as e:
#         print("[EXCEPTION]", e)

#     time.sleep(1)