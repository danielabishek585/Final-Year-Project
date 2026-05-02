import time
import RPi.GPIO as GPIO
from mpu6050 import mpu6050
import Adafruit_DHT
from RPLCD.i2c import CharLCD
import pandas as pd
import joblib
import numpy as np
import requests

GPIO.setmode(GPIO.BCM)

VIB_PIN = 18
FIRE_PIN = 17
FLOAT_PIN = 26
TRIG = 23
ECHO = 24

GPIO.setup(VIB_PIN, GPIO.IN)
GPIO.setup(FIRE_PIN, GPIO.IN)
GPIO.setup(FLOAT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

DHT_PIN = 4
DHT_SENSOR = Adafruit_DHT.DHT11
mpu = mpu6050(0x68)
lcd = CharLCD('PCF8574', 0x27)

model = joblib.load("disaster_multilabel_model.pkl")

BLYNK_AUTH = "GcTe_llA3usC1ajKfic7UrIqDBM7ZRt3"

previous_state = {
    "fire": False,
    "flood": False,
    "vibration": False,
    "landslide": False,
    "high_temp": False
}

alert_messages = {
    "fire": "FIRE DETECTED!",
    "flood": "FLOOD ALERT!",
    "vibration": "EARTHQUAKE/VIBRATION!",
    "landslide": "LANDSLIDE ALERT!",
    "high_temp": "HIGH TEMPERATURE!"
}

blynk_pins = {
    "fire": 0,
    "flood": 1,
    "vibration": 2,
    "landslide": 3,
    "high_temp": 4
}

def send_blynk_alert(alert_type, message):
    pin = blynk_pins[alert_type]
    url = f"https://blynk.cloud/external/api/logEvent?token={BLYNK_AUTH}&code=alert&description={message}"
    try:
        requests.get(url)
    except:
        pass

def read_dht():
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    if humidity is None or temperature is None:
        return (0, 0)
    return (temperature, humidity)

def distance_ultrasonic():
    GPIO.output(TRIG, False)
    time.sleep(0.05)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    pulse_start = time.time()
    pulse_end = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    duration = pulse_end - pulse_start
    distance = int(duration * 17150)
    return distance

def get_mpu_tilt():
    accel = mpu.get_accel_data()
    ax, ay, az = accel['x'], accel['y'], accel['z']
    tilt_angle = ((ax**2 + ay**2)**0.5)
    return ax, ay, az, tilt_angle

def show_alert(message):
    lcd.clear()
    lcd.write_string(" !!! ALERT !!! ")
    lcd.cursor_pos = (1, 0)
    lcd.write_string(message[:16])
    time.sleep(2)

page = 0

try:
    while True:
        vib = GPIO.input(VIB_PIN)
        fire = not GPIO.input(FIRE_PIN)
        water_float = GPIO.input(FLOAT_PIN)
        dist = distance_ultrasonic()
        temp, hum = read_dht()
        ax, ay, az, tilt_angle = get_mpu_tilt()

        data = {
            "acc_x": ax,
            "acc_y": ay,
            "acc_z": az,
            "tilt_angle": tilt_angle,
            "temperature": temp,
            "humidity": hum,
            "fire": int(fire),
            "vibration": int(vib),
            "water_float": int(water_float),
            "ultrasonic_distance": dist
        }

        df = pd.DataFrame([data])
        pred = model.predict(df)[0]

        landslide, high_temp, vibration_alert, fire_alert, flood_alert = pred

        current_state = {
            "fire": fire_alert,
            "flood": flood_alert,
            "vibration": vibration_alert,
            "landslide": landslide,
            "high_temp": high_temp
        }

        for alert_type in current_state:
            if current_state[alert_type] and not previous_state[alert_type]:
                send_blynk_alert(alert_type, alert_messages[alert_type])
                show_alert(alert_messages[alert_type])
                previous_state[alert_type] = True
            elif not current_state[alert_type] and previous_state[alert_type]:
                previous_state[alert_type] = False

        lcd.clear()
        if page == 0:
            lcd.write_string(f"T:{temp}C H:{hum}%")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(f"D:{dist}cm W:{'HIGH' if water_float else 'OK'}")
        elif page == 1:
            lcd.write_string(f"Vib:{vibration_alert} Fire:{fire_alert}")
            lcd.cursor_pos = (1, 0)
            lcd.write_string(f"Tilt:{'DANGER' if landslide else 'OK'}")

        page = (page + 1) % 2

        print("-------------------------------")
        print(f"Temp: {temp}C High={high_temp}")
        print(f"Humidity: {hum}%")
        print(f"Distance: {dist}cm")
        print(f"Vibration: {vibration_alert}")
        print(f"Fire: {fire_alert}")
        print(f"Flood: {flood_alert}")
        print(f"Landslide: {landslide}")
        print("-------------------------------\n")

        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
