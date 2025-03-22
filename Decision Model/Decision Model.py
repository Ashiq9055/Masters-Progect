import socket
import threading
import tkinter as tk
from tkinter import scrolledtext

# Function to handle incoming connections and display data in the text box
def handle_client(conn, address, text_box, last_action):
    with conn:
        while True:
            data = conn.recv(1024).decode()
            if not data:
                break
            if data != last_action[0]:
                text_box.insert(tk.END, f"Predicted action: {data}\n")
                text_box.see(tk.END)  # Scroll to the end
                last_action[0] = data
            conn.send(data.encode())

# Function to start the server and accept connections
def server_program(text_box):
    host = '0.0.0.0'  # or '192.168.1.2' if you want to specify
    port = 5000

    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)
    text_box.insert(tk.END, f"Server listening on port: {port}\n")
    text_box.see(tk.END)  # Scroll to the end

    last_action = ['']  # List to store the last action

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, address, text_box, last_action))
        client_thread.start()

# Function to create and start the GUI
def start_gui():
    root = tk.Tk()
    root.title("Server Log")
    root.geometry("600x400")

    text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text_box.pack(expand=True, fill='both')

    # Start the server in a separate thread
    server_thread = threading.Thread(target=server_program, args=(text_box,))
    server_thread.daemon = True
    server_thread.start()

    root.mainloop()

if __name__ == '__main__':
    start_gui()


















# import socket
# import threading
# import tkinter as tk
# from tkinter import scrolledtext

# # Function to handle incoming connections and display data in the text box
# def handle_client(conn, address, text_box, last_action):
#     with conn:
#         while True:
#             data = conn.recv(1024).decode()
#             if not data:
#                 break
#             if data != last_action[0]:
#                 text_box.insert(tk.END, f"Predicted action: {data}\n")
#                 text_box.see(tk.END)  # Scroll to the end
#                 last_action[0] = data
#             # Optionally send some response to the client
#             conn.send(data.encode())

# # Function to start the server and accept connections
# def server_program(text_box):
#     host = '0.0.0.0'
#     port = 5000

#     server_socket = socket.socket()
#     server_socket.bind((host, port))
#     server_socket.listen(2)
#     text_box.insert(tk.END, f"Server listening on port: {port}\n")
#     text_box.see(tk.END)  # Scroll to the end

#     last_action = ['']  # List to store the last action

#     while True:
#         conn, address = server_socket.accept()
#         client_thread = threading.Thread(target=handle_client, args=(conn, address, text_box, last_action))
#         client_thread.start()

# # Function to create and start the GUI
# def start_gui():
#     root = tk.Tk()
#     root.title("Server Log")
#     root.geometry("600x400")

#     text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD)
#     text_box.pack(expand=True, fill='both')

#     # Start the server in a separate thread
#     server_thread = threading.Thread(target=server_program, args=(text_box,))
#     server_thread.daemon = True
#     server_thread.start()

#     root.mainloop()

# if __name__ == '__main__':
#     start_gui()






import asyncio
import socket
import threading
import tkinter as tk
from tkinter import scrolledtext
import websockets
import json

# Global variable for the text box
text_box = None

# Shared data structure to hold the latest predictions
latest_prediction = {
    'tcp': '',
    'websocket': ''
}

# Update the GUI with predictions
def update_gui(prediction_type, prediction):
    global latest_prediction
    latest_prediction[prediction_type] = prediction
    text_box.insert(tk.END, f"{prediction_type.capitalize()} Prediction: {prediction}\n")
    text_box.see(tk.END)  # Scroll to the end

# Function to handle incoming TCP connections
def handle_client(conn, address):
    with conn:
        while True:
            data = conn.recv(1024).decode()
            if not data:
                break
            update_gui('tcp', data)
            conn.send(data.encode())

# Function to start the TCP server
def start_tcp_server():
    host = '0.0.0.0'
    port = 5000
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)
    print(f"TCP Server listening on port: {port}")

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client, args=(conn, address))
        client_thread.start()

# WebSocket client handler
async def ws_handler(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                json_data = json.loads(message)
                predicted_action = json_data.get('predicted_action', '')
                update_gui('websocket', predicted_action)
            except Exception as e:
                print(f"Error handling WebSocket message: {e}")

# Function to start the WebSocket client
def start_websocket_client():
    ws_uri = "ws://localhost:8765"  # WebSocket server URI
    asyncio.run(ws_handler(ws_uri))

# Function to create and start the GUI
def start_gui():
    global text_box
    root = tk.Tk()
    root.title("Predictions")
    root.geometry("600x400")

    text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text_box.pack(expand=True, fill='both')

    # Start TCP server in a separate thread
    tcp_thread = threading.Thread(target=start_tcp_server)
    tcp_thread.daemon = True
    tcp_thread.start()

    # Start WebSocket client in a separate thread
    ws_thread = threading.Thread(target=start_websocket_client)
    ws_thread.daemon = True
    ws_thread.start()

    root.mainloop()

if __name__ == '__main__':
    start_gui()













































import asyncio
import json
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import websockets
import socket
import threading
import tkinter as tk
from tkinter import scrolledtext
from io import StringIO
import queue
import cv2
import mediapipe as mp

# Define the LSTM model
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the last timestep output
        out = self.fc(lstm_out)
        return self.softmax(out)

def load_model(model_class, input_size, hidden_size, output_size, file_path):
    model = model_class(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# Constants
WEBSOCKET_PORT = 8765
TCP_PORT = 5000
sequence_length = 10
input_size = 38
hidden_size = 128
output_size = 5
model_path = 'D:/iMobie/CoolSo/Final assembly data/Lego Assembly data/Data/Training Data/Data/SimpleLSTMModel_seq10_epochs150.PTH'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(SimpleLSTMModel, input_size, hidden_size, output_size, model_path).to(device)

# Mapping of predicted hand action numbers to action names
hand_action_mapping = {
    0: 'Rest',
    1: 'Installing Base',
    2: 'Installing Shaft',
    3: 'Install Tayers',
    4: 'None'
}

def preprocess_data(json_data):
    csv_data = json_data.get('data', '')
    if not csv_data.strip():
        raise ValueError("No CSV data found in JSON payload")
    
    csv_buffer = StringIO(csv_data)
    df = pd.read_csv(csv_buffer)
    
    all_columns = [
        'DuctionRate', 'Fingerlabel_Grab', 'Fingerlabel_Idle', 'Fingerlabel_IndexClutch', 
        'Fingerlabel_IndexTap', 'Fingerlabel_Misc_', 'Fingerlabel_Rotate', 'Fingerlabel_Snap', 
        'Fingerlabel_Spread', 'Fingerlabel_ThumbIn', 'Fingerlabel_ThumbOut', 'Fingerlabel_WaveIn', 
        'Fingerlabel_WaveOut', 'FlexionRate', 'ForearmAngle_Arm3', 'ForearmAngle_Arm4', 
        'ForearmAngle_Arm5', 'ForearmAngle_Arm6', 'ForearmAngle_Arm7', 'ForearmAngle_Arm8', 
        'ForearmAngle_Arm9', 'ForearmAngle_ArmA', 'ForearmAngle_ArmB', 'ForearmAngle_ArmC', 
        'HorizontalRate', 'RotationRate', 'StrengthAmplitude', 'VerticalRate', 'WristAngle_Wrist0', 
        'WristAngle_Wrist1', 'WristAngle_Wrist2', 'WristAngle_Wrist3', 'WristAngle_Wrist4', 
        'WristAngle_Wrist5', 'WristAngle_Wrist6', 'WristAngle_Wrist7', 'WristAngle_Wrist8', 
        'WristAngle_Wrist9'
    ]

    for col in all_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[all_columns].astype(float).fillna(method='ffill').fillna(method='bfill')

    data = df.values
    if len(data) < sequence_length:
        data = np.pad(data, ((0, sequence_length - len(data)), (0, 0)), mode='constant')
    elif len(data) > sequence_length:
        data = data[:sequence_length]
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Global variables
text_box = None
latest_actions = {'body': None, 'hand': None, 'predicted': None}
message_queue = queue.Queue()
predicted_action_keypoints = None
frame = None  # Global variable for the current frame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam feed
cap = cv2.VideoCapture(1)  # Adjust the index if needed

# Define the rectangle coordinates (top-left and bottom-right)
rect_start = (250, 150)
rect_end = (620, 420)

# def detect_hand_keypoints(frame):
#     if frame is None:
#         return False
#     cropped_frame = frame[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
#     cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(cropped_frame_rgb)

#     if results.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#             hand_label = results.multi_handedness[idx].classification[0].label
#             if hand_label == "Right":
#                 mp_drawing.draw_landmarks(cropped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 return True  # Key points detected
#     return False
def detect_hand_keypoints(frame):
    if frame is None:
        return False
    cropped_frame = frame[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
    cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(cropped_frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            if hand_label == "Left":  # Check specifically for the right hand
                mp_drawing.draw_landmarks(cropped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                return True  # Key points detected for the right hand
    return False

# WebSocket handler for hand actions
async def ws_handler(websocket, path):
    global predicted_action_keypoints
    async for message in websocket:
        try:
            json_data = json.loads(message)
            input_data = preprocess_data(json_data)
            
            with torch.no_grad():
                model.to(device)
                input_data = input_data.to(device)
                outputs = model(input_data)
                _, predicted = torch.max(outputs, 1)
                predicted_class = int(predicted.item())
                predicted_action = hand_action_mapping.get(predicted_class, 'Unknown')

            # Check if key points are detected
            if detect_hand_keypoints(frame):
                predicted_action_keypoints = predicted_action
                latest_actions['predicted'] = predicted_action
                message_queue.put(f"Predicted Action: {predicted_action_keypoints}\n")
            else:
                predicted_action_keypoints = None
                latest_actions['predicted'] = None

            await websocket.send(json.dumps({'predicted_action': predicted_action}))

            if latest_actions['body'] == "Human working":
                if predicted_action != latest_actions['hand']:
                    latest_actions['hand'] = predicted_action
                    formatted_message = f"Predicted body action: {latest_actions['body']}   Predicted hand action: {latest_actions['hand']}   Predicted action: {latest_actions['predicted']}\n"
                    message_queue.put(formatted_message)

        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

# Function to handle incoming TCP connections for body actions
def handle_tcp_client(conn, address):
    try:
        with conn:
            while True:
                try:
                    data = conn.recv(1024).decode()
                    if not data:
                        break
                    if data != latest_actions['body']:
                        latest_actions['body'] = data
                        if data != "Human working":
                            latest_actions['hand'] = None
                            latest_actions['predicted'] = None
                        formatted_message = f"Predicted body action: {latest_actions['body']}   Predicted hand action: {latest_actions['hand']}   Predicted action: {latest_actions['predicted']}\n"
                        message_queue.put(formatted_message)
                    conn.send(data.encode())
                except ConnectionAbortedError:
                    print(f"Connection aborted by {address}")
                    break
                except ConnectionResetError:
                    print(f"Connection reset by {address}")
                    break
    except Exception as e:
        print(f"Error handling TCP client {address}: {e}")

# Function to start the TCP server for body actions
def start_tcp_server():
    host = '0.0.0.0'
    server_socket = socket.socket()
    server_socket.bind((host, TCP_PORT))
    server_socket.listen(2)
    message_queue.put(f"TCP Server listening on port: {TCP_PORT}\n")

    while True:
        conn, address = server_socket.accept()
        client_thread = threading.Thread(target=handle_tcp_client, args=(conn, address))
        client_thread.start()

# Function to start the WebSocket server for hand actions
async def start_websocket_server():
    message_queue.put(f'Starting WebSocket Server at port {WEBSOCKET_PORT}\n')
    async with websockets.serve(ws_handler, "localhost", WEBSOCKET_PORT):
        await asyncio.Future()

# Function to update the GUI
def update_gui():
    global text_box
    while True:
        message = message_queue.get()
        if text_box:
            # Check if the message contains 'Predicted Action'
            if 'Predicted Action' in message:
                # Insert text with red color
                text_box.insert(tk.END, message, 'red')
            else:
                # Insert text with default color
                text_box.insert(tk.END, message)
            text_box.yview(tk.END)
        message_queue.task_done()

# Function to capture and process frames from the webcam
def update_frame():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.resize(frame, (640, 480))
        cv2.rectangle(frame, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    global text_box

    root = tk.Tk()
    root.title("Action Prediction")
    root.geometry("800x400")  # Increased width to accommodate the new column

    text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=100)  # Increased width
    text_box.pack(padx=10, pady=10)

    # Define a tag for red color
    text_box.tag_configure('red', foreground='red')

    tcp_thread = threading.Thread(target=start_tcp_server)
    tcp_thread.daemon = True
    tcp_thread.start()

    websocket_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(websocket_loop)
    websocket_thread = threading.Thread(target=lambda: websocket_loop.run_until_complete(start_websocket_server()))
    websocket_thread.daemon = True
    websocket_thread.start()

    frame_thread = threading.Thread(target=update_frame)
    frame_thread.daemon = True
    frame_thread.start()

    gui_thread = threading.Thread(target=update_gui)
    gui_thread.daemon = True
    gui_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()


