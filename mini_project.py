import tkinter as tk
import pyttsx3
from tkinter import filedialog

# Initialize engine
engine = pyttsx3.init()

# Functions
def speak_text():
    text = text_area.get("1.0", tk.END)
    engine.say(text)
    engine.runAndWait()

def stop_speech():
    engine.stopAndRunAndWait()

def save_audio():
    text = text_area.get("1.0", tk.END)
    file_path = filedialog.asksaveasfilename(defaultextension=".mp3")
    if file_path:
        engine.save_to_file(text, file_path)
        engine.runAndWait()

def set_voice():
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_var.get()].id)

def set_speed(val):
    engine.setProperty('rate', int(val))

def set_volume(val):
    engine.setProperty('volume', float(val))

# GUI Window
root = tk.Tk()
root.title("Text-to-Speech App")
root.geometry("600x500")
root.configure(bg="#1e1e2f") 

# Title Label
title = tk.Label(root, text="Text to Speech Converter",
                 font=("Helvetica", 18, "bold"),
                 bg="#1e1e2f", fg="white")
title.pack(pady=10)

# Text Area
text_area = tk.Text(root, height=8, width=50,
                    font=("Arial", 14),
                    bg="#f0f0f0")
text_area.pack(pady=10)

# Buttons Frame
frame = tk.Frame(root, bg="#05050b")
frame.pack()

# Buttons
tk.Button(frame, text="▶ Play", command=speak_text,
          font=("Arial", 12, "bold"),
          bg="#4CAF50", fg="white", width=10).grid(row=0, column=0, padx=5)

tk.Button(frame, text="⏹ Stop", command=stop_speech,
          font=("Arial", 12, "bold"),
          bg="#f44336", fg="white", width=10).grid(row=0, column=1, padx=5)

tk.Button(frame, text="💾 Save", command=save_audio,
          font=("Arial", 12, "bold"),
          bg="#2196F3", fg="white", width=10).grid(row=0, column=2, padx=5)

# Voice Selection
voice_var = tk.IntVar()
tk.Label(root, text="Select Voice",
         font=("Arial", 12, "bold"),
         bg="#1e1e2f", fg="white").pack()

tk.Radiobutton(root, text="Male", variable=voice_var, value=0,
               command=set_voice,
               bg="#1e1e2f", fg="white",
               selectcolor="#333").pack()

tk.Radiobutton(root, text="Female", variable=voice_var, value=1,
               command=set_voice,
               bg="#1e1e2f", fg="white",
               selectcolor="#333").pack()

# Speed Control
tk.Label(root, text="Speed",
         font=("Arial", 12, "bold"),
         bg="#1e1e2f", fg="white").pack()

tk.Scale(root, from_=100, to=300, orient="horizontal",
         command=set_speed, bg="#1e1e2f", fg="white",
         highlightthickness=0).pack()

# Volume Control
tk.Label(root, text="Volume",
         font=("Arial", 12, "bold"),
         bg="#1e1e2f", fg="white").pack()

tk.Scale(root, from_=0, to=1, resolution=0.1,
         orient="horizontal", command=set_volume,
         bg="#1e1e2f", fg="white",
         highlightthickness=0).pack()

root.mainloop()