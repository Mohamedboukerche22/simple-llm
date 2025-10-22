import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf


MODEL_PATH = "digit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


window = tk.Tk()
window.title("Handwritten Calculator")
window.geometry("320x420")


canvas = tk.Canvas(window, width=200, height=200, bg="white")
canvas.pack(pady=10)

drawn_image = Image.new("L", (200, 200), color=255)
draw = ImageDraw.Draw(drawn_image)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 200, 200), fill=255)


expression = ""

def predict_symbol():
    global expression
    img = drawn_image.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    pred = model.predict(arr)
    result = np.argmax(pred)

    symbol = str(result)
    expression += symbol
    result_label.config(text=f"Expression: {expression}")
    clear_canvas()

def add_operator(op):
    global expression
    expression += op
    result_label.config(text=f"Expression: {expression}")

def calculate():
    global expression
    try:
        result = eval(expression)
        result_label.config(text=f"{expression} = {result}")
        expression = str(result)
    except Exception:
        result_label.config(text="Error!")
        expression = ""


btn_frame = tk.Frame(window)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Predict", command=predict_symbol, width=10).pack(side="left", padx=5)
tk.Button(btn_frame, text="Clear", command=clear_canvas, width=10).pack(side="left", padx=5)


op_frame = tk.Frame(window)
op_frame.pack(pady=5)

for op in ["+", "-", "*", "/"]:
    tk.Button(op_frame, text=op, width=5, command=lambda o=op: add_operator(o)).pack(side="left", padx=3)

tk.Button(window, text="=", width=20, command=calculate).pack(pady=5)

result_label = tk.Label(window, text="Draw a digit, then press Predict", font=("Arial", 14))
result_label.pack(pady=10)

window.mainloop()

