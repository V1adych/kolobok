from pynput import mouse, keyboard
import math
import time

radius = 30

# Получение текущей позиции курсора
mouse_controller = mouse.Controller()
keyboard_controller = keyboard.Controller()

print("Наведите курсор в центр десятиугольника и нажмите 'g'. Для выхода нажмите 'esc'.")

# Флаг для выхода
running = True

def click_decagon(center_x, center_y, radius=30):
    points = []
    for i in range(10):
        angle = 2 * math.pi * i / 10
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))
    for x, y in points:
        mouse_controller.position = (x, y)
        time.sleep(0.05)
        mouse_controller.click(mouse.Button.left, 1)
        time.sleep(0.1)
    # Одиннадцатый клик — в первую точку
    mouse_controller.position = (points[0][0], points[0][1])
    time.sleep(0.05)
    mouse_controller.click(mouse.Button.left, 1)
    time.sleep(0.1)

def on_press(key):
    global running
    try:
        if key.char == 'g':
            x, y = mouse_controller.position
            click_decagon(x, y, radius)
        elif key.char == 'q':
            # Эмуляция Ctrl + '-'
            with keyboard_controller.pressed(keyboard.Key.ctrl):
                keyboard_controller.press('-')
                keyboard_controller.release('-')
        elif key.char == 'w':
            # Эмуляция Ctrl + '+'
            with keyboard_controller.pressed(keyboard.Key.ctrl):
                keyboard_controller.press('=')
                keyboard_controller.release('=')
        elif key == keyboard.Key.esc:
            running = False
            return False
    except AttributeError:
        if key == keyboard.Key.esc:
            running = False
            return False

with keyboard.Listener(on_press=on_press) as listener:
    while running:
        time.sleep(0.1) 