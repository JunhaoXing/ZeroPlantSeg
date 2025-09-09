import numpy as np
import cv2

def palette(color_name: str):
    my_palette = {"akane": (101, 83, 209), 
                  "benihi": (41, 57, 232),
                  "webblue": (255, 104, 0)}
    return my_palette[color_name]

def random_color():
    return tuple(np.random.randint(0, 256, size=3))

def angle2rgb(angle):
    """_summary_

    Args:
        angle (_type_): 0 ~ 360
        s (int, optional): _description_. Defaults to 255.
        v (int, optional): _description_. Defaults to 255.
    """
    N_theta = 360
    luv = np.zeros((1, N_theta, 3)).astype(np.float32)
    theta = np.linspace(0, 2*np.pi, N_theta)
    luv[:, :, 0] = 55 # L
    luv[:, :, 1] = np.cos(theta)*100 # u
    luv[:, :, 2] = np.sin(theta)*100 # v

    rgb = cv2.cvtColor(luv, cv2.COLOR_Luv2RGB)
    # get coordinates
    theta = np.linspace(0, 2*np.pi, rgb.shape[1]+1)

    # get color
    color = rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]))
    color_from_angle = color[round(angle)] * 255
    color_from_angle = color_from_angle.astype(np.uint8)
    return color_from_angle

def angle2bgr(angle):
    """_summary_

    Args:
        angle (_type_): 0 ~ 360
        s (int, optional): _description_. Defaults to 255.
        v (int, optional): _description_. Defaults to 255.
    """
    N_theta = 360
    luv = np.zeros((1, N_theta, 3)).astype(np.float32)
    theta = np.linspace(0, 2*np.pi, N_theta)
    luv[:, :, 0] = 55 # L
    luv[:, :, 1] = np.cos(theta)*100 # u
    luv[:, :, 2] = np.sin(theta)*100 # v

    rgb = cv2.cvtColor(luv, cv2.COLOR_Luv2BGR)
    # get coordinates
    theta = np.linspace(0, 2*np.pi, rgb.shape[1]+1)

    # get color
    color = rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]))
    color_from_angle = color[round(angle)] * 255
    color_from_angle = color_from_angle.astype(np.uint8)
    return color_from_angle
