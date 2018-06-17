from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import adam
from keras.initializers import RandomNormal,RandomUniform
import os
import numpy as np
import cv2
from PIL import Image

gif_folder = "gif_files"

def generate_input(width=800, height=800, scale=1, offset=0.0):
    values = []
    for i in range(width):
        for j in range(height):
            x = scale*(i/width - 0.5)
            y = scale*(j/height - 0.5)
            value = []
            # value.append(x)
            value.append(y)
            value.append(np.sqrt((x+0.1)**2 + y**2))
            value.append(offset)
            # value.append(np.sqrt((x-0.5)**2 + (y-0.5)**2))
            # value.append(np.sign(y)*np.sqrt(abs(y)))
            # value.append(np.sign(x)*np.sqrt(abs(x)))
            # value.append(1/(np.sqrt((x+0.4)**2 + y**2)+1))
            # value.append(1/(np.sqrt((x-0.4)**2 + y**2)+1))
            # value.append(np.sqrt((x-0.5)**2 + y**2))
            # value.append(np.sqrt((x+0.1)**2 + (y+ 0.1)**2))
            # value.append(np.sqrt((x+offset)**2 + (y)**2))
            # value.append(np.sqrt((x-offset)**2 + (y)**2))
            # value.append(np.sqrt((x+0.5)**2 + (y-0.5)**2))
            # value.append(np.sqrt((x+0.5)**2 + (y-0.5)**2))
            # value.append(np.sqrt((x-0.5)**2 + (y+0.5)**2))

            values.append(value)
    return np.array(values)

def create_model(in_shape):
    inp = Input(shape=in_shape)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-1,maxval=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(inp)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-1,maxval=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-1,maxval=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-1,maxval=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-4,maxval=4), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-4,maxval=4), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-4,maxval=4), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-4,maxval=4), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    # l = Dense(40, activation="tanh",kernel_initializer=RandomNormal(mean=0,stddev=1), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    l = Dense(40, activation="tanh", kernel_initializer=RandomUniform(minval=-3,maxval=4), bias_initializer=RandomNormal(mean=0.0, stddev=0.0))(l)
    out = Dense(3, activation="sigmoid")(l)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=adam(),loss='mean_squared_error')
    return model


def create_video(x=800, y=800, n_frames=100):
    values = generate_input(x, y, scale=1, offset=0)
    model = create_model(np.shape(values)[1:])
    print(model.summary())
    print("predicting")
    offset = 0.001
    frames = []
    for i in range(n_frames):
        print(i)
        values = generate_input(x,y,scale=1, offset=i*offset)
        rgb = model.predict(values, batch_size=400)
        rgb = np.reshape(rgb,(x,y,3))
        rgb = np.uint8((rgb)*255)
        im = Image.fromarray(rgb)
        file = os.path.join(gif_folder,"out{}.png".format(i))
        im.save(file)
        frames.append(rgb)

    write_frames_to_video("neural_output.avi", "MJPG", frames, dim=(x,y))
    write_frames_to_video("neural_output2.avi", "X264", frames, dim=(x,y))

def write_frames_to_video(name, codec, frames, dim=(800, 800)):
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(name, fourcc, 30, dim)

        for frame in frames:
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
    except Exception as e:
        print("failed for:", codec)
        print(e)

def create_png(x=800, y=800):
    values = generate_input(x, y, scale=1, offset=0)
    model = create_model(np.shape(values)[1:])
    print(model.summary())
    print("predicting")
    values = generate_input(x, y, scale=1)
    rgb = model.predict(values, batch_size=400)
    rgb = np.reshape(rgb, (x, y, 3))
    rgb = np.uint8((rgb) * 255)
    im = Image.fromarray(rgb)
    file = "out.png"
    im.save(file)

if __name__ == '__main__':
    create_video(n_frames=100)



