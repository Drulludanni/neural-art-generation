from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import adam
from keras.initializers import RandomNormal,RandomUniform
import os
import numpy as np
import cv2
from PIL import Image

gif_folder = "gif_files"

def generate_input(width=800, height=800, zoom=1, dt=0.0):
    """
    :param width: the width of the picture/frame
    :param height: the height of the picture/frame
    :param zoom: how much zoomed out the picture is
    :param dt:
    :return:
    """
    aspect = height/width
    Y, X = np.mgrid[0:height,0:width]
    X = 2*zoom*(np.reshape(X, (height, width, 1))/width - 0.5)
    Y = 2*zoom*(np.reshape(Y, (height, width, 1))/height - 0.5)*aspect
    dt = np.zeros(X.shape) + dt
    values = [
        X,
        Y,
        X+Y+1,
        X-Y-1,
        # np.sqrt(X**2 + Y**2),
        dt
              ]
    conc = np.concatenate(values, axis=2)
    print(np.shape(conc))
    return conc


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


def create_video(x=800, y=800,zoom=1, n_frames=100, dt=0.001 ):
    # generate the data for the first frame early so that
    # the model can be built to accept it
    values = generate_input(x, y, dt=0)
    model = create_model(np.shape(values)[1:])
    print("predicting")
    frames = []
    for i in range(1, n_frames):
        print(i)
        rgb = model.predict(values, batch_size=400)
        rgb = np.reshape(rgb,(y,x,3))
        rgb = np.uint8((rgb)*255)
        im = Image.fromarray(rgb)
        file = os.path.join(gif_folder,"out{}.png".format(i))
        im.save(file)
        frames.append(rgb)
        values = generate_input(x,y,zoom=zoom, dt=(i)*dt)

    write_frames_to_video("neural_output.avi", "MJPG", frames, dim=(x,y))
    # write_frames_to_video("neural_output2.avi", "X264", frames, dim=(x,y))

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

def create_png(x=800, y=800, zoom=1):
    values = generate_input(x, y, zoom)
    model = create_model(np.shape(values)[1:])
    print("predicting")
    rgb = model.predict(values, batch_size=400)
    rgb = np.reshape(rgb, (y, x, 3))
    rgb = np.uint8((rgb) * 255)
    im = Image.fromarray(rgb)
    file = "out.png"
    im.save(file)

if __name__ == '__main__':
    #create_video(1000, 500,n_frames=100)
    create_png(1000, 500, zoom=5)


