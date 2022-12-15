import os
import ctypes
import numpy as np
import math
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from random import random
from perlin_noise import PerlinNoise

vertexShaderCode = """
    attribute vec3 position;
    uniform mat4 transformationMatrix[2];
    uniform mat4 projectionMatrix;
    attribute vec4 color;
    varying vec4 vColor;
    
    void main(){
        gl_Position = projectionMatrix * transformationMatrix[0] * transformationMatrix[1] * vec4(position, 1.0);
        vColor = color;
    }
    """

fragmentShaderCode = """
    varying vec4 vColor;
    void main(){
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    """

# function to request and compiler shader slots from GPU
def createShader(source, type):
    # request shader
    shader = gl.glCreateShader(type)

    # set shader source using the code
    gl.glShaderSource(shader, source)

    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        error = gl.glGetShaderInfoLog(shader).decode()
        print(error)
        raise RuntimeError(f"{source} shader compilation error")

    return shader


# func to build and activate program
def createProgram(vertex, fragment):
    program = gl.glCreateProgram()

    # attach shader objects to the program
    gl.glAttachShader(program, vertex)
    gl.glAttachShader(program, fragment)

    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        print(gl.glGetProgramInfoLog(program))
        raise RuntimeError("Linking error")

    # Get rid of shaders (no more needed)
    gl.glDetachShader(program, vertex)
    gl.glDetachShader(program, fragment)

    return program


# -- Building Data --

def generateTerrainData(size, resolution, scale, perlinFactor = 3):
    noise = PerlinNoise()
    data = []
    for j in range(-size[1] * scale, (size[1] * scale)):
        for i in range(-size[0] * scale, (size[0] * scale)):
            data.append([i/(scale * resolution[0]), j/(scale * resolution[1]), noise([i * perlinFactor/(scale * resolution[0]), j * perlinFactor/(scale * resolution[1])])])
            data.append([i/(scale * resolution[0]), (j + 1)/(scale * resolution[1]), noise([i * perlinFactor/(scale * resolution[0]), (j + 1) * perlinFactor/(scale * resolution[1])])])
            data.append([(i+1)/(scale * resolution[0]), (j)/(scale * resolution[1]), noise([(i + 1) * perlinFactor/(scale * resolution[0]), j * perlinFactor/(scale * resolution[1])])])
        
    data = np.array(data, dtype = np.float32)
    return data

def generateRotation(transformationData):
    if not transformationData or transformationData[0] == "":
        transformationMatrix = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )
        return transformationMatrix
    cTheta = np.cos(transformationData[1] / 180 * math.pi)
    sTheta = np.sin(transformationData[1] / 180 * math.pi)

    # x - axis rotation
    if transformationData[0] == "pitch":
        transformationMatrix = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                cTheta,
                -sTheta,
                0.0,
                0.0,
                sTheta,
                cTheta,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )

    # y - axis rotation
    elif transformationData[0] == "yaw":
        transformationMatrix = np.array(
            [
                cTheta,
                0.0,
                sTheta,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                -sTheta,
                0.0,
                cTheta,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )

    # z - axis rotation
    elif transformationData[0] == "roll":
        transformationMatrix = np.array(
            [
                cTheta,
                -sTheta,
                0.0,
                0.0,
                sTheta,
                cTheta,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )
    
    return transformationMatrix


def generateTranslation(translationData):
    if not translationData or translationData[0] == "":
        transformationMatrix = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )

        return transformationMatrix

    transformationMatrix = np.array(
            [
                1.0,
                0.0,
                0.0,
                translationData[0],
                0.0,
                1.0,
                0.0,
                translationData[1],
                0.0,
                0.0,
                1.0,
                translationData[2],
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            np.float32,
        )

    return transformationMatrix
# initialization function
def initialize():
    global program
    global data

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glLoadIdentity()

    program = createProgram(
        createShader(vertexShaderCode, gl.GL_VERTEX_SHADER),
        createShader(fragmentShaderCode, gl.GL_FRAGMENT_SHADER),
    )
    
    data = generateTerrainData([64, 36], [16, 9], 3, 1.75)

    # data = np.array([[-0.5,-0.5,1.0],[-0.5, -0.4, 1.0], [-0.4,-0.5,1.0], [-0.4,-0.4,1.0], [-0.3,-0.5,1.0], [-0.3,-0.4,1.0]], dtype = np.float32)
    color = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    
    fieldOfView = (90 * math.pi)/180
    tanFOVHalf = np.tan(fieldOfView / 2.0)
    f = 1/tanFOVHalf
    projectionMatrix = np.array([f, 0.0, 0.0, 0.0,
                                0.0, 1, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0, 1], dtype = np.float32)
    
    translationMatrix =generateTranslation([0, 0.5, 0.4])
    rotationMatrix = generateRotation(["pitch", 45])
    transformationMatrix = np.array([rotationMatrix, translationMatrix])
    # make program the default program
    gl.glUseProgram(program)

    buffer = gl.glGenBuffers(1)

    # make these buffer the default one
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

    # bind the position attribute
    stride = data.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)


    stride = color.strides[0]
    offset = ctypes.c_void_p(0)
    loc = gl.glGetAttribLocation(program, "color")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    loc = gl.glGetUniformLocation(program, "transformationMatrix")
    gl.glUniformMatrix4fv(loc, 2, gl.GL_TRUE, transformationMatrix)

    loc = gl.glGetUniformLocation(program, "projectionMatrix")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, projectionMatrix)

    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE);
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, data.shape[0])
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL);
    glut.glutSwapBuffers()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b"\x1b":
        os._exit(1)


# GLUT init
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
glut.glutCreateWindow("Graphics Window")
glut.glutReshapeWindow(1920, 1080)
glut.glutReshapeFunc(reshape)

initialize()

glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)

# enter the mainloop
glut.glutMainLoop()