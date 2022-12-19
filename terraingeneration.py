import os
import ctypes
import numpy as np
import math
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from random import random
from perlin_noise import PerlinNoise

noise = PerlinNoise()
fieldOfView = 80
byteOffset = 0
# Paramters [ Mesh Size, Screen Reoslution, Scaling Size, Perlin Factor]
parameters = [48, 25, 8, 4, 4, 0.12]
startPoint = 0


vertexShaderCode = """
    attribute vec3 position;
    uniform mat4 transformationMatrix[2];
    uniform mat4 projectionMatrix;
    uniform mat4 translate;
    
    void main(){
        gl_Position = projectionMatrix * transformationMatrix[0] * transformationMatrix[1] * translate * vec4(position, 1.0);
    }
    """

fragmentShaderCode = """
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
def generateTerrain(value):
    global data 
    global noise
    
    data = []

    halfDataX = int(value[0] * value[4] / 2)
    halfDataY = int(value[1] * value[4] / 2)

    scaledResolutionX = int(value[2] * value[4] / 2)
    scaledResolutionY = int(value[3] * value[4] / 2)

    startX = 0
    startY = 0
    for j in range(-halfDataY, halfDataY):
        for i in range(-halfDataX, halfDataX):
            iData = i / scaledResolutionX
            jData = j / scaledResolutionY
            nextiData = (i + 1) / scaledResolutionX
            nextjData = (j + 1) / scaledResolutionY
            data.append([iData, jData, mapNoise(noise([startX, startY]))])
            data.append([iData, nextjData, mapNoise(noise([startX, (startY + value[5])]))])
            data.append([nextiData, jData, mapNoise(noise([(startX + value[5]), startY]))])
            startX = startX + value[5]
        
        startY = startY + value[5]
        startX = 0

    data = np.array(data, dtype = np.float32)
    return data

def mapNoise(noiseValue, maxMap = 1, minMap = 0.75):
    print(noiseValue)
    # return noiseValue

    if noiseValue > 0:
        return noiseValue * maxMap
    else:
        return noiseValue * minMap

def generateRotation(transformationData = None):
    if not transformationData or transformationData[0] == "":
        transformationMatrix = np.array(
            [
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0,
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
                1.0,0.0,0.0,0.0,
                0.0,cTheta,-sTheta,0.0,
                0.0,sTheta,cTheta,0.0,
                0.0,0.0,0.0,1.0,
            ],
            np.float32,
        )

    # y - axis rotation
    elif transformationData[0] == "yaw":
        transformationMatrix = np.array(
            [
                cTheta,0.0,sTheta,0.0,
                0.0,1.0,0.0,0.0,
                -sTheta,0.0,cTheta,0.0,
                0.0,0.0,0.0,1.0,
            ],
            np.float32,
        )

    # z - axis rotation
    elif transformationData[0] == "roll":
        transformationMatrix = np.array(
            [
                cTheta,-sTheta,0.0,0.0,
                sTheta,cTheta,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0,
            ],
            np.float32,
        )
    
    return transformationMatrix


def generateTranslation(translationData = None):
    if not translationData or translationData[0] == "":
        transformationMatrix = np.array(
            [
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0,
            ],
            np.float32,
        )

        return transformationMatrix

    transformationMatrix = np.array(
            [
                1.0,0.0,0.0,translationData[0],
                0.0,1.0,0.0,translationData[1],
                0.0,0.0,1.0,translationData[2],
                0.0,0.0,0.0,1.0,
            ],
            np.float32,
        )

    return transformationMatrix

def animate(offset):
    global noise
    global byteOffset
    global parameters
    global fieldOfView

    offsetY = int(parameters[1] * parameters[4] / 2) + offset
    
    halfDataX = int(parameters[0] * parameters[4] / 2)

    scaledResolutionX = int(parameters[2] * parameters[4] / 2)
    scaledResolutionY = int(parameters[3] * parameters[4] / 2)

    noOfData = 2 * halfDataX * 3

    translation = generateTranslation([0.0, -offset/scaledResolutionY, 0.0])

    loc = gl.glGetUniformLocation(program, "translate")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, translation)

    fov = (fieldOfView * math.pi)/180
    tanFOVHalf = np.tan(fov / 2.0)
    f = 1/tanFOVHalf
    projectionMatrix = np.array([f, 0.0, 0.0, 0.0,
                                0.0, 1, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0, 1], dtype = np.float32)    

    loc = gl.glGetUniformLocation(program, "projectionMatrix")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, projectionMatrix)

    data = []
    nextjData = (offsetY + 1) / scaledResolutionY
    jData = offsetY / scaledResolutionY
    startY = ((parameters[1] * parameters[4]) + offset) * parameters[5]
    startX = 0
    for i in range(-halfDataX, halfDataX):
        iData = i / scaledResolutionX
        nextiData = (i + 1) / scaledResolutionX
        data.append([iData, jData, mapNoise(noise([startX, startY]))])
        data.append([iData, nextjData, mapNoise(noise([startX, (startY + parameters[5])]))])
        data.append([nextiData, jData, mapNoise(noise([(startX + parameters[5]), startY]))])
        startX = startX + parameters[5]
    data = np.array(data, dtype = np.float32)

    rowBytes = noOfData * data.strides[0]
    maxBuffer = int(parameters[1] * parameters[4]) -1 

    if  byteOffset > maxBuffer:
        byteOffset = 0

    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, byteOffset * rowBytes, rowBytes, data)
    
    byteOffset = byteOffset + 1

    glut.glutTimerFunc(1, animate, offset + 1)


# initialization function
def initialize():
    global program
    global data
    global parameters
    global vertexBuffer
    global fieldOfView

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glLoadIdentity()

    program = createProgram(
        createShader(vertexShaderCode, gl.GL_VERTEX_SHADER),
        createShader(fragmentShaderCode, gl.GL_FRAGMENT_SHADER),
    )
    
    data = generateTerrain(parameters)
    
    fov = (fieldOfView * math.pi)/180
    tanFOVHalf = np.tan(fov / 2.0)
    f = 1/tanFOVHalf
    projectionMatrix = np.array([f, 0.0, 0.0, 0.0,
                                0.0, 1, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0, 1], dtype = np.float32)
    
    translationMatrix =generateTranslation([0, 0.5, 0.4])
    translation = generateTranslation()
    rotationMatrix = generateRotation(["pitch", 45])
    transformationMatrix = np.array([rotationMatrix, translationMatrix])
    
    # make program the default program
    gl.glUseProgram(program)

    vertexBuffer = gl.glGenBuffers(1)

    # make these buffer the default one
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)

    # bind the position attribute
    stride = data.strides[0]
    offset = ctypes.c_void_p(0)

    loc = gl.glGetAttribLocation(program, "position")
    gl.glEnableVertexAttribArray(loc)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

    loc = gl.glGetUniformLocation(program, "transformationMatrix")
    gl.glUniformMatrix4fv(loc, 2, gl.GL_TRUE, transformationMatrix)

    loc = gl.glGetUniformLocation(program, "projectionMatrix")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, projectionMatrix)

    loc = gl.glGetUniformLocation(program, "translate")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, translation)

    # Upload data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, data.shape[0])
    glut.glutSwapBuffers()
    glut.glutPostRedisplay()



def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b"\x1b":
        os._exit(1)

def mouse(x ,y):
    global startPoint
    global fieldOfView

    delX = x - startPoint
    print("released")

    if (delX == 0):
        fieldOfView = 70
    else:
        fieldOfView = (fieldOfView + (delX * 0.0050)) % 360
    print(fieldOfView)

# GLUT init
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH)
glut.glutCreateWindow("Graphics Window")
glut.glutReshapeWindow(1920, 1080)
glut.glutReshapeFunc(reshape)

initialize()
animate(0)

# animate()
glut.glutDisplayFunc(display)
glut.glutPostRedisplay()
glut.glutKeyboardFunc(keyboard)
glut.glutMotionFunc(mouse)
# enter the mainloop
glut.glutMainLoop()