import sys
from PIL import Image
from math import pi, sin, cos, tan


def cot(angle):
    return 1 / tan(angle)


# Project polar coordinates onto a surrounding cube
# assume ranges theta is [0,pi] with 0 the north poll, pi south poll
# phi is in range [0,2pi]
def projection(theta, phi):
    # if theta < 0.615:
    #     return projectTop(theta, phi)
    # elif theta > 2.527:
    #     return projectBottom(theta, phi)
    # if phi <= pi / 4 or phi > 7 * pi / 4:
    if (0.615 <= theta <= 2.527) and (phi <= pi / 4 or phi > 7 * pi / 4):
        return projectLeft(theta, phi)
    else:
        return ('None')
    # elif phi > pi / 4 and phi <= 3 * pi / 4:
    #     return projectFront(theta, phi)
    # elif phi > 3 * pi / 4 and phi <= 5 * pi / 4:
    #     return projectRight(theta, phi)
    # elif phi > 5 * pi / 4 and phi <= 7 * pi / 4:
    #     return projectBack(theta, phi)


def projectLeft(theta, phi):
    x = 1
    y = tan(phi)
    z = cot(theta) / cos(phi)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ("Left", x, y, z)


def projectFront(theta, phi):
    x = tan(phi - pi / 2)
    y = 1
    z = cot(theta) / cos(phi - pi / 2)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ("Front", x, y, z)


def projectRight(theta, phi):
    x = -1
    y = tan(phi)
    z = -cot(theta) / cos(phi)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ("Right", x, -y, z)


def projectBack(theta, phi):
    x = tan(phi - 3 * pi / 2)
    y = -1
    z = cot(theta) / cos(phi - 3 * pi / 2)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ("Back", -x, y, z)


def projectTop(theta, phi):
    # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,1)
    a = 1 / cos(theta)
    x = tan(theta) * cos(phi)
    y = tan(theta) * sin(phi)
    z = 1
    return ("Top", x, y, z)


def projectBottom(theta, phi):
    # (a sin θ cos ø, a sin θ sin ø, a cos θ) = (x,y,-1)
    a = -1 / cos(theta)
    x = -tan(theta) * cos(phi)
    y = -tan(theta) * sin(phi)
    z = -1
    return ("Bottom", x, y, z)


# Convert coords in cube to image coords
# coords is a tuple with the side and x,y,z coords
# edge is the length of an edge of the cube in pixels
def cubeToImg(coords, edge):
    if coords[0] == "Left":
        (x, y) = (int(edge * (coords[2] + 1) / 2), int(edge * (3 - coords[3]) / 2))
    elif coords[0] == "Front":
        (x, y) = (int(edge * (coords[1] + 3) / 2), int(edge * (3 - coords[3]) / 2))
    elif coords[0] == "Right":
        (x, y) = (int(edge * (5 - coords[2]) / 2), int(edge * (3 - coords[3]) / 2))
    elif coords[0] == "Back":
        (x, y) = (int(edge * (7 - coords[1]) / 2), int(edge * (3 - coords[3]) / 2))
    elif coords[0] == "Top":
        (x, y) = (int(edge * (3 - coords[1]) / 2), int(edge * (1 + coords[2]) / 2))
    elif coords[0] == "Bottom":
        (x, y) = (int(edge * (3 - coords[1]) / 2), int(edge * (5 - coords[2]) / 2))
    else:
        (x, y) = (0, 0)
    return (x, y)


# convert the in image to out image
def convert(imgIn, imgOut):
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0] / 4  # the length of each edge in pixels
    for i in range(inSize[0]):  # x
        for j in range(inSize[1]):  # y
            pixel = inPix[i, j]
            phi = i * 2 * pi / inSize[0]
            theta = j * pi / inSize[1]
            res = projection(theta, phi)
            if res[0] == 'None':
                continue

            (x, y) = cubeToImg(res, edge)
            # print('cord:', (x, y))
            # if i % 100 == 0 and j % 100 == 0:
            #   print i,j,phi,theta,res,x,y
            if x >= outSize[0]:
                # print "x out of range ",x,res
                x = outSize[0] - 1
            if y >= outSize[1]:
                # print "y out of range ",y,res
                y = outSize[1] - 1
            outPix[x, y] = pixel


imgIn = Image.open(sys.argv[1])
inSize = imgIn.size
imgOut = Image.new("RGB", (inSize[0], int(inSize[0] * 3 / 4)), "black")
convert(imgIn, imgOut)
imgOut.show()
