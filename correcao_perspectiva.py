import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import math

def unwarp(img, src, dst, testing):
    #pega a altura e comprimento da imagem

    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    #se a flag testing estiver ativa, vai mostrar o resultado da transformacao

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)

        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        ax2.set_ylim([h, 0])
        ax2.set_xlim([0, w])
        plt.show()
    else:
        #retorna a imagem com a perspectiva corrigida, que nao precisa.
        return warped, M

#funcao para extrair a matriz src para a funcao anterior
#tem como parametros de entrada uma img, que sera corrgida a perspectiva, e a flag testing.
def pega_pontos(img, testing = False):

    #aplica um filtro de cinza, suavizacao e da um destaque na imagem a ser processada
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    # usa o threshold para binarizar a imagem agucada
    thresh = cv2.threshold(sharpen,50,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    #pega os contoronos pretos na imagem
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #define uma area minima para que o contorno possa ser trabalhado.
    min_area = 400
    coordenadas = []


    for c in cnts:
        #veirifica os contornos na imagem
        #se os mesmos possuem uma area maior que a area minima vai extrair o centro de massa do contorno
        #e armazenar no array coordenadas
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        if area > min_area:

            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordenadas.append([cX, cY])
            except:
                None

    print(coordenadas)
    coordsorted = coordenadas.sort(key = lambda x:x[0])
    #ordena o vetor coordenadas em ordem crescente
    print(coordenadas)

    if testing:
        cv2.imshow('sharpen', sharpen)
        cv2.imshow('close', close)
        cv2.imshow('thresh', thresh)
        cv2.imshow('image', img)
        cv2.waitKey()
    #retorna o vetor coordenadas
    return coordenadas

im1 = cv2.imread('/content/imagem9.jpg')
pontos1 = pega_pontos(im1)

src = np.float32([(600, 80),
                  (80, 80),
                  (600, 300),
                  (80, 300)])

f_scale = 1.0
comprim = 500
larg = 500

dst = np.float32([(700, 0),
                  (0, 0),
                  (700, 400),
                  (0, 400)])

warpData = unwarp(im1, src, dst, False)
warpImg  = warpData[0]

plt.subplot(121),plt.imshow(im1)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(warpImg)
plt.title('Tranformada'), plt.xticks([]), plt.yticks([])

plt.show()

edges = cv2.Canny(warpImg,100,200)
plt.subplot(121),plt.imshow(warpImg,cmap = 'gray')
plt.title('Original warpImg'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge warpImg'), plt.xticks([]), plt.yticks([])

plt.show()

# Pontos do contorno
imgray = cv2.cvtColor(warpImg, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)