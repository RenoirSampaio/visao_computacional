import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
import math
from PIL import Image

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

# Ajuste do brilho
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

x = '/content/imagem9.jpg' 
original = cv2.imread(x, 1)
cv2_imshow(original)

# Imagem mais escura
gamma = 0.5                             
adjusted1 = adjust_gamma(original, gamma=gamma)
cv2.putText(adjusted1, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2_imshow(adjusted1)
im = Image.fromarray(adjusted1)
im.save("adjusted1.jpg")

# Imagem mais clara
gamma = 2                     
adjusted2 = adjust_gamma(original, gamma=gamma)
cv2.putText(adjusted2, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2_imshow(adjusted2)
im = Image.fromarray(adjusted2)
im.save("adjusted2.jpg")

# Caminho da imagem
im1 = cv2.imread('/content/imagem9.jpg')
baseheight = 560
img = Image.open('/content/imagem9.jpg')
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), Image.ANTIALIAS)
img.save('resized_image.jpg')
im2 = cv2.imread('/content/resized_image.jpg')

# Captura dos pontos na imagem
pontos1 = pega_pontos(im1)
pontos2 = pega_pontos(im2)

# Fonte
src1 = np.float32([(500, 85),
                  (120, 85),
                  (500, 285),
                  (120, 285)])
src2 = np.float32([(560, 100),
                  (140, 100),
                  (560, 320),
                  (140, 320)])

# Destino
dst1 = np.float32([(735, 0),
                  (0, 0),
                  (735, 440),
                  (0, 440)])
dst2 = np.float32([(700, 0),
                  (0, 0),
                  (700, 415),
                  (0, 415)])

warpData = unwarp(im1, src1, dst1, False)
warpImg  = warpData[0]
warpData2 = unwarp(im2, src2, dst2, False)
warpImg2  = warpData2[0]

# Plotagem das imagens
plt.subplot(121),plt.imshow(im1)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(warpImg2)
plt.title('Tranformada'), plt.xticks([]), plt.yticks([])
plt.show()

edges = cv2.Canny(warpImg2,100,200)
plt.subplot(121),plt.imshow(warpImg2,cmap = 'gray')
plt.title('Original warpImg'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge warpImg'), plt.xticks([]), plt.yticks([])
plt.show()

# Plotagem do contorno na imagem
imgray = cv2.cvtColor(warpImg,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 120,255,cv2.THRESH_BINARY)
contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
for contour in contours:
  cv2.drawContours(warpImg2, contour, -5, (0, 255, 100), 3)
plt.figure()
plt.imshow(warpImg2)

# Escrita de arquivo do contorno
arquivo = open("contornos.txt", "a")
for e in contour:
  str_sub = str(e)
  arquivo.write(str_sub[1:-1])