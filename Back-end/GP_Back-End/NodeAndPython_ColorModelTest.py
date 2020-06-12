# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:24:34 2019

@author: yousra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:49:33 2019

@author: yousra
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    
    
def BGR2RGB(img):
    RGBimg=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    RGBimg[:,:,0]=R
    RGBimg[:,:,1]=G
    RGBimg[:,:,2]=B
    return RGBimg

def HSLImage (BGRImage,modifiedtags):
    HSLImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2HLS).astype('float')
    Hue = HSLImage[:,:,0] * 2
    Sat = HSLImage[:,:,1] / 255
    Light = HSLImage[:,:,2] / 255
     
    imgRows = HSLImage.shape[0]
    imgCols = HSLImage.shape[1]
    
    piece = np.zeros((imgRows,imgCols), bool)
    piece[:,:]=False
    piece[modifiedtags == 255] = True
    totalPixels = np.sum(piece.astype('int'))

    ColorsMat = np.zeros((imgRows,imgCols), np.uint8)
    ColorsLabels = np.array(['null','black','white','darkGray','lightGray','darkRed','lightRed','brown','lightOrange','darkYellow','lightYellow','darkGreen', 'lightGreen','darkCyan','lightCyan','darkBlue','lightBlue','darkPurple','lightPurple','darkPink','lightPink'])
    
    black =  (Light >= 0) & (Light < 0.125)  #0
    black = black & piece
    blackCount = np.sum(black.astype('int')) / totalPixels
    
    white = (Light >= 0.875) & (Light <= 1 )  #1
    whiteCount = np.sum(white.astype('int')) /totalPixels
    
    dark_gray = (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.125) & (Light <0.5) #2
    dark_grayCount = np.sum(dark_gray.astype('int')) / totalPixels
    
    light_gray  =  (Sat >=0) & (Sat < 0.2 ) & (Light >= 0.5) & (Light < 0.875) #3
    light_grayCount = np.sum(light_gray.astype('int')) / totalPixels
    
    dark_red   = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #4
    dark_redCount = np.sum(dark_red.astype('int')) / totalPixels
    
    light_red = np.logical_or((Hue >= 0 ) & (Hue < 18),(Hue >=340)) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #5
    light_redCount = np.sum(light_red.astype('int')) / totalPixels
    
    brown = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #6
    brownCount = np.sum(brown.astype('int')) / totalPixels
    
    light_orange = (Hue >= 18 ) & (Hue < 45) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #7
    light_orangeCount = np.sum(light_orange.astype('int')) / totalPixels
    
    dark_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5)  #8
    dark_yellowCount = np.sum(dark_yellow.astype('int')) / totalPixels

    light_yellow = (Hue >= 45 ) & (Hue < 75) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #9
    light_yellowCount = np.sum(light_yellow.astype('int')) / totalPixels

    dark_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #10
    dark_greenCount = np.sum(dark_green.astype('int')) / totalPixels

    light_green = (Hue >= 75 ) & (Hue <155 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #11
    light_greenCount = np.sum(light_green.astype('int')) / totalPixels

    dark_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #12
    dark_cyanCount = np.sum(dark_cyan.astype('int')) / totalPixels

    light_cyan = (Hue >= 155 ) & (Hue <200 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #13
    light_cyanCount = np.sum(light_cyan.astype('int')) / totalPixels
    
    dark_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #14
    dark_blueCount = np.sum(dark_blue.astype('int')) / totalPixels
    
    light_blue = (Hue >= 200 ) & (Hue <260 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #15
    light_blueCount = np.sum(light_blue.astype('int')) / totalPixels
    
    dark_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #16
    dark_purpleCount = np.sum(dark_purple.astype('int')) / totalPixels
    
    light_purple = (Hue >= 260 ) & (Hue <310 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #17
    light_purpleCount = np.sum(light_purple.astype('int')) / totalPixels
    
    dark_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.125) & (Light <0.5) #18
    dark_pinkCount = np.sum(dark_pink.astype('int')) / totalPixels
    
    light_pink = (Hue >= 310 ) & (Hue <340 ) & (Sat >=0.2) & (Sat<=1 ) & (Light >= 0.5) & (Light < 0.875) #19
    light_pinkCount = np.sum(light_pink.astype('int')) / totalPixels
   
       
    ColorsMat[black] = 1
    ColorsMat[white] = 2
    ColorsMat[dark_gray] = 3
    ColorsMat[light_gray] = 4
    ColorsMat[dark_red] = 5
    ColorsMat[light_red] = 6
    ColorsMat[brown] = 7
    ColorsMat[light_orange] = 8
    ColorsMat[dark_yellow] = 9
    ColorsMat[light_yellow] = 10
    ColorsMat[dark_green] = 11
    ColorsMat[light_green] = 12
    ColorsMat[dark_cyan] = 13
    ColorsMat[light_cyan] = 14
    ColorsMat[dark_blue] = 15
    ColorsMat[light_blue] = 16
    ColorsMat[dark_purple] = 17
    ColorsMat[light_purple] = 18
    ColorsMat[dark_pink] = 19
    ColorsMat[light_pink] = 20
    
    ColorsRatios = np.array([0,blackCount,whiteCount,dark_grayCount,light_grayCount,dark_redCount,light_redCount,brownCount,light_orangeCount,dark_yellowCount,light_yellowCount,dark_greenCount,light_greenCount,dark_cyanCount,light_cyanCount,dark_blueCount,light_blueCount,dark_purpleCount,light_purpleCount,dark_pinkCount,light_pinkCount])
    colors = [ColorsLabels[i] for i in np.unique(ColorsMat)]
    colors.remove('null')
    frequencies = [ColorsRatios[i] for i in np.unique(ColorsMat)]
    frequencies.remove(0)
    #x = list(zip(colors,frequencies))
    #print(x)
    return colors, frequencies









def getImageColors(image,matPixelLevel,labelsList,discardedTags):
    imageColors = []
    tags = matPixelLevel.get('groundtruth')  #array
    modifiedtags = np.copy(tags,np.uint8)
    uniqueTags = np.unique(tags)
    for tag in uniqueTags:
        pieceLabel = labelsList[tag]
        if(tag in discardedTags):
            continue
        modifiedtags[tags == tag] = 255
        modifiedtags[tags != tag] = 0
        newSegmentedImg = np.zeros((modifiedtags.shape[0],modifiedtags.shape[1],3),np.uint8)
        newSegmentedImg[:,:,0] = modifiedtags
        newSegmentedImg[:,:,1] = modifiedtags
        newSegmentedImg[:,:,2] = modifiedtags
    
        newImage = np.bitwise_and(newSegmentedImg.astype('uint8'), image.astype('uint8'))
        pieceColors,colorsFrequencies = HSLImage(newImage,modifiedtags)
        pieceColors=np.array(pieceColors)
        colorsFrequencies=np.array(colorsFrequencies)
        thresholdedColors =  pieceColors[colorsFrequencies>0.15]
        pieceColors = ''
        for color in thresholdedColors:
            pieceColors = color + '-' + pieceColors 
        pieceColors = pieceColors + pieceLabel
        imageColors.append(pieceColors)
        
    return imageColors

    
   
    


