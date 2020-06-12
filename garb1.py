

# unique = np.unique(label1)
#
# # print (tags[0,unique])



#print(len(labels[0]))
#imgs=loadLabels("annotations\pixel-level",10)
#print(imgs[0]['groundtruth'])



# x = loadmat('0001.mat')
# imlevel= loadmat('1005.mat')
# tags=np.array(imlevel['tags'])
# print(tags.shape)
# listmap=loadmat('label_list.mat')
# labels=np.array(listmap['label_list'])
# print(labels[0,tags])
#
# print(x['groundtruth'])
# z=np.array(x['groundtruth'])
# print(z.shape)

#
# #resizing
# img=imgs[0]
# unique = np.unique(img)
#
# print (len(unique))
#
# img1=cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
# unique = np.unique(img1)
#
# print (len(unique))
# listmap=loadmat('label_list.mat')
# tags=np.array(listmap['label_list'])
#
# j=180
# for i in range(120):
#     img1[j,i,0]=255
#     img1[j,i,1]=0
#     img1[j,i,2]=0
# #show_images([img1])