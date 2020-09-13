import cv2
import numpy as np

cap=cv2.VideoCapture(0)
whT=320
confThreshold=0.5
nmsTreshold=0.3 #değeri düşük tutmaya çalışın aksi halde kutu içinde kutular çıkar !

classFile="coco.names"
classNames=[]

with open(classFile,"rt") as f:
    classNames=f.read().rstrip("\n").split("\n")
# print(className)
# print(len(className))

#Ağırlık ve yapılandırıcı dosyalarımızın yolunu ve türünü belirliyoruz
modelConfiguration="yolov3-tiny.cfg"
modelWeights="yolov3-tiny.weights"

#Modelimizin sinir ağları ile iletişeme gecmemiz gerek
net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
#Opencv arka uçta cpu ile kullanmak için:
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def findObject(outputs,img):
    hT,wT,cT=img.shape
    bbox=[] #sınırlayıcı kutularımızı tutan liste
    classIds=[] #sınıflarımızın ne oluğunu tutan id listesti
    confs=[] #Bulunan nesnelerin güven değerini tutan listemiz

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId] #Burda nesne tanırken güven değeri en yüksek olanları kaydediyoruz
            if confidence>confThreshold:
                w,h=int(det[2]*wT),int(det[3]*hT)
                x,y=int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    #İç içe kutuların oluşmasını önlemek için ;
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsTreshold)
    print(indices)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,255),2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success,img=cap.read()
    #Darknet blob ile haberleşmek için
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False) #genişlik ve yükseklik ayarlarını yapıyoruz
    #girdilerimizi blob olarak belirliyoruz
    net.setInput(blob)
    #Yolo mimarisini daha iyi anlamak adına katman adlarımıza göz atmalıyız .Ve 3 adet çıktı katmanımız olduğunu unutmayalım
    layerNames=net.getLayerNames()
    #print(layerNames)
    #print(net.getUnconnectedOutLayers())
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs=net.forward(outputNames)
    #print(len(outputs))
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    findObject(outputs,img)

    cv2.imshow("DemoMete",img)
    cv2.waitKey(1)