import cv2
import numpy as np
import os



####################config#####################################
BaseDir = r"C:\Users\guoqingpei\Documents\Tencent Files\578930228\FileRecv\4120-4520"
SaveDir = r"C:\Users\guoqingpei\Desktop\FaceRecog\marked"

#########global variables#################
drawing = False # true if mouse is pressed
ix,iy = -1,-1
buffered_image  = None
img = None
marktimes = 3
window_name = None
i = 0
marks = []



files = [f for f in os.listdir(BaseDir) if f.endswith(".jpeg")]
total_imgs = len(files)
img = cv2.imread(os.path.join(BaseDir,files[i]))
img = cv2.resize(img, (0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)  
window_name = 'image {}/{}'.format(i+1,total_imgs)
marktimes = 3
buffered_image =  img.copy()

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) #Facultative
cv2.moveWindow(window_name, 0, 0)
# cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN ,cv2.WINDOW_FULLSCREEN )
cv2.imshow(window_name,img)

#create save dir
if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)

#write landmark files
def Onchange(image_to_save,image_name, lands):
    img_save_path = os.path.join(SaveDir, image_name)
    #save marked images
    cv2.imwrite(img_save_path, image_to_save)
    #save lands
    landsfile  = img_save_path.replace(".jpeg",".txt")
    open(landsfile,"w+").write("\n".join(lands))



# mouse callback function
def draw_circle(event,x,y,flags,param):
    global marks,ix,iy,drawing,img,buffered_image,marktimes,BaseDir,files,total_imgs,i,window_name
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        # print "EVENT_LBUTTONDOWN"
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            drawing_tmp = buffered_image.copy()
            cv2.rectangle(drawing_tmp,(ix,iy),(x,y),(0,255,0),1)
            cv2.imshow(window_name,drawing_tmp)
            cv2.moveWindow(window_name, 0, 0)
            # cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN ,cv2.WINDOW_FULLSCREEN )
            # print "EVENT_MOUSEMOVE"
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        buffered_image = img.copy()
        cv2.imshow(window_name,img)
        # cv2.waitKey()
        marktimes-=1
        marks.append("{} {}\t{} {}".format(ix,iy,x,y))
        # print "EVENT_LBUTTONUP:marktimes {}".format(marktimes)
        if marktimes==0:
            Onchange(img, files[i], marks)
            i+=1
            marks = []
            cv2.destroyWindow(window_name)
            img_path = os.path.join(BaseDir,files[i])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
            window_name = 'image {}/{}'.format(i+1,total_imgs)
            marktimes = 3
            buffered_image =  img.copy()    
            cv2.imshow(window_name,img)
            cv2.moveWindow(window_name, 0, 0)
            # cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN ,cv2.WINDOW_FULLSCREEN )
            cv2.setMouseCallback(window_name,draw_circle)



cv2.setMouseCallback(window_name,draw_circle)
cv2.waitKey()
cv2.destroyAllWindows()

