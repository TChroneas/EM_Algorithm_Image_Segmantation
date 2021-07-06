import numpy as np
from numpy.random import random
import random
import imageio
import scipy.stats as st
from PIL import Image

def EM_init(K,img):
     P=np.full(shape=K,fill_value=1/K)
     M = np.asarray(random.sample(list(img), K))
     S = [np.cov(img.T) for z in range(K)]
     return M,P,S
def update_Gamma(img,M,S,P,K):
    N,D=img.shape
    L=np.ones((N,K))
    for i in range(K):
        dist=st.multivariate_normal(M[i],S[i])
        L[:,i]=dist.pdf(img)
    top=L*P
    bottom=np.sum(top,axis=1)
    bottom=np.reshape(bottom,(len(bottom),1))
    Gamma=top/bottom
    return Gamma
def update_P(Gamma):
    p = np.sum(Gamma, axis = 0) / Gamma.shape[0]
    return p
def update_M(img,Gamma,K):
    M=[]
    for j in range(K):
        G=Gamma[:,[j]]
        G=np.reshape(G, (1,len(G)))
        bottom = np.sum(G)
        top=np.sum(G.dot(img),axis=0)
        mean=top/bottom
        mean=np.array(mean)
        mean=np.reshape(mean,(1,len(mean)))
        M.append(mean[0])
    M=np.array(M)
    return M
def load_image(file):
    img=imageio.imread(file)
    return img
def flatten_image(image):
    x,y,z=image.shape
    fl_image=image.reshape(x*y,z)
    fl_image=np.array(fl_image,dtype=float)
    return fl_image
def unflatten_image(image,X,Y,vis=False):
    img=(image).astype(np.uint8)
    new_img=img.reshape(X,Y,3)
    return new_img
def update_covariance(img,M,Gamma,K):
    S=[]
    for k in range(K):
        G = Gamma[:, [k]]
        G = np.reshape(G, (1, len(G)))
        Mean = M[[k], :]
        diff=img-Mean
        diff=np.square(diff)
        top = np.sum(G.dot(diff),axis=1)
        top=np.sum(top)
        bottom = np.sum(G)
        bottom=bottom*3
        sigma=top/bottom
        cov=np.zeros((3,3))
        np.fill_diagonal(cov,sigma)
        S.append(cov)
    S=np.array(S)
    return S
def update_Log(img,M,S,P,K):
    pdf=np.ones((img.shape[0],K))
    for i in range(K):
        dist=st.multivariate_normal(M[i],S[i])
        pdf[:,i]=dist.pdf(img)
    log=np.log(np.sum(P*pdf,axis=1))
    sum_log=np.sum(log)
    return sum_log
def predict(Gamma):
    pr=np.argmax(Gamma,axis=1)
    return pr
def main():
    K=2
    error=10e-4
    image=load_image('im.jpg')
    img=flatten_image(image)
    M,P,S=EM_init(K,img)
    Gamma=update_Gamma(img,M,S,P,K)
    old_likelihood=0
    new_likelihood=1
    cnt=0
    while(abs(old_likelihood-new_likelihood)>error):
         cnt+=1
         old_likelihood=new_likelihood
         labels=predict(Gamma)
         Gamma=update_Gamma(img,M,S,P,K)
         M=update_M(img,Gamma,K)
         S=update_covariance(img,M,Gamma,K)
         P=update_P(Gamma)
         new_likelihood=update_Log(img,M,S,P,K)
         print("Iteration :",cnt)
         print("Diff = ",abs(old_likelihood-new_likelihood))
    print(labels)
    new_img=M[labels]
    new_image=unflatten_image(new_img,image.shape[0],image.shape[1])
    ni=Image.fromarray(new_image)
    ni.save("newim.jpg")
    norm=np.square(img-new_img)
    segError=np.sum(norm)/img.shape[0]
    print('Error for K = ',K,' is ',segError)
if __name__ == '__main__':
    main()
