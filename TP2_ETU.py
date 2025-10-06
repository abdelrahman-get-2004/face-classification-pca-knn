# TP2 – ACP, classification & reconstruction (version claire)
# Auteur: (Abdelmagid Abdelrahman) – M1 SDV – 2024/25

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import time 
from time import perf_counter

#%%
def plotGallery(images, n=16, title=None):
    # Affiche les n premières images contenues dans images
    # images est de taille Nb image*Ny*Nx
    n = min(n, images.shape[0])
    nSubplots = int(np.ceil(np.sqrt(n)))
    fig, axs = plt.subplots(nSubplots, nSubplots)
    for i in range(n):
        axs[i // nSubplots, i % nSubplots].imshow(images[i], cmap=plt.cm.gray)
        axs[i // nSubplots, i % nSubplots].set_xticks([])
        axs[i // nSubplots, i % nSubplots].set_yticks([])
    if title:
        plt.suptitle(title)
    plt.show()
    print(f"Le nombre de classes est : {len(name)}") 
   
[imgs, lbls, name] = np.load("TP1.npy", allow_pickle=True)

print("Identité des 16 personnes :")
for i in range(16):
    print(f"La personne sur l'image {i} est : {name[lbls[i]]}")
    

# nb de images :966+322
#nb de classes 7
#tail (62x47)
def plotHistoClasses(lbls):
    # Affiche le nombre d'exemples par classe
    nLbls = np.array([[i, np.where(lbls == i)[0].shape[0]] for i in np.unique(lbls)])
    plt.figure()
    plt.bar(nLbls[:, 0], nLbls[:, 1])
    plt.title("Nombre d'exemples par classe")
    plt.grid(axis='y')
    plt.show()
    
    


[imgs, lbls, name]=np.load("TP1.npy",allow_pickle=True )
plotGallery(imgs)
plotHistoClasses(lbls)
X_train, X_test, y_train,y_test =train_test_split(imgs,lbls,random_state=342)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

print(X_test.ndim)

#N=966 et n =2914
X_train =X_train.reshape(X_train.shape[0],-1)
X_test=X_test.reshape(X_test.shape[0],-1)
#X_test=np.reshape(X_test,(938308,1))
scaler=StandardScaler()
scaler.fit(X_train)
# partie 2:pour centre et reduire ,avoir la moyenne et faire l ecart type 
#standardiser et eviter le data leakage 
#Chaque pixel (chaque feature) a une distribution différente : certains ont des 
#valeurs fortes, d’autres faibles.
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

print("moyennes:", scaler.mean_)
print("Ecarts-types :", scaler.scale_)

knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn1.fit(X_train_std, y_train)
y_pred1 = knn1.predict(X_test_std)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred1)
print("Matrice de confusion (1-PPV):\n", cm)

acc1 = accuracy_score(y_test, y_pred1)
print("Taux de reconnaissance (1-PPV):", acc1)
#la matrice de confusion ?
#c’est un tableau qui compare les predictions du classifieur aux vraies 
#etiquettes. Chaque ligne correspond aux exemples d’une classe reelle, 
#chaque colonne aux classes prdites. L’élément (i,j) est le nombre d’exemples
# de la classe reelle i classes en j. Les diagonales = bien classés, les hors-diagonales = erreurs.
#la somme de tous les éléments de la matrice = le nombre total d’images dans la base de test.
#La somme de chaque ligne = le nombre d’exemples dans la base de test appartenant à cette classe particulière.
#est-ce que les classes sont équilibrées dans la base de test ?
#Non, pas vraiment. Comme indiqué dans le PDF, certaines personnes apparaissent
# beaucoup plus souvent que d’autres. Les classes ne sont donc pas équiprobables :
#certaines lignes de la matrice de confusion auront beaucoup plus d’exemples que d’autres.
# ---- Classifieur K-PPV ----
Kmax = 47
accs = []
for k in range(1, Kmax+1):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    accs.append(accuracy_score(y_test, y_pred))
# Courbe du taux de reconnaissance en fonction de K
plt.figure()
plt.plot(range(1, Kmax+1), accs, marker='o')
plt.xlabel("valeur de K")
plt.ylabel("taux de reconnaissance")
plt.title("performance des KPPV")
plt.grid(True)
plt.show()

print("meilleur taux :", max(accs), "pour K =", np.argmax(accs)+1)
#évolution montre qu’un petit K améliore la robustesse par rapport au cas 
#1-PPV, mais qu’un K trop grand dégrade les résultats car il efface les differences entre classes.
I = Image.open("Bush.jpg")
I = np.array(I)
I = rgb2gray(I)
print("taille originale :", I.shape)
I = np.resize(I, (62, 47))
print("nouvelle taille  :", I.shape)
plt.imshow(I, cmap='gray')
plt.show()
I_vec = I.reshape(1, -1)       
I_vec_std = scaler.transform(I_vec)   

# prediction
y_pred = knn1.predict(I_vec_std)  
print("la personne prédite est :", name[y_pred[0]])

#########
#%%
print ("Aprés redimensionnement, le nombre et la dimension des données en Train et en Test :")
# Combien y a-t-il d'mages en train et en test : 
print ("nb img X_train : " , len(X_train))
print ("nb img  X_test : ", len(X_test), )
print ("dim X_train : ", np.shape(X_train[0]))
print ("dim X_test :", np.shape(X_test[0]))
print ("dim y_train : ", len(y_train), " labels")
print ("dim y_test: ", len(y_test), " labels")

#le nombr dans chaque set n'a pas changé, mais la dimension elle si, donc c bien un redimensionnement


#%%
pca = PCA(n_components=100)
tps1 = time.time()
pca.fit(X_train_std)
tps2 = time.time()
print("Durée de classification",tps2 - tps1)
plt.figure()
plt.xlabel("nb comp principle")
plt.ylabel("variance explique cumule")

plt.show()
X_train1 = pca.transform(X_train_std)
X_test1 = pca.transform(X_test_std)
print("tps1",tps1)
print("tps2",tps2)

print("nouvelle dim X_train1 :", X_train1.shape)
print("nouvelle dim  X_test1 :", X_test1.shape)


pca = PCA(n_components=None)
tps1 = time.time()
print("tps1:",)
pca.fit(X_train)
tps2 = time.time()
print("durree classification",tps2 - tps1)
plt.figure()
plt.plot(pca.explained_variance_ratio_)

plt.title("courbe des variance en fonction des dimennsions")
plt.xlabel("composantes principal")
plt.ylabel("variance expliquée")
plt.grid()
#%%
#sans pca 

tps1 = time.time()
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train_std, y_train)
y_pred_std = knn_manhattan.predict(X_test_std)
tps2 = time.time()

print("Sans PCA ")
print("taux reconaissance  :", accuracy_score(y_test, y_pred_std))
print("durée classification ", tps2 - tps1, "secondes")

# avec pca
tps1 = time.time()
knn_manhattan_pca = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan_pca.fit(X_train1, y_train)
y_pred_pca = knn_manhattan_pca.predict(X_test1)
tps2 = time.time()

print("PCA 100 comp")
print("taux recon :", accuracy_score(y_test, y_pred_pca))
print("duree de classif :", tps2 - tps1, "s")

#La PCA réduit le temps de calcul de façon significative.

#%%

###III. Analyse en composantes principales et reconstruction


#%%
pca50 = PCA(n_components=50)
pca50.fit(X_train_std)

# Vecteurs propres
image50 = pca50.components_.reshape((50, 62, 47))

plotGallery(image50)
print("dimension :",np.shape(image50))

#%%



#%%
# Compression
X_test_comp = pca50.transform(X_test_std)

# Reconstruction
X_test_reconst = pca50.inverse_transform(X_test_comp)

# Comparaison visuelle
plotGallery(X_test_std.reshape((-1,62,47))[:16], title="originale")
plotGallery(X_test_reconst.reshape((-1,62,47))[:16], title="reconstr")
#%%
E = X_test_reconst - X_test_std
erreur = np.mean(np.sqrt(np.sum(E**2, axis=0)))
print("erreur moy de constr :", erreur)
#%%
erreurs = []
for d in range(10, 951, 50):
    pca_d = PCA(n_components=d)
    X_test_comp_d = pca_d.fit_transform(X_train_std)
    X_test_reconst_d = pca_d.inverse_transform(pca_d.transform(X_test_std))
    E = X_test_reconst_d - X_test_std
    erreurs.append(np.mean(np.sqrt(np.sum(E**2, axis=1))))
plt.figure()
plt.plot(range(10,951,50), erreurs, marker='o')
plt.grid()
plt.show()

#%%
