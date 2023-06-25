import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def readFiles(folder_name):
    faces = {}
    #Walk thru each dir and file to get image
    for root, dirs, files in os.walk(folder_name, topdown=False):
        for filename in files:
            # Skip if it's not a .pgm file
            if not filename.endswith(".pgm"):
                continue
            #Get the path to image  
            file_path = os.path.join(root, filename)
        
            #Get the file name with the class name, Format the name
            filename = "/".join(file_path.split("/")[1:])

            #Read the image at that location
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            #Add to list
            faces[filename] = img

    return faces


def findMatchingFace(toFind, faces, count):
    #Image size
    L_ImageSize = list(faces.values())[0].shape


    faceMatrix = []
    faceLabel = []

    for key,val in faces.items():
        if key.startswith("s1/"):
            continue # exclude for testing

        if key == 's18/1.pgm':
            continue # exclude for testing
        

        faceMatrix.append(val.flatten())
        faceLabel.append(key.split("/")[0])

    #Create face Matrix as (n_samples x n_pixels) matrix
    faceMatrix = np.array(faceMatrix)

    #Use pca to find eigenfaces
    pca = PCA().fit(faceMatrix)



    #Take the first k principal components as eigenFaces
    KCompoments = 50

    eigenFaces = pca.components_[0:KCompoments]


    #generate weights as KxN matrix - K is the num of eigenFaces and N the num of samples
    meanSubtractedMatrix = faceMatrix - pca.mean_
    weights = eigenFaces @ np.transpose(meanSubtractedMatrix)

    #Testing 
    #Reshape to 1D array
    inputIm = faces[toFind].reshape(1, -1)
    
    #Find input image's weight
    meanSubtractedInput = inputIm - pca.mean_
    inputWeight = eigenFaces @ np.transpose(meanSubtractedInput)
    
    #Find the euclidean Distance to identify the closest matching image
    euclideanDistance = np.linalg.norm(weights - inputWeight, axis = 0)
    

    #set up for finding best match
    best_match = euclideanDistance[0]
    best_matchPos = 0

    #Find the min value in euclideanDistance array
    for i in range (euclideanDistance.shape[0]):
        if best_match > euclideanDistance[i]:
            best_match = euclideanDistance[i]
            best_matchPos = i

    #If the value is so small
    if euclideanDistance[best_matchPos] < 1e-10:
        euclideanDistance[best_matchPos] = 0

    #VISUALISATION
    print(f"The best match for input image is {faceLabel[best_matchPos]} with Euclidean distance {euclideanDistance[best_matchPos]}")
    print("")    

    fig, axes = plt.subplots(1, 2)
    plt.suptitle(f"Test case {count}", fontsize = 20)
    # plt.title(title)
    axes[0].imshow(inputIm.reshape(L_ImageSize), cmap = 'gray')
    axes[0].set_title("Input")

    axes[1].imshow(faceMatrix[best_matchPos].reshape(L_ImageSize), cmap = 'gray')
    axes[1].set_title("Best match")

    plt.show()



def main():
    #Get faces from data folder
    faces = readFiles("faces")

    #Get faces from input folder
    toFind = readFiles("toFind")

    for index, testCase in enumerate(toFind):
        findMatchingFace(testCase, faces, index + 1)

if __name__ == '__main__':
    main()