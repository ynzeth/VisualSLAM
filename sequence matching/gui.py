import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

DATASETPATH = "../similarityComputation/slice4/"

def showMatch(databaseId, queryId):
    fig = plt.figure()
    fig.canvas.set_window_title("Database " + str(databaseId) + " -> Query " + str(queryId))

    imageContainer1 = plt.subplot(121)
    imageContainer2 = plt.subplot(122)

    databaseImage = mpimg.imread(DATASETPATH + "database/class1/" + sorted(os.listdir(DATASETPATH + "database/class1/"))[int(databaseId)])
    queryImage = mpimg.imread(DATASETPATH + "query/class1/" + sorted(os.listdir(DATASETPATH + "query/class1/"))[int(queryId)])
    
    _ = imageContainer1.imshow(databaseImage)
    _ = imageContainer2.imshow(queryImage)

    imageContainer1.axis("off")
    imageContainer2.axis("off")
    
    
    plt.show()