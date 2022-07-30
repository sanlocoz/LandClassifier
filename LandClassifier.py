import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class RasterMap:
    """ RasterMap is a class that represents raster map with a total pixel of totalX * totalY.
        Each cell is pixelsizeX by pixelsizeY in size.
        Each cell can contains multivalue that given in array-like value in **kwargs.
        
        Parameters
        ----------
        totalX : int
            Number of pixels in X axis.
        totalY : int
            Number of pixels in Y axis.
        pixelSizeX : float
            Size of each pixel in X axis.
        pixelSizeY : float
            Size of each pixel in Y axis.
        **kwargs : string -> array-like values (totalX * totalY in dimension)
            Name of the layer or band (key) to the given raster (value), the dimension must be totalX * totalY.

        Attributes
        -------
        totalX : int
            Number of pixels in X axis.
        totalY : int
            Number of pixels in Y axis.
        pixelSizeX : float
            Size of each pixel in X axis.
        pixelSizeY : float
            Size of each pixel in Y axis.
        **kwargs : string -> array-like values (totalX * totalY in dimension)
            Name of the layer or band (key) to the given raster (value).
        """
    
    def __init__(self, totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):

        self.totalX = totalX
        self.totalY = totalY
        self.pixelSizeX = pixelSizeX
        self.pixelSizeY = pixelSizeY
        self.__dict__.update(kwargs)
    
    def NDVI(self, nameNIR, nameRED, show = False):
        """Calculating NDVI (Normalized Difference Vegetation Index) value given NIR and RED value.

        Parameters
        ----------
        nameNIR : str
            Name of the NIR band in RasterMap.
        nameRED : str
            Name of the RED band in RasterMap.
        show : boolean, optional
            Show KDE plot of NDVI in seaborn. The default is False.

        Returns
        -------
        np.array
            NDVI value in a numpy-array.
        np.nan
            If there is no such nameNIR or nameRED band in the RasterMap instance.
        """
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if('NIR' in locals() and 'RED' in locals()):
            NDVI = (NIR-RED)/(NIR+RED)
            
            if(show):
                fig = sns.kdeplot(data = NDVI.reshape(np.shape(NDVI)[0] * np.shape(NDVI)[0]), shade=True)
                fig.figure.suptitle("NDVI Distribution", fontsize = 20)
                plt.xlabel('NDVI', fontsize=18)
                plt.ylabel('Distribution', fontsize=16)
                
            return NDVI
        else:
            return np.nan
        
    def LandClustering(self, nameNIR, nameRED, outputName, nClusters = 3):
        """LandClustering is a method to cluster each cells based on NIR and RED to nClusters usually 3 (vegetation, water and soil).
        KMeans clustering is used to cluster scatter plot of nameNIR and nameRED.
        
        Parameters
        ----------
        nameNIR : str
            Name of the NIR band in RasterMap.
        nameRED : str
            Name of the RED band in RasterMap.
        outputName : str
            Name of the layer in RasterMap output.
        nClusters : int
            Number of distinct clusters assigned.

        Returns
        -------
        RasterMap 
            If nameNIR and nameRED is defined return a RasterMap with outputName attribute that is filled with 0 to nClusters - 1 value.
        np.nan
            If there is no such nameNIR or nameRED band in the RasterMap instance.
        """
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if('NIR' in locals() and 'RED' in locals()):
            size = self.totalX * self.totalY
            
            NIR1D = NIR.reshape([size])
            RED1D = RED.reshape([size])
            
            X = []
            for i in range(0, size):
                X.append([NIR1D[i],RED1D[i]])
            
            kmeans = KMeans(n_clusters=nClusters, random_state = 0).fit(X)
            landUse = kmeans.labels_.reshape([self.totalX, self.totalY])
            kwargs = {}
            kwargs[outputName] = landUse
        
            return RasterMap(self.totalX, self.totalY, self.pixelSizeX, self.pixelSizeY, **kwargs)
        else:
            return np.nan
    
    def ClassifySoilVegetationWater (self, nameNIR, nameRED, landCluster, clusterName, soilCode, vegetationCode, waterCode, outputName):
        """ClassifySoilVegetationWater will clasify landCluster that contains 3 distinct values to soil or vegetation or water.\n
        • Water is defined first as the cell which has low reflectance in NIR and RED.\n
        • Vegetation is defined as the cell which has a jump in high NIR to low RED reflectance.\n
        • Soil is defined as the remaining cells.\n
        The value that is used as the parameter is the average of the cells with the same value in clusterName in landCluster argument.
        
        Parameters
        ----------
        nameNIR : str
            Name of the NIR band in RasterMap.
        nameRED : str
            Name of the RED band in RasterMap.
        landCluster : RasterMap
            RasterMap that contains 3 distinct values in clusterName band.
        clusterName : str
            clusterName is the band in landCluster which contains 3 distinct values.
        soilCode : str
            Character of string that describe soil in outputName.
        vegetationCode : str
            Character of string that describe vegetation in outputName.
        waterCode : str
            Character of string that describe vegetation in outputName.
        outputName : str
            Name of the layer in RasterMap output.
            
        Returns
        -------
        RasterMap
            If landCluster contains 3 distinct values and nameNIR, nameRED are defined in the instance. 
            
        np.nan
            If the requirements above is not met. 
        """
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
            land = np.array(landCluster.__getattribute__(clusterName))
        except:
            AttributeError
        
        if('NIR' in locals() and 'RED' in locals() and 'land' in locals()):
            uniqueElement = np.unique(land)
            
            #It must contains 3 elements, we cannot classify <> 3 elements to vegetation, water and soil.
            #If it contains more or less than 3 elements then, we have to manually classifying the raster, by using visualization.
            if(len(uniqueElement) == 3): 
                NIRMean = []
                REDMean = []
                for element in uniqueElement:
                    NIRCurrent = NIR[land == element]
                    REDCurrent = RED[land == element]
                    
                    NIRMean.append(np.mean(NIRCurrent))
                    REDMean.append(np.mean(REDCurrent))
                
                NIRMean = np.array(NIRMean)
                REDMean = np.array(REDMean)
                NIRREDMeanAddition = NIRMean + REDMean
                NIRREDMeanSubstraction = NIRMean - REDMean
                classificationCode = ["","",""]
                
                #Water Classification, the lowest NIRREDMeanAddition value
                classificationCode[NIRREDMeanAddition.argmin()] = waterCode
                
                #Vegetation Classification, the highest NIRREDMeanSubstraction value
                classificationCode[NIRREDMeanSubstraction.argmax()] = vegetationCode
                
                #Soil Classification, the rest index
                for i in range(0, len(uniqueElement)):
                    if(classificationCode[i] == ""):
                        classificationCode[i] = soilCode
                
                landClusterCode = []
                for i in range(0, np.shape(land)[0]):
                    for j in range(0, np.shape(land)[1]):
                        landClusterCode.append(classificationCode[land[i][j]])
                        
                landClusterCode = np.array(landClusterCode).reshape([np.shape(land)[0], np.shape(land)[1]])
                
                kwargs = {}
                kwargs[outputName] = landClusterCode
                return RasterMap(self.totalX, self.totalY, self.pixelSizeX, self.pixelSizeY, **kwargs)
                
        return np.nan

class ConfusionMatrix:
    """Confusion Matrix is a that compares between 2 RasterMap layer, the true value and predicted value.
    True value (mapTrue) and predicted value (mapPredicted) layer have to be the same size in dimension.
    
    Parameters
    ----------
    mapTrue : RasterMap
        RasterMap of true value.
    mapPredicted : RasterMap
        RasterMap of predicted value.
    nameTrue : str
        Name of true value layer in mapTrue.
    namePredicted : str
        Name of predicted value layer in mapPredicted.

    Attributes
    -------
    commissionError_ : ndarray of shape (nClasses_)
        Percentage of commission error for each class.
    confMat_ : ndarray of shape (nClasses_, nClasses_)
        Confusion matrix with dimension of nClasses_ * nClasses_.
    labels_ : ndarray of shape (nClasses_)
        Unique labels in mapTrue and mapPredicted.
    nClasses_ : int
        Size of classes derived from mapTrue and mapPredicted.
    nTrials_ : int
        Number of all pixels in mapTrue or mapPredicted.
    nTrue_ : int
        Number of all matching pixels (correct prediction) of mapTrue and mapPredicted.
    omissionError_ : ndarray of shape (nClasses_)
        Percentage of omission error for each class.
    overallAccuracy_ : float
        Percentage of true prediction over all trials (nTrue_ / nTrials_).
    """
    
    def __init__(self, mapTrue, mapPredicted, nameTrue, namePredicted):
        yTrue = mapTrue.__getattribute__(nameTrue)
        yPred = mapPredicted.__getattribute__(namePredicted)
        
        self.nTrials_ = yTrue.shape[0] * yTrue.shape[1]
        
        yTrue = yTrue.reshape(yTrue.shape[0] * yTrue.shape[1])
        yPred = yPred.reshape(yPred.shape[0] * yPred.shape[1])

        self.labels_ = np.unique(np.concatenate((yTrue, yPred)))
        self.confMat_ = confusion_matrix(yTrue, yPred, labels = self.labels_)
        self.nClasses_ = np.shape(self.labels_)[0]
        
        self.nTrue_ = 0
        for i in range(self.nClasses_):
            self.nTrue_ += self.confMat_[i][i]
            
        self.overallAccuracy_ = self.nTrue_ / self.nTrials_

        self.omissionError_ = []
        for i in range(self.nClasses_):
            correctPrediction = 0
            falsePrediction = 0
            for j in range(self.nClasses_):
                if(i != j):
                    falsePrediction += self.confMat_[i][j]
                else:
                    correctPrediction += self.confMat_[i][j]
            self.omissionError_.append(falsePrediction / (falsePrediction + correctPrediction))

        self.omissionError_ = np.array(self.omissionError_)
        
        self.commissionError_ = []
        for i in range(self.nClasses_):
            correctPrediction = 0
            falsePrediction = 0
            for j in range(self.nClasses_):
                if(i != j):
                    falsePrediction += self.confMat_[j][i]
                else:
                    correctPrediction += self.confMat_[j][i]
            self.commissionError_.append(falsePrediction / (falsePrediction + correctPrediction))

        self.commissionError_ = np.array(self.commissionError_)
        
    def DisplayConfusionMatrix(self):
        """Displaying the confusion matrix.
        
        Returns
        -------
        None.
        """
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confMat_, display_labels=self.labels_)
        disp.plot()
        plt.show()

def ClassifiedNDVIPlot(NDVI, landUseMap, landUseLayer, KDE = True, histogram = False, swarmPlot = False):
    """Showing KDE, histogram or swarm plot of the given NDVI for each corresponding land use.
    
    Parameters
    ----------
    NDVI : 2D array-like values
        2D array matrix that contains NDVI values for corresponding landUseMap layer.
    landUseMap : RasterMap
        RasterMap that contains semantic value of land use in landUseLayer.
    landUseLayer : str
        Name of layer in landUseMap that contains semantic value of land use.
    KDE : boolean, optional
        Showing KDE plot with categorical data of land use. The default is True.
    histogram : boolean, optional
        Showing histogram plot with categorical data of land use. The default is False.
    swarmPlot : boolean, optional
        Showing swarm plot with categorical data of land use. The default is False.

    Returns
    -------
    None.
    """
    NDVIFlattened = np.array(NDVI)
    NDVIFLattened = NDVIFlattened.reshape(np.shape(NDVIFlattened)[0] * np.shape(NDVIFlattened)[1])
    landUseFlattened = np.array(landUseMap.__getattribute__(landUseLayer)) 
    landUseFlattened = landUseFlattened.reshape(np.shape(landUseFlattened)[0] * np.shape(landUseFlattened)[1])
    
    data = np.array([NDVIFLattened, landUseFlattened])
    data = data.transpose()
    data = pd.DataFrame(data, columns = ["NDVI", "landUse"])
    data.NDVI = pd.to_numeric(data.NDVI)
    
    if(KDE):
        plt.figure()
        sns.displot(data = data, x = "NDVI", hue = "landUse", kind = "kde")
    
    if(histogram):
        plt.figure()
        sns.displot(data = data, x = "NDVI", hue = "landUse", kind = "hist")
    
    if(swarmPlot):
        plt.figure()
        sns.swarmplot(x = data['landUse'], y = data['NDVI'])
    
def GenerateRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):
    """Generating random RasterMap, with the upper and lower-bound of the band given in **kwargs.\n
    Used for testing only.
    
    Parameters
    ----------
    totalX : int
        Number of pixels in X axis.
    totalY : int
        Number of pixels in Y axis.
    pixelSizeX : float
        Size of each pixel in X axis.
    pixelSizeY : float
        Size of each pixel in Y axis.
    **kwargs : string -> (float,float)
        Pair of nameBand -> (lowerBound, upperBound)\n
        Creating RasterMap with given each band consisting of nameBand within the range of lowerBound and upperBound, inclusive.

    Returns
    -------
    RasterMap
        RasterMap with bands defined in **kwargs and each pixel of the band consisting the value between lowerBound and upperBound, inclusive.
    """
    
    nkwargs = {}
    for nameBand in kwargs:
        nkwargs[nameBand] = np.random.uniform(kwargs[nameBand][0],kwargs[nameBand][1],[totalX,totalY])
    
    return RasterMap(totalX, totalY, pixelSizeX, pixelSizeY, **nkwargs)

def GenerateWaterSoilVegetationRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, soilCode, vegetationCode, waterCode, nameLand, nameNIR, nameRED):
    """Generating Random RasterMap with soilCode, vegetationCode, and waterCode value that is saved in nameLand layer.
    NIR and RED value is assigned in nameNIR and nameRED layer based on this following criteria:\n
    • Water, NIR <= 0.1 and RED <= 0.1\n
    • Vegetation, 0.3 <= NIR <= 0.8 and RED <= 0.1\n
    • Soil,  0.20 <= NIR <= 0.40 and 0.20 <= RED <= 0.40\n
    Used for testing only.
    
    Parameters
    ----------
    totalX : int
        Number of pixels in X axis.
    totalY : int
        Number of pixels in Y axis.
    pixelSizeX : float
        Size of each pixel in X axis.
    pixelSizeY : float
        Size of each pixel in Y axis.
    soilCode : str
        Character of string that describe soil in nameLand.
    vegetationCode : str
        Character of string that describe vegetation in nameLand.
    waterCode : str
        Character of string that describe vegetation in nameLand.
    nameLAND : str
        Name of layer that contains soilCode, vegetationCode and waterCode.
    nameNIR : str
        Name of layer that contains NIR value.
    nameRED : str
        Name of layer that contains RED value.

    Returns
    -------
    RasterMap
        RasterMap with nameLand, nameNIR and nameRED layers.
    """
    code = [soilCode, vegetationCode, waterCode]
    lowerNIR = [0.20, 0.30, 0.00]
    upperNIR = [0.40, 0.80, 0.10]
    lowerRED = [0.20, 0.00, 0.00]
    upperRED = [0.40, 0.10, 0.10]
    
    land = np.random.randint(0, len(code), [totalX, totalY])
    landCode = []
    NIR = []
    RED = []
    
    for i in range(0, np.shape(land)[0]):
        for j in range(0, np.shape(land)[1]):
            landCode.append(code[land[i][j]])
            NIR.append(np.random.uniform(lowerNIR[land[i][j]], upperNIR[land[i][j]]))
            RED.append(np.random.uniform(lowerRED[land[i][j]], upperRED[land[i][j]]))
    
    landCode = np.array(landCode).reshape((totalX, totalY))
    NIR = np.array(NIR).reshape((totalX, totalY))
    RED = np.array(RED).reshape((totalX, totalY))
    
    kwargs = {}
    kwargs[nameLand] = landCode
    kwargs[nameNIR] = NIR
    kwargs[nameRED] = RED
    
    return RasterMap(totalX, totalY, pixelSizeX, pixelSizeY, **kwargs)

#Main program

#sampleMap1
#Definition of nkwargs in RasterMap
nkwargs = {"ref800nm": [[0.5,0.2,0.1,0.3],
                       [0.5,0.2,0.1,0.3]],
           "ref680nm": [[0.3,0.1,0.6,0.4],
                       [0.2,0.3,0.7,0.6]]}
sampleMap1 = RasterMap(4, 2, 1, 1, **nkwargs) #sampleMap1 from nkwargs

#sampleMap2
#Definition of kwargs in GenerateRandomMap (including lower and upper bound for each band)
kwargs = {"ref800nm": (0,1), "ref680nm":(0,1)}
sampleMap2 = GenerateRandomMap(50, 50, 4, 4, **kwargs) #sampleMap2 from kwargs using GenerateRandomMap function
sampleMap2NDVI = sampleMap2.NDVI("ref800nm","ref680nm", show = False) #NDVI value from sampleMap2

#sampleMap3
sampleMap3 = GenerateWaterSoilVegetationRandomMap(10, 10, 4, 4, "sol", "veg", "wtr", "landUse", "ref800nm", "ref680nm") #sampleMap3 from GenerateWaterSoilVegetationRandomMap function
sampleMap3NDVI = sampleMap3.NDVI("ref800nm","ref680nm", show = True) #NDVI value from sampleMap3
landCluster = sampleMap3.LandClustering("ref800nm","ref680nm", "land") #Clustering Map from LandClustering method
landClusterCode  = sampleMap3.ClassifySoilVegetationWater("ref800nm", "ref680nm", landCluster, "land", "sol", "veg", "wtr", "landCode") #Determining soil, vegetation and water
confMatrix = ConfusionMatrix(sampleMap3, landClusterCode, "landUse", "landCode") #Calculating confusion matrix and its attribut of sampleMap3
confMatrix.DisplayConfusionMatrix() #Displaying confustion matrix
ClassifiedNDVIPlot(sampleMap3NDVI, sampleMap3, "landUse", KDE = True, histogram = True, swarmPlot = True) #Displaying KDE, histogram and swarmPlot of NDVI