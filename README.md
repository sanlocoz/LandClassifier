# LandClassifier
Classify land use using K Nearest Neighbors Algorithm

<header>
<h1 class="title">Module <code>LandClassifier</code></h1>
</header>
<section id="section-intro">
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class RasterMap:
    &#34;&#34;&#34; RasterMap is a class that represents raster map with a total pixel of totalX * totalY.
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
        **kwargs : string -&gt; array-like values (totalX * totalY in dimension)
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
        **kwargs : string -&gt; array-like values (totalX * totalY in dimension)
            Name of the layer or band (key) to the given raster (value).
        &#34;&#34;&#34;
    
    def __init__(self, totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):

        self.totalX = totalX
        self.totalY = totalY
        self.pixelSizeX = pixelSizeX
        self.pixelSizeY = pixelSizeY
        self.__dict__.update(kwargs)
    
    def NDVI(self, nameNIR, nameRED, show = False):
        &#34;&#34;&#34;Calculating NDVI (Normalized Difference Vegetation Index) value given NIR and RED value.

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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
            NDVI = (NIR-RED)/(NIR+RED)
            
            if(show):
                fig = sns.kdeplot(data = NDVI.reshape(np.shape(NDVI)[0] * np.shape(NDVI)[0]), shade=True)
                fig.figure.suptitle(&#34;NDVI Distribution&#34;, fontsize = 20)
                plt.xlabel(&#39;NDVI&#39;, fontsize=18)
                plt.ylabel(&#39;Distribution&#39;, fontsize=16)
                
            return NDVI
        else:
            return np.nan
        
    def LandClustering(self, nameNIR, nameRED, outputName, nClusters = 3):
        &#34;&#34;&#34;LandClustering is a method to cluster each cells based on NIR and RED to nClusters usually 3 (vegetation, water and soil).
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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
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
        &#34;&#34;&#34;ClassifySoilVegetationWater will clasify landCluster that contains 3 distinct values to soil or vegetation or water.\n
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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
            land = np.array(landCluster.__getattribute__(clusterName))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals() and &#39;land&#39; in locals()):
            uniqueElement = np.unique(land)
            
            #It must contains 3 elements, we cannot classify &lt;&gt; 3 elements to vegetation, water and soil.
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
                classificationCode = [&#34;&#34;,&#34;&#34;,&#34;&#34;]
                
                #Water Classification, the lowest NIRREDMeanAddition value
                classificationCode[NIRREDMeanAddition.argmin()] = waterCode
                
                #Vegetation Classification, the highest NIRREDMeanSubstraction value
                classificationCode[NIRREDMeanSubstraction.argmax()] = vegetationCode
                
                #Soil Classification, the rest index
                for i in range(0, len(uniqueElement)):
                    if(classificationCode[i] == &#34;&#34;):
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
    &#34;&#34;&#34;Confusion Matrix is a that compares between 2 RasterMap layer, the true value and predicted value.
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
    &#34;&#34;&#34;
    
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
        &#34;&#34;&#34;Displaying the confusion matrix.
        
        Returns
        -------
        None.
        &#34;&#34;&#34;
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confMat_, display_labels=self.labels_)
        disp.plot()
        plt.show()

def ClassifiedNDVIPlot(NDVI, landUseMap, landUseLayer, KDE = True, histogram = False, swarmPlot = False):
    &#34;&#34;&#34;Showing KDE, histogram or swarm plot of the given NDVI for each corresponding land use.
    
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
    &#34;&#34;&#34;
    NDVIFlattened = np.array(NDVI)
    NDVIFLattened = NDVIFlattened.reshape(np.shape(NDVIFlattened)[0] * np.shape(NDVIFlattened)[1])
    landUseFlattened = np.array(landUseMap.__getattribute__(landUseLayer)) 
    landUseFlattened = landUseFlattened.reshape(np.shape(landUseFlattened)[0] * np.shape(landUseFlattened)[1])
    
    data = np.array([NDVIFLattened, landUseFlattened])
    data = data.transpose()
    data = pd.DataFrame(data, columns = [&#34;NDVI&#34;, &#34;landUse&#34;])
    data.NDVI = pd.to_numeric(data.NDVI)
    
    if(KDE):
        plt.figure()
        sns.displot(data = data, x = &#34;NDVI&#34;, hue = &#34;landUse&#34;, kind = &#34;kde&#34;)
    
    if(histogram):
        plt.figure()
        sns.displot(data = data, x = &#34;NDVI&#34;, hue = &#34;landUse&#34;, kind = &#34;hist&#34;)
    
    if(swarmPlot):
        plt.figure()
        sns.swarmplot(x = data[&#39;landUse&#39;], y = data[&#39;NDVI&#39;])
    
def GenerateRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):
    &#34;&#34;&#34;Generating random RasterMap, with the upper and lower-bound of the band given in **kwargs.\n
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
    **kwargs : string -&gt; (float,float)
        Pair of nameBand -&gt; (lowerBound, upperBound)\n
        Creating RasterMap with given each band consisting of nameBand within the range of lowerBound and upperBound, inclusive.

    Returns
    -------
    RasterMap
        RasterMap with bands defined in **kwargs and each pixel of the band consisting the value between lowerBound and upperBound, inclusive.
    &#34;&#34;&#34;
    
    nkwargs = {}
    for nameBand in kwargs:
        nkwargs[nameBand] = np.random.uniform(kwargs[nameBand][0],kwargs[nameBand][1],[totalX,totalY])
    
    return RasterMap(totalX, totalY, pixelSizeX, pixelSizeY, **nkwargs)

def GenerateWaterSoilVegetationRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, soilCode, vegetationCode, waterCode, nameLand, nameNIR, nameRED):
    &#34;&#34;&#34;Generating Random RasterMap with soilCode, vegetationCode, and waterCode value that is saved in nameLand layer.
    NIR and RED value is assigned in nameNIR and nameRED layer based on this following criteria:\n
    • Water, NIR &lt;= 0.1 and RED &lt;= 0.1\n
    • Vegetation, 0.3 &lt;= NIR &lt;= 0.8 and RED &lt;= 0.1\n
    • Soil,  0.20 &lt;= NIR &lt;= 0.40 and 0.20 &lt;= RED &lt;= 0.40\n
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
    &#34;&#34;&#34;
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
nkwargs = {&#34;ref800nm&#34;: [[0.5,0.2,0.1,0.3],
                       [0.5,0.2,0.1,0.3]],
           &#34;ref680nm&#34;: [[0.3,0.1,0.6,0.4],
                       [0.2,0.3,0.7,0.6]]}
sampleMap1 = RasterMap(4, 2, 1, 1, **nkwargs) #sampleMap1 from nkwargs

#sampleMap2
#Definition of kwargs in GenerateRandomMap (including lower and upper bound for each band)
kwargs = {&#34;ref800nm&#34;: (0,1), &#34;ref680nm&#34;:(0,1)}
sampleMap2 = GenerateRandomMap(50, 50, 4, 4, **kwargs) #sampleMap2 from kwargs using GenerateRandomMap function
sampleMap2NDVI = sampleMap2.NDVI(&#34;ref800nm&#34;,&#34;ref680nm&#34;, show = False) #NDVI value from sampleMap2

#sampleMap3
sampleMap3 = GenerateWaterSoilVegetationRandomMap(10, 10, 4, 4, &#34;sol&#34;, &#34;veg&#34;, &#34;wtr&#34;, &#34;landUse&#34;, &#34;ref800nm&#34;, &#34;ref680nm&#34;) #sampleMap3 from GenerateWaterSoilVegetationRandomMap function
sampleMap3NDVI = sampleMap3.NDVI(&#34;ref800nm&#34;,&#34;ref680nm&#34;, show = True) #NDVI value from sampleMap3
landCluster = sampleMap3.LandClustering(&#34;ref800nm&#34;,&#34;ref680nm&#34;, &#34;land&#34;) #Clustering Map from LandClustering method
landClusterCode  = sampleMap3.ClassifySoilVegetationWater(&#34;ref800nm&#34;, &#34;ref680nm&#34;, landCluster, &#34;land&#34;, &#34;sol&#34;, &#34;veg&#34;, &#34;wtr&#34;, &#34;landCode&#34;) #Determining soil, vegetation and water
confMatrix = ConfusionMatrix(sampleMap3, landClusterCode, &#34;landUse&#34;, &#34;landCode&#34;) #Calculating confusion matrix and its attribut of sampleMap3
confMatrix.DisplayConfusionMatrix() #Displaying confustion matrix
ClassifiedNDVIPlot(sampleMap3NDVI, sampleMap3, &#34;landUse&#34;, KDE = True, histogram = True, swarmPlot = True) #Displaying KDE, histogram and swarmPlot of NDVI</code></pre>
</details>
</section>
<section>
</section>
<section>
</section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="LandClassifier.ClassifiedNDVIPlot"><code class="name flex">
<span>def <span class="ident">ClassifiedNDVIPlot</span></span>(<span>NDVI, landUseMap, landUseLayer, KDE=True, histogram=False, swarmPlot=False)</span>
</code></dt>
<dd>
<div class="desc"><p>Showing KDE, histogram or swarm plot of the given NDVI for each corresponding land use.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>NDVI</code></strong> :&ensp;<code>2D array-like values</code></dt>
<dd>2D array matrix that contains NDVI values for corresponding landUseMap layer.</dd>
<dt><strong><code>landUseMap</code></strong> :&ensp;<code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap that contains semantic value of land use in landUseLayer.</dd>
<dt><strong><code>landUseLayer</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of layer in landUseMap that contains semantic value of land use.</dd>
<dt><strong><code>KDE</code></strong> :&ensp;<code>boolean</code>, optional</dt>
<dd>Showing KDE plot with categorical data of land use. The default is True.</dd>
<dt><strong><code>histogram</code></strong> :&ensp;<code>boolean</code>, optional</dt>
<dd>Showing histogram plot with categorical data of land use. The default is False.</dd>
<dt><strong><code>swarmPlot</code></strong> :&ensp;<code>boolean</code>, optional</dt>
<dd>Showing swarm plot with categorical data of land use. The default is False.</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def ClassifiedNDVIPlot(NDVI, landUseMap, landUseLayer, KDE = True, histogram = False, swarmPlot = False):
    &#34;&#34;&#34;Showing KDE, histogram or swarm plot of the given NDVI for each corresponding land use.
    
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
    &#34;&#34;&#34;
    NDVIFlattened = np.array(NDVI)
    NDVIFLattened = NDVIFlattened.reshape(np.shape(NDVIFlattened)[0] * np.shape(NDVIFlattened)[1])
    landUseFlattened = np.array(landUseMap.__getattribute__(landUseLayer)) 
    landUseFlattened = landUseFlattened.reshape(np.shape(landUseFlattened)[0] * np.shape(landUseFlattened)[1])
    
    data = np.array([NDVIFLattened, landUseFlattened])
    data = data.transpose()
    data = pd.DataFrame(data, columns = [&#34;NDVI&#34;, &#34;landUse&#34;])
    data.NDVI = pd.to_numeric(data.NDVI)
    
    if(KDE):
        plt.figure()
        sns.displot(data = data, x = &#34;NDVI&#34;, hue = &#34;landUse&#34;, kind = &#34;kde&#34;)
    
    if(histogram):
        plt.figure()
        sns.displot(data = data, x = &#34;NDVI&#34;, hue = &#34;landUse&#34;, kind = &#34;hist&#34;)
    
    if(swarmPlot):
        plt.figure()
        sns.swarmplot(x = data[&#39;landUse&#39;], y = data[&#39;NDVI&#39;])</code></pre>
</details>
</dd>
<dt id="LandClassifier.GenerateRandomMap"><code class="name flex">
<span>def <span class="ident">GenerateRandomMap</span></span>(<span>totalX, totalY, pixelSizeX, pixelSizeY, **kwargs)</span>
</code></dt>
<dd>
<div class="desc"><p>Generating random RasterMap, with the upper and lower-bound of the band given in **kwargs.</p>
<p>Used for testing only.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>totalX</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in X axis.</dd>
<dt><strong><code>totalY</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in Y axis.</dd>
<dt><strong><code>pixelSizeX</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in X axis.</dd>
<dt><strong><code>pixelSizeY</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in Y axis.</dd>
<dt><strong><code>**kwargs</code></strong> :&ensp;<code>string -&gt; (float,float)</code></dt>
<dd>
<p>Pair of nameBand -&gt; (lowerBound, upperBound)</p>
<p>Creating RasterMap with given each band consisting of nameBand within the range of lowerBound and upperBound, inclusive.</p>
</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap with bands defined in **kwargs and each pixel of the band consisting the value between lowerBound and upperBound, inclusive.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def GenerateRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):
    &#34;&#34;&#34;Generating random RasterMap, with the upper and lower-bound of the band given in **kwargs.\n
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
    **kwargs : string -&gt; (float,float)
        Pair of nameBand -&gt; (lowerBound, upperBound)\n
        Creating RasterMap with given each band consisting of nameBand within the range of lowerBound and upperBound, inclusive.

    Returns
    -------
    RasterMap
        RasterMap with bands defined in **kwargs and each pixel of the band consisting the value between lowerBound and upperBound, inclusive.
    &#34;&#34;&#34;
    
    nkwargs = {}
    for nameBand in kwargs:
        nkwargs[nameBand] = np.random.uniform(kwargs[nameBand][0],kwargs[nameBand][1],[totalX,totalY])
    
    return RasterMap(totalX, totalY, pixelSizeX, pixelSizeY, **nkwargs)</code></pre>
</details>
</dd>
<dt id="LandClassifier.GenerateWaterSoilVegetationRandomMap"><code class="name flex">
<span>def <span class="ident">GenerateWaterSoilVegetationRandomMap</span></span>(<span>totalX, totalY, pixelSizeX, pixelSizeY, soilCode, vegetationCode, waterCode, nameLand, nameNIR, nameRED)</span>
</code></dt>
<dd>
<div class="desc"><p>Generating Random RasterMap with soilCode, vegetationCode, and waterCode value that is saved in nameLand layer.
NIR and RED value is assigned in nameNIR and nameRED layer based on this following criteria:</p>
<p>• Water, NIR &lt;= 0.1 and RED &lt;= 0.1</p>
<p>• Vegetation, 0.3 &lt;= NIR &lt;= 0.8 and RED &lt;= 0.1</p>
<p>• Soil,
0.20 &lt;= NIR &lt;= 0.40 and 0.20 &lt;= RED &lt;= 0.40</p>
<p>Used for testing only.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>totalX</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in X axis.</dd>
<dt><strong><code>totalY</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in Y axis.</dd>
<dt><strong><code>pixelSizeX</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in X axis.</dd>
<dt><strong><code>pixelSizeY</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in Y axis.</dd>
<dt><strong><code>soilCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe soil in nameLand.</dd>
<dt><strong><code>vegetationCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe vegetation in nameLand.</dd>
<dt><strong><code>waterCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe vegetation in nameLand.</dd>
<dt><strong><code>nameLAND</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of layer that contains soilCode, vegetationCode and waterCode.</dd>
<dt><strong><code>nameNIR</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of layer that contains NIR value.</dd>
<dt><strong><code>nameRED</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of layer that contains RED value.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap with nameLand, nameNIR and nameRED layers.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def GenerateWaterSoilVegetationRandomMap(totalX, totalY, pixelSizeX, pixelSizeY, soilCode, vegetationCode, waterCode, nameLand, nameNIR, nameRED):
    &#34;&#34;&#34;Generating Random RasterMap with soilCode, vegetationCode, and waterCode value that is saved in nameLand layer.
    NIR and RED value is assigned in nameNIR and nameRED layer based on this following criteria:\n
    • Water, NIR &lt;= 0.1 and RED &lt;= 0.1\n
    • Vegetation, 0.3 &lt;= NIR &lt;= 0.8 and RED &lt;= 0.1\n
    • Soil,  0.20 &lt;= NIR &lt;= 0.40 and 0.20 &lt;= RED &lt;= 0.40\n
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
    &#34;&#34;&#34;
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
    
    return RasterMap(totalX, totalY, pixelSizeX, pixelSizeY, **kwargs)</code></pre>
</details>
</dd>
</dl>
</section>
<section>
<h2 class="section-title" id="header-classes">Classes</h2>
<dl>
<dt id="LandClassifier.ConfusionMatrix"><code class="flex name class">
<span>class <span class="ident">ConfusionMatrix</span></span>
<span>(</span><span>mapTrue, mapPredicted, nameTrue, namePredicted)</span>
</code></dt>
<dd>
<div class="desc"><p>Confusion Matrix is a that compares between 2 RasterMap layer, the true value and predicted value.
True value (mapTrue) and predicted value (mapPredicted) layer have to be the same size in dimension.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>mapTrue</code></strong> :&ensp;<code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap of true value.</dd>
<dt><strong><code>mapPredicted</code></strong> :&ensp;<code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap of predicted value.</dd>
<dt><strong><code>nameTrue</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of true value layer in mapTrue.</dd>
<dt><strong><code>namePredicted</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of predicted value layer in mapPredicted.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>commissionError_</code></strong> :&ensp;<code>ndarray</code> of <code>shape (nClasses_)</code></dt>
<dd>Percentage of commission error for each class.</dd>
<dt><strong><code>confMat_</code></strong> :&ensp;<code>ndarray</code> of <code>shape (nClasses_, nClasses_)</code></dt>
<dd>Confusion matrix with dimension of nClasses_ * nClasses_.</dd>
<dt><strong><code>labels_</code></strong> :&ensp;<code>ndarray</code> of <code>shape (nClasses_)</code></dt>
<dd>Unique labels in mapTrue and mapPredicted.</dd>
<dt><strong><code>nClasses_</code></strong> :&ensp;<code>int</code></dt>
<dd>Size of classes derived from mapTrue and mapPredicted.</dd>
<dt><strong><code>nTrials_</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of all pixels in mapTrue or mapPredicted.</dd>
<dt><strong><code>nTrue_</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of all matching pixels (correct prediction) of mapTrue and mapPredicted.</dd>
<dt><strong><code>omissionError_</code></strong> :&ensp;<code>ndarray</code> of <code>shape (nClasses_)</code></dt>
<dd>Percentage of omission error for each class.</dd>
<dt><strong><code>overallAccuracy_</code></strong> :&ensp;<code>float</code></dt>
<dd>Percentage of true prediction over all trials (nTrue_ / nTrials_).</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">class ConfusionMatrix:
    &#34;&#34;&#34;Confusion Matrix is a that compares between 2 RasterMap layer, the true value and predicted value.
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
    &#34;&#34;&#34;
    
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
        &#34;&#34;&#34;Displaying the confusion matrix.
        
        Returns
        -------
        None.
        &#34;&#34;&#34;
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confMat_, display_labels=self.labels_)
        disp.plot()
        plt.show()</code></pre>
</details>
<h3>Methods</h3>
<dl>
<dt id="LandClassifier.ConfusionMatrix.DisplayConfusionMatrix"><code class="name flex">
<span>def <span class="ident">DisplayConfusionMatrix</span></span>(<span>self)</span>
</code></dt>
<dd>
<div class="desc"><p>Displaying the confusion matrix.</p>
<h2 id="returns">Returns</h2>
<p>None.</p></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def DisplayConfusionMatrix(self):
    &#34;&#34;&#34;Displaying the confusion matrix.
    
    Returns
    -------
    None.
    &#34;&#34;&#34;
    disp = ConfusionMatrixDisplay(confusion_matrix=self.confMat_, display_labels=self.labels_)
    disp.plot()
    plt.show()</code></pre>
</details>
</dd>
</dl>
</dd>
<dt id="LandClassifier.RasterMap"><code class="flex name class">
<span>class <span class="ident">RasterMap</span></span>
<span>(</span><span>totalX, totalY, pixelSizeX, pixelSizeY, **kwargs)</span>
</code></dt>
<dd>
<div class="desc"><p>RasterMap is a class that represents raster map with a total pixel of totalX * totalY.
Each cell is pixelsizeX by pixelsizeY in size.
Each cell can contains multivalue that given in array-like value in **kwargs.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>totalX</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in X axis.</dd>
<dt><strong><code>totalY</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in Y axis.</dd>
<dt><strong><code>pixelSizeX</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in X axis.</dd>
<dt><strong><code>pixelSizeY</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in Y axis.</dd>
<dt><strong><code>**kwargs</code></strong> :&ensp;<code>string -&gt; array-like values (totalX * totalY in dimension)</code></dt>
<dd>Name of the layer or band (key) to the given raster (value), the dimension must be totalX * totalY.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>totalX</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in X axis.</dd>
<dt><strong><code>totalY</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of pixels in Y axis.</dd>
<dt><strong><code>pixelSizeX</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in X axis.</dd>
<dt><strong><code>pixelSizeY</code></strong> :&ensp;<code>float</code></dt>
<dd>Size of each pixel in Y axis.</dd>
<dt><strong><code>**kwargs</code></strong> :&ensp;<code>string -&gt; array-like values (totalX * totalY in dimension)</code></dt>
<dd>Name of the layer or band (key) to the given raster (value).</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">class RasterMap:
    &#34;&#34;&#34; RasterMap is a class that represents raster map with a total pixel of totalX * totalY.
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
        **kwargs : string -&gt; array-like values (totalX * totalY in dimension)
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
        **kwargs : string -&gt; array-like values (totalX * totalY in dimension)
            Name of the layer or band (key) to the given raster (value).
        &#34;&#34;&#34;
    
    def __init__(self, totalX, totalY, pixelSizeX, pixelSizeY, **kwargs):

        self.totalX = totalX
        self.totalY = totalY
        self.pixelSizeX = pixelSizeX
        self.pixelSizeY = pixelSizeY
        self.__dict__.update(kwargs)
    
    def NDVI(self, nameNIR, nameRED, show = False):
        &#34;&#34;&#34;Calculating NDVI (Normalized Difference Vegetation Index) value given NIR and RED value.

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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
            NDVI = (NIR-RED)/(NIR+RED)
            
            if(show):
                fig = sns.kdeplot(data = NDVI.reshape(np.shape(NDVI)[0] * np.shape(NDVI)[0]), shade=True)
                fig.figure.suptitle(&#34;NDVI Distribution&#34;, fontsize = 20)
                plt.xlabel(&#39;NDVI&#39;, fontsize=18)
                plt.ylabel(&#39;Distribution&#39;, fontsize=16)
                
            return NDVI
        else:
            return np.nan
        
    def LandClustering(self, nameNIR, nameRED, outputName, nClusters = 3):
        &#34;&#34;&#34;LandClustering is a method to cluster each cells based on NIR and RED to nClusters usually 3 (vegetation, water and soil).
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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
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
        &#34;&#34;&#34;ClassifySoilVegetationWater will clasify landCluster that contains 3 distinct values to soil or vegetation or water.\n
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
        &#34;&#34;&#34;
        
        try:
            NIR = np.array(self.__getattribute__(nameNIR))
            RED = np.array(self.__getattribute__(nameRED))
            land = np.array(landCluster.__getattribute__(clusterName))
        except:
            AttributeError
        
        if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals() and &#39;land&#39; in locals()):
            uniqueElement = np.unique(land)
            
            #It must contains 3 elements, we cannot classify &lt;&gt; 3 elements to vegetation, water and soil.
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
                classificationCode = [&#34;&#34;,&#34;&#34;,&#34;&#34;]
                
                #Water Classification, the lowest NIRREDMeanAddition value
                classificationCode[NIRREDMeanAddition.argmin()] = waterCode
                
                #Vegetation Classification, the highest NIRREDMeanSubstraction value
                classificationCode[NIRREDMeanSubstraction.argmax()] = vegetationCode
                
                #Soil Classification, the rest index
                for i in range(0, len(uniqueElement)):
                    if(classificationCode[i] == &#34;&#34;):
                        classificationCode[i] = soilCode
                
                landClusterCode = []
                for i in range(0, np.shape(land)[0]):
                    for j in range(0, np.shape(land)[1]):
                        landClusterCode.append(classificationCode[land[i][j]])
                        
                landClusterCode = np.array(landClusterCode).reshape([np.shape(land)[0], np.shape(land)[1]])
                
                kwargs = {}
                kwargs[outputName] = landClusterCode
                return RasterMap(self.totalX, self.totalY, self.pixelSizeX, self.pixelSizeY, **kwargs)
                
        return np.nan</code></pre>
</details>
<h3>Methods</h3>
<dl>
<dt id="LandClassifier.RasterMap.ClassifySoilVegetationWater"><code class="name flex">
<span>def <span class="ident">ClassifySoilVegetationWater</span></span>(<span>self, nameNIR, nameRED, landCluster, clusterName, soilCode, vegetationCode, waterCode, outputName)</span>
</code></dt>
<dd>
<div class="desc"><p>ClassifySoilVegetationWater will clasify landCluster that contains 3 distinct values to soil or vegetation or water.</p>
<p>• Water is defined first as the cell which has low reflectance in NIR and RED.</p>
<p>• Vegetation is defined as the cell which has a jump in high NIR to low RED reflectance.</p>
<p>• Soil is defined as the remaining cells.</p>
<p>The value that is used as the parameter is the average of the cells with the same value in clusterName in landCluster argument.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>nameNIR</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the NIR band in RasterMap.</dd>
<dt><strong><code>nameRED</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the RED band in RasterMap.</dd>
<dt><strong><code>landCluster</code></strong> :&ensp;<code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>RasterMap that contains 3 distinct values in clusterName band.</dd>
<dt><strong><code>clusterName</code></strong> :&ensp;<code>str</code></dt>
<dd>clusterName is the band in landCluster which contains 3 distinct values.</dd>
<dt><strong><code>soilCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe soil in outputName.</dd>
<dt><strong><code>vegetationCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe vegetation in outputName.</dd>
<dt><strong><code>waterCode</code></strong> :&ensp;<code>str</code></dt>
<dd>Character of string that describe vegetation in outputName.</dd>
<dt><strong><code>outputName</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the layer in RasterMap output.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></dt>
<dd>If landCluster contains 3 distinct values and nameNIR, nameRED are defined in the instance.</dd>
<dt><code>np.nan</code></dt>
<dd>If the requirements above is not met.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def ClassifySoilVegetationWater (self, nameNIR, nameRED, landCluster, clusterName, soilCode, vegetationCode, waterCode, outputName):
    &#34;&#34;&#34;ClassifySoilVegetationWater will clasify landCluster that contains 3 distinct values to soil or vegetation or water.\n
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
    &#34;&#34;&#34;
    
    try:
        NIR = np.array(self.__getattribute__(nameNIR))
        RED = np.array(self.__getattribute__(nameRED))
        land = np.array(landCluster.__getattribute__(clusterName))
    except:
        AttributeError
    
    if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals() and &#39;land&#39; in locals()):
        uniqueElement = np.unique(land)
        
        #It must contains 3 elements, we cannot classify &lt;&gt; 3 elements to vegetation, water and soil.
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
            classificationCode = [&#34;&#34;,&#34;&#34;,&#34;&#34;]
            
            #Water Classification, the lowest NIRREDMeanAddition value
            classificationCode[NIRREDMeanAddition.argmin()] = waterCode
            
            #Vegetation Classification, the highest NIRREDMeanSubstraction value
            classificationCode[NIRREDMeanSubstraction.argmax()] = vegetationCode
            
            #Soil Classification, the rest index
            for i in range(0, len(uniqueElement)):
                if(classificationCode[i] == &#34;&#34;):
                    classificationCode[i] = soilCode
            
            landClusterCode = []
            for i in range(0, np.shape(land)[0]):
                for j in range(0, np.shape(land)[1]):
                    landClusterCode.append(classificationCode[land[i][j]])
                    
            landClusterCode = np.array(landClusterCode).reshape([np.shape(land)[0], np.shape(land)[1]])
            
            kwargs = {}
            kwargs[outputName] = landClusterCode
            return RasterMap(self.totalX, self.totalY, self.pixelSizeX, self.pixelSizeY, **kwargs)
            
    return np.nan</code></pre>
</details>
</dd>
<dt id="LandClassifier.RasterMap.LandClustering"><code class="name flex">
<span>def <span class="ident">LandClustering</span></span>(<span>self, nameNIR, nameRED, outputName, nClusters=3)</span>
</code></dt>
<dd>
<div class="desc"><p>LandClustering is a method to cluster each cells based on NIR and RED to nClusters usually 3 (vegetation, water and soil).
KMeans clustering is used to cluster scatter plot of nameNIR and nameRED.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>nameNIR</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the NIR band in RasterMap.</dd>
<dt><strong><code>nameRED</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the RED band in RasterMap.</dd>
<dt><strong><code>outputName</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the layer in RasterMap output.</dd>
<dt><strong><code>nClusters</code></strong> :&ensp;<code>int</code></dt>
<dd>Number of distinct clusters assigned.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a> </code></dt>
<dd>If nameNIR and nameRED is defined return a RasterMap with outputName attribute that is filled with 0 to nClusters - 1 value.</dd>
<dt><code>np.nan</code></dt>
<dd>If there is no such nameNIR or nameRED band in the RasterMap instance.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def LandClustering(self, nameNIR, nameRED, outputName, nClusters = 3):
    &#34;&#34;&#34;LandClustering is a method to cluster each cells based on NIR and RED to nClusters usually 3 (vegetation, water and soil).
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
    &#34;&#34;&#34;
    
    try:
        NIR = np.array(self.__getattribute__(nameNIR))
        RED = np.array(self.__getattribute__(nameRED))
    except:
        AttributeError
    
    if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
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
        return np.nan</code></pre>
</details>
</dd>
<dt id="LandClassifier.RasterMap.NDVI"><code class="name flex">
<span>def <span class="ident">NDVI</span></span>(<span>self, nameNIR, nameRED, show=False)</span>
</code></dt>
<dd>
<div class="desc"><p>Calculating NDVI (Normalized Difference Vegetation Index) value given NIR and RED value.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>nameNIR</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the NIR band in RasterMap.</dd>
<dt><strong><code>nameRED</code></strong> :&ensp;<code>str</code></dt>
<dd>Name of the RED band in RasterMap.</dd>
<dt><strong><code>show</code></strong> :&ensp;<code>boolean</code>, optional</dt>
<dd>Show KDE plot of NDVI in seaborn. The default is False.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>np.array</code></dt>
<dd>NDVI value in a numpy-array.</dd>
<dt><code>np.nan</code></dt>
<dd>If there is no such nameNIR or nameRED band in the RasterMap instance.</dd>
</dl></div>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def NDVI(self, nameNIR, nameRED, show = False):
    &#34;&#34;&#34;Calculating NDVI (Normalized Difference Vegetation Index) value given NIR and RED value.

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
    &#34;&#34;&#34;
    
    try:
        NIR = np.array(self.__getattribute__(nameNIR))
        RED = np.array(self.__getattribute__(nameRED))
    except:
        AttributeError
    
    if(&#39;NIR&#39; in locals() and &#39;RED&#39; in locals()):
        NDVI = (NIR-RED)/(NIR+RED)
        
        if(show):
            fig = sns.kdeplot(data = NDVI.reshape(np.shape(NDVI)[0] * np.shape(NDVI)[0]), shade=True)
            fig.figure.suptitle(&#34;NDVI Distribution&#34;, fontsize = 20)
            plt.xlabel(&#39;NDVI&#39;, fontsize=18)
            plt.ylabel(&#39;Distribution&#39;, fontsize=16)
            
        return NDVI
    else:
        return np.nan</code></pre>
</details>
</dd>
</dl>
</dd>
</dl>
</section>
</article>
<nav id="sidebar">
<h1>Index</h1>
<div class="toc">
<ul></ul>
</div>
<ul id="index">
<li><h3><a href="#header-functions">Functions</a></h3>
<ul class="">
<li><code><a title="LandClassifier.ClassifiedNDVIPlot" href="#LandClassifier.ClassifiedNDVIPlot">ClassifiedNDVIPlot</a></code></li>
<li><code><a title="LandClassifier.GenerateRandomMap" href="#LandClassifier.GenerateRandomMap">GenerateRandomMap</a></code></li>
<li><code><a title="LandClassifier.GenerateWaterSoilVegetationRandomMap" href="#LandClassifier.GenerateWaterSoilVegetationRandomMap">GenerateWaterSoilVegetationRandomMap</a></code></li>
</ul>
</li>
<li><h3><a href="#header-classes">Classes</a></h3>
<ul>
<li>
<h4><code><a title="LandClassifier.ConfusionMatrix" href="#LandClassifier.ConfusionMatrix">ConfusionMatrix</a></code></h4>
<ul class="">
<li><code><a title="LandClassifier.ConfusionMatrix.DisplayConfusionMatrix" href="#LandClassifier.ConfusionMatrix.DisplayConfusionMatrix">DisplayConfusionMatrix</a></code></li>
</ul>
</li>
<li>
<h4><code><a title="LandClassifier.RasterMap" href="#LandClassifier.RasterMap">RasterMap</a></code></h4>
<ul class="">
<li><code><a title="LandClassifier.RasterMap.ClassifySoilVegetationWater" href="#LandClassifier.RasterMap.ClassifySoilVegetationWater">ClassifySoilVegetationWater</a></code></li>
<li><code><a title="LandClassifier.RasterMap.LandClustering" href="#LandClassifier.RasterMap.LandClustering">LandClustering</a></code></li>
<li><code><a title="LandClassifier.RasterMap.NDVI" href="#LandClassifier.RasterMap.NDVI">NDVI</a></code></li>
</ul>
</li>
</ul>
</li>
</ul>
</nav>
</main>
<footer id="footer">
<p>Generated by <a href="https://pdoc3.github.io/pdoc" title="pdoc: Python API documentation generator"><cite>pdoc</cite> 0.10.0</a>.</p>
</footer>
</body>
</html>
