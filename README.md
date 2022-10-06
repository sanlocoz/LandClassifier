# LandClassifier

## What is this program about?
This program is to implement basic remote sensing for land use system.
I also try to use object-oriented paradigm in developing the module.


## Algorithms and concepts
The module comprises a main data structure model called <a href= "https://sanlocoz.github.io/LandClassifier/#LandClassifier.RasterMap"> RasterMap</a>.
>RasterMap is a class that represents raster map with a total pixel of totalX * totalY. Each cell is pixelsizeX by pixelsizeY in size. Each cell can contains multivalue that given in array-like value in **kwargs.

RasterMap is basically like other raster object, where it holds pixel values that are given in dictionary of name of the layer and the raster map (that hold the values) that could be supplied through **kwargs. 

<img src="img/1.gif" alt="Raster Map" width="300"/>

*Basic concepts of raster object (ArcGIS Resources)*

In remote sensing of land cover, one of the well known parameters is NDVI (Normalized difference vegetation index).
NDVI use NIR (near-infrared) and RED reflectance to differentiate vegetation and other land cover.

There are differences in terms of distribution of NIR-RED plots for each land cover type. The characteristics of each land cover (soil, vegetation and water bodies) could be seen in the figure below:

<img src="img/2.jpg" alt="Reflectance" width="600"/>

*Reflectance characteristics of soil, vegetation and water bodies (SEOS project)*

From the figure above there are some characteristics that we could derived:
- Water only reflects visible light range and the value is significantly lower than other two land use (soil and vegetation).
- Vegetation has a jump in the reflectance from RED (visible light) to NIR (near infrared). It is because visible blue and red light is used in photosynthesis.
Meanwhile near infrared is significantly reflected because it is not used in the process of photosynthesis.
- Soil reflectance increases slightly from visible light range to near infrared light. 
The reflectance is also dependent on type of the soil itself, moisture content and the mineral it is composed of.

For artificial map generation that is used for testing purposes, the pixel of the map is generated with these underlying criteria:

$$Water \begin{cases}
  NIR \leq 0.1 \\
  RED  \leq 0.1
\end{cases}$$

$$Vegetation \begin{cases}
  0.3 \leq NIR \leq 0.8 \\
  RED  \leq 0.1
\end{cases}$$

$$Soil \begin{cases}
  0.2 \leq NIR \leq 0.4 \\
  0.2 \leq RED \leq 0.4
\end{cases}$$

The implementation is in <a href = "https://sanlocoz.github.io/LandClassifier/#LandClassifier.GenerateWaterSoilVegetationRandomMap">GenerateWaterSoilVegetationRandomMap</a> function within the module.

The map is then classifed with K nearest neighbor algoritm within scikit-learn and confusion matrix is chosen as the error metrics.
The implementation of classifying the land use is given in <a href = "https://sanlocoz.github.io/LandClassifier/#LandClassifier.RasterMap.LandClustering">LandClustering</a> and <a href = "https://sanlocoz.github.io/LandClassifier/#LandClassifier.RasterMap.ClassifySoilVegetationWater">ClassifySoilVegetationWater</a> attributes within <a href= "https://sanlocoz.github.io/LandClassifier/#LandClassifier.RasterMap">RasterMap</a> class.

K nearest neighbor as one of the method in unsupervised learning is used because the characteristics of the problem is similar to what the algorithm results.
From the theory above, we can imagine that the scatter plot of NIR vs RED are separated into 3 main areas where each of them corresponds to water, vegetation and soil land use (It also shown in the figure from the next section).

## Sample graphical output of the classification
<img src="img/5.png" alt="Reflectance" width="300"/>

*3 distinct areas (output of KNN algorithm, where 0 = water, 1 = vegetation, 3 = soil)*

<img src="img/3.png" alt="Reflectance" width="300"/>

*Confusion matrix for error measurement*

<img src="img/4.png" alt="Reflectance" width="300"/>

*NDVI density plot of NDVI*

## What to do next?
This raster object is implemented for learning purposes. There are many advanced algorithms that has been implemented in GIS softwares and GDAL codebase.
For learning purposes, one could implements algorithm in raster object such as delineation of catchment or even flood fill algorithm given bounding box in polygon (for creating ground truthing map).
Other than that algorithm to check if a point is inside a polygon (insideness check) will be useful as shown in the figure below.

<img src="img/6.gif" alt="Reflectance" width="300"/>

*Flood fill algorithm illustration (wikipedia)*

## Further information
The detail functionality of the module can be found <a href= "https://sanlocoz.github.io/LandClassifier/"> here</a>.

## References
SEOS-project.eu. n.d. SEOS Tutorials. [online] Available at: <https://seos-project.eu/> [Accessed March 2022]. 