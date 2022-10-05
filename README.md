# LandClassifier

## What is this program about?
This program is to implement basic remote sensing for land use system.
I also try to use object-oriented paradigm in developing the module.


## Algorithms and concepts
The module comprises a main data structure model called <a href= "https://sanlocoz.github.io/LandClassifier/#LandClassifier.RasterMap"> RasterMap</a>.
>RasterMap is a class that represents raster map with a total pixel of totalX * totalY. Each cell is pixelsizeX by pixelsizeY in size. Each cell can contains multivalue that given in array-like value in **kwargs.

RasterMap is basically like other raster object, where it holds pixel values that are given in dictionary of name of the layer and the raster map that could be supplied through **kwargs. 

<img src="img/1.gif" alt="Raster Map" width="300"/>

*Basic concepts of raster object (ArcGIS Resources)*

## Sample graphical output of the classification

## What to do next?
This raster object is implemented for learning purposes. There are many advanced algorithms that has been implemented in GIS softwares and GDAL codebase.
For learning purposes, one could implements algorithm in raster object such as delineation of catchment or even flood fill algorithm given bounding box in polygon (for example).



## Further information
The detail functionality of the module can be found <a href= "https://sanlocoz.github.io/LandClassifier/"> here</a>.
