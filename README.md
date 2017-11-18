# rrim

## Reference
> [Red relief image map: New visualization method for three dimensional data](http://www.isprs.org/proceedings/XXXVII/congress/2_pdf/11_ThS-6/08.pdf)</br>
> T. Chiba, S. Kaneta, Y. Suzuki, 2008, ISPRS</br>

## Enviroment
- Python: 3.6
- OpenCV: 3.3

## Description
**Generate a rrim image from depth data, such as DEM, raster, point cloud data.**</br>
Example:</br>
DEM data:</br>
![dem](/data/ASTGTM2_N29E111_dem_review.jpg)
RRIM image:</br>
![rrim](/data/ASTGTM2_N29E111_dem_rrim.png)
A more detailed image of the northwestern part:</br>
![detail](/data/ASTGTM2_N29E111_dem_rrim_detail.png)

## Parameters
Main function:</br>

    rrim(depth, cellSize, L, output_fname, color_size=(50, 50, 3))

- depth</br>
numpy.array: Depth data</br>
- cellSize</br>
float: The actual distance a cell (pixel, point) cross</br>
- L</br>
float: The distance within which points are used to compute openness</br>
- output_fname</br>
string: The filename of the output rrim image</br>
- color_size</br>
numpy.array, shape = (int, int, 3): The range of color corresponding to the ranges of slope and openness</br>