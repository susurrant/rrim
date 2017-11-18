# rrim

## Reference
> [Red relief image map: New visualization method for three dimensional data](www.isprs.org/proceedings/XXXVII/congress/2_pdf/11_ThS-6/08.pdf)</br>
> T. Chiba, S. Kaneta, Y. Suzuki, 2008, ISPRS</br>

## Enviroment
· Python: 3.6
· OpenCV: 3.3

## Description
*Generate a rrim image from depth data, such as DEM, raster, point cloud data.*</br>
Example:</br>
DEM data:</br>
![dem](/data/ASTGTM2_N29E111_dem.tif)
rrim image:</br>
![rrim](/data/)

## Parameters
Main function:</br>
    rrim(depth, cellSize, L, output_fname)
· depth
Depth data: numpy.array</br>
· cellSize
The actual distance a cell (pixel, point) cross</br>
· L
The distance within which points are used to compute openness</br>
· output_fname
The filename of the output rrim image</br>