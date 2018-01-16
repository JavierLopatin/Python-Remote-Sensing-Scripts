
# Python-Remote-Sensing-Scripts

## A set of python scripts for remote sensing processing

### As an example, I will use a UAV-based hyperspectral image of a peatland in central-south Chile. The image have 41 bands (10 nm width) ranging form 480-880 nm and a pixel size of 10 cm. The green dots correspond to plots where measurements of biomass, species composition and carbon stock information were taken:

<img src="peatland.png", alt="Drawing" style="width: 600px;"/>

### With our scripts you can first extract the spectra in the plots location in a few seconds

```terminal
python ExtractValues.py -r peatland.tif -s plots.shp -i ID
```


```python
## Che
import pandas as pd

df = pd.read_csv("plots.csv")
df[:5]
```

![alt text](https://github.com/JavierLopatin/Python-Remote-Sensing-Scripts/tree/master/README/peatland.PNG)

 ### You can also perform a MNF transformation of the data. This function have several options, like applying Savitzky Golay filtering and brightness normalization of the spectra. The basic function is like (image resample to 2m in the example):

```terminal
python MNF.py -i peatland.tif 
```

<img src="MNF.png", alt="Drawing" style="width: 600px;"/>

### Get the Gray-Level Co-Occurrence Matrix (GLCM) textures from the image. Use the first MNF component as imput and a moving window of 5 X 5 pixels (default):

```terminal
python GLCM.py -i peatland_MNF.tif  
```

<img src="GLCM.png", alt="Drawing" style="width: 600px;"/>

### Finally, we can also obtain texture information from point clouds (in this case based in the UAV photogrametric point cloud) based on the Canupo algorithm proposed in [This paper](3D terrestrial lidar data classification of complex natural scenes using a multi-scale dimensionality criterion: Applications in geomorphology), also implemented in the [CloudCompare](http://www.danielgm.net/cc/) LiDAR software. Nevertheless, both the paper and the software implemented the transformation to generate poin-based classification while this python script produces texture rasters to be use in any application: 

```terminal
python canupo.py -i lidar.txt -s 1 5 1 -r 1 
# scales: 1,2,3,4,5 m; output resolution 1 m
```

<img src="canupo.png", alt="Drawing" style="width: 600px;"/>
