
# Python-Remote-Sensing-Scripts

## A set of python scripts for remote sensing processing

### As an example, I will use a UAV-based hyperspectral image of a peatland in central-south Chile. The image have 41 bands (10 nm width) ranging form 480-880 nm and a pixel size of 10 cm. The green dots correspond to plots where measurements of biomass, species composition and carbon stock information were taken:

![alt text](/README/peatland.PNG){:height="600px" width="600px"}

### With our scripts you can first extract the spectra in the plots location in a few seconds

```terminal
python ExtractValues.py -r peatland.tif -s plots.shp -i ID
```

### Check output

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>B1</th>
      <th>B2</th>
      <th>B3</th>
      <th>B4</th>
      <th>B5</th>
      <th>B6</th>
      <th>B7</th>
      <th>B8</th>
      <th>B9</th>
      <th>...</th>
      <th>B32</th>
      <th>B33</th>
      <th>B34</th>
      <th>B35</th>
      <th>B36</th>
      <th>B37</th>
      <th>B38</th>
      <th>B39</th>
      <th>B40</th>
      <th>B41</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1509.813451</td>
      <td>2291.564899</td>
      <td>3109.637130</td>
      <td>3962.194325</td>
      <td>4674.092289</td>
      <td>5177.553225</td>
      <td>5526.183572</td>
      <td>5698.716701</td>
      <td>5730.444486</td>
      <td>...</td>
      <td>15122.021006</td>
      <td>15696.353105</td>
      <td>15488.906034</td>
      <td>15259.373692</td>
      <td>15370.818661</td>
      <td>15988.557692</td>
      <td>16801.212295</td>
      <td>16728.998597</td>
      <td>16846.117603</td>
      <td>17324.593108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1708.011608</td>
      <td>2617.267914</td>
      <td>3579.341584</td>
      <td>4624.986406</td>
      <td>5542.596761</td>
      <td>6221.865500</td>
      <td>6708.542956</td>
      <td>6970.212103</td>
      <td>7036.433086</td>
      <td>...</td>
      <td>20393.572983</td>
      <td>20991.396063</td>
      <td>20751.881674</td>
      <td>20494.922936</td>
      <td>20547.978802</td>
      <td>21333.448745</td>
      <td>21717.872021</td>
      <td>21661.447804</td>
      <td>21420.735874</td>
      <td>21518.602349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>12.875854</td>
      <td>211.439792</td>
      <td>1201.264700</td>
      <td>2444.936965</td>
      <td>3603.781002</td>
      <td>4468.655196</td>
      <td>5054.639060</td>
      <td>5373.096807</td>
      <td>5479.100261</td>
      <td>...</td>
      <td>40287.680262</td>
      <td>40283.248111</td>
      <td>39078.153197</td>
      <td>37895.060076</td>
      <td>38651.953385</td>
      <td>39584.352791</td>
      <td>40869.073804</td>
      <td>39726.095768</td>
      <td>39378.668239</td>
      <td>37695.083542</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>214.950019</td>
      <td>852.952500</td>
      <td>1827.763722</td>
      <td>2946.394303</td>
      <td>3962.429581</td>
      <td>4700.627478</td>
      <td>5189.396823</td>
      <td>5428.162923</td>
      <td>5479.461922</td>
      <td>...</td>
      <td>16865.061517</td>
      <td>17382.712572</td>
      <td>17022.812516</td>
      <td>16595.132046</td>
      <td>16618.242789</td>
      <td>17052.205065</td>
      <td>17653.929336</td>
      <td>17440.521948</td>
      <td>17410.552106</td>
      <td>17704.214614</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>2704.663992</td>
      <td>3474.524823</td>
      <td>4291.852040</td>
      <td>5139.117787</td>
      <td>5834.414719</td>
      <td>6324.879534</td>
      <td>6673.001772</td>
      <td>6841.418896</td>
      <td>6859.041616</td>
      <td>...</td>
      <td>19038.206233</td>
      <td>19714.628088</td>
      <td>19361.587963</td>
      <td>19008.547838</td>
      <td>19256.278283</td>
      <td>20024.457928</td>
      <td>20623.883552</td>
      <td>20655.435636</td>
      <td>20865.324231</td>
      <td>20861.972693</td>
    </tr>
  </tbody>
</table>
</div>

 ### You can also perform a MNF transformation of the data. This function have several options, like applying Savitzky Golay filtering and brightness normalization of the spectra. The basic function is like (image resample to 2m in the example):

```terminal
python MNF.py -i peatland.tif 
```

![alt text](/README/MNF.png)

### Get the Gray-Level Co-Occurrence Matrix (GLCM) textures from the image. Use the first MNF component as imput and a moving window of 5 X 5 pixels (default):

```terminal
python GLCM.py -i peatland_MNF.tif  
```

![alt text](/README/GLCM.png)

### Finally, we can also obtain texture information from point clouds (in this case based in the UAV photogrametric point cloud) based on the Canupo algorithm proposed in [This paper](3D terrestrial lidar data classification of complex natural scenes using a multi-scale dimensionality criterion: Applications in geomorphology), also implemented in the [CloudCompare](http://www.danielgm.net/cc/) LiDAR software. Nevertheless, both the paper and the software implemented the transformation to generate poin-based classification while this python script produces texture rasters to be use in any application: 

```terminal
python canupo.py -i lidar.txt -s 1 5 1 -r 1 
# scales: 1,2,3,4,5 m; output resolution 1 m
```

![alt text](/README/canupo.png)
