
###############################################################################
#
# Vegetation indices for Google Earth Engine Python API
#
###############################################################################

import ee

# Functions to derive vegetation indices and other raster operations
def NDVI(image):
    return image.normalizedDifference(['B5', 'B4'])

def SAM(image):
    band1 = image.select("B1")
    bandn = image.select("B2","B3","B4","B5","B6","B7","B8","B9");
    maxObjSize = 256;
    b = band1.divide(bandn);
    spectralAngleMap = b.atan();
    spectralAngleMap_sin = spectralAngleMap.sin();
    spectralAngleMap_cos = spectralAngleMap.cos();
    sum_cos = spectralAngleMap_cos.reduce(ee.call("Reducer.sum"));
    sum_sin = spectralAngleMap_sin.reduce(ee.call("Reducer.sum"));
    return ee.Image.cat(sum_sin, sum_cos, spectralAngleMap_sin, spectralAngleMap_cos);

#Enhanced Vegetation Index
def EVI(image):
    # L(Canopy background)
    # C1,C2(Coefficients of aerosol resistance term)
    # GainFactor(Gain or scaling factor)
    gain_factor = ee.Image(2.5);
    coefficient_1 = ee.Image(6);
    coefficient_2 = ee.Image(7.5);
    l = ee.Image(1);
    nir = image.select("B5");
    red = image.select("B4");
    blue = image.select("B2");
    evi = image.expression(
        "Gain_Factor*((NIR-RED)/(NIR+C1*RED-C2*BLUE+L))",
        {
            "Gain_Factor":gain_factor,
            "NIR":nir,
            "RED":red,
            "C1":coefficient_1,
            "C2":coefficient_2,
            "BLUE":blue,
            "L":l
        }
    )
    return evi

#Atmospherically Resistant Vegetation Index
def ARVI(image):
    red = image.select("B4")
    blue = image.select("B2")
    nir = image.select("B5")
    red_square = red.multiply(red)
    arvi = image.expression(
        "NIR - (REDsq - BLUE)/(NIR+(REDsq-BLUE))",{
            "NIR": nir,
            "REDsq": red_square,
            "BLUE": blue
        }
    )
    return arvi

#Leaf Area Index
def LAI(image):
    nir = image.select("B5")
    red = image.select("B4")
    coeff1 = ee.Image(0.0305);
    coeff2 = ee.Image(1.2640);
    lai = image.expression(
        "(((NIR/RED)*COEFF1)+COEFF2)",
        {
            "NIR":nir,
            "RED":red,
            "COEFF1":coeff1,
            "COEFF2":coeff2
        }
    )
    return lai

def tasseled_cap_transformation(image):
    #Tasseled Cap Transformation for Landsat 8 based on the
    #scientfic work "Derivation of a tasselled cap transformation based on Landsat 8 at-satellite reflectance"
    #by M.Baigab, L.Zhang, T.Shuai & Q.Tong (2014). The bands of the output image are the brightness index,
    #greenness index and wetness index.
    b = image.select("B2", "B3", "B4", "B5", "B6", "B7");
    #Coefficients are only for Landsat 8 TOA
    brightness_coefficents= ee.Image([0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872])
    greenness_coefficents= ee.Image([-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]);
    wetness_coefficents= ee.Image([0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]);
    fourth_coefficents= ee.Image([-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773]);
    fifth_coefficents= ee.Image([-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085]);
    sixth_coefficents= ee.Image([0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252]);

    #Calculate tasseled cap transformation
    brightness = image.expression(
        '(B * BRIGHTNESS)',
        {
            'B':b,
            'BRIGHTNESS': brightness_coefficents
        })
    greenness = image.expression(
        '(B * GREENNESS)',
        {
            'B':b,
            'GREENNESS': greenness_coefficents
        })
    wetness = image.expression(
        '(B * WETNESS)',
        {
            'B':b,
            'WETNESS': wetness_coefficents
        })
    fourth = image.expression(
        '(B * FOURTH)',
        {
            'B':b,
            'FOURTH': fourth_coefficents
        })
    fifth = image.expression(
        '(B * FIFTH)',
        {
            'B':b,
            'FIFTH': fifth_coefficents
        })
    sixth = image.expression(
        '(B * SIXTH)',
        {
            'B':b,
            'SIXTH': sixth_coefficents
        })
    bright = brightness.reduce(ee.call("Reducer.sum"));
    green = greenness.reduce(ee.call("Reducer.sum"));
    wet = wetness.reduce(ee.call("Reducer.sum"));
    four = fourth.reduce(ee.call("Reducer.sum"));
    five = fifth.reduce(ee.call("Reducer.sum"));
    six = sixth.reduce(ee.call("Reducer.sum"));
    tasseled_cap = ee.Image(bright).addBands(green).addBands(wet).addBands(four).addBands(five).addBands(six)
    return tasseled_cap.rename('brightness','greenness','wetness','fourth','fifth','sixth')
