'''Normalize image intensity for classification'''

import enum
import numpy as np
import skimage
from skimage.exposure import equalize_adapthist

class NormalizeMethod(enum.Enum):
    '''The algorithm to use to normalize image planes'''

    '''Use a local adaptive histogram filter to normalize'''
    EQUALIZE_ADAPTHIST=1,
    '''Rescale to -.5, .5, discarding outliers'''
    RESCALE=2,
    '''Rescale 0-255 to 0-1 and otherwise do no normalization'''
    NONE=3,
    '''Match histogram of tile against that of whole image'''
    MATCH=4,
    '''Match histogram of tile against representative ECS images'''
    MATCH_ECS=5
    '''Match the histogram of 2016_10 / R0'''
    MATCH_ECS_R0=6

def normalize_image_adapthist(img, offset=.5):
    '''Normalize image using a locally adaptive histogram
    
    :param img: image to be normalized
    :returns: normalized image
    '''
    version = tuple(map(int, skimage.__version__.split(".")))
    if version < (0, 12, 0):
        img = img.astype(np.uint16)
    img = equalize_adapthist(img)
    if version < (0, 12, 0):
        # Scale image if prior to 0.12
        imax = img.max()
        imin = img.min()
        img = (img.astype(np.float32) - imin) / \
            (imax - imin + np.finfo(np.float32).eps)
    return img - offset

def normalize_image_rescale(img, saturation_level=0.05, offset=.5):
    '''Normalize the image by rescaling after discaring outliers
    
    :param img: the image to normalize
    :param saturation_level: the fraction of outliers to discard from the
    two extrema
    :param offset: the offset to subtract from the result, scaled to 0-1
    '''
    sortedValues = np.sort( img.ravel())                                        
    minVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])                                                                      
    maxVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])                                                                  
    normImg = np.float32(img - minVal) * \
        (255 / (maxVal-minVal + np.finfo(np.float32).eps))                
    normImg[normImg<0] = 0                                                      
    normImg[normImg>255] = 255                                                  
    return (np.float32(normImg) / 255.0) - offset

#
# The normalization for the histogram matching method was done on the
# Kasthuri AC4 dataset. This will change in the future, but until then,
# these are numbers calculated on AC4
#
'''The fractional cumulative sum of the 256-bin histogram of the AC4 volume'''
uim_quantiles = np.array([
    0.00310061,  0.00440498,  0.00555032,  0.00663368,  0.00767656,
    0.00869805,  0.00970942,  0.01070947,  0.01170988,  0.01271539,
    0.01371903,  0.01472565,  0.01574396,  0.01676395,  0.0177929 ,
    0.01882508,  0.01986628,  0.0209087 ,  0.02196501,  0.02303138,
    0.02410706,  0.02519341,  0.02629363,  0.0274055 ,  0.02853512,
    0.02968522,  0.0308412 ,  0.03200819,  0.03319989,  0.03440769,
    0.03563792,  0.03688377,  0.03816102,  0.03945598,  0.04078296,
    0.04213454,  0.04351143,  0.04491549,  0.0463609 ,  0.04783278,
    0.04934895,  0.05090058,  0.05090058,  0.05248948,  0.054125  ,
    0.0558062 ,  0.05752796,  0.05930098,  0.0611254 ,  0.06299638,
    0.06492598,  0.06691724,  0.06896171,  0.07107004,  0.07324983,
    0.07548998,  0.07780748,  0.08018692,  0.08263973,  0.08516398,
    0.08776816,  0.09044994,  0.09322005,  0.0960726 ,  0.09900747,
    0.10202567,  0.10512624,  0.10830801,  0.11158661,  0.11495885,
    0.11841599,  0.12197106,  0.12560893,  0.12934012,  0.13316071,
    0.13707697,  0.14109526,  0.14518867,  0.14937853,  0.1536528 ,
    0.15802087,  0.16248119,  0.16703333,  0.17166367,  0.17166367,
    0.17638687,  0.18118765,  0.18607435,  0.19105545,  0.19611351,
    0.20125732,  0.20648499,  0.21177473,  0.21714657,  0.22259899,
    0.22812031,  0.23372444,  0.23941671,  0.24516632,  0.25100155,
    0.25690378,  0.26285519,  0.26888803,  0.27499084,  0.28116581,
    0.2874054 ,  0.29371513,  0.30009258,  0.30655565,  0.31306488,
    0.31966661,  0.32630558,  0.33303936,  0.33982559,  0.34667343,
    0.35360199,  0.36058441,  0.3676321 ,  0.37474991,  0.38195053,
    0.38921055,  0.39654446,  0.40393688,  0.41137901,  0.41891357,
    0.42650764,  0.43416054,  0.43416054,  0.44187874,  0.44966515,
    0.45751411,  0.46544174,  0.47341164,  0.48144543,  0.48953384,
    0.49769569,  0.50589523,  0.51414093,  0.5224382 ,  0.53078075,
    0.539189  ,  0.54762028,  0.55610168,  0.56459259,  0.57310806,
    0.5816436 ,  0.5902063 ,  0.59876457,  0.6073621 ,  0.61594475,
    0.62453465,  0.63311203,  0.64168579,  0.65025902,  0.65879616,
    0.66729897,  0.67575653,  0.68416847,  0.69254677,  0.70086403,
    0.70913139,  0.71731743,  0.72544136,  0.73346817,  0.74142937,
    0.74927505,  0.75704323,  0.76470222,  0.77226547,  0.77226547,
    0.77969444,  0.78700714,  0.79421417,  0.80127708,  0.80820435,
    0.81498451,  0.82162949,  0.82811073,  0.83443024,  0.84060791,
    0.84662529,  0.85248856,  0.85819031,  0.86374435,  0.86912201,
    0.87434761,  0.87942329,  0.8843338 ,  0.88909401,  0.89369148,
    0.89815559,  0.90245972,  0.9066111 ,  0.91062599,  0.91448959,
    0.91822899,  0.92181877,  0.92528351,  0.92861458,  0.93181053,
    0.93488434,  0.93782837,  0.94065483,  0.94335876,  0.94595963,
    0.94843666,  0.95080948,  0.95306999,  0.95523628,  0.95729874,
    0.95926689,  0.96115028,  0.96115028,  0.96293602,  0.96465065,
    0.96628387,  0.96785042,  0.96934616,  0.97076797,  0.97213493,
    0.97344238,  0.97469673,  0.97589958,  0.97705765,  0.97817192,
    0.9792392 ,  0.98027466,  0.98127647,  0.98224525,  0.98318115,
    0.98408562,  0.984963  ,  0.98581657,  0.98664909,  0.98746017,
    0.98824966,  0.98901604,  0.98976982,  0.99049309,  0.99120354,
    0.99189186,  0.99256538,  0.9932238 ,  0.993862  ,  0.99447968,
    0.99508301,  0.99566254,  0.99621597,  0.99675003,  0.99725327,
    0.99772911,  0.99816704,  0.9985701 ,  0.99893394,  0.99893394,  1.])
'''the %.1 percentile bin'''
uim_low = 0
'''the 99.9% percentile bin'''
uim_high = 249

ecs_quantiles = np.array([
    0.004030, 0.004309, 0.004309, 0.004605, 0.004605,
    0.004923, 0.004923, 0.004923, 0.005270, 0.005270,
    0.005644, 0.005644, 0.006047, 0.006047, 0.006485,
    0.006485, 0.006956, 0.006956, 0.007463, 0.007463,
    0.008009, 0.008009, 0.008598, 0.008598, 0.009239,
    0.009239, 0.009924, 0.009924, 0.010667, 0.010667,
    0.010667, 0.011471, 0.011471, 0.012340, 0.012340,
    0.013279, 0.013279, 0.014285, 0.014285, 0.015369,
    0.015369, 0.016543, 0.016543, 0.017801, 0.017801,
    0.019158, 0.019158, 0.020619, 0.020619, 0.022194,
    0.022194, 0.023889, 0.023889, 0.023889, 0.025706,
    0.025706, 0.027659, 0.027659, 0.029761, 0.029761,
    0.032015, 0.032015, 0.034427, 0.034427, 0.037004,
    0.037004, 0.039768, 0.039768, 0.042723, 0.042723,
    0.045882, 0.045882, 0.049252, 0.049252, 0.052842,
    0.052842, 0.052842, 0.056661, 0.056661, 0.060722,
    0.060722, 0.065026, 0.065026, 0.069602, 0.069602,
    0.074441, 0.074441, 0.079554, 0.079554, 0.084955,
    0.084955, 0.090656, 0.090656, 0.096653, 0.096653,
    0.102939, 0.102939, 0.109549, 0.109549, 0.109549,
    0.116488, 0.116488, 0.123721, 0.123721, 0.131275,
    0.131275, 0.139132, 0.139132, 0.147320, 0.147320,
    0.155815, 0.155815, 0.164625, 0.164625, 0.173728,
    0.173728, 0.183116, 0.183116, 0.192793, 0.192793,
    0.202747, 0.202747, 0.202747, 0.212931, 0.212931,
    0.223386, 0.223386, 0.234081, 0.234081, 0.244985,
    0.244985, 0.256082, 0.256082, 0.267387, 0.267387,
    0.278854, 0.278854, 0.290503, 0.290503, 0.302296,
    0.302296, 0.314243, 0.314243, 0.314243, 0.326305,
    0.326305, 0.338492, 0.338492, 0.350782, 0.350782,
    0.363186, 0.363186, 0.375712, 0.375712, 0.388344,
    0.388344, 0.401090, 0.401090, 0.413957, 0.413957,
    0.426965, 0.426965, 0.440107, 0.440107, 0.453369,
    0.453369, 0.453369, 0.466788, 0.466788, 0.480387,
    0.480387, 0.494181, 0.494181, 0.508167, 0.508167,
    0.522370, 0.522370, 0.536776, 0.536776, 0.551446,
    0.551446, 0.566340, 0.566340, 0.581491, 0.581491,
    0.596888, 0.596888, 0.612548, 0.612548, 0.612548,
    0.628444, 0.628444, 0.644582, 0.644582, 0.660933,
    0.660933, 0.677469, 0.677469, 0.694168, 0.694168,
    0.711000, 0.711000, 0.727927, 0.727927, 0.744874,
    0.744874, 0.761796, 0.761796, 0.778615, 0.778615,
    0.795256, 0.795256, 0.795256, 0.811664, 0.811664,
    0.827760, 0.827760, 0.843449, 0.843449, 0.858660,
    0.858660, 0.873288, 0.873288, 0.887257, 0.887257,
    0.900504, 0.900504, 0.912960, 0.912960, 0.924559,
    0.924559, 0.935264, 0.935264, 0.945047, 0.945047,
    0.945047, 0.953855, 0.953855, 0.961721, 0.961721,
    0.968647, 0.968647, 0.974633, 0.974633, 0.979747,
    0.979747, 0.984050, 0.984050, 0.987624, 0.987624,
    0.990527, 0.990527, 0.992853, 0.992853, 0.994691,
    1.0])

ecs_quantile_r0=np.array([
    0.004020, 0.004286, 0.004286, 0.004570, 0.004570,
    0.004876, 0.004876, 0.005204, 0.005204, 0.005553,
    0.005553, 0.005925, 0.005925, 0.006329, 0.006329,
    0.006758, 0.006758, 0.007214, 0.007214, 0.007703,
    0.007703, 0.008228, 0.008228, 0.008790, 0.008790,
    0.009393, 0.009393, 0.010035, 0.010035, 0.010721,
    0.010721, 0.011454, 0.011454, 0.012237, 0.012237,
    0.013076, 0.013076, 0.013973, 0.013973, 0.014930,
    0.014930, 0.015948, 0.015948, 0.017037, 0.017037,
    0.018196, 0.018196, 0.019430, 0.019430, 0.020744,
    0.020744, 0.022143, 0.023634, 0.023634, 0.025224,
    0.025224, 0.026911, 0.026911, 0.028692, 0.028692,
    0.030578, 0.030578, 0.032584, 0.032584, 0.034716,
    0.034716, 0.036962, 0.036962, 0.039344, 0.039344,
    0.041862, 0.041862, 0.044523, 0.044523, 0.047322,
    0.047322, 0.050283, 0.050283, 0.053402, 0.053402,
    0.056682, 0.056682, 0.060136, 0.060136, 0.063761,
    0.063761, 0.067577, 0.067577, 0.071568, 0.071568,
    0.075755, 0.075755, 0.080148, 0.080148, 0.084751,
    0.084751, 0.089555, 0.089555, 0.094577, 0.094577,
    0.099811, 0.099811, 0.105270, 0.110949, 0.110949,
    0.116863, 0.116863, 0.123011, 0.123011, 0.129404,
    0.129404, 0.136056, 0.136056, 0.142956, 0.142956,
    0.150086, 0.150086, 0.157479, 0.157479, 0.165125,
    0.165125, 0.173027, 0.173027, 0.181178, 0.181178,
    0.189574, 0.189574, 0.198220, 0.198220, 0.207106,
    0.207106, 0.216249, 0.216249, 0.225625, 0.225625,
    0.235229, 0.235229, 0.245075, 0.245075, 0.255140,
    0.255140, 0.265431, 0.265431, 0.275929, 0.275929,
    0.286630, 0.286630, 0.297538, 0.297538, 0.308635,
    0.308635, 0.319928, 0.319928, 0.331416, 0.343062,
    0.343062, 0.354880, 0.354880, 0.366871, 0.366871,
    0.379037, 0.379037, 0.391370, 0.391370, 0.403892,
    0.403892, 0.416572, 0.416572, 0.429446, 0.429446,
    0.442521, 0.442521, 0.455772, 0.455772, 0.469254,
    0.469254, 0.482965, 0.482965, 0.496936, 0.496936,
    0.511170, 0.511170, 0.525682, 0.525682, 0.540506,
    0.540506, 0.555656, 0.555656, 0.571130, 0.571130,
    0.586961, 0.586961, 0.603121, 0.603121, 0.619618,
    0.619618, 0.636464, 0.636464, 0.653631, 0.653631,
    0.671089, 0.671089, 0.688798, 0.688798, 0.706716,
    0.724770, 0.724770, 0.742897, 0.742897, 0.761035,
    0.761035, 0.779067, 0.779067, 0.796911, 0.796911,
    0.814475, 0.814475, 0.831647, 0.831647, 0.848316,
    0.848316, 0.864396, 0.864396, 0.879776, 0.879776,
    0.894365, 0.894365, 0.908061, 0.908061, 0.920785,
    0.920785, 0.932487, 0.932487, 0.943137, 0.943137,
    0.952694, 0.952694, 0.961157, 0.961157, 0.968554,
    0.968554, 0.974905, 0.974905, 0.980286, 0.980286,
    0.984757, 0.984757, 0.988421, 0.988421, 0.991355,
    0.991355, 0.993666, 0.993666, 0.995448, 0.995448,
    1.0])

# The bins at 0.4 %
ecs_low = 57.15
ecs_high = 179.07

ecs_low_r0 = 41
ecs_high_r0 = 171

def normalize_image_match(img, offset=0):
    '''Match individual planes histograms against that of the global dist'''
    result = []
    for plane in img:
        plane = plane.copy()
        plane[plane < int(uim_low)] = int(uim_low)
        plane[plane > int(uim_high)] = int(uim_high)
        plane = ((plane.astype(np.float32) - uim_low) / (uim_high - uim_low)).astype(np.uint8)
        p_bincount = np.bincount(plane.flatten(), minlength=256)
        p_quantiles = \
            np.cumsum(p_bincount).astype(np.float32) / np.prod(plane.shape)
        tbl = np.interp(p_quantiles, uim_quantiles, 
                        np.linspace(-offset, 1-offset, 256).astype(np.float32))
        result.append(tbl[plane])
    
    return np.array(result)

def ecs_stretch(a):
    '''Stretch an array to 0 - 255 with clipping
    
    :param a: an array or matrix
    :returns: the array, clipped to ecs_low and ecs_high, then stretched
              to the range, 0 - 255
    '''
    a = a.copy().astype(np.float64)
    a[a < ecs_low] = ecs_low
    a[a > ecs_high] = ecs_high
    a = 255. * (a - ecs_low) / (ecs_high - ecs_low)
    return a.astype(np.uint8)

def normalize_image_match_ecs(img, offset=0):
    '''Match individual planes against the ECS distribution'''
    result = []

    for plane in img:
        plane = ecs_stretch(plane)
        p_bincount = np.bincount(plane.flatten(), minlength=256)
        p_quantiles = \
            np.cumsum(p_bincount).astype(np.float32) / np.prod(plane.shape)
        tbl = np.interp(p_quantiles, ecs_quantiles, 
                        np.linspace(-offset, 1-offset, 256).astype(np.float32))
        result.append(tbl[plane])
    
    return np.array(result)


def ecs_stretch_r0(a):
    '''Stretch an array to 0 - 255 with clipping
    
    :param a: an array or matrix
    :returns: the array, clipped to ecs_low and ecs_high, then stretched
              to the range, 0 - 255
    '''
    a = a.copy().astype(np.float64)
    a[a < ecs_low_r0] = ecs_low_r0
    a[a > ecs_high_r0] = ecs_high_r0
    a = 255. * (a - ecs_low_r0) / (ecs_high_r0 - ecs_low_r0)
    return a.astype(np.uint8)

def normalize_image_match_ecs_r0(img, offset=0):
    '''Match individual planes against the ECS distribution'''
    result = []

    for plane in img:
        plane = ecs_stretch_r0(plane)
        p_bincount = np.bincount(plane.flatten(), minlength=256)
        p_quantiles = \
            np.cumsum(p_bincount).astype(np.float32) / np.prod(plane.shape)
        tbl = np.interp(p_quantiles, ecs_quantile_r0, 
                        np.linspace(-offset, 1-offset, 256).astype(np.float32))
        result.append(tbl[plane])
    
    return np.array(result)    

def normalize_image(img, normalize_method, 
                    saturation_level=0.05,
                    offset=.5):
    '''Normalize an image plane's intensity
    
    :param img: the image to normalize
    :param normalize_method: one of the image normalization enumerations
    :param saturation_level: for the rescaling method, the fraction of outliers
    to discard from the distribution (both min and max).
    :param offset: the offset to subtract.
    '''
    if normalize_method == NormalizeMethod.EQUALIZE_ADAPTHIST:
        return np.array([normalize_image_adapthist(_, offset) for _ in img])
    elif normalize_method == NormalizeMethod.RESCALE:
        return np.array([
            normalize_image_rescale(_, saturation_level, offset) for _ in img])
    elif normalize_method == NormalizeMethod.MATCH:
        return normalize_image_match(img)
    elif normalize_method == NormalizeMethod.MATCH_ECS:
        return normalize_image_match_ecs(img)
    else:
        return img.astype(float) / 255.0


