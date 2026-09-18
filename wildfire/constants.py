"""Task constants kept separate from model code."""

AF_CLASS_IDS = (1,)
BS_CLASS_IDS = (1, 2, 3)

AF_SHAPE = (256, 256)
BS_SHAPE = (512, 512)

# ESA WorldCover class ids used only as contextual priors in the deterministic baseline.
LC_TREE = 10
LC_SHRUB = 20
LC_GRASS = 30
LC_CROP = 40
LC_BUILT = 50
LC_BARE = 60
LC_SNOW = 70
LC_WATER = 80
LC_WETLAND = 90
LC_MANGROVE = 95
LC_MOSS = 100

# Sentinel-2 L2A SCL classes. 4/5/7 are the optical classes that reproduce
# the previous deterministic baseline. Cloud/shadow/snow/no-data pixels can
# still be handled by an independently observed SAR fallback when OOF
# calibration proves that useful.
OPTICAL_BASELINE_SCL = frozenset({4, 5, 7})
SCL_CAST_SHADOW = 2
SCL_CLOUD_SHADOW = 3
SCL_WATER = 6
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_SNOW_ICE = 11
