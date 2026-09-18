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

# Sentinel-2 L2A Scene Classification Layer values that should not contribute
# to burn-severity estimation.
INVALID_SCL = frozenset({0, 1, 2, 3, 6, 8, 9, 10, 11})
