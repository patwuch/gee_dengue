# workflow/products.py

PRODUCT_REGISTRY: dict[str, dict] = {
    "CHIRPS": {
        "ee_collection": "UCSB-CHG/CHIRPS/DAILY",
        "min_date": "1981-01-01",
        "max_date": "2026-02-28",
        "scale": 5566,
        "cadence": "daily",
        "categorical": False,
        "content": {
            "precipitation": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
        },
        "label": "CHIRPS Daily Precipitation",
        "description": "Global precipitation (0.05° resolution).",
        "resolution_m": 5566,
    },
    "ERA5_LAND": {
        "ee_collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "min_date": "1950-01-01",
        "max_date": "2026-02-28",
        "scale": 11132,
        "cadence": "daily",
        "categorical": False,
        "content": {
            "temperature_2m":               {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"], "band_transform": {"scale": 1.0, "offset": -273.15}},
            "temperature_2m_min":           {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"], "band_transform": {"scale": 1.0, "offset": -273.15}},
            "temperature_2m_max":           {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"], "band_transform": {"scale": 1.0, "offset": -273.15}},
            "total_precipitation_sum":      {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "total_evaporation_sum":        {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "potential_evaporation_sum":    {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "snow_evaporation_sum":         {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "evaporation_from_bare_soil_sum": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "evaporation_from_open_water_surfaces_excluding_oceans_sum": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "evaporation_from_the_top_of_canopy_sum": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
            "evaporation_from_vegetation_transpiration_sum": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
        },
        "label": "ERA5-Land Climate",
        "description": "Reanalysis data for land variables (0.1° / ~11km native resolution).",
        "resolution_m": 11132,
    },
    "MODIS_LST": {
        "ee_collection": "MODIS/061/MOD11A2",
        "min_date": "2000-02-18",
        "max_date": "2026-02-10",
        "scale": 1000,
        "cadence": "composite",
        "categorical": False,
        "content": {
            "LST_Day_1km": {
                "stats": ["sum", "mean", "min", "max", "std"],
                "default_stats": ["sum", "mean", "min", "max", "std"],
                # MOD11A2 QC_Day bits 0–1: 00=good, 01=other quality — keep both.
                "qa_mask": {
                    "band": "QC_Day",
                    "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}],
                },
                # Raw DN → °C: value × 0.02 gives Kelvin; subtract 273.15 for Celsius.
                "band_transform": {"scale": 0.02, "offset": -273.15},
            },
            "LST_Night_1km": {
                "stats": ["sum", "mean", "min", "max", "std"],
                "default_stats": ["sum", "mean", "min", "max", "std"],
                # MOD11A2 QC_Night bits 0–1: same encoding as QC_Day.
                "qa_mask": {
                    "band": "QC_Night",
                    "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}],
                },
                # Raw DN → °C: value × 0.02 gives Kelvin; subtract 273.15 for Celsius.
                "band_transform": {"scale": 0.02, "offset": -273.15},
            },
            "LST_Mean": {
                "stats": ["sum", "mean", "min", "max", "std"],
                "default_stats": ["mean", "min", "max", "std"],
                # Pixel-wise average of Day and Night; only valid where both pass QA.
                "band_compute": {
                    "type": "mean",
                    "input_bands": ["LST_Day_1km", "LST_Night_1km"],
                    "input_qa_masks": [
                        {"band": "QC_Day",   "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}]},
                        {"band": "QC_Night", "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}]},
                    ],
                },
                # Raw DN → °C applied after averaging.
                "band_transform": {"scale": 0.02, "offset": -273.15},
            },
        },
        "label": "MODIS Land Surface Temperature",
        "description": "Land Surface Temperature (1km resolution).",
        "resolution_m": 1000,
    },
    "MODIS_NDVI_EVI": {
        "ee_collection": "MODIS/061/MOD13Q1",
        "min_date": "2000-02-18",
        "max_date": "2026-02-02",
        "scale": 250,
        "cadence": "composite",
        "categorical": False,
        "content": {
            "NDVI": {
                "stats": ["sum", "mean", "min", "max", "std"],
                "default_stats": ["sum", "mean", "min", "max", "std"],
                # MOD13Q1 SummaryQA bits 0–1: 00=good, 01=marginal — keep both.
                "qa_mask": {
                    "band": "SummaryQA",
                    "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}],
                },
                # MOD13Q1 raw values are scaled integers; multiply by 0.0001 to get [-1, 1].
                "band_transform": {"scale": 0.0001, "offset": 0.0},
            },
            "EVI": {
                "stats": ["sum", "mean", "min", "max", "std"],
                "default_stats": ["sum", "mean", "min", "max", "std"],
                "qa_mask": {
                    "band": "SummaryQA",
                    "tests": [{"start": 0, "end": 1, "good_values": [0, 1]}],
                },
                # MOD13Q1 raw values are scaled integers; multiply by 0.0001 to get [-1, 1].
                "band_transform": {"scale": 0.0001, "offset": 0.0},
            },
        },
        "label": "MODIS NDVI / EVI",
        "description": "Vegetation Indices (250m resolution).",
        "resolution_m": 250,
    },
    "WorldCover_v100": {
        "ee_collection": "ESA/WorldCover/v100",
        "min_date": "2020-01-01",
        "max_date": "2020-12-31",
        "scale": 10,
        "cadence": "annual",
        "categorical": True,
        "normalize_histogram": True,
        "content": {
            "Map": {"stats": ["histogram"], "default_stats": ["histogram"]},
        },
        "label": "ESA WorldCover 2020 (v1.0)",
        "description": "ESA WorldCover landcover classification v100 (10m, 2020), adjust in 'products.py' if too complex.",
        "resolution_m": 10,
    },
    "WorldCover_v200": {
        "ee_collection": "ESA/WorldCover/v200",
        "min_date": "2021-01-01",
        "max_date": "2021-12-31",
        "scale": 10,
        "cadence": "annual",
        "categorical": True,
        "normalize_histogram": True,
        "content": {
            "Map": {"stats": ["histogram"], "default_stats": ["histogram"]},
        },
        "label": "ESA WorldCover 2021 (v2.0)",
        "description": "ESA WorldCover landcover classification v200 (10m, 2021), adjust in 'products.py' if too complex",
        "resolution_m": 10,
    },
    "MODIS_LULC": {
        "ee_collection": "MODIS/061/MCD12Q1",
        "min_date": "2001-01-01",
        "max_date": "2023-12-31",
        "scale": 500,
        "cadence": "annual",
        "categorical": True,
        "normalize_histogram": True,
        "content": {
            "LC_Type1": {"stats": ["histogram"], "default_stats": ["histogram"]},
            "LC_Type2": {"stats": ["histogram"], "default_stats": ["histogram"]},
            "LC_Type3": {"stats": ["histogram"], "default_stats": ["histogram"]},
            "LC_Type4": {"stats": ["histogram"], "default_stats": ["histogram"]},
            "LC_Type5": {"stats": ["histogram"], "default_stats": ["histogram"]},
        },
        "label": "MODIS Land Use / Land Cover",
        "description": "MODIS Land Cover Type (500m resolution, annual, multiple schemes), adjust in 'products.py' if too complex",
        "resolution_m": 500,
    },
    "Dynamic World V1": {
        "ee_collection": "GOOGLE/DYNAMICWORLD/V1",
        "min_date": "2015-06-27",
        "max_date": "2026-04-27",
        "scale": 10,
        "cadence": "daily",
        "categorical": True,
        "normalize_histogram": True,
        "content": {
            "label": {"stats": ["histogram"], "default_stats": ["histogram"]}},
        "label": "Dynamic World v1 Land Use / Land Cover",
        "description": "Dynamic World v1 landcover classification (10m resolution, daily), adjust in 'products.py' if too complex",
        "resolution_m": 10
        },
    "WorldPop": {
        "ee_collection": "WorldPop/GP/100m/pop",
        "min_date": "2000-01-01",
        "max_date": "2020-12-31",
        "scale": 100,
        "cadence": "annual",
        "categorical": False,
        "content": {
            "population": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
        },
        "label": "WorldPop Population (100m)",
        "description": "Global unconstrained population count (100m resolution, annual 2000–2020).",
        "resolution_m": 100,
    },
    "Landsat NDBI": {
        # No single ee_collection — worker uses multi_collections to build a merged NDBI series.
        # gee_weight=4: computed from 30m Landsat on-the-fly,  more expensive per job than
        # precomputed products.  With gee_concurrency=10, caps concurrent NDBI jobs to
        # floor(10/5)=2 . If still experiencing computational problems, drop GEE worker max count 
        # to 5 so only one NDBI could ever be active at a time + it blocks any other workers from 
        # getting schedule at the same time.
        "gee_weight": 5,
        "tile_scale": 4,
        "ee_collection": None,
        "multi_collections": [
            {
                # LT05 routine operations suspended 2011-11-18 after a power anomaly;
                # no usable imagery after November 2011.
                "id":               "LANDSAT/LT05/C02/T1_L2",
                "date_start":       "1984-01-01",
                "date_end":         "2011-11-30",
                "swir_band":        "SR_B5",
                "nir_band":         "SR_B4",
                # Landsat C2 L2 surface reflectance scale factors (USGS).
                "reflectance_scale":  0.0000275,
                "reflectance_offset": -0.2,
                # Landsat C02 L2 QA_PIXEL: keep pixels where cloud/shadow/snow bits are clear.
                # bit 1=dilated cloud, bit 3=cloud shadow, bit 4=snow, bit 5=cloud — all must be 0.
                "qa_mask": {
                    "band": "QA_PIXEL",
                    "tests": [
                        {"start": 1, "end": 1, "good_values": [0]},
                        {"start": 3, "end": 3, "good_values": [0]},
                        {"start": 4, "end": 4, "good_values": [0]},
                        {"start": 5, "end": 5, "good_values": [0]},
                    ],
                },
            },
            {
                # LE07 (SLC-off since 2003 but usable) bridges LT05 end to LC08 start.
                "id":               "LANDSAT/LE07/C02/T1_L2",
                "date_start":       "2011-12-01",
                "date_end":         "2013-04-10",
                "swir_band":        "SR_B5",
                "nir_band":         "SR_B4",
                "reflectance_scale":  0.0000275,
                "reflectance_offset": -0.2,
                "qa_mask": {
                    "band": "QA_PIXEL",
                    "tests": [
                        {"start": 1, "end": 1, "good_values": [0]},
                        {"start": 3, "end": 3, "good_values": [0]},
                        {"start": 4, "end": 4, "good_values": [0]},
                        {"start": 5, "end": 5, "good_values": [0]},
                    ],
                },
            },
            {
                # LC08 first operational data: 2013-04-11.
                "id":               "LANDSAT/LC08/C02/T1_L2",
                "date_start":       "2013-04-11",
                "date_end":         "2025-12-31",
                "swir_band":        "SR_B6",
                "nir_band":         "SR_B5",
                "reflectance_scale":  0.0000275,
                "reflectance_offset": -0.2,
                "qa_mask": {
                    "band": "QA_PIXEL",
                    "tests": [
                        {"start": 1, "end": 1, "good_values": [0]},
                        {"start": 3, "end": 3, "good_values": [0]},
                        {"start": 4, "end": 4, "good_values": [0]},
                        {"start": 5, "end": 5, "good_values": [0]},
                    ],
                },
            },
        ],
        "min_date": "1984-01-01",
        "max_date": "2025-12-31",
        "scale": 30,
        "cadence": "seasonal",
        "categorical": False,
        "content": {
            "NDBI": {"stats": ["sum", "mean", "min", "max", "std", "variance"], "default_stats": ["sum", "mean", "min", "max", "std", "variance"]},
        },
        "label": "NDBI (Landsat 5 / 7 / 8)",
        "description": (
            "Normalized Difference Built-up Index (30m, per-scene), adjust in 'products.py' if too complex"
            "Landsat 5 TM 1984–Nov 2011, Landsat 7 ETM+ (SLC-off) Dec 2011–Apr 2013, "
            "Landsat 8 OLI Apr 2013–2025. Formula: (SWIR − NIR) / (SWIR + NIR)."
        ),
        "resolution_m": 30,
    },
    "MODIS_NDBI": {
        "gee_weight": 2,
        "tile_scale": 2,
        "ee_collection": None,
        "multi_collections": [
            {
                # Terra — MOD09GA daily surface reflectance, available from 2000-02-24.
                "id":         "MODIS/061/MOD09GA",
                "date_start": "2000-02-24",
                "date_end":   "2025-12-31",
                "swir_band":  "sur_refl_b06",
                "nir_band":   "sur_refl_b02",
                # MOD09GA state_1km bits 0–1: 00=clear, 01=cloudy, 10=mixed, 11=not set.
                # Keep only clear (00).
                "qa_mask": {
                    "band": "state_1km",
                    "tests": [
                        {"start": 0, "end": 1, "good_values": [0]},
                    ],
                },
            },
            {
                # Aqua — MYD09GA daily surface reflectance, available from 2002-07-04.
                # Merged with Terra to increase observation density and reduce cloud gaps.
                "id":         "MODIS/061/MYD09GA",
                "date_start": "2002-07-04",
                "date_end":   "2025-12-31",
                "swir_band":  "sur_refl_b06",
                "nir_band":   "sur_refl_b02",
                "qa_mask": {
                    "band": "state_1km",
                    "tests": [
                        {"start": 0, "end": 1, "good_values": [0]},
                    ],
                },
            },
        ],
        "min_date": "2000-02-24",
        "max_date": "2025-12-31",
        "scale": 500,
        "cadence": "seasonal",
        "categorical": False,
        "content": {
            "NDBI": {
                "stats": ["sum", "mean", "min", "max", "std", "variance"],
                "default_stats": ["sum", "mean", "min", "max", "std", "variance"],
            },
        },
        "label": "NDBI (MODIS Terra + Aqua)",
        "description": (
            "Normalized Difference Built-up Index (500m, per-scene), "
            "merged from MODIS Terra (MOD09GA, 2000–) and Aqua (MYD09GA, 2002–) "
            "daily surface reflectance. Formula: (SWIR − NIR) / (SWIR + NIR)."
        ),
        "resolution_m": 500,
    },
}
