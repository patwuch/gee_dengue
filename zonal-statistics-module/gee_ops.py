# workflow/gee_ops.py
"""
GEE computation layer
"""
import ee
from datetime import datetime, timedelta



def apply_qa_mask(img, qa_mask_cfg):
    """
    Apply config-driven bit-level quality masking to a single ee.Image.

    qa_mask_cfg schema (from workflow/products.py):
        {
            "band": "<QA band name in the image>",
            "tests": [
                {"start": <lsb>, "end": <msb>, "good_values": [<int>, ...]},
                ...
            ]
        }

    Each test extracts bits [start..end] (both inclusive) and checks that the
    result is one of the listed good_values.  All tests are ANDed together —
    a pixel must pass every test to remain unmasked.

    Examples from products.py:
      MODIS LST  — QC_Day/QC_Night bits 0-1: 00=good, 01=other quality → good_values=[0,1]
      MODIS VI   — SummaryQA bits 0-1: 00=good, 01=marginal → good_values=[0,1]
      Landsat C2 — QA_PIXEL bit 5: 0=not cloud → good_values=[0]
    """
    qa = img.select(qa_mask_cfg["band"])
    mask = None
    for test in qa_mask_cfg["tests"]:
        start_bit = test["start"]
        end_bit   = test["end"]
        good      = test["good_values"]
        n_bits    = end_bit - start_bit + 1
        bit_mask  = (1 << n_bits) - 1
        extracted = qa.rightShift(start_bit).bitwiseAnd(bit_mask)
        # OR together all acceptable values for this bit field.
        test_mask = extracted.eq(good[0])
        for v in good[1:]:
            test_mask = test_mask.Or(extracted.eq(v))
        mask = test_mask if mask is None else mask.And(test_mask)
    return img.updateMask(mask)


def build_multi_ndbi_collection(multi_collections, start, end_dt, region=None):
    collection_list = []

    for sensor in multi_collections:
        # Date windowing
        s_start = sensor["date_start"]
        s_end_date = datetime.strptime(sensor["date_end"], "%Y-%m-%d") + timedelta(days=1)
        s_end = s_end_date.strftime("%Y-%m-%d")
        seg_start = max(start, s_start)
        seg_end = min(end_dt, s_end)
        if seg_start >= seg_end:
            continue

        # Stage 2 — Pre-builder: resolve bitmask integer constants in Python now,
        # before the collection map.  The Python loop runs once here; GEE receives
        # the full bitwise expression chain as a single compiled instruction.
        #
        # Fast path: if every test is a single-bit "must be 0" check, collapse all
        # of them into one combined_clear_mask integer.  On the server this becomes
        # qa.bitwiseAnd(mask).eq(0) — two ops instead of 4×(rightShift+bitwiseAnd+eq)
        # AND-reduced together (~12 ops).
        pre_built_tests = []
        combined_clear_mask = None
        if sensor.get("qa_mask"):
            raw_tests = sensor["qa_mask"]["tests"]
            if all(t["start"] == t["end"] and t["good_values"] == [0] for t in raw_tests):
                combined_clear_mask = 0
                for t in raw_tests:
                    combined_clear_mask |= (1 << t["start"])
            else:
                for test in raw_tests:
                    n_bits = test["end"] - test["start"] + 1
                    pre_built_tests.append({
                        "start":       test["start"],
                        "bit_mask":    (1 << n_bits) - 1,
                        "good_values": test["good_values"],
                    })

        # Stage 1 — Early Selection: keep only the 3 bands needed (SWIR, NIR, QA)
        # before the Aggregator moves data between GEE's internal servers.
        select_bands = [sensor["swir_band"], sensor["nir_band"]]
        qa_band = None
        if sensor.get("qa_mask"):
            qa_band = sensor["qa_mask"]["band"]
            select_bands.append(qa_band)

        seg_col = (
            ee.ImageCollection(sensor["id"])
            .filterDate(seg_start, seg_end)
            .select(select_bands)
        )
        if region:
            seg_col = seg_col.filterBounds(region)

        # Factory function captures sensor-specific values by value, avoiding
        # the Python closure-over-loop-variable bug.
        def _make_to_ndbi(swir_b, nir_b, qa_b, tests, clear_mask, ref_scale, ref_offset):
            def _to_ndbi(img):
                # Apply QA mask — fast path uses a single bitwiseAnd+eq (2 ops);
                # fallback uses the pre-built per-test chain for complex configs.
                if clear_mask is not None:
                    img = img.updateMask(img.select(qa_b).bitwiseAnd(clear_mask).eq(0))
                elif tests:
                    qa = img.select(qa_b)
                    mask = None
                    for t in tests:
                        extracted = qa.rightShift(t["start"]).bitwiseAnd(t["bit_mask"])
                        test_mask = extracted.eq(t["good_values"][0])
                        for v in t["good_values"][1:]:
                            test_mask = test_mask.Or(extracted.eq(v))
                        mask = test_mask if mask is None else mask.And(test_mask)
                    img = img.updateMask(mask)

                # Stage 3 — Math Engine: normalizedDifference is a pre-compiled
                # C++ atomic op on GEE servers; avoids subtract().divide() overhead.
                # Scale/offset only applied for sensors that store packed integers
                # (e.g. Landsat C2 L2); MODIS bands are omitted from the sensor
                # config and left as raw integers (the factor cancels in the ratio).
                swir_ref = img.select(swir_b)
                nir_ref  = img.select(nir_b)
                if ref_scale is not None:
                    swir_ref = swir_ref.multiply(ref_scale).add(ref_offset)
                    nir_ref  = nir_ref.multiply(ref_scale).add(ref_offset)
                ndbi = (
                    ee.Image([swir_ref, nir_ref])
                    .rename(["swir", "nir"])
                    .normalizedDifference(["swir", "nir"])
                    .rename("NDBI")
                )
                return ndbi.copyProperties(img, ["system:time_start", "system:index"])
            return _to_ndbi

        collection_list.append(
            seg_col.map(_make_to_ndbi(
                sensor["swir_band"], sensor["nir_band"], qa_band, pre_built_tests, combined_clear_mask,
                sensor.get("reflectance_scale"), sensor.get("reflectance_offset"),
            ))
        )

    if not collection_list:
        return None

    merged = collection_list[0]
    for col in collection_list[1:]:
        merged = merged.merge(col)

    # Return all scenes directly — no per-date mosaic.
    # Multiple path/row tiles covering the same AOI on the same date each
    # produce a row from reduceRegions; geojson_to_parquet.py collapses them
    # via GROUP BY (region_id, Date) AVG, which is cheaper than building the
    # aggregate_array().distinct().map() computation graph on GEE.
    return merged

def build_reducer(stat_name):
    """Return the EE reducer for a given stat name string."""
    return {
        "SUM":    ee.Reducer.sum(),
        "MEAN":   ee.Reducer.mean(),
        "MAX":    ee.Reducer.max(),
        "MIN":    ee.Reducer.min(),
        "MEDIAN": ee.Reducer.median(),
        "STD":      ee.Reducer.stdDev(),
        "VARIANCE": ee.Reducer.variance(),
    }.get(stat_name.upper(), ee.Reducer.mean())


def build_compound_reducer(stats_list):
    """
    Build a combined reducer for all requested stats.

    Single stat  → returns that reducer directly.
    Multiple stats → combines with sharedInputs=True so all receive the same
    input band and each outputs a separate '{band}_{stat}' property.
    """
    if len(stats_list) == 1:
        return build_reducer(stats_list[0])
    base = build_reducer(stats_list[0])
    for s in stats_list[1:]:
        base = base.combine(build_reducer(s), sharedInputs=True)
    return base


# Stats where applying the same reducer twice (temporal then spatial) produces
# a meaningless result.  For these, the spatial pass always uses mean so that
# e.g. 'sum' yields mean per-pixel cumulative index (region-size-invariant)
# rather than spatial-sum-of-temporal-sums (which scales with pixel count).
# min/max compose correctly (min-of-mins = true overall min) so they are excluded.
_DECOUPLE_SPATIAL_STATS = {"sum", "std", "variance"}


def build_seasonal_stats(collection, regions, scale, stats_list, band, tile_scale=1):
    """
    Compute seasonal (quarterly) zonal statistics with decoupled temporal and
    spatial reducers.

    Temporal pass: stat-specific reducer collapses all scenes in the window
    into one value per pixel.
    Spatial pass: mean for sum/std/variance (so results are region-size-invariant);
    same reducer as temporal for min/max/mean (they compose correctly).
    """
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joiner = ee.Join.saveFirst('_right')

    def _reduce_stat(s):
        temporal_reducer = build_reducer(s)
        spatial_reducer  = ee.Reducer.mean() if s.lower() in _DECOUPLE_SPATIAL_STATS else temporal_reducer
        band_name = f"{band}_{s.lower()}"
        img = collection.reduce(temporal_reducer).rename([band_name])
        return img.reduceRegions(
            collection=regions,
            reducer=spatial_reducer,
            scale=scale,
            tileScale=tile_scale,
        )

    result = _reduce_stat(stats_list[0])
    for s in stats_list[1:]:
        fc = _reduce_stat(s)
        band_name = f"{band}_{s.lower()}"
        joined = joiner.apply(result, fc, join_filter)
        def _copy(f, bn=band_name):
            return f.copyProperties(f.get('_right'), [bn])
        result = joined.map(_copy)
    return result


def build_annual_stats(collection, regions, scale, stats_list, band, tile_scale=1):
    """
    Compute annual zonal statistics using the same reducer for both temporal
    (across images) and spatial (across pixels within a region) aggregation.

    For multiple stats the results are joined on system:index so the returned
    FeatureCollection has one feature per region with all '{band}_{stat}'
    properties present.
    """
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joiner = ee.Join.saveFirst('_right')

    def _reduce_stat(s):
        reducer = build_reducer(s)
        band_name = f"{band}_{s.lower()}"
        img = collection.reduce(reducer).rename([band_name])
        return img.reduceRegions(
            collection=regions,
            reducer=reducer,
            scale=scale,
            tileScale=tile_scale,
        )

    result = _reduce_stat(stats_list[0])
    for s in stats_list[1:]:
        fc = _reduce_stat(s)
        band_name = f"{band}_{s.lower()}"
        joined = joiner.apply(result, fc, join_filter)
        def _copy(f, bn=band_name):
            return f.copyProperties(f.get('_right'), [bn])
        result = joined.map(_copy)
    return result


def build_daily_stats(collection, regions, scale, spatial_reducer, tile_scale=1):
    """
    Reduce each image in the collection over the AOI regions independently,
    tagging each output feature with the image acquisition date.

    Used for both daily cadence (CHIRPS, ERA5) and composite cadence products
    (MODIS 8-day, Landsat 16-day) — each image carries its own system:time_start.

    tile_scale: passed to reduceRegions to control GEE tile size. Higher values
    (e.g. 4) break the computation into smaller tiles GEE can parallelise more
    aggressively — recommended for expensive computed products like NDBI.

    Note: spatial filtering (filterBounds) should be applied to the collection
    before it is passed in, ideally in workflow/products.py where the collection
    is constructed.
    """
    def reduce_image(img):
        date_str = img.date().format("YYYY-MM-dd")
        return img.reduceRegions(
            collection=regions,
            reducer=spatial_reducer,
            scale=scale,
            tileScale=tile_scale,
        ).map(lambda f: f.set("Date", date_str))
    return collection.map(reduce_image).flatten()


def build_histogram_stats(collection, regions, scale, band):
    """
    Compute per-class pixel counts for a categorical band using frequencyHistogram.

    Mosaics the collection to a single image (appropriate for annual products
    like WorldCover and MODIS LULC where only one image covers each year),
    then counts pixels per class value within each AOI region.

    Returns a FeatureCollection where each feature has a property named after
    the band containing a dict of {class_value: pixel_count, ...}.

    Note: spatial filtering (filterBounds) should be applied to the collection
    before it is passed in, ideally in workflow/products.py where the collection
    is constructed.
    """
    image = collection.mosaic().select([band])
    return image.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=scale,
    )


def build_daily_histogram_stats(collection, regions, scale, band, tile_scale=1):
    """
    Compute per-class pixel counts for each image in a daily/composite categorical
    collection. One row per region per image date, tagged with a 'Date' property.

    Use instead of build_histogram_stats when the collection contains multiple
    images (daily or composite cadence) — mosaic would silently discard all but
    the first image per pixel location.
    """
    def reduce_image(img):
        date_str = img.date().format("YYYY-MM-dd")
        return img.select([band]).reduceRegions(
            collection=regions,
            reducer=ee.Reducer.frequencyHistogram(),
            scale=scale,
            tileScale=tile_scale,
        ).map(lambda f: f.set("Date", date_str))
    return collection.map(reduce_image).flatten()