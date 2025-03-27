# GEE_Dengue_ML documentation!

## Description

A short description of the project.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://gee_dengue_tmu/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://gee_dengue_tmu/data/` to `data/`.


