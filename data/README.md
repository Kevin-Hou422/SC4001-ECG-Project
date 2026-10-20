# Dataset files

`raw/mitdb/` contains 44 original MIT-BIH waveform/header/annotation triplets
(132 files, approximately 86 MB), plus publisher SHA256 checksums and a manifest.
The four paced records excluded by the project plan are not mirrored.

Source: MIT-BIH Arrhythmia Database v1.0.0, George Moody and Roger Mark,
PhysioNet, https://doi.org/10.13026/C2F305.
Terms: Open Data Commons Attribution License v1.0,
https://opendatacommons.org/licenses/by/1-0/.
See [complete attribution and references](../docs/DATA_SOURCES.md).

Clone the repository to obtain the raw files. The Colab notebook verifies every
raw file and generates the `.npz` files in `processed/`. These large derived
files are ignored by Git and can be regenerated or shared using Google Drive.
Their lightweight class statistics, manifest and audit summary are versioned.
