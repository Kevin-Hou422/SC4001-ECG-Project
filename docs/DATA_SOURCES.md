# Data sources, attribution and terms

This repository redistributes an **unaltered 44-record subset** of the
MIT-BIH Arrhythmia Database, version 1.0.0, by **George Moody and Roger Mark**,
available through **PhysioNet**:

- Dataset: https://physionet.org/content/mitdb/1.0.0/
- Persistent identifier: https://doi.org/10.13026/C2F305
- File source: https://physionet.org/files/mitdb/1.0.0/
- Dataset terms: **Open Data Commons Attribution License v1.0 (ODC-By)**,
  https://opendatacommons.org/licenses/by/1-0/
- Publisher checksum list: `data/raw/mitdb/SHA256SUMS.txt`.
- Mirrored-file checksum list and source URL: `data/raw/mitdb/download_manifest.json`.

The included `.hea`, `.dat` and `.atr` files are original source files, verified
against the publisher's SHA-256 values. They have not been relabelled or edited.
The source licence and attribution apply to the dataset; this repository does not
claim to have collected the data or apply a different licence to these files.
When redistributing the raw data or derived datasets, retain this attribution and
the link to its original terms. No additional restriction is imposed on the data.

The full database has 48 recordings from 47 subjects. This project omits paced
records 102, 104, 107 and 217. The original records contain two leads; processing
uses MLII only. Records 201 and 202 originate from the same person. See the
original record directory for source limitations:
https://physionet.org/physiobank/database/html/mitdbdir/records.htm

The generated NPZs are **derived project datasets**: MLII segmentation,
four-class mapping, exclusions, splitting, normalisation and RR feature extraction
are performed by `src/data.py`. Those transformations and their parameters are
documented in README.md and the generated manifest. Report the derived counts,
not merely counts copied from an unrelated implementation.

## References to cite in the report

1. Moody, G., & Mark, R. (2005). *MIT-BIH Arrhythmia Database* (version 1.0.0).
   PhysioNet. https://doi.org/10.13026/C2F305
2. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
   Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
   https://doi.org/10.1109/51.932724
3. de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification
   of heartbeats using ECG morphology and heartbeat interval features.
   *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206.
   https://doi.org/10.1109/TBME.2004.827359

Reference 3 motivates the conventional split; the strict variant is explicitly
a project modification and should not be described as the unmodified protocol.
