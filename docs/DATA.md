# Dataset and provenance

Original source: [Solar Power Generation Data by Ani Kannal](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data).

The source archive contains generation and weather CSVs for two plants. This version uses only Plant 1. Its generation file has 68,778 inverter-level rows, and its weather file has 3,182 rows. The period represented is May 15–June 17, 2020. Local inspection found 22 inverter identifiers and 3,158 distinct generation timestamps.

The target is the sum of AC_POWER across the complete inverter roster at a timestamp. It is instantaneous/interval-reported power, not accumulated energy. DAILY_YIELD and TOTAL_YIELD are not model inputs. IRRADIATION is used unchanged in source sensor units; do not feed W/m² numbers into this model without confirming the source's scale.

The original page is accessible but its field-level documentation and license text were not extractable in this session. Raw data is therefore excluded from the deliverable. Obtain it from the original provider and check applicable terms before redistributing it. The source data license is separate from this project's MIT code license.

The downloader records the original archive and selected-file SHA-256 hashes in data/raw/provenance.json. A copy of the exact run manifest is included in reports/data_provenance.json. A changed upstream archive may generate different results; compare file hashes when reproducing.

Quality filtering retains 3,057 complete timestamps. 101 generation timestamps are excluded for incomplete coverage; all retained timestamps match weather. This does not imply a continuous 15-minute time series: missing periods remain missing. Training and testing split whole observed days rather than relying on row counts or fixed interval continuity.
