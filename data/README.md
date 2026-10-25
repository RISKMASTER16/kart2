# Data

The hourly pageview data is not stored in the repository: the full file is around 500 MB, above
GitHub's file-size limit. Recreate it with one of:

```bash
python download.py --start "2026-10-18 00:00" --end "2026-10-23 02:00"   # the period used in our results
python make_fake_data.py                                                  # small synthetic data set
```

| File | Contents |
|---|---|
| `views.parquet` | One row per page per hour: `hour`, `page`, `views` (format in [INTERFACES.md](../INTERFACES.md#2-dataviewsparquet)) |
| `missing_hours.txt` | Hours that could not be downloaded, one per line (empty when none are missing) |
| `hours/` | Per-hour files written while downloading; merged into `views.parquet` |

Source: [Wikimedia hourly pageviews](https://dumps.wikimedia.org/other/pageviews/), released under CC0.
Run `python checks.py` after downloading to report missing hours, incomplete hours and suspected bot traffic.
