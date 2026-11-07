# Speed benchmark

Machine: x86_64, Python 3.13.18. Peak memory for the whole run: 934 MB.

| Rows | profile() seconds | clean() seconds (includes the re-check) | Issues found |
|---:|---:|---:|---:|
| 10,100 | 0.5 | 0.2 | 10 |
| 101,000 | 2.3 | 1.6 | 10 |
| 505,000 | 10.9 | 8.4 | 12 |
| 1,010,000 | 23.8 | 17.8 | 11 |
