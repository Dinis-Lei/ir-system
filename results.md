## Before Ranking Implementation

### 4GB RAM available, min_length = 3, no stemmer

|          | Token. Time | Merge Time | Total Time | Temp. Blocks | Index Size | Vocab. Size |
| -------- | ----------- | ---------- | ---------- | ------------ | ---------- | ---------- |
| Tiny     | 73          | 15         | 88         | 1            | 94MB (1 file) | 200276
| Small    | 1648        | 330        | 1978       | 10           | 953MB (2 files) | 810294
| Medium   | 7586        | 600        | 8186       | 33           | 3.2GB (17 files) | 1711539
| Large    | 13077       | 1176       | 14253      | 80           | 6.9GB (36 files) | 2799756

## After Ranking Implementation

### 4GB RAM available, min_length = 3, no stemmer, 8 concurrent processes, lnc.ltc

|          | Token. Time | Merge Time | Total Time | Temp. Blocks | Index Size | Vocab. Size |
| -------- | ----------- | ---------- | ---------- | ------------ | ---------- | ---------- |
| Tiny     | 15          | 88         | 107         | 8            | 148MB | 243724
| Small    | 265        | 1226        | 1522       | 24           | 1.6GB | 944501
| Medium   | 1423        | 3291        | 4878       | 71           | 5.4GB | 1931932
| Large    | 5015       |    9855    | 15223      |   145  |  11.9GB | 3093352

### 4GB RAM available, min_length = 3, no stemmer, 8 concurrent processes, ltc.ltc

|          | Token. Time | Merge Time | Total Time | Temp. Blocks | Index Size | Vocab. Size |
| -------- | ----------- | ---------- | ---------- | ------------ | ---------- | ---------- |
| Tiny     | 47          | 97         | 147         | 8            | 144MB | 243724
| Small    | 595        | 1183        | 1811       | 24           | 1.5GB | 944501
| Medium   |         |         |        |            | | 
| Large    |        |        |       |            |   | 