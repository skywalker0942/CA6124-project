# Included datasets

The submission copy includes the raw rating files used by DaisyRec's built-in MovieLens readers:

| Dataset | Included file | Format | Official source |
|---|---|---|---|
| MovieLens 100K | `DaisyRec/data/ml-100k/u.data` | tab-separated user, item, rating, timestamp | https://grouplens.org/datasets/movielens/100k/ |
| MovieLens 1M | `DaisyRec/data/ml-1m/ratings.dat` | `UserID::MovieID::Rating::Timestamp` | https://grouplens.org/datasets/movielens/1m/ |

Only files read by the assignment pipeline are included; the original rating-prediction folds, demographic files, movie metadata, tutorials, and unrelated datasets were omitted. The MovieLens 100K dataset's original README is retained in its data directory. The experiment applies DaisyRec preprocessing (`5filter`, positive rating threshold 4) at runtime.
