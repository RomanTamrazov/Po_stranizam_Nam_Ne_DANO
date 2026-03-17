import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRanker, Pool

HAS_IMPLICIT = False
BASE_DIR = "." # Сюда можете поставить ваш путь так как данный код выполнен был в colab
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBMIT_DIR = os.path.join(BASE_DIR, "submit")

def load_data():
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"), parse_dates=["event_ts"])
    editions = pd.read_csv(os.path.join(DATA_DIR, "editions.csv"))
    book_genres = pd.read_csv(os.path.join(DATA_DIR, "book_genres.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    candidates = pd.read_csv(os.path.join(SUBMIT_DIR, "candidates.csv"))
    targets = pd.read_csv(os.path.join(SUBMIT_DIR, "targets.csv"))
    return interactions, editions, book_genres, users, candidates, targets

def preprocess_text(editions, n_components=20, max_features=5000):
    ed = editions.copy()
    ed["text"] = (ed["title"].fillna("") + " " + ed["description"].fillna("")).str.strip()
    tfidf = TfidfVectorizer(max_features=max_features, min_df=2, ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(ed["text"].values)
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=927)
    text_emb = svd.fit_transform(tfidf_matrix)
    emb_cols = [f"text_svd_{i}" for i in range(n_comp)]
    df_emb = pd.DataFrame(text_emb, columns=emb_cols)
    df_emb["edition_id"] = ed["edition_id"].values
    return df_emb, emb_cols

def build_svd_cf(interactions, n_factors=48, use_log=True):
    print(f"  Building SVD CF (n_factors={n_factors}, log={use_log})...")
    inter = interactions.copy()
    inter["w"] = inter["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    agg = inter.groupby(["user_id", "edition_id"])["w"].sum().reset_index()

    user_ids = sorted(agg["user_id"].unique())
    item_ids = sorted(agg["edition_id"].unique())
    uid_map = {u: i for i, u in enumerate(user_ids)}
    iid_map = {e: i for i, e in enumerate(item_ids)}

    rows = agg["user_id"].map(uid_map).values
    cols = agg["edition_id"].map(iid_map).values
    vals = np.log1p(agg["w"].values) if use_log else agg["w"].values

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    k = min(n_factors, min(len(user_ids), len(item_ids)) - 1)
    if k <= 0:
        k = 1
    U, S, Vt = svds(matrix.astype(np.float64), k=k)

    idx = np.argsort(-S)
    U, S, Vt = U[:, idx], S[idx], Vt[idx, :]
    sqrt_S = np.sqrt(S)
    user_emb = U * sqrt_S[np.newaxis, :]
    item_emb = Vt.T * sqrt_S[np.newaxis, :]

    user_emb_dict = {uid: user_emb[i] for uid, i in uid_map.items()}
    item_emb_dict = {eid: item_emb[i] for eid, i in iid_map.items()}

    print(f"  SVD done: {len(user_ids)} users, {len(item_ids)} items, {k} factors")
    return user_emb_dict, item_emb_dict, k

def build_item_stats(interactions):
    inter = interactions.copy()
    inter["w"] = inter["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
    stats = inter.groupby("edition_id").agg(
        item_pop=("w", "sum"),
        item_events=("w", "count"),
        item_read_cnt=("event_type", lambda x: (x == 2).sum()),
        item_wish_cnt=("event_type", lambda x: (x == 1).sum()),
        item_rating_mean=("rating", "mean"),
        item_rating_cnt=("rating", lambda x: x.notna().sum()),
        item_rating_std=("rating", "std"),
    ).reset_index()
    stats["item_read_ratio"] = stats["item_read_cnt"] / (stats["item_events"] + 1e-9)
    C = stats["item_rating_mean"].mean()
    m = 5
    stats["item_bayesian_rating"] = (
        (stats["item_rating_cnt"] * stats["item_rating_mean"] + m * C)
        / (stats["item_rating_cnt"] + m)
    )
    return stats

def build_user_stats(interactions):
    inter = interactions.copy()
    inter["w"] = inter["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
    stats = inter.groupby("user_id").agg(
        user_events=("w", "count"),
        user_read_cnt=("event_type", lambda x: (x == 2).sum()),
        user_wish_cnt=("event_type", lambda x: (x == 1).sum()),
        user_rating_mean=("rating", "mean"),
        user_rating_cnt=("rating", lambda x: x.notna().sum()),
        user_rating_std=("rating", "std"),
        user_last_ts=("event_ts", "max"),
    ).reset_index()
    stats["user_read_ratio"] = stats["user_read_cnt"] / (stats["user_events"] + 1e-9)
    item_pop = interactions.groupby("edition_id").size().reset_index(name="ipop")
    user_item_pop = interactions.merge(item_pop, on="edition_id", how="left")
    user_avg_pop = user_item_pop.groupby("user_id")["ipop"].mean().reset_index(name="user_avg_item_pop")
    stats = stats.merge(user_avg_pop, on="user_id", how="left")
    return stats

def build_genre_features(interactions, editions, book_genres):
    ed_to_book = editions[["edition_id", "book_id"]].copy()
    user_items = interactions[["user_id", "edition_id", "event_type"]].copy()
    user_items = user_items.merge(ed_to_book, on="edition_id", how="left")
    user_items = user_items.merge(book_genres, on="book_id", how="left")

    user_genre_div = user_items.groupby("user_id")["genre_id"].nunique().reset_index()
    user_genre_div.columns = ["user_id", "user_genre_diversity"]

    user_items["gw"] = user_items["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    user_genre_weighted = user_items.groupby(["user_id", "genre_id"])["gw"].sum().reset_index(name="genre_weight")

    user_top = user_genre_weighted.sort_values(["user_id", "genre_weight"], ascending=[True, False])
    user_top = user_top.groupby("user_id").head(5)
    user_top_genre_sets = user_top.groupby("user_id")["genre_id"].apply(set).to_dict()

    user_total = user_genre_weighted.groupby("user_id")["genre_weight"].sum().reset_index(name="total_w")
    user_genre_weighted = user_genre_weighted.merge(user_total, on="user_id")
    user_genre_weighted["genre_affinity"] = user_genre_weighted["genre_weight"] / (user_genre_weighted["total_w"] + 1e-9)
    user_genre_aff_df = user_genre_weighted[["user_id", "genre_id", "genre_affinity"]].copy()

    item_genre_cnt = book_genres.merge(ed_to_book, on="book_id", how="right")
    item_genre_cnt = item_genre_cnt.groupby("edition_id")["genre_id"].nunique().reset_index()
    item_genre_cnt.columns = ["edition_id", "item_genre_count"]

    item_genres_df = book_genres.merge(ed_to_book, on="book_id", how="right")
    item_genre_sets = item_genres_df.groupby("edition_id")["genre_id"].apply(set).to_dict()
    item_genre_expanded = item_genres_df[["edition_id", "genre_id"]].dropna().copy()

    rows = []
    for uid, gset in user_top_genre_sets.items():
        for g in gset:
            rows.append((uid, g))
    user_top_expanded = pd.DataFrame(rows, columns=["user_id", "genre_id"])

    return (user_genre_div, user_top_genre_sets, item_genre_cnt, item_genre_sets,
            user_genre_aff_df, item_genre_expanded, user_top_expanded)

def build_author_features(interactions, editions):
    user_items = interactions[["user_id", "edition_id"]].copy()
    user_items = user_items.merge(editions[["edition_id", "author_id"]], on="edition_id", how="left")
    user_author_counts = user_items.groupby(["user_id", "author_id"]).size().reset_index(name="user_author_cnt")
    user_author_div = user_items.groupby("user_id")["author_id"].nunique().reset_index()
    user_author_div.columns = ["user_id", "user_author_diversity"]
    return user_author_counts, user_author_div

def build_author_stats(interactions, editions):
    inter_ed = interactions.merge(editions[["edition_id", "author_id"]], on="edition_id", how="left")
    stats = inter_ed.groupby("author_id").agg(
        author_pop=("event_type", "count"),
        author_read_cnt=("event_type", lambda x: (x == 2).sum()),
        author_rating_mean=("rating", "mean"),
        author_rating_cnt=("rating", lambda x: x.notna().sum()),
    ).reset_index()
    stats["author_read_ratio"] = stats["author_read_cnt"] / (stats["author_pop"] + 1e-9)
    return stats

def build_publisher_stats(interactions, editions):
    inter_ed = interactions.merge(editions[["edition_id", "publisher_id"]], on="edition_id", how="left")
    stats = inter_ed.groupby("publisher_id").agg(
        pub_pop=("event_type", "count"),
        pub_rating_mean=("rating", "mean"),
    ).reset_index()
    return stats

def build_user_language(interactions, editions):
    inter_ed = interactions.merge(editions[["edition_id", "language_id"]], on="edition_id", how="left")
    user_lang = inter_ed.groupby("user_id")["language_id"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else -1
    ).reset_index()
    user_lang.columns = ["user_id", "user_main_lang"]
    return user_lang

def build_time_weighted_pop(interactions, current_time, halflife_days=30):
    inter = interactions.copy()
    days_ago = (current_time - inter["event_ts"]).dt.total_seconds() / 86400.0
    inter["time_weight"] = np.exp(-np.log(2) * days_ago / halflife_days)
    inter["tw"] = inter["time_weight"] * inter["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
    tw_pop = inter.groupby("edition_id")["tw"].sum().reset_index(name="item_tw_pop")
    return tw_pop

def build_user_text_profile(interactions, text_emb, emb_cols):
    inter = interactions[["user_id", "edition_id", "event_type"]].copy()
    inter["w"] = inter["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
    tmp = inter.merge(text_emb, on="edition_id", how="left")
    tmp[emb_cols] = tmp[emb_cols].fillna(0.0)
    for c in emb_cols:
        tmp[c] = tmp[c] * tmp["w"]
    sum_vec = tmp.groupby("user_id")[emb_cols].sum()
    sum_w = tmp.groupby("user_id")["w"].sum().replace(0, np.nan)
    return (sum_vec.T / sum_w).T.fillna(0.0)

def build_item_pop_windows(interactions, current_time):
    outs = {}
    for days in [7, 30, 90]:
        start = current_time - pd.Timedelta(days=int(days))
        tmp = interactions[(interactions["event_ts"] >= start) & (interactions["event_ts"] < current_time)].copy()
        tmp["w"] = tmp["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
        agg = tmp.groupby("edition_id")["w"].sum().reset_index(name=f"item_pop_{days}d")
        outs[f"item_pop_{days}d"] = agg
    df = outs["item_pop_7d"].merge(outs["item_pop_30d"], on="edition_id", how="outer")
    df = df.merge(outs["item_pop_90d"], on="edition_id", how="outer").fillna(0.0)
    df["item_pop_7_over_30"] = df["item_pop_7d"] / (df["item_pop_30d"] + 1e-9)
    df["item_pop_30_over_90"] = df["item_pop_30d"] / (df["item_pop_90d"] + 1e-9)
    df["item_pop_slope_proxy"] = df["item_pop_7_over_30"] - df["item_pop_30_over_90"]
    df["item_pop_7d_z"] = (df["item_pop_7d"] - df["item_pop_7d"].mean()) / (df["item_pop_7d"].std() + 1e-9)
    return df

def build_author_user_affinity(interactions, editions, current_time, halflife_days=30):
    inter = interactions.copy()
    inter = inter.merge(editions[["edition_id", "author_id"]], on="edition_id", how="left")
    days_ago = (current_time - inter["event_ts"]).dt.total_seconds() / 86400.0
    inter["time_weight"] = np.exp(-np.log(2) * days_ago / halflife_days)
    inter["w"] = inter["time_weight"] * inter["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    agg = inter.groupby(["user_id", "author_id"])["w"].sum().reset_index(name="user_author_aff_tw")
    return agg

def build_author_user_affinity_norm(interactions, editions, current_time, halflife_days=30):
    agg = build_author_user_affinity(interactions, editions, current_time, halflife_days=halflife_days)
    author_stats = build_author_stats(interactions, editions)
    agg = agg.merge(author_stats[["author_id", "author_pop"]], on="author_id", how="left")
    eps = 1e-6
    agg["user_author_aff_norm"] = agg["user_author_aff_tw"] / (np.sqrt(agg["author_pop"].fillna(0.0)) + eps)
    agg["user_author_aff_norm"] = agg["user_author_aff_norm"].fillna(0.0)
    return agg[["user_id", "author_id", "user_author_aff_norm"]]

def build_user_popularity_preference(interactions, item_stats):
    inter = interactions.copy()
    inter["w"] = inter["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    ip = item_stats[["edition_id", "item_pop"]].copy()
    if ip.shape[0] == 0:
        return pd.DataFrame(columns=["user_id", "user_pref_pop"])
    ip["item_pop_rank_pct"] = ip["item_pop"].rank(pct=True)
    merged = inter.merge(ip[["edition_id", "item_pop_rank_pct"]], on="edition_id", how="left")
    merged["item_pop_rank_pct"] = merged["item_pop_rank_pct"].fillna(0.5)
    agg = merged.groupby("user_id").apply(lambda df: np.average(df["item_pop_rank_pct"], weights=(df["w"].values + 1e-9))).reset_index(name="user_pref_pop")
    agg["user_pref_pop"] = agg["user_pref_pop"].fillna(0.5)
    return agg

def build_publisher_user_affinity(interactions, editions, current_time, halflife_days=30):
    inter = interactions.copy()
    inter = inter.merge(editions[["edition_id", "publisher_id"]], on="edition_id", how="left")
    days_ago = (current_time - inter["event_ts"]).dt.total_seconds() / 86400.0
    inter["time_weight"] = np.exp(-np.log(2) * days_ago / halflife_days)
    inter["w"] = inter["time_weight"] * inter["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    agg = inter.groupby(["user_id", "publisher_id"])["w"].sum().reset_index(name="user_publisher_aff_tw")
    return agg

def build_genre_entropy(interactions, editions, book_genres):
    ed_to_book = editions[["edition_id", "book_id"]].copy()
    ig = book_genres.merge(ed_to_book, on="book_id", how="right")
    item_g = ig.groupby(["edition_id", "genre_id"]).size().reset_index(name="cnt")
    item_tot = item_g.groupby("edition_id")["cnt"].sum().reset_index(name="tot")
    item_g = item_g.merge(item_tot, on="edition_id")
    item_g["p"] = item_g["cnt"] / (item_g["tot"] + 1e-9)
    item_entropy = item_g.groupby("edition_id").apply(lambda df: -np.sum(df["p"] * np.log(df["p"] + 1e-9))).reset_index(name="item_genre_entropy")
    ui = interactions[["user_id", "edition_id"]].merge(ig[["edition_id", "genre_id"]], on="edition_id", how="left").dropna()
    user_g = ui.groupby(["user_id", "genre_id"]).size().reset_index(name="cnt")
    user_tot = user_g.groupby("user_id")["cnt"].sum().reset_index(name="tot")
    user_g = user_g.merge(user_tot, on="user_id")
    user_g["p"] = user_g["cnt"] / (user_g["tot"] + 1e-9)
    user_entropy = user_g.groupby("user_id").apply(lambda df: -np.sum(df["p"] * np.log(df["p"] + 1e-9))).reset_index(name="user_genre_entropy")
    return item_entropy, user_entropy

def build_user_novelty_pref(interactions, current_time, low_pop_quantile=0.2, window_days=30):
    start = current_time - pd.Timedelta(days=int(window_days))
    tmp = interactions[(interactions["event_ts"] >= start) & (interactions["event_ts"] < current_time)].copy()
    tmp["w"] = tmp["event_type"].map({1: 1.5, 2: 3.0}).fillna(0.0)
    item_pop = tmp.groupby("edition_id")["w"].sum().reset_index(name="pop_w")
    if item_pop.shape[0] == 0:
        return pd.DataFrame(columns=["user_id", "user_novelty_pref"])
    q = item_pop["pop_w"].quantile(low_pop_quantile)
    low_items = set(item_pop[item_pop["pop_w"] <= q]["edition_id"].values)
    tmp["is_low_pop"] = tmp["edition_id"].isin(low_items).astype(int)
    user_novelty = tmp.groupby("user_id")["is_low_pop"].mean().reset_index(name="user_novelty_pref")
    return user_novelty

def build_user_book_history(interactions, editions):
    inter = interactions.merge(editions[["edition_id", "book_id"]], on="edition_id", how="left")
    book_hist = inter.groupby(["user_id", "book_id"]).agg(
        user_book_cnt=("event_type", "count"),
        user_book_read=("event_type", lambda x: (x == 2).sum()),
        user_book_wish=("event_type", lambda x: (x == 1).sum()),
        user_book_rating_mean=("rating", "mean"),
    ).reset_index()
    book_hist["user_book_any_read"] = (book_hist["user_book_read"] > 0).astype(int)
    book_hist["user_book_any_wish"] = (book_hist["user_book_wish"] > 0).astype(int)
    book_hist["user_book_max_signal"] = book_hist["user_book_read"].clip(upper=1) * 3 + \
        book_hist["user_book_wish"].clip(upper=1) * (1 - book_hist["user_book_read"].clip(upper=1))
    return book_hist

def build_als_cf(interactions, n_factors=64, iterations=20, regularization=0.1, alpha=40.0):
    if not HAS_IMPLICIT:
        return {}, {}, n_factors

    from implicit.als import AlternatingLeastSquares

    inter = interactions.copy()
    inter["w"] = inter["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    agg = inter.groupby(["user_id", "edition_id"])["w"].sum().reset_index()

    user_ids = sorted(agg["user_id"].unique())
    item_ids = sorted(agg["edition_id"].unique())
    uid_map = {u: i for i, u in enumerate(user_ids)}
    iid_map = {e: i for i, e in enumerate(item_ids)}

    rows = agg["user_id"].map(uid_map).values
    cols = agg["edition_id"].map(iid_map).values
    conf_vals = 1.0 + alpha * agg["w"].values

    item_user = csr_matrix((conf_vals, (cols, rows)), shape=(len(item_ids), len(user_ids)))

    k = min(n_factors, min(len(user_ids), len(item_ids)) - 1)
    model = AlternatingLeastSquares(
        factors=k,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
        use_gpu=False,
    )
    model.fit(item_user)

    item_factors = model.item_factors
    user_factors = model.user_factors

    user_emb_dict = {uid: user_factors[i] for uid, i in uid_map.items()}
    item_emb_dict = {eid: item_factors[i] for eid, i in iid_map.items()}

    print(f"  ALS done: {len(user_ids)} users, {len(item_ids)} items, {k} factors")
    return user_emb_dict, item_emb_dict, k

def build_user_time_windows(interactions, current_time, windows=(7, 30)):
    outs = []
    for w in windows:
        start = current_time - pd.Timedelta(days=int(w))
        tmp = interactions[(interactions["event_ts"] >= start) & (interactions["event_ts"] < current_time)].copy()
        if tmp.shape[0] == 0:
            agg = pd.DataFrame(columns=["user_id", f"events_{w}d", f"reads_{w}d", f"wishes_{w}d"])
        else:
            agg = tmp.groupby("user_id").agg(
                **{
                    f"events_{w}d": ("event_type", "count"),
                    f"reads_{w}d": ("event_type", lambda x: (x == 2).sum()),
                    f"wishes_{w}d": ("event_type", lambda x: (x == 1).sum()),
                }
            ).reset_index()
        agg[f"read_ratio_{w}d"] = agg[f"reads_{w}d"].astype(float) / (agg[f"events_{w}d"].astype(float) + 1e-9)
        outs.append(agg)
    from functools import reduce
    if len(outs) == 0:
        return pd.DataFrame(columns=["user_id"])
    dfw = reduce(lambda a, b: a.merge(b, on="user_id", how="outer"), outs).fillna(0.0)
    if "events_30d" in dfw.columns and "events_7d" in dfw.columns:
        dfw["events_7_over_30"] = dfw["events_7d"] / (dfw["events_30d"] + 1e-9)
    else:
        dfw["events_7_over_30"] = 0.0
    if "read_ratio_7d" in dfw.columns and "read_ratio_30d" in dfw.columns:
        dfw["read_ratio_diff_7_30"] = dfw["read_ratio_7d"] - dfw["read_ratio_30d"]
    else:
        dfw["read_ratio_diff_7_30"] = 0.0
    return dfw

def add_user_zscore(df, user_col="user_id", cols=None):
    if cols is None:
        return df
    for c in cols:
        if c not in df.columns:
            continue
        mean = df.groupby(user_col)[c].transform("mean")
        std = df.groupby(user_col)[c].transform("std").replace(0, np.nan)
        df[f"z_{c}"] = (df[c] - mean) / (std + 1e-9)
        df[f"z_{c}"] = df[f"z_{c}"].fillna(0.0)
    return df

def make_train_pairs(interactions, candidates, label_start, label_end,
                     k_neg=200, seed=927, only_users_with_pos=True):
    inter_lbl = interactions[
        (interactions["event_ts"] >= label_start) & (interactions["event_ts"] < label_end)
    ].copy()

    pos = inter_lbl[["user_id", "edition_id", "event_type"]].copy()
    pos["target"] = pos["event_type"].map({1: 1.0, 2: 3.0}).fillna(0.0)
    pos = pos.groupby(["user_id", "edition_id"])["target"].max().reset_index()

    pos_pairs = set(zip(pos["user_id"].values, pos["edition_id"].values))
    cand_g = candidates.groupby("user_id")["edition_id"].apply(list).to_dict()

    if only_users_with_pos:
        cand_set = set()
        for uid, eids in cand_g.items():
            for eid in eids:
                cand_set.add((uid, eid))
        users_with_pos_in_cand = set()
        for uid, eid in pos_pairs:
            if (uid, eid) in cand_set:
                users_with_pos_in_cand.add(uid)
        users_with_any_pos = set(pos["user_id"].unique())
        active_users = users_with_any_pos
    else:
        active_users = set(cand_g.keys())

    rng = np.random.RandomState(seed)
    neg_rows = []
    for uid, eids in cand_g.items():
        if uid not in active_users:
            continue
        eids_neg = [eid for eid in eids if (uid, eid) not in pos_pairs]
        if not eids_neg:
            continue
        take = min(k_neg, len(eids_neg))
        if take < len(eids_neg):
            chosen = rng.choice(eids_neg, size=take, replace=False)
        else:
            chosen = eids_neg
        neg_rows.extend([(uid, int(eid), 0.0) for eid in chosen])

    neg = pd.DataFrame(neg_rows, columns=["user_id", "edition_id", "target"])
    result = pd.concat([pos[["user_id", "edition_id", "target"]], neg], ignore_index=True)

    user_max_target = result.groupby("user_id")["target"].max()
    users_keep = user_max_target[user_max_target > 0].index
    result = result[result["user_id"].isin(users_keep)]

    print(f"  Training users: {result['user_id'].nunique()}, "
          f"rows: {len(result)}, "
          f"pos: {(result['target'] > 0).sum()}, "
          f"neg: {(result['target'] == 0).sum()}")

    return result

N_SVD_ELEMWISE = 10

def get_features(df_pairs, editions, users, text_emb, emb_cols,
                 item_stats, user_stats, user_profile,
                 user_genre_div, item_genre_cnt,
                 user_genre_aff_df, item_genre_expanded, user_top_expanded,
                 user_author_counts, user_author_div,
                 author_stats, publisher_stats, user_lang,
                 svd_user_emb, svd_item_emb, svd_k,
                 svd_user_emb_2, svd_item_emb_2, svd_k_2,
                 tw_pop,
                 current_time,
                 item_pop_windows=None,
                 author_user_aff=None,
                 publisher_user_aff=None,
                 item_genre_entropy=None,
                 user_genre_entropy=None,
                 user_novelty_pref=None,
                 user_book_history=None,
                 als_user_emb=None,
                 als_item_emb=None,
                 als_k=None,
                 user_pop_pref=None,
                 author_user_aff_norm=None,
                 ):

    df = df_pairs[["user_id", "edition_id", "target"]].copy()
    df = df.merge(users, on="user_id", how="left")

    item_features = editions[[
        "edition_id", "book_id", "author_id", "language_id", "publisher_id",
        "publication_year", "age_restriction",
    ]].copy()

    item_features = item_features.merge(text_emb, on="edition_id", how="left")
    item_features = item_features.merge(item_stats, on="edition_id", how="left")
    item_features = item_features.merge(tw_pop, on="edition_id", how="left")

    if item_pop_windows is not None:
        item_features = item_features.merge(item_pop_windows, on="edition_id", how="left")

    df = df.merge(item_features, on="edition_id", how="left")
    df = df.merge(user_stats, on="user_id", how="left")
    if user_pop_pref is not None:
        df = df.merge(user_pop_pref, on="user_id", how="left")
        df["user_pref_pop"] = df["user_pref_pop"].fillna(0.5)
    else:
        df["user_pref_pop"] = 0.5

    df = df.merge(user_genre_div, on="user_id", how="left")
    df = df.merge(item_genre_cnt, on="edition_id", how="left")

    overlap_df = df[["user_id", "edition_id"]].drop_duplicates().merge(
        user_top_expanded, on="user_id"
    ).merge(
        item_genre_expanded, on=["edition_id", "genre_id"]
    ).groupby(["user_id", "edition_id"]).size().reset_index(name="genre_overlap")
    df = df.merge(overlap_df, on=["user_id", "edition_id"], how="left")
    df["genre_overlap"] = df["genre_overlap"].fillna(0)

    affinity_df = df[["user_id", "edition_id"]].drop_duplicates().merge(
        item_genre_expanded, on="edition_id"
    ).merge(
        user_genre_aff_df, on=["user_id", "genre_id"], how="left"
    )
    affinity_df["genre_affinity"] = affinity_df["genre_affinity"].fillna(0)
    affinity_agg = affinity_df.groupby(["user_id", "edition_id"])["genre_affinity"].sum().reset_index()
    df = df.merge(affinity_agg, on=["user_id", "edition_id"], how="left")
    df["genre_affinity"] = df["genre_affinity"].fillna(0)

    df = df.merge(user_author_div, on="user_id", how="left")
    df = df.merge(user_author_counts, on=["user_id", "author_id"], how="left")
    df["user_author_cnt"] = df["user_author_cnt"].fillna(0)
    df["author_familiar"] = (df["user_author_cnt"] > 0).astype(int)
    df = df.merge(author_stats, on="author_id", how="left")

    df = df.merge(publisher_stats, on="publisher_id", how="left")

    df = df.merge(user_lang, on="user_id", how="left")
    df["lang_match"] = (df["language_id"] == df["user_main_lang"]).astype(int)

    zero_emb = np.zeros(svd_k)
    u_embs = np.array([svd_user_emb.get(uid, zero_emb) for uid in df["user_id"].values])
    i_embs = np.array([svd_item_emb.get(eid, zero_emb) for eid in df["edition_id"].values])

    df["svd_score"] = np.sum(u_embs * i_embs, axis=1)
    n_ew = min(N_SVD_ELEMWISE, svd_k)
    elemwise = u_embs[:, :n_ew] * i_embs[:, :n_ew]
    for j in range(n_ew):
        df[f"svd_ew_{j}"] = elemwise[:, j]

    zero_emb_2 = np.zeros(svd_k_2)
    u_embs_2 = np.array([svd_user_emb_2.get(uid, zero_emb_2) for uid in df["user_id"].values])
    i_embs_2 = np.array([svd_item_emb_2.get(eid, zero_emb_2) for eid in df["edition_id"].values])
    df["svd_score_2"] = np.sum(u_embs_2 * i_embs_2, axis=1)

    u_norms = np.linalg.norm(u_embs, axis=1)
    i_norms = np.linalg.norm(i_embs, axis=1)
    df["svd_u_norm"] = u_norms
    df["svd_i_norm"] = i_norms
    df["svd_euc_dist"] = np.linalg.norm(u_embs - i_embs, axis=1)
    df["svd_euc_dist2"] = np.sum((u_embs - i_embs) ** 2, axis=1)
    df["svd_cos"] = df["svd_score"] / (u_norms * i_norms + 1e-9)
    df["svd_cos"] = np.clip(df["svd_cos"], -1.0, 1.0)
    df["svd_angle"] = np.arccos(df["svd_cos"])
    df["svd_proj_on_user"] = df["svd_score"] / (u_norms + 1e-9)
    df["svd_item_orth_mag"] = np.sqrt(np.maximum(0.0, i_norms ** 2 - df["svd_proj_on_user"] ** 2))

    u2_norms = np.linalg.norm(u_embs_2, axis=1)
    i2_norms = np.linalg.norm(i_embs_2, axis=1)
    df["svd2_u_norm"] = u2_norms
    df["svd2_i_norm"] = i2_norms
    df["svd2_euc_dist"] = np.linalg.norm(u_embs_2 - i_embs_2, axis=1)
    df["svd2_euc_dist2"] = np.sum((u_embs_2 - i_embs_2) ** 2, axis=1)
    df["svd2_cos"] = df["svd_score_2"] / (u2_norms * i2_norms + 1e-9)
    df["svd2_cos"] = np.clip(df["svd2_cos"], -1.0, 1.0)
    df["svd2_angle"] = np.arccos(df["svd2_cos"])
    df["svd2_proj_on_user"] = df["svd_score_2"] / (u2_norms + 1e-9)
    df["svd2_item_orth_mag"] = np.sqrt(np.maximum(0.0, i2_norms ** 2 - df["svd2_proj_on_user"] ** 2))

    u = np.nan_to_num(user_profile.reindex(df["user_id"]).values)
    v = df[emb_cols].fillna(0.0).values
    dot = np.sum(u * v, axis=1)
    df["text_sim"] = dot / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    df["text_l2"] = np.linalg.norm(u - v, axis=1)

    df["age_num"] = df["age"].astype(float)
    df["age_restr"] = df["age_restriction"].astype(float).fillna(0.0)
    df["is_age_ok"] = ((df["age_num"].isna()) | (df["age_num"] >= df["age_restr"])).astype(int)

    df["pub_year"] = df["publication_year"].astype(float).fillna(0.0)
    df.loc[df["pub_year"] < 1900, "pub_year"] = 2016.0
    df["year_old"] = (2025.0 - df["pub_year"]).clip(0, 100)
    df["pub_recent"] = (df["pub_year"] >= 2020).astype(int)

    if current_time is not None and "user_last_ts" in df.columns:
        df["days_since_last"] = (current_time - df["user_last_ts"]).dt.total_seconds() / 86400
        df["days_since_last"] = df["days_since_last"].fillna(999)
        df["user_last_hour"] = df["user_last_ts"].dt.hour.fillna(-1).astype(int)
        df["user_last_dow"] = df["user_last_ts"].dt.dayofweek.fillna(-1).astype(int)
    else:
        df["days_since_last"] = 0
        df["user_last_hour"] = -1
        df["user_last_dow"] = -1

    df["rating_diff"] = np.abs(
        df["user_rating_mean"].fillna(8.3) - df["item_rating_mean"].fillna(8.3)
    )

    df["item_pop_vs_user"] = df["item_pop"] / (df["user_avg_item_pop"].fillna(1) + 1e-9)
    df["item_vs_author_pop"] = df["item_pop"] / (df["author_pop"].fillna(1) + 1e-9)

    if author_user_aff is not None:
        df = df.merge(author_user_aff, on=["user_id", "author_id"], how="left")
        df["user_author_aff_tw"] = df["user_author_aff_tw"].fillna(0.0)
    else:
        df["user_author_aff_tw"] = 0.0

    if author_user_aff_norm is not None:
        df = df.merge(author_user_aff_norm, on=["user_id", "author_id"], how="left")
        df["user_author_aff_norm"] = df["user_author_aff_norm"].fillna(0.0)
    else:
        df["user_author_aff_norm"] = 0.0

    if publisher_user_aff is not None:
        df = df.merge(publisher_user_aff, on=["user_id", "publisher_id"], how="left")
        df["user_publisher_aff_tw"] = df["user_publisher_aff_tw"].fillna(0.0)
    else:
        df["user_publisher_aff_tw"] = 0.0

    if item_genre_entropy is not None:
        df = df.merge(item_genre_entropy, on="edition_id", how="left")
        df["item_genre_entropy"] = df["item_genre_entropy"].fillna(0.0)
    else:
        df["item_genre_entropy"] = 0.0
    if user_genre_entropy is not None:
        df = df.merge(user_genre_entropy, on="user_id", how="left")
        df["user_genre_entropy"] = df["user_genre_entropy"].fillna(0.0)
    else:
        df["user_genre_entropy"] = 0.0

    if user_novelty_pref is not None:
        df = df.merge(user_novelty_pref, on="user_id", how="left")
        df["user_novelty_pref"] = df["user_novelty_pref"].fillna(0.0)
    else:
        df["user_novelty_pref"] = 0.0

    if user_book_history is not None:
        df = df.merge(user_book_history, on=["user_id", "book_id"], how="left")
        for c in ["user_book_cnt", "user_book_read", "user_book_wish",
                  "user_book_rating_mean", "user_book_any_read", "user_book_any_wish",
                  "user_book_max_signal"]:
            df[c] = df[c].fillna(0.0)
    else:
        for c in ["user_book_cnt", "user_book_read", "user_book_wish",
                  "user_book_rating_mean", "user_book_any_read", "user_book_any_wish",
                  "user_book_max_signal"]:
            df[c] = 0.0

    if als_user_emb is not None and als_item_emb is not None and als_k:
        _zero_als = np.zeros(als_k)
        _u_als = np.array([als_user_emb.get(uid, _zero_als) for uid in df["user_id"].values])
        _i_als = np.array([als_item_emb.get(eid, _zero_als) for eid in df["edition_id"].values])
        df["als_score"] = np.sum(_u_als * _i_als, axis=1)
        _u_norms_als = np.linalg.norm(_u_als, axis=1)
        _i_norms_als = np.linalg.norm(_i_als, axis=1)
        df["als_cos"] = df["als_score"] / (_u_norms_als * _i_norms_als + 1e-9)
        df["als_cos"] = np.clip(df["als_cos"], -1.0, 1.0)
        df["als_u_norm"] = _u_norms_als
        df["als_i_norm"] = _i_norms_als
    else:
        df["als_score"] = 0.0
        df["als_cos"] = 0.0
        df["als_u_norm"] = 0.0
        df["als_i_norm"] = 0.0

    for col in ["item_pop", "svd_score", "text_sim", "genre_affinity",
                "item_bayesian_rating", "item_tw_pop", "als_score"]:
        if col in df.columns:
            df[f"cand_pct_{col}"] = df.groupby("user_id")[col].rank(pct=True, ascending=True)
        else:
            df[f"cand_pct_{col}"] = 0.0

    z_cols = ["item_pop", "svd_score", "svd_score_2", "text_sim",
              "genre_affinity", "item_bayesian_rating", "item_tw_pop", "als_score",
              "user_author_aff_norm", "user_pref_pop"]
    z_cols_present = [c for c in z_cols if c in df.columns]
    df = add_user_zscore(df, user_col="user_id", cols=z_cols_present)

    if item_pop_windows is not None:
        for c in ["item_pop_7d", "item_pop_30d", "item_pop_90d",
                  "item_pop_7_over_30", "item_pop_30_over_90", "item_pop_slope_proxy", "item_pop_7d_z"]:
            if c in df.columns:
                continue
    if "item_pop" in df.columns:
        df["item_pop_q"] = pd.qcut(df["item_pop"].rank(method="first"), q=5, labels=False, duplicates="drop")
        df["item_pop_q"] = df["item_pop_q"].fillna(2).astype(int)
    else:
        df["item_pop_q"] = 2

    return df.fillna(0.0)

def force_cat_to_str(df, cat_cols):
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64").astype(str).replace("<NA>", "__nan__")
    return df

def gentle_diversity_rerank(df_scored, item_genre_sets, top_k=20, lam=0.97):
    results = []
    for uid, group in df_scored.groupby("user_id"):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        cands = group[["edition_id", "score"]].values.tolist()

        scores_arr = np.array([s for _, s in cands])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max > s_min:
            norm = {int(eid): (s - s_min) / (s_max - s_min) for eid, s in cands}
        else:
            norm = {int(eid): 1.0 for eid, s in cands}

        selected = []
        selected_genres = set()
        remaining = [int(eid) for eid, _ in cands]

        for _ in range(top_k):
            best_eid = None
            best_mmr = -1e9
            for eid in remaining:
                rel = norm[eid]
                ig = item_genre_sets.get(eid, set())
                div = len(ig - selected_genres) / max(len(ig), 1) if ig else 0.0
                mmr = lam * rel + (1 - lam) * div
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_eid = eid
            if best_eid is not None:
                selected.append(best_eid)
                remaining.remove(best_eid)
                selected_genres |= item_genre_sets.get(best_eid, set())

        for rank_idx, eid in enumerate(selected):
            results.append((uid, eid, rank_idx + 1))

    return pd.DataFrame(results, columns=["user_id", "edition_id", "rank"])

def make_submission_from_scores(df_scored, top_k=20):
    df_scored = df_scored.sort_values(["user_id", "score"], ascending=[True, False])
    top = df_scored.groupby("user_id", sort=False).head(top_k).copy()
    top["rank"] = top.groupby("user_id", sort=False).cumcount() + 1
    return top[["user_id", "edition_id", "rank"]]

def build_window(inter_feat, ed, bg, users, text_emb, emb_cols,
                 df_pairs, current_time):

    item_stats = build_item_stats(inter_feat)
    user_stats = build_user_stats(inter_feat)
    user_profile = build_user_text_profile(inter_feat, text_emb, emb_cols)

    (user_genre_div, user_top_genre_sets, item_genre_cnt, item_genre_sets,
     user_genre_aff_df, item_genre_expanded, user_top_expanded) = \
        build_genre_features(inter_feat, ed, bg)
    user_author_counts, user_author_div = build_author_features(inter_feat, ed)
    author_stats = build_author_stats(inter_feat, ed)
    publisher_stats = build_publisher_stats(inter_feat, ed)
    user_lang = build_user_language(inter_feat, ed)
    tw_pop = build_time_weighted_pop(inter_feat, current_time, halflife_days=30)

    svd_u1, svd_i1, k1 = build_svd_cf(inter_feat, n_factors=48, use_log=True)
    svd_u2, svd_i2, k2 = build_svd_cf(inter_feat, n_factors=20, use_log=False)

    user_time_windows = build_user_time_windows(inter_feat, current_time, windows=(7, 30))

    item_pop_windows = build_item_pop_windows(inter_feat, current_time)
    author_user_aff = build_author_user_affinity(inter_feat, ed, current_time)
    publisher_user_aff = build_publisher_user_affinity(inter_feat, ed, current_time)
    item_genre_entropy, user_genre_entropy = build_genre_entropy(inter_feat, ed, bg)
    user_novelty_pref = build_user_novelty_pref(inter_feat, current_time, low_pop_quantile=0.2, window_days=30)
    user_book_history = build_user_book_history(inter_feat, ed)
    als_user_emb, als_item_emb, als_k = build_als_cf(inter_feat, n_factors=64, iterations=20)

    user_pop_pref = build_user_popularity_preference(inter_feat, item_stats)
    author_user_aff_norm = build_author_user_affinity_norm(inter_feat, ed, current_time)

    X = get_features(
        df_pairs, ed, users, text_emb, emb_cols,
        item_stats, user_stats, user_profile,
        user_genre_div, item_genre_cnt,
        user_genre_aff_df, item_genre_expanded, user_top_expanded,
        user_author_counts, user_author_div,
        author_stats, publisher_stats, user_lang,
        svd_u1, svd_i1, k1,
        svd_u2, svd_i2, k2,
        tw_pop, current_time,
        item_pop_windows=item_pop_windows,
        author_user_aff=author_user_aff,
        publisher_user_aff=publisher_user_aff,
        item_genre_entropy=item_genre_entropy,
        user_genre_entropy=user_genre_entropy,
        user_novelty_pref=user_novelty_pref,
        user_book_history=user_book_history,
        als_user_emb=als_user_emb,
        als_item_emb=als_item_emb,
        als_k=als_k,
        user_pop_pref=user_pop_pref,                      
        author_user_aff_norm=author_user_aff_norm,      
    )

    X = X.merge(user_time_windows, on="user_id", how="left").fillna(0.0)

    return X, item_genre_sets

def main():
    inter, ed, bg, users, cand, targets = load_data()

    HORIZON_DAYS = 30
    HISTORY_DAYS = 180

    T0 = inter["event_ts"].max() + pd.Timedelta(seconds=1)

    train_label_end = T0
    train_label_start = T0 - pd.Timedelta(days=HORIZON_DAYS)
    train_feat_end = train_label_start
    train_feat_start = train_feat_end - pd.Timedelta(days=HISTORY_DAYS)
    test_feat_end = T0
    test_feat_start = T0 - pd.Timedelta(days=HISTORY_DAYS)

    print("TRAIN FEATURES:", train_feat_start, "->", train_feat_end)
    print("TRAIN LABEL:   ", train_label_start, "->", train_label_end)
    print("TEST FEATURES: ", test_feat_start, "->", test_feat_end)

    text_emb, emb_cols = preprocess_text(ed, n_components=20, max_features=5000)

    inter_train_feat = inter[
        (inter["event_ts"] >= train_feat_start) & (inter["event_ts"] < train_feat_end)
    ].copy()
    inter_test_feat = inter[
        (inter["event_ts"] >= test_feat_start) & (inter["event_ts"] < test_feat_end)
    ].copy()

    svd_ew_cols = [f"svd_ew_{j}" for j in range(N_SVD_ELEMWISE)]
    cand_pct_cols = [f"cand_pct_{c}" for c in
                     ["item_pop", "svd_score", "text_sim", "genre_affinity",
                      "item_bayesian_rating", "item_tw_pop", "als_score"]]

    features = [
        "gender", "age_num",
        "user_events", "user_read_cnt", "user_wish_cnt",
        "user_rating_mean", "user_rating_cnt", "user_rating_std", "user_read_ratio",
        "user_avg_item_pop",
        "book_id", "author_id", "language_id", "publisher_id",
        "age_restr", "is_age_ok", "pub_year", "year_old",
        "item_pop", "item_events", "item_read_cnt", "item_wish_cnt",
        "item_rating_mean", "item_rating_cnt", "item_rating_std",
        "item_read_ratio", "item_bayesian_rating", "item_tw_pop",
        "item_pop_7d", "item_pop_30d", "item_pop_90d", "item_pop_7_over_30", "item_pop_30_over_90", "item_pop_slope_proxy", "item_pop_7d_z",
        "text_sim", "text_l2",
        "user_genre_diversity", "item_genre_count",
        "genre_overlap", "genre_affinity",
        "item_genre_entropy", "user_genre_entropy",
        "user_author_diversity", "user_author_cnt", "author_familiar",
        "author_pop", "author_read_cnt", "author_rating_mean", "author_rating_cnt",
        "author_read_ratio",
        "user_author_aff_tw",
        "pub_pop", "pub_rating_mean", "user_publisher_aff_tw",
        "lang_match",
        "svd_score", "svd_score_2",
        "days_since_last", "rating_diff",
        "item_pop_vs_user", "item_vs_author_pop",
        "user_novelty_pref", "item_pop_q",
        "user_last_hour", "user_last_dow", "pub_recent",
        "user_book_cnt", "user_book_read", "user_book_wish",
        "user_book_rating_mean", "user_book_any_read", "user_book_any_wish",
        "user_book_max_signal",
        "als_score", "als_cos", "als_u_norm", "als_i_norm",
        "user_pref_pop", "user_author_aff_norm",
    ] + svd_ew_cols + cand_pct_cols + emb_cols

    features += [
        "svd_u_norm", "svd_i_norm", "svd_euc_dist", "svd_euc_dist2", "svd_cos", "svd_angle",
        "svd_proj_on_user", "svd_item_orth_mag",
        "svd2_u_norm", "svd2_i_norm", "svd2_euc_dist", "svd2_euc_dist2", "svd2_cos", "svd2_angle",
        "svd2_proj_on_user", "svd2_item_orth_mag",
    ]
    for c in ["item_pop", "svd_score", "svd_score_2", "text_sim", "genre_affinity", "item_bayesian_rating", "item_tw_pop", "als_score", "user_author_aff_norm", "user_pref_pop"]:
        features.append(f"z_{c}")
    features += [
        "events_7d", "reads_7d", "wishes_7d", "read_ratio_7d",
        "events_30d", "reads_30d", "wishes_30d", "read_ratio_30d",
        "events_7_over_30", "read_ratio_diff_7_30"
    ]

    cat_cols = ["gender", "book_id", "author_id", "language_id", "publisher_id"]

    print("\n=== Precomputing TRAIN window features ===")
    tr_item_stats = build_item_stats(inter_train_feat)
    tr_user_stats = build_user_stats(inter_train_feat)
    tr_user_profile = build_user_text_profile(inter_train_feat, text_emb, emb_cols)
    (tr_ugd, tr_utgs, tr_igc, tr_igs,
     tr_ugadf, tr_ige, tr_ute) = build_genre_features(inter_train_feat, ed, bg)
    tr_uac, tr_uad = build_author_features(inter_train_feat, ed)
    tr_as = build_author_stats(inter_train_feat, ed)
    tr_ps = build_publisher_stats(inter_train_feat, ed)
    tr_ul = build_user_language(inter_train_feat, ed)
    tr_twp = build_time_weighted_pop(inter_train_feat, train_feat_end, halflife_days=30)
    tr_svd1_u, tr_svd1_i, tr_k1 = build_svd_cf(inter_train_feat, n_factors=48, use_log=True)
    tr_svd2_u, tr_svd2_i, tr_k2 = build_svd_cf(inter_train_feat, n_factors=20, use_log=False)
    tr_utw = build_user_time_windows(inter_train_feat, train_feat_end, windows=(7, 30))

    tr_item_pop_windows = build_item_pop_windows(inter_train_feat, train_feat_end)
    tr_author_user_aff = build_author_user_affinity(inter_train_feat, ed, train_feat_end)
    tr_publisher_user_aff = build_publisher_user_affinity(inter_train_feat, ed, train_feat_end)
    tr_item_genre_entropy, tr_user_genre_entropy = build_genre_entropy(inter_train_feat, ed, bg)
    tr_user_novelty_pref = build_user_novelty_pref(inter_train_feat, train_feat_end, low_pop_quantile=0.2, window_days=30)
    tr_user_book_history = build_user_book_history(inter_train_feat, ed)
    print("  Building ALS CF train")
    tr_als_u, tr_als_i, tr_als_k = build_als_cf(inter_train_feat, n_factors=64, iterations=20)
    tr_user_pop_pref = build_user_popularity_preference(inter_train_feat, tr_item_stats)
    tr_author_user_aff_norm = build_author_user_affinity_norm(inter_train_feat, ed, train_feat_end)

    print("\n=== Precomputing TEST window features ===")
    te_item_stats = build_item_stats(inter_test_feat)
    te_user_stats = build_user_stats(inter_test_feat)
    te_user_profile = build_user_text_profile(inter_test_feat, text_emb, emb_cols)
    (te_ugd, te_utgs, te_igc, te_igs,
     te_ugadf, te_ige, te_ute) = build_genre_features(inter_test_feat, ed, bg)
    te_uac, te_uad = build_author_features(inter_test_feat, ed)
    te_as = build_author_stats(inter_test_feat, ed)
    te_ps = build_publisher_stats(inter_test_feat, ed)
    te_ul = build_user_language(inter_test_feat, ed)
    te_twp = build_time_weighted_pop(inter_test_feat, test_feat_end, halflife_days=30)
    te_svd1_u, te_svd1_i, te_k1 = build_svd_cf(inter_test_feat, n_factors=48, use_log=True)
    te_svd2_u, te_svd2_i, te_k2 = build_svd_cf(inter_test_feat, n_factors=20, use_log=False)
    te_utw = build_user_time_windows(inter_test_feat, test_feat_end, windows=(7, 30))

    te_item_pop_windows = build_item_pop_windows(inter_test_feat, test_feat_end)
    te_author_user_aff = build_author_user_affinity(inter_test_feat, ed, test_feat_end)
    te_publisher_user_aff = build_publisher_user_affinity(inter_test_feat, ed, test_feat_end)
    te_item_genre_entropy, te_user_genre_entropy = build_genre_entropy(inter_test_feat, ed, bg)
    te_user_novelty_pref = build_user_novelty_pref(inter_test_feat, test_feat_end, low_pop_quantile=0.2, window_days=30)
    te_user_book_history = build_user_book_history(inter_test_feat, ed)
    print("  Building ALS CF test")
    te_als_u, te_als_i, te_als_k = build_als_cf(inter_test_feat, n_factors=64, iterations=20)
    te_user_pop_pref = build_user_popularity_preference(inter_test_feat, te_item_stats)
    te_author_user_aff_norm = build_author_user_affinity_norm(inter_test_feat, ed, test_feat_end)

    print("\n=== Building TEST feature matrix ===")
    X_test = get_features(
        cand.assign(target=0.0), ed, users, text_emb, emb_cols,
        te_item_stats, te_user_stats, te_user_profile,
        te_ugd, te_igc, te_ugadf, te_ige, te_ute,
        te_uac, te_uad, te_as, te_ps, te_ul,
        te_svd1_u, te_svd1_i, te_k1,
        te_svd2_u, te_svd2_i, te_k2,
        te_twp, test_feat_end,
        item_pop_windows=te_item_pop_windows,
        author_user_aff=te_author_user_aff,
        publisher_user_aff=te_publisher_user_aff,
        item_genre_entropy=te_item_genre_entropy,
        user_genre_entropy=te_user_genre_entropy,
        user_novelty_pref=te_user_novelty_pref,
        user_book_history=te_user_book_history,
        als_user_emb=te_als_u,
        als_item_emb=te_als_i,
        als_k=te_als_k,
        user_pop_pref=te_user_pop_pref,                          
        author_user_aff_norm=te_author_user_aff_norm,
    )
    X_test = X_test.merge(te_utw, on="user_id", how="left").fillna(0.0)
    X_test = force_cat_to_str(X_test, cat_cols)

    configs = [
        {"seed": 2024, "k_neg": 195, "lr": 0.1, "depth": 9, "iters": 4000, "l2": 5.5, "subsample": 0.85},
    ]

    all_preds = []
    all_ranks = []

    for i, cfg in enumerate(configs):
        print(f"\n=== Model {i+1}/{len(configs)}: seed={cfg['seed']} ===")

        train_pairs = make_train_pairs(
            inter, cand,
            label_start=train_label_start, label_end=train_label_end,
            k_neg=cfg["k_neg"], seed=cfg["seed"],
            only_users_with_pos=True
        )

        X_train = get_features(
            train_pairs, ed, users, text_emb, emb_cols,
            tr_item_stats, tr_user_stats, tr_user_profile,
            tr_ugd, tr_igc, tr_ugadf, tr_ige, tr_ute,
            tr_uac, tr_uad, tr_as, tr_ps, tr_ul,
            tr_svd1_u, tr_svd1_i, tr_k1,
            tr_svd2_u, tr_svd2_i, tr_k2,
            tr_twp, train_feat_end,
            item_pop_windows=tr_item_pop_windows,
            author_user_aff=tr_author_user_aff,
            publisher_user_aff=tr_publisher_user_aff,
            item_genre_entropy=tr_item_genre_entropy,
            user_genre_entropy=tr_user_genre_entropy,
            user_novelty_pref=tr_user_novelty_pref,
            user_book_history=tr_user_book_history,
            als_user_emb=tr_als_u,
            als_item_emb=tr_als_i,
            als_k=tr_als_k,
            user_pop_pref=tr_user_pop_pref,
            author_user_aff_norm=tr_author_user_aff_norm,
        )
        X_train = X_train.merge(tr_utw, on="user_id", how="left").fillna(0.0)

        X_train = force_cat_to_str(X_train, cat_cols)
        X_train = X_train.sort_values("user_id").reset_index(drop=True)

        cat_idx = [features.index(c) for c in cat_cols if c in features]

        model = CatBoostRanker(
            iterations=cfg["iters"],
            learning_rate=cfg["lr"],
            depth=cfg["depth"],
            loss_function="YetiRank",
            random_seed=cfg["seed"],
            verbose=200,
            l2_leaf_reg=cfg["l2"],
            subsample=cfg["subsample"],
            min_data_in_leaf=15,
            task_type="GPU",
            devices="0",
            bootstrap_type="Bernoulli"
        )

        train_pool = Pool(
            data=X_train[features],
            label=X_train["target"],
            group_id=X_train["user_id"],
            cat_features=cat_idx,
        )
        model.fit(train_pool)

        preds = model.predict(X_test[features])
        all_preds.append(preds)

        tmp = X_test[["user_id"]].copy()
        tmp["pred"] = preds
        tmp["rank"] = tmp.groupby("user_id")["pred"].rank(ascending=False)
        all_ranks.append(tmp["rank"].values)
    X_test["score_avg"] = np.mean(all_preds, axis=0)
    X_test["rank_avg"] = np.mean(all_ranks, axis=0)
    X_test["score_rank"] = -X_test["rank_avg"]

    ed_to_book = ed[["edition_id", "book_id"]]
    all_ig_df = bg.merge(ed_to_book, on="book_id", how="right")
    all_item_genre_sets = all_ig_df.groupby("edition_id")["genre_id"].apply(set).to_dict()

    sub = make_submission_from_scores(X_test[["user_id", "edition_id"]].assign(score=X_test["score_avg"]), top_k=20)
    sub.to_csv("submission_Werserk_I_Love_U.csv", index=False)


if __name__ == "__main__":
    main()