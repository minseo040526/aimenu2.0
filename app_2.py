import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# =========================
# 유틸 함수
# =========================
def normalize_columns(df: pd.DataFrame, is_drink: bool = False) -> pd.DataFrame:
    menu_type = "음료" if is_drink else "베이커리"
    req = ['name','price','sweetness','tags']
    if is_drink: req.append('category')
    miss = [c for c in req if c not in df.columns]
    if miss:
        st.error(f"🚨 {menu_type} 파일에 필수 컬럼({', '.join(miss)})이 없습니다.")
        st.stop()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce')
    if df['price'].isnull().any() or df['sweetness'].isnull().any():
        st.error(f"🚨 {menu_type} 파일의 숫자형 컬럼에 잘못된 값이 있습니다.")
        st.stop()
    return df

def preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:
    df['tags_list'] = (
        df['tags'].fillna('').astype(str)
        .str.replace('#','')
        .str.replace(';',',')
        .str.split(r'\s*,\s*')
        .apply(lambda xs: [t.strip() for t in xs if t.strip()])
    )
    return df

def assign_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(lambda ts: 10 if '인기' in ts else 5)
    return df

def uniq_tags(df: pd.DataFrame) -> set:
    return set(t for sub in df['tags_list'] for t in sub if t)

def load_image(path: str):
    try: return Image.open(path)
    except Exception: return None

# =========================
# 데이터 로드
# =========================
try:
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df  = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True)
except Exception as e:
    st.error(f"🚨 CSV 파일 오류: {e}")
    st.stop()

bakery_df = assign_popularity_score(preprocess_tags(bakery_df))
drink_df  = assign_popularity_score(preprocess_tags(drink_df))

FLAVOR_TAGS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','초코','치즈'}
BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS  = uniq_tags(drink_df)
ui_bakery_utility_tags = sorted(BAKERY_TAGS - FLAVOR_TAGS)
ui_drink_flavor_tags   = sorted(DRINK_TAGS & FLAVOR_TAGS)
all_drink_categories   = sorted(drink_df['category'].astype(str).unique())

# =========================
# 추천 로직
# =========================
def _match_all(row_tags, selected): 
    return set(selected).issubset(set(row_tags))

def _match_any(row_tags, selected):
    return not set(row_tags).isdisjoint(set(selected))

def _filter_base(df, min_s, max_s, sel_tags, max_price=None, sel_cats=None, require_all=True):
    f = df.copy()
    is_drink = 'category' in f.columns
    if is_drink and sel_cats:
        f = f[f['category'].isin(sel_cats)]
    f = f[(f['sweetness'] >= min_s) & (f['sweetness'] <= max_s)]
    if sel_tags:
        if require_all:
            f = f[f['tags_list'].apply(lambda t: _match_all(t, sel_tags))]
        else:
            f = f[f['tags_list'].apply(lambda t: _match_any(t, sel_tags))]
    if max_price is not None:
        f = f[f['price'] <= max_price]
    return f

def _make_recs_from_filtered(f, n_items, max_price=None):
    recs = []
    if f.empty: return recs
    if n_items == 1:
        for _, r in f.sort_values(['popularity_score','price'], ascending=[False,True]).iterrows():
            recs.append([r.to_dict()])
            if len(recs) >= 200: break
        return recs
    if len(f) < n_items:
        top = f.sort_values('popularity_score', ascending=False)
        recs.append([r.to_dict() for _, r in top.iterrows()])
        return recs
    pool = f.sort_values('popularity_score', ascending=False).head(30)
    for combo in itertools.combinations(pool.itertuples(index=False), n_items):
        price_sum = sum(c.price for c in combo)
        if (max_price is None) or (price_sum <= max_price):
            recs.append([{k:getattr(c,k) for k in f.columns} for c in combo])
            if len(recs) >= 200: break
    return recs

def recommend_menu_strict(df, min_s, max_s, sel_tags, n_items, max_price=None, sel_cats=None):
    f = _filter_base(df, min_s, max_s, sel_tags, max_price, sel_cats, require_all=True)
    return _make_recs_from_filtered(f, n_items, max_price)

def recommend_menu_relaxed(df, min_s, max_s, sel_tags, n_items, max_price=None, sel_cats=None):
    f = _filter_base(df, min_s, max_s, sel_tags, max_price, sel_cats, require_all=False)
    if not f.empty: return _make_recs_from_filtered(f, n_items, max_price)
    f = _filter_base(df, min_s, max_s, [], max_price, sel_cats, require_all=True)
    if not f.empty: return _make_recs_from_filtered(f, n_items, max_price)
    new_min = max(1, min_s-1); new_max = min(5, max_s+1)
    f = _filter_base(df, new_min, new_max, [], max_price, sel_cats, require_all=True)
    if not f.empty: return _make_recs_from_filtered(f, n_items, max_price)
    f = df.copy()
    if max_price is not None: f = f[f['price'] <= max_price]
    return _make_recs_from_filtered(f, n_items, max_price)

def calculate_weighted_score(items, selected_tags):
    if not selected_tags:
        tag_score = 100.0
    else:
        total = len(items)
        match = sum(1 for it in items if not set(it['tags_list']).isdisjoint(selected_tags))
        tag_score = (match/total)*100.0 if total else 0.0
    avg_pop = sum(it['popularity_score'] for it in items)/len(items) if items else 0.0
    return round(tag_score*0.7 + (avg_pop*10)*0.3, 1)

# =========================
# UI
# =========================
tab_reco, tab_board = st.tabs(["AI 메뉴 추천", "메뉴판"])

with tab_reco:
    st.title("AI 메뉴 추천 시스템")
    st.caption("고객님의 취향과 인기 메뉴 정보를 반영해 AI가 메뉴를 추천합니다.")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader("인원/예산")
        n_people = st.number_input("인원수(인원수만큼 음료를 추천해드려요)", 1, 10, 2)
        unlimited = st.checkbox("예산 무제한", value=True)
        if unlimited:
            max_budget = None
            st.slider("최대 예산(1인)", 5000, 50000, 50000, 1000, disabled=True)
        else:
            max_budget = st.slider("최대 예산(1인)", 5000, 50000, 15000, 1000)

    with c2:
        st.subheader("베이커리")
        n_bakery = st.slider("베이커리 개수", 1, 5, 2)
        min_bak, max_bak = st.slider("당도(베이커리)", 1, 5, (1,5))
        sel_bak_tags = st.multiselect("베이커리 태그", ui_bakery_utility_tags, max_selections=3)

    with c3:
        st.subheader("음료")
        sel_cats = st.multiselect("카테고리", all_drink_categories, default=all_drink_categories)
        min_drk, max_drk = st.slider("당도(음료)", 1, 5, (1,5))
        sel_drk_tags = st.multiselect("맛 태그", ui_drink_flavor_tags, max_selections=3)

    st.markdown("---")

    if st.button("AI 추천받기", type="primary", use_container_width=True):
        drink_recs = recommend_menu_strict(drink_df, min_drk, max_drk, sel_drk_tags, 1, max_budget, sel_cats)
        bakery_recs = recommend_menu_strict(bakery_df, min_bak, max_bak, sel_bak_tags, n_bakery, max_budget)
        relaxed_used = False
        if not drink_recs:
            drink_recs = recommend_menu_relaxed(drink_df, min_drk, max_drk, sel_drk_tags, 1, max_budget, sel_cats)
            relaxed_used = True
        if not bakery_recs:
            bakery_recs = recommend_menu_relaxed(bakery_df, min_bak, max_bak, sel_bak_tags, n_bakery, max_budget)
            relaxed_used = True

        if not drink_recs and not bakery_recs:
            st.warning("현재 조건에 맞는 메뉴를 찾지 못했습니다. 태그나 당도 범위를 완화해 주세요.")
            st.stop()

        results = []
        pairs = list(itertools.product(drink_recs or [[]], bakery_recs or [[]]))
        random.shuffle(pairs)
        selected_all_tags = sel_bak_tags + sel_drk_tags
        for d_combo, b_combo in pairs:
            price_d = d_combo[0]['price'] if d_combo else 0
            price_b = sum(x['price'] for x in b_combo) if b_combo else 0
            per_price = price_d + price_b
            if (max_budget is None) or (per_price <= max_budget):
                items = (d_combo or []) + (b_combo or [])
                score = calculate_weighted_score(items, selected_all_tags)
                results.append({
                    "score": score,
                    "drink": d_combo[0] if d_combo else None,
                    "bakery": b_combo,
                    "total_price_per_set": per_price
                })
            if len(results) >= 200: break

        if not results:
            if max_budget is None:
                st.warning("현재 조건에 맞는 메뉴가 없습니다. 태그/당도 조건을 완화해 주세요.")
            else:
                st.warning("예산 범위 때문에 추천이 제한되었습니다. 예산을 높이거나 조건을 완화해 주세요.")
            st.stop()

        st.markdown("""
<style>
.card{padding:14px 16px;margin-bottom:12px;border-radius:12px;border:1px solid #eee;background:#fff}
.card h4{margin:0 0 6px 0;font-size:1.05rem}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #ff5a5f;margin-right:6px;font-size:0.85rem}
.kv{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:8px 10px;margin-top:6px}
.small{color:#666;font-size:0.9rem}
.tag{display:inline-block;background:#fff4f4;color:#c44;border:1px solid #fbb;padding:2px 6px;border-radius:6px;margin:2px;font-size:0.85rem}
</style>
        """, unsafe_allow_html=True)

        results.sort(key=lambda x:x['score'], reverse=True)
        top_k = 5
        if relaxed_used:
            st.info("조건에 정확히 맞는 메뉴가 부족하여, AI가 유사한 옵션을 함께 추천했습니다.")

        for rank, r in enumerate(results[:top_k], start=1):
            base_drink = r['drink']
            bakery_list = r['bakery']
            per_price = r['total_price_per_set']
            total_price = per_price * n_people

            # --- 인원수만큼 음료 추천 ---
            other_drinks = []
            if n_people > 1:
                available = drink_df[drink_df['name'] != base_drink['name']]
                available = available[
                    (available['sweetness'] >= min_drk) &
                    (available['sweetness'] <= max_drk)
                ]
                if sel_cats:
                    available = available[available['category'].isin(sel_cats)]
                if sel_drk_tags:
                    available = available[available['tags_list'].apply(lambda t: any(tag in sel_drk_tags for tag in t))]
                available = available.sort_values(by='popularity_score', ascending=False)
                num_add = min(n_people - 1, len(available))
                for _, row in available.head(num_add).iterrows():
                    other_drinks.append(row.to_dict())
            drink_list = [base_drink] + other_drinks

            def tag_html(tags):
                tags = [t for t in tags if t != '인기']
                return "".join([f"<span class='tag'>#{t}</span>" for t in tags]) if tags else "<span class='small'>태그 없음</span>"

            drink_html = "<br>".join([f"- {d['name']} ({d['price']:,}원)<br>{tag_html(d['tags_list'])}" for d in drink_list])
            bakery_html = "<br>".join([f"- {b['name']} ({b['price']:,}원)<br>{tag_html(b['tags_list'])}" for b in bakery_list])

            st.markdown(f"""
<div class="card">
  <h4>추천 세트 {rank} · 점수 {r['score']}점</h4>
  <span class="badge">1인 {per_price:,}원</span>
  <span class="badge">{n_people}명 총 {total_price:,}원</span>
  <div class="kv"><b>음료</b><br>{drink_html}</div>
  <div class="kv"><b>베이커리</b><br>{bakery_html}</div>
  <div class="small">※ 취향 태그와 메뉴 인기를 함께 반영해 순위를 매겼습니다.</div>
</div>
            """, unsafe_allow_html=True)

with tab_board:
    st.title("메뉴판")
    img1,img2 = load_image("menu_board_1.png"),load_image("menu_board_2.png")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("베이커리")
        if img1: st.image(img1, caption="Bakery 메뉴판", use_column_width=True)
        else: st.dataframe(bakery_df)
    with c2:
        st.subheader("음료")
        if img2: st.image(img2, caption="Drink 메뉴판", use_column_width=True)
        else: st.dataframe(drink_df)
