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
    required_cols = ['name', 'price', 'sweetness', 'tags']
    if is_drink:
        required_cols.append('category')

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"🚨 {menu_type} 파일에 필수 컬럼({', '.join(missing)})이 없습니다.")
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
        .str.replace('#', '')
        .str.replace(';', ',')
        .str.split(r'\s*,\s*')
        .apply(lambda xs: [t.strip() for t in xs if t.strip()])
    )
    return df

def assign_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(lambda ts: 10 if '인기' in ts else 5)
    return df

def uniq_tags(df: pd.DataFrame) -> set:
    return set(t for sub in df['tags_list'] for t in sub if t)  # '인기' 포함

def load_image(image_path: str):
    try:
        return Image.open(image_path)
    except Exception:
        return None

# =========================
# 데이터 로드
# =========================
try:
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df  = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True)
except Exception as e:
    st.error(f"🚨 CSV 파일 오류: {e}")
    st.stop()

bakery_df = preprocess_tags(bakery_df)
drink_df  = preprocess_tags(drink_df)
bakery_df = assign_popularity_score(bakery_df)
drink_df  = assign_popularity_score(drink_df)

FLAVOR_TAGS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','씁쓸한','초코','치즈'}
BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS  = uniq_tags(drink_df)
ui_bakery_utility_tags = sorted(BAKERY_TAGS - FLAVOR_TAGS)
ui_drink_flavor_tags   = sorted(DRINK_TAGS & FLAVOR_TAGS)
all_drink_categories   = sorted(drink_df['category'].astype(str).unique())

# =========================
# 추천 로직
# =========================
def _match_all_tags(row_tags, selected_tags) -> bool:
    if not selected_tags:
        return True
    return set(selected_tags).issubset(set(row_tags))

def recommend_menu(df, min_sweet, max_sweet, selected_tags, n_items, max_price=None, selected_categories=None):
    f = df.copy()
    is_drink = 'category' in f.columns
    if is_drink and selected_categories:
        f = f[f['category'].isin(selected_categories)]
    f = f[(f['sweetness'] >= min_sweet) & (f['sweetness'] <= max_sweet)]
    if selected_tags:
        f = f[f['tags_list'].apply(lambda t: _match_all_tags(t, selected_tags))]
    if f.empty:
        return []
    recs = []
    if n_items == 1:
        for _, r in f.iterrows():
            if (max_price is None) or (r['price'] <= max_price):
                recs.append([r.to_dict()])
                if len(recs) >= 200:
                    break
        return recs
    if len(f) < n_items:
        top = f.sort_values('popularity_score', ascending=False)
        recs.append([r.to_dict() for _, r in top.iterrows()])
        return recs
    subset = f.sort_values('popularity_score', ascending=False).head(30)
    for combo in itertools.combinations(subset.itertuples(index=False), n_items):
        price_sum = sum(c.price for c in combo)
        if (max_price is None) or (price_sum <= max_price):
            recs.append([{k: getattr(c, k) for k in f.columns} for c in combo])
            if len(recs) >= 200:
                break
    return recs

def calculate_weighted_score(combo_items, selected_tags):
    if not selected_tags:
        tag_score = 100
    else:
        total = len(combo_items)
        match = sum(1 for i in combo_items if not set(i['tags_list']).isdisjoint(selected_tags))
        tag_score = (match / total) * 100 if total else 0
    avg_pop = sum(i['popularity_score'] for i in combo_items)/len(combo_items) if combo_items else 0
    return round(tag_score*0.7 + avg_pop*10*0.3, 1)

# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["AI 메뉴 추천", "메뉴판"])

with tab1:
    st.title("💡 AI 메뉴 추천 시스템")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 인원/예산")
        n_people = st.number_input("인원수", 1, 10, 2)
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        if budget_unlimited:
            max_budget = None
            st.slider("최대 예산(1인)", 5000, 50000, 50000, 1000, disabled=True)
        else:
            max_budget = st.slider("최대 예산(1인)", 5000, 50000, 15000, 1000)

    with col2:
        st.markdown("#### 🍞 베이커리")
        n_bakery = st.slider("베이커리 개수", 1, 5, 2)
        min_bak, max_bak = st.slider("베이커리 당도", 1, 5, (1,5))
        selected_bakery_tags = st.multiselect("베이커리 태그", ui_bakery_utility_tags, max_selections=3)

    with col3:
        st.markdown("#### ☕ 음료")
        selected_categories = st.multiselect("음료 카테고리", all_drink_categories, default=all_drink_categories)
        min_drk, max_drk = st.slider("음료 당도", 1, 5, (1,5))
        selected_drink_tags = st.multiselect("음료 맛 태그", ui_drink_flavor_tags, max_selections=3)

    st.markdown("---")

    if st.button("AI 추천 메뉴 조합 받기", use_container_width=True, type="primary"):
        st.markdown("### 🏆 AI 추천 메뉴 조합")

        drink_recs = recommend_menu(drink_df, min_drk, max_drk, selected_drink_tags, 1, max_budget, selected_categories)
        bakery_recs = recommend_menu(bakery_df, min_bak, max_bak, selected_bakery_tags, n_bakery, max_budget)
        if not drink_recs and not bakery_recs:
            st.warning("일치하는 메뉴가 없어요.")
            st.stop()

        results = []
        all_pairs = list(itertools.product(drink_recs or [[]], bakery_recs or [[]]))
        random.shuffle(all_pairs)

        for d_combo, b_combo in all_pairs:
            total_price = sum(i['price'] for i in (d_combo+b_combo))
            if (max_budget is None) or (total_price <= max_budget):
                score = calculate_weighted_score(d_combo+b_combo, selected_bakery_tags+selected_drink_tags)
                results.append({
                    "score": score,
                    "drink": d_combo[0] if d_combo else None,
                    "bakery": b_combo,
                    "total_price": total_price
                })

        if not results:
            if max_budget is None:
                st.warning("태그/당도 조건을 완화해주세요.")
            else:
                st.warning("예산이 적습니다. 조건을 다시 설정해주세요")
            st.stop()

        compact = st.checkbox("요약 보기", value=True)
        results.sort(key=lambda x:x['score'], reverse=True)
        topN = st.slider("표시 개수", 3, 20, 5)
        top = results[:topN]

        df = pd.DataFrame([{
            "순위":i+1,
            "점수":r['score'],
            "음료": r['drink']['name'] if r['drink'] else "-",
            "베이커리": ", ".join(b['name'] for b in r['bakery']),
            "1인세트(원)": r['total_price'],
            f"{n_people}명합계(원)": r['total_price']*n_people
        } for i,r in enumerate(top)])
        st.dataframe(df, use_container_width=True, height=300)

        pick = st.selectbox("상세보기 순위 선택", [r["순위"] for _,r in df.iterrows()], index=0)
        chosen = top[pick-1]

        if compact:
            st.markdown(f"#### 세트 {pick} — {chosen['score']}점")
            c1,c2=st.columns(2)
            with c1:
                d=chosen['drink']
                if d: st.info(f"{d['name']} ({d['price']:,}원) 당도 {d['sweetness']}")
            with c2:
                for b in chosen['bakery']:
                    st.success(f"{b['name']} ({b['price']:,}원) 당도 {b['sweetness']}")
        else:
            with st.expander("세부 보기", expanded=True):
                if chosen['drink']:
                    d=chosen['drink']
                    st.markdown(f"**음료:** {d['name']} ({d['price']:,}원)")
                for b in chosen['bakery']:
                    st.markdown(f"**베이커리:** {b['name']} ({b['price']:,}원)")

with tab2:
    st.title("📋 메뉴판")
    img1,img2=load_image("menu_board_1.png"),load_image("menu_board_2.png")
    c1,c2=st.columns(2)
    with c1:
        if img1: st.image(img1,caption="베이커리 메뉴")
        else: st.dataframe(bakery_df)
    with c2:
        if img2: st.image(img2,caption="음료 메뉴")
        else: st.dataframe(drink_df)
