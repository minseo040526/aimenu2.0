import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")
FONT_NAME = "Jua"  # 사용 폰트(스트림릿 테마에서 설정했다면 생략 가능)

# =========================
# 유틸 함수
# =========================
def normalize_columns(df: pd.DataFrame, is_drink: bool = False) -> pd.DataFrame:
    """필수 컬럼/타입 확인 및 정규화"""
    menu_type = "음료" if is_drink else "베이커리"

    required_cols = ['name', 'price', 'sweetness', 'tags']
    if is_drink:
        required_cols.append('category')

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"🚨 오류: {menu_type} 파일에 필수 컬럼({', '.join(missing)})이 없습니다.")
        st.stop()

    # 타입 정리
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce')

    if df['price'].isnull().any():
        st.error(f"🚨 오류: {menu_type} 파일의 'price'에 숫자가 아닌 값이 있습니다.")
        st.stop()
    if df['sweetness'].isnull().any():
        st.error(f"🚨 오류: {menu_type} 파일의 'sweetness'에 숫자가 아닌 값이 있습니다(1~5).")
        st.stop()

    return df

def preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:
    """tags -> tags_list (리스트화)"""
    df['tags_list'] = (
        df['tags'].fillna('').astype(str)
        .str.replace('#', '')
        .str.replace(';', ',')
        .str.split(r'\s*,\s*')
        .apply(lambda xs: [t.strip() for t in xs if t.strip()])
    )
    return df

def assign_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    """#인기 태그 있으면 10, 없으면 5"""
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(lambda ts: 10 if '인기' in ts else 5)
    return df

def uniq_tags(df: pd.DataFrame) -> set:
    return set(t for sub in df['tags_list'] for t in sub if t)

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
    if bakery_df.empty or drink_df.empty:
        st.error("🚨 오류: 메뉴 데이터가 비어 있습니다.")
        st.stop()
except FileNotFoundError:
    st.error("🚨 오류: CSV 파일을 찾을 수 없습니다. (Bakery_menu.csv, Drink_menu.csv)")
    st.stop()
except Exception as e:
    st.error(f"🚨 CSV 로드 오류: {e}")
    st.stop()

# 전처리
bakery_df = preprocess_tags(bakery_df)
drink_df  = preprocess_tags(drink_df)
bakery_df = assign_popularity_score(bakery_df)
drink_df  = assign_popularity_score(drink_df)

# 태그 집합
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
    """선택 태그 ALL 매칭 (모두 포함해야 통과)"""
    if not selected_tags:
        return True
    return set(selected_tags).issubset(set(row_tags))

def recommend_menu(df: pd.DataFrame,
                   min_sweetness: int, max_sweetness: int,
                   selected_tags: list, n_items: int,
                   max_price: float = None, selected_categories: list = None):
    """
    필터링 후 추천 조합 반환
    - 태그는 ALL 매칭
    - 조합 불가 시 가능한 만큼이라도 반환(폴백)
    """
    filtered_df = df.copy()
    is_drink = 'category' in filtered_df.columns

    if is_drink and selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    filtered_df = filtered_df[
        (filtered_df['sweetness'] >= min_sweetness) &
        (filtered_df['sweetness'] <= max_sweetness)
    ]

    if selected_tags:
        filtered_df = filtered_df[filtered_df['tags_list'].apply(lambda t: _match_all_tags(t, selected_tags))]

    if filtered_df.empty:
        return []

    recs = []

    if n_items == 1:
        items = filtered_df.sort_values(by=['popularity_score', 'price'], ascending=[False, True])
        for _, r in items.iterrows():
            if (max_price is None) or (r['price'] <= max_price):
                recs.append([{
                    'name': r['name'],
                    'price': r['price'],
                    'tags': r['tags_list'],
                    'popularity': r['popularity_score'],
                    'sweetness': r['sweetness']
                }])
                if len(recs) >= 200:
                    break
        return recs

    # 복수 조합(베이커리 등)
    if len(filtered_df) < n_items:
        # 조합 불가 → 가능한 만큼 묶어서 1세트처럼 반환
        top = filtered_df.sort_values('popularity_score', ascending=False).head(max(1, len(filtered_df)))
        combo = [{
            'name': r['name'], 'price': r['price'], 'tags': r['tags_list'],
            'popularity': r['popularity_score'], 'sweetness': r['sweetness']
        } for _, r in top.iterrows()]
        recs.append(combo)
        return recs

    subset = filtered_df.sort_values('popularity_score', ascending=False).head(30) if len(filtered_df) > 30 else filtered_df
    combos = list(itertools.combinations(subset.itertuples(index=False), n_items))
    random.shuffle(combos)

    for c in combos:
        total_price = sum(i.price for i in c)
        if (max_price is None) or (total_price <= max_price):
            recs.append([{
                'name': i.name, 'price': i.price, 'tags': i.tags_list,
                'popularity': i.popularity_score, 'sweetness': i.sweetness
            } for i in c])
            if len(recs) >= 200:
                break
    return recs

def calculate_weighted_score(combo_items: list, selected_tags: list) -> float:
    """태그 일치(70) + 인기(30) 가중 평균"""
    # 태그 일치도
    if not selected_tags:
        tag_match_score = 100.0
    else:
        total = len(combo_items)
        if total == 0:
            tag_match_score = 0.0
        else:
            s = set(selected_tags)
            matches = 0
            for item in combo_items:
                if not s.isdisjoint(set(item['tags'])):
                    matches += 1
            tag_match_score = (matches / total) * 100.0

    # 인기 점수
    if not combo_items:
        pop100 = 0.0
    else:
        avg_pop = sum(x['popularity'] for x in combo_items) / len(combo_items)
        pop100 = avg_pop * 10.0  # 10점 만점 → 100 환산

    final_score = round(tag_match_score * 0.7 + pop100 * 0.3, 1)
    return final_score

# =========================
# UI
# =========================
tab_reco, tab_board = st.tabs(["AI 메뉴 추천", "메뉴판"])

with tab_reco:
    st.title("💡 AI 메뉴 추천 시스템")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 👤 인원 & 💰 예산 (인원 수만큼 음료를 추천해드려요!)")
        n_people = st.number_input("인원수", min_value=1, max_value=10, value=2, step=1)
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        if budget_unlimited:
            max_budget = float('inf')
            st.slider("최대 예산(1인)", 5000, 50000, 50000, 1000, disabled=True)
        else:
            max_budget = st.slider("최대 예산(1인)", 5000, 50000, 15000, 1000)

    with col2:
        st.markdown("#### 🍞 베이커리")
        n_bakery = st.slider("베이커리 개수", 1, 5, 2, 1)
        min_bak, max_bak = st.slider("베이커리 당도", 1, 5, (1, 5), 1)
        selected_bakery_tags = st.multiselect(
            "베이커리 태그(모두 포함, 최대3개)",
            options=ui_bakery_utility_tags, default=[], max_selections=3,
            placeholder="예: 든든한, 겉바속촉, 가벼운"
        )

    with col3:
        st.markdown("#### ☕ 음료")
        selected_categories = st.multiselect("카테고리", options=all_drink_categories, default=all_drink_categories)
        min_drk, max_drk = st.slider("음료 당도", 1, 5, (1, 5), 1)
        selected_drink_tags = st.multiselect(
            "음료 맛 태그(모두 포함, 최대3개)",
            options=ui_drink_flavor_tags, default=[], max_selections=3,
            placeholder="예: 깔끔한, 쌉싸름한, 상큼한"
        )

    st.markdown("---")

if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
    st.markdown("### 🏆 AI 추천 메뉴 조합")
        # --- 추천 조합 생성 ---
    # 음료 추천
    drink_recs = recommend_menu(
        drink_df, 
        min_drk, max_drk, 
        selected_drink_tags, 
        1, 
        max_price=max_budget, 
        selected_categories=selected_categories
    )

    # 베이커리 추천
    bakery_recs = recommend_menu(
        bakery_df, 
        min_bak, max_bak, 
        selected_bakery_tags, 
        n_bakery, 
        max_price=max_budget
    )

    results = []  # ✅ 결과 저장용 리스트
    if not results:
        st.warning("예산에 맞는 메뉴가 없습니다. 조건 설정을 다시 해주세요")
else:
    # ====== 컴팩트 모드 ======
    compact = st.checkbox("컴팩트 보기", value=True)

    # 점수순 정렬 + 상위 N 선택
    results.sort(key=lambda x: x['score'], reverse=True)
    topN = st.slider("표시 개수", 3, 20, 6, 1)
    top = results[:topN]

    import pandas as pd
    table = []
    for i, r in enumerate(top, start=1):
        drink_name = r['drink']['name']
        bakery_names = ", ".join([b['name'] for b in r['bakery']])
        table.append({
            "순위": i,
            "점수": r['score'],
            "음료": drink_name,
            "베이커리": bakery_names,
            "1인가격(원)": r['total_price_per_set'],
            f"{n_people}명합계(원)": r['total_price_for_n_people'],
        })
    df = pd.DataFrame(table)

    st.dataframe(df, use_container_width=True, height=min(320, 60 + 35 * len(df)))

    pick = st.selectbox("상세 볼 세트 선택(순위)", options=[row["순위"] for row in table], index=0)
    chosen = top[pick-1]

    if compact:
        st.markdown(f"#### 세트 {pick} — 점수 **{chosen['score']} / 100**")
        st.markdown(f"- 1인 세트: **{chosen['total_price_per_set']:,}원** / {n_people}명 합계: **{chosen['total_price_for_n_people']:,}원**")

        c1, c2 = st.columns(2)
        with c1:
            d = chosen['drink']
            st.subheader("🥤 음료")
            st.info(f"• {d['name']} ({d['price']:,}원) | 당도 {d['sweetness']}")
        with c2:
            st.subheader("🍞 베이커리")
            for b in chosen['bakery']:
                st.success(f"• {b['name']} ({b['price']:,}원) | 당도 {b['sweetness']}")
    else:
        st.markdown(f"#### 세트 {pick} — 점수 **{chosen['score']} / 100**")
        st.markdown(f"- 1인 세트: **{chosen['total_price_per_set']:,}원** / {n_people}명 합계: **{chosen['total_price_for_n_people']:,}원**")
        with st.expander("상세 메뉴 보기", expanded=True):
            st.markdown("##### 음료 🥤")
            d = chosen['drink']
            st.info(f"• {d['name']} ({d['price']:,}원) — 당도 {d['sweetness']}")
            st.markdown("##### 베이커리 🍞")
            for item in chosen['bakery']:
                st.success(f"• {item['name']} ({item['price']:,}원) — 당도 {item['sweetness']}")
with tab_board:
    st.title("📋 메뉴판")
    st.markdown("---")
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("베이커리 메뉴")
        if img1: st.image(img1, caption="Bakery 메뉴판", use_column_width=True)
        else:
            df = bakery_df.rename(columns={'name':'메뉴','price':'가격','sweetness':'당도','tags':'태그'})
            df['인기점수'] = df['popularity_score']
            st.dataframe(df[['메뉴','가격','당도','태그','인기점수']], use_container_width=True)

    with c2:
        st.subheader("음료 메뉴")
        if img2: st.image(img2, caption="Drink 메뉴판", use_column_width=True)
        else:
            df = drink_df.rename(columns={'name':'메뉴','price':'가격','sweetness':'당도','tags':'태그','category':'카테고리'})
            df['인기점수'] = df['popularity_score']
            st.dataframe(df[['메뉴','가격','카테고리','당도','태그','인기점수']], use_container_width=True)
