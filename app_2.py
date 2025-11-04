import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="🤖AI 베이커리 메뉴 추천 시스템", layout="wide")

# =========================
# 공용 유틸
# =========================
def normalize_columns(df: pd.DataFrame, is_drink: bool = False) -> pd.DataFrame:
    """필수 컬럼/타입 정규화"""
    menu_type = "음료" if is_drink else "베이커리"
    df.columns = [c.strip().lower() for c in df.columns]  # 대소문자/공백 정리
    required = ['name', 'price', 'sweetness', 'tags']
    if is_drink:
        required.append('category')
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"🚨 {menu_type} 파일에 필수 컬럼({', '.join(missing)})이 없습니다.")
        st.stop()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce')
    if df['price'].isnull().any() or df['sweetness'].isnull().any():
        st.error(f"🚨 {menu_type} 파일의 price/sweetness 컬럼에 잘못된 값이 있습니다.")
        st.stop()
    # 🔒 카테고리 문자열 표준화
    if is_drink and 'category' in df.columns:
        df['category'] = (
            df['category'].astype(str)
            .str.strip()
            .str.replace('  ', ' ', regex=False)
        )
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
    return set(t for sub in df['tags_list'] for t in sub if t)

def load_image(path: str):
    try:
        return Image.open(path)
    except Exception:
        return None

# =========================
# 데이터 로드
# =========================
try:
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df  = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True)
except Exception as e:
    st.error(f"🚨 CSV 파일 로드 오류: {e}")
    st.stop()

# 태그/인기도 전처리
bakery_df = assign_popularity_score(preprocess_tags(bakery_df))
drink_df  = assign_popularity_score(preprocess_tags(drink_df))

# 카테고리 옵션(공백 정리 후 유일값)
all_drink_categories = sorted(drink_df['category'].astype(str).str.strip().unique())

# 태그 후보
FLAVOR_TAGS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','초코','치즈'}
BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS  = uniq_tags(drink_df)
ui_bakery_utility_tags = sorted(BAKERY_TAGS - FLAVOR_TAGS)
ui_drink_flavor_tags   = sorted(DRINK_TAGS & FLAVOR_TAGS)

# =========================
# 추천 로직
# =========================
def filter_base(df, min_s, max_s, tags, max_price=None, categories=None, require_all=True):
    f = df.copy()

    # ✅ 음료 카테고리는 무조건 일치(공백 정리 포함)
    if 'category' in f.columns:
        if categories and len(categories) > 0:
            cats = [str(c).strip() for c in categories]
            f = f[f['category'].astype(str).str.strip().isin(cats)]
        else:
            # 카테고리 미선택이면 음료 추천 자체를 막아 혼동 방지
            return pd.DataFrame(columns=f.columns)

    # 당도 범위
    f = f[(f['sweetness'] >= min_s) & (f['sweetness'] <= max_s)]

    # 태그 필터
    if tags:
        if require_all:
            f = f[f['tags_list'].apply(lambda x: set(tags).issubset(set(x)))]
        else:
            f = f[f['tags_list'].apply(lambda x: not set(x).isdisjoint(set(tags)))]

    # 예산(단품 기준) 필터
    if max_price is not None and 'price' in f.columns:
        f = f[f['price'] <= max_price]

    return f

def make_recs(f, n_items, max_price=None):
    recs = []
    if f.empty:
        return recs
    if n_items == 1:
        for _, r in f.sort_values(['popularity_score','price'], ascending=[False,True]).iterrows():
            recs.append([r.to_dict()])
            if len(recs) >= 200: break
        return recs
    pool = f.sort_values('popularity_score', ascending=False).head(30)
    if len(pool) < n_items:
        recs.append([r.to_dict() for _, r in pool.iterrows()])
        return recs
    for combo in itertools.combinations(pool.itertuples(index=False), n_items):
        total_price = sum(c.price for c in combo)
        if (max_price is None) or (total_price <= max_price):
            recs.append([{col: getattr(c, col) for col in f.columns} for c in combo])
            if len(recs) >= 200: break
    return recs

def recommend_strict(df, min_s, max_s, tags, n_items, max_price=None, categories=None):
    f = filter_base(df, min_s, max_s, tags, max_price, categories, require_all=True)
    return make_recs(f, n_items, max_price)

def recommend_relaxed(df, min_s, max_s, tags, n_items, max_price=None, categories=None):
    # 1) ANY 태그
    f = filter_base(df, min_s, max_s, tags, max_price, categories, require_all=False)
    if not f.empty: return make_recs(f, n_items, max_price)
    # 2) 태그 무시
    f = filter_base(df, min_s, max_s, [], max_price, categories, require_all=True)
    if not f.empty: return make_recs(f, n_items, max_price)
    # 3) 당도 ±1
    f = filter_base(df, max(1, min_s-1), min(5, max_s+1), [], max_price, categories, require_all=True)
    if not f.empty: return make_recs(f, n_items, max_price)
    # 4) 인기순(카테고리+예산만)
    f = df.copy()
    if 'category' in f.columns and categories:
        cats = [str(c).strip() for c in categories]
        f = f[f['category'].astype(str).str.strip().isin(cats)]
    if max_price is not None:
        f = f[f['price'] <= max_price]
    return make_recs(f.sort_values('popularity_score', ascending=False), n_items, max_price)

def calc_score(items, selected_tags):
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
    st.title("🤖AI 메뉴 추천 시스템")
    st.caption("고객님의 취향과 인기 정보를 함께 반영해 AI가 추천해드립니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("인원/예산")
        n_people = st.number_input("인원 수", 1, 10, 2)
        unlimited = st.checkbox("예산 무제한", value=True)
        if unlimited:
            max_budget = None
            st.slider("최대 예산(1인)", 5000, 50000, 50000, 1000, disabled=True)
        else:
            max_budget = st.slider("최대 예산(1인)", 5000, 50000, 15000, 1000)

    with c2:
        st.subheader("베이커리")
        n_bakery = st.slider("베이커리 개수", 1, 5, 2)
        min_bak, max_bak = st.slider("베이커리 당도", 1, 5, (1,5))
        sel_bak_tags = st.multiselect("베이커리 태그", ui_bakery_utility_tags, max_selections=3)

    with c3:
        st.subheader("음료")
        sel_cats = st.multiselect("음료 카테고리", all_drink_categories, default=all_drink_categories)
        min_drk, max_drk = st.slider("음료 당도", 1, 5, (1,5))
        sel_drk_tags = st.multiselect("음료 맛 태그", ui_drink_flavor_tags, max_selections=3)

    st.markdown("---")

    if st.button("AI 추천 메뉴 보기👇", type="primary", use_container_width=True):
        # 1) 엄격 매칭
        drink_recs  = recommend_strict(drink_df,  min_drk, max_drk, sel_drk_tags, 1,        max_budget, sel_cats)
        bakery_recs = recommend_strict(bakery_df, min_bak, max_bak, sel_bak_tags, n_bakery, max_budget)
        relaxed_used = False
        # 2) 부족하면 완화
        if not drink_recs:
            drink_recs = recommend_relaxed(drink_df,  min_drk, max_drk, sel_drk_tags, 1,        max_budget, sel_cats)
            relaxed_used = True
        if not bakery_recs:
            bakery_recs = recommend_relaxed(bakery_df, min_bak, max_bak, sel_bak_tags, n_bakery, max_budget)
            relaxed_used = True

        if not drink_recs and not bakery_recs:
            st.warning("조건에 맞는 메뉴가 없습니다. 태그나 당도를 완화해 주세요.")
            st.stop()

        # 조합 생성 + 점수
        results = []
        for d_combo, b_combo in itertools.product(drink_recs or [[]], bakery_recs or [[]]):
            per_price = (d_combo[0]['price'] if d_combo else 0) + sum(x['price'] for x in b_combo)
            if (max_budget is None) or (per_price <= max_budget):
                items = (d_combo or []) + b_combo
                score = calc_score(items, sel_drk_tags + sel_bak_tags)
                results.append({
                    "score": score,
                    "drink": d_combo[0] if d_combo else None,
                    "bakery": b_combo,
                    "per_price": per_price
                })
            if len(results) >= 200: break

        if not results:
            st.warning("예산에 맞는 메뉴가 없습니다. 조건을 완화해 주세요.")
            st.stop()

        # 스타일
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

        results.sort(key=lambda x: x['score'], reverse=True)
        if relaxed_used:
            st.info("조건에 정확히 맞는 메뉴가 부족해요😅 AI가 유사한 메뉴를 함께 추천했습니다🥰")

        for rank, r in enumerate(results[:5], start=1):
            base_drink = r['drink']
            bakery_list = r['bakery']
            per_price   = r['per_price']
            total_price = per_price * n_people

            # 인원수만큼 음료 추천(카테고리/태그/당도 강제)
            drink_list = []
            if base_drink:
                drink_list.append(base_drink)
            if n_people > 1:
                available = drink_df[drink_df['name'] != (base_drink['name'] if base_drink else "")]
                # 카테고리 강제(공백 정리)
                cats = [str(c).strip() for c in sel_cats] if sel_cats else []
                if cats:
                    available = available[available['category'].astype(str).str.strip().isin(cats)]
                available = available[(available['sweetness'] >= min_drk) & (available['sweetness'] <= max_drk)]
                if sel_drk_tags:
                    available = available[available['tags_list'].apply(lambda t: not set(t).isdisjoint(set(sel_drk_tags)))]
                available = available.sort_values('popularity_score', ascending=False)
                need = max(0, n_people - len(drink_list))
                for _, row in available.head(need).iterrows():
                    drink_list.append(row.to_dict())

            def tags_html(tags):
                t = [f"<span class='tag'>#{x}</span>" for x in tags if x != '인기']
                return "".join(t) if t else "<span class='small'>태그 없음</span>"

            drink_html  = "<br>".join([f"- {d['name']} ({d['price']:,}원)<br>{tags_html(d['tags_list'])}" for d in drink_list])
            bakery_html = "<br>".join([f"- {b['name']} ({b['price']:,}원)<br>{tags_html(b['tags_list'])}" for b in bakery_list])

            st.markdown(f"""
<div class="card">
  <h4>추천 세트 {rank} · 점수 {r['score']}점</h4>
  <span class="badge">1인 {per_price:,}원</span>
  <span class="badge">{n_people}명 총 {total_price:,}원</span>
  <div class="kv"><b>음료</b><br>{drink_html}</div>
  <div class="kv"><b>베이커리</b><br>{bakery_html}</div>
  <div class="small">※ AI가 취향 태그와 인기 정보를 바탕으로 추천했습니다.</div>
</div>
            """, unsafe_allow_html=True)

with tab_board:
    st.title("메뉴판")
    img1, img2 = load_image("menu_board_1.png"), load_image("menu_board_2.png")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("베이커리 메뉴")
        if img1: st.image(img1, caption="Bakery 메뉴판", use_column_width=True)
        else: st.dataframe(bakery_df)
    with c2:
        st.subheader("음료 메뉴")
        if img2: st.image(img2, caption="Drink 메뉴판", use_column_width=True)
        else: st.dataframe(drink_df)
