import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

font = "Jua"
# --- 데이터 로드 및 컬럼 정규화 함수 ---

def normalize_columns(df, is_drink=False):
    """'sweetness' 컬럼 및 필수 컬럼을 확인합니다."""
    
    # 2. 필수 컬럼 확인
    required_cols = ['name', 'price', 'sweetness', 'tags']
    if is_drink:
        required_cols.append('category')

    # 현재 df에 없는 필수 컬럼 목록
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        col_names = ", ".join(missing_cols)
        menu_type = "음료" if is_drink else "베이커리"
        st.error(f"🚨 오류: {menu_type} 파일에 필수 컬럼({col_names})이 없습니다. 'name', 'price', 'sweetness', 'tags' 컬럼을 확인해주세요. 음료는 추가로 'category' 컬럼이 필요합니다.")
        st.stop()
    
    # 'sweetness' 컬럼이 숫자인지 확인
    try:
        df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce')
        if df['sweetness'].isnull().any():
            st.error(f"🚨 오류: {menu_type} 파일의 'sweetness' 컬럼에 숫자가 아닌 값이 포함되어 있습니다. 1~5 사이의 숫자로 입력해주세요.")
            st.stop()
    except Exception:
        st.error(f"🚨 오류: {menu_type} 파일의 'sweetness' 컬럼을 숫자로 변환할 수 없습니다.")
        st.stop()
        
    return df

try:
    # CSV 파일 로드 및 컬럼 정규화 적용
    # 'Bakery_menu.csv'와 'Drink_menu.csv' 파일이 필요합니다.
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True) 

    if drink_df.empty or bakery_df.empty:
        st.error("🚨 오류: 메뉴 데이터가 비어 있습니다. 파일 내용을 확인해주세요.")
        st.stop()

except FileNotFoundError:
    st.error("🚨 오류: 메뉴 CSV 파일을 찾을 수 없습니다. 파일 이름과 경로(Bakery_menu.csv, Drink_menu.csv)를 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 오류: 메뉴 CSV 파일을 로드하는 중 알 수 없는 오류가 발생했습니다: {e}")
    st.stop()

# --- 데이터 전처리 및 태그 추출 ---
def preprocess_tags(df):
    """CSV의 tags 컬럼을 클린징하고 리스트로 변환합니다."""
    # 태그 컬럼에서 #을 제거하고 쉼표(,)를 기준으로 분리하며 공백을 제거합니다.
    df['tags_list'] = df['tags'].fillna('').astype(str).str.strip().str.replace('#', '').str.split(r'\s*,\s*')
    df['tags_list'] = df['tags_list'].apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    return df

bakery_df = preprocess_tags(bakery_df)
drink_df = preprocess_tags(drink_df)

# --- 인기도 점수 동적 생성 로직 ---
def assign_popularity_score(df):
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(
            lambda tags: 10 if '인기' in tags else 5
        )
    return df

bakery_df = assign_popularity_score(bakery_df)
drink_df = assign_popularity_score(drink_df)
# --------------------------------------------------------

# 전체 사용 가능한 태그 및 카테고리 추출
def uniq_tags(df):
    return set(t for sub in df['tags_list'] for t in sub if t and t != '인기')

BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS = uniq_tags(drink_df)
all_drink_categories = sorted(drink_df['category'].unique())

# --- 태그 그룹 분리 ---
# NOTE: 맛/당도 태그는 이제 CSV의 'sweetness' 컬럼(숫자)과 슬라이더로 조절되지만, 
# '달콤한', '쌉싸름한' 등 문자열 태그가 남아있다면 별도로 분리해야 합니다.

# 베이커리/음료 공통으로 사용되는 '맛/당도' 관련 태그 (필터링이 아닌, 사용자 선택을 위한 리스트)
# CSV에 있는 태그 중, 맛과 관련된 태그를 정의합니다. (주로 음료나 디저트에 사용)
FLAVOR_TAGS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','씁쓸한', '초코', '치즈'} 

# 1. 음료 전용 '맛' 태그 (DRINK_TAGS 중 FLAVOR_TAGS만)
ui_drink_flavor_tags = sorted(DRINK_TAGS & FLAVOR_TAGS)

# 2. 베이커리 전용 '식감/용도' 태그 (BAKERY_TAGS 중 FLAVOR_TAGS를 제외한 나머지)
ui_bakery_utility_tags = sorted(BAKERY_TAGS - FLAVOR_TAGS) 


# --- 추천 로직 함수 ---
def recommend_menu(df, min_sweetness, max_sweetness, selected_tags, n_items, max_price=None, selected_categories=None):
    """
    주어진 조건으로 메뉴 조합을 추천합니다.
    - min_sweetness, max_sweetness: 당도 슬라이더 범위 (숫자)
    - selected_tags: 메뉴별 선호 태그 (문자열 리스트)
    """

    filtered_df = df.copy()
    is_drink_menu = 'category' in filtered_df.columns
    
    # 1. 카테고리 필터링 (음료에만 해당)
    if is_drink_menu and selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    # 2. 당도 필터링 (숫자 슬라이더)
    filtered_df = filtered_df[
        (filtered_df['sweetness'] >= min_sweetness) & 
        (filtered_df['sweetness'] <= max_sweetness)
    ]
    
    # 3. 선호 태그 필터링
    if selected_tags:
        # 선택된 태그 중 하나라도 메뉴의 tags_list에 포함되면 유지
        temp_filtered_df = filtered_df[filtered_df['tags_list'].apply(lambda tags: any(tag in selected_tags for tag in tags))]
        if not temp_filtered_df.empty:
            filtered_df = temp_filtered_df

    # 필터링 결과가 없으면 종료
    if filtered_df.empty:
        return []

    recommendations = []
    
    # --- 4. 조합 또는 단일 아이템 추천 ---
    
    if n_items == 1: # 단일 아이템 추천 (음료)
        items = filtered_df.sort_values(by=['popularity_score', 'price'], ascending=[False, True])
        
        for _, row in items.iterrows():
            if max_price is None or row['price'] <= max_price:
                recommendations.append([{
                    'name': row['name'], 
                    'price': row['price'], 
                    'tags': row['tags_list'],
                    'popularity': row['popularity_score'],
                    'sweetness': row['sweetness'] 
                }])
                if len(recommendations) >= 200: 
                    break
    else: # n_items > 1 (베이커리 조합)
        if len(filtered_df) > 30: 
            subset = filtered_df.sort_values(by='popularity_score', ascending=False).head(30)
        else:
            subset = filtered_df

        all_combinations = list(itertools.combinations(subset.itertuples(index=False), n_items))
        random.shuffle(all_combinations)

        for combo in all_combinations:
            total_price = sum(item.price for item in combo)
            if max_price is None or total_price <= max_price:
                combo_result = [{
                    'name': item.name, 
                    'price': item.price, 
                    'tags': item.tags_list,
                    'popularity': item.popularity_score,
                    'sweetness': item.sweetness 
                } for item in combo]
                recommendations.append(combo_result)
                if len(recommendations) >= 200:
                    break
    
    return recommendations


# --- 가중치 기반 점수 계산 함수 ---
# 선택된 '유틸리티/식감' 태그에 대해서만 일치도를 계산합니다.
def calculate_weighted_score(combo_items, selected_tags):
    """
    선택된 태그 일치도(70%)와 인기 점수(30%)를 가중 평균하여 최종 점수(100점 만점)를 계산
    """
    
    # --- 1. 태그 일치도 (Tag Match Score) 계산 (70% 가중치) ---
    if not selected_tags:
        # 태그를 선택하지 않은 경우, 태그 일치도를 최고점인 100점으로 간주
        tag_match_score = 100 
    else:
        total_items = len(combo_items)
        if total_items == 0:
            tag_match_score = 0
        else:
            total_matches = 0
            selected_tags_set = set(selected_tags)

            for item in combo_items:
                item_tags_set = set(item['tags'])
                # 선택된 태그 중 하나라도 메뉴의 태그에 포함되면 일치로 간주
                if not item_tags_set.isdisjoint(selected_tags_set):
                    total_matches += 1 
            
            # (일치하는 메뉴 수 / 전체 메뉴 수) * 100
            tag_match_score = (total_matches / total_items) * 100

    # --- 2. 인기 점수 (Popularity Score) 계산 (30% 가중치) ---
    if not combo_items:
        avg_popularity_score = 0
    else:
        total_popularity = sum(item['popularity'] for item in combo_items)
        avg_popularity_score = total_popularity / len(combo_items) 
    
    # 인기 점수를 100점 만점으로 변환 (인기 10점 만점 기준)
    popularity_score_100 = avg_popularity_score * 10 
    
    # --- 3. 최종 가중치 점수 계산 (100점 만점) ---
    WEIGHT_TAG = 0.7
    WEIGHT_POPULARITY = 0.3
    
    final_score = (tag_match_score * WEIGHT_TAG) + (popularity_score_100 * WEIGHT_POPULARITY)
    
    return round(final_score, 1)


# --- Streamlit 앱 구성 ---

st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# Image loading function (in case file is missing)
def load_image(image_path):
    try:
        # NOTE: 이 코드는 파일 시스템에 "menu_board_1.png"와 "menu_board_2.png" 파일이 존재해야 정상 작동합니다.
        return Image.open(image_path)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# --- 탭 구성 ---
tab_recommendation, tab_menu_board = st.tabs(["AI 메뉴 추천", "메뉴판"])


with tab_recommendation:
    st.title("💡 AI 메뉴 추천 시스템")
    st.subheader("예산, 당도, 카테고리, 취향, 인기를 고려한 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션 (5개의 컬럼으로 분할)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown("#### 👤 인원수 & 💰 예산")
        n_people = st.number_input("인원수", min_value=1, max_value=10, value=2, step=1)
        
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        if budget_unlimited:
            budget = float('inf') 
            st.slider("최대 예산 설정", min_value=5000, max_value=50000, value=50000, step=1000, disabled=True)
        else:
            budget = st.slider("최대 예산 설정 (1인 기준)", min_value=5000, max_value=50000, value=15000, step=1000)

    with col2:
        st.markdown("#### 🍞 베이커리 옵션")
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)
        
        bakery_sweetness_range = st.slider(
            "선호 베이커리 당도 레벨",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
            key='bakery_sweetness_slider'
        )
        min_bakery_sweetness, max_bakery_sweetness = bakery_sweetness_range

        selected_bakery_tags = st.multiselect(
            "선호 베이커리 태그 (식감/용도, 최대 3개)",
            options=ui_bakery_utility_tags,
            default=[],
            max_selections=3,
            placeholder="예: 든든한, 겉바속촉, 가벼운",
            key='bakery_tags_multiselect'
        )
        
    with col3:
        st.markdown("#### ☕ 음료 옵션")
        selected_categories = st.multiselect(
            "선호 음료 카테고리",
            options=all_drink_categories,
            default=all_drink_categories,
            placeholder="예: 커피, 티",
        )

        drink_sweetness_range = st.slider(
            "선호 음료 당도 레벨",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
            key='drink_sweetness_slider'
        )
        min_drink_sweetness, max_drink_sweetness = drink_sweetness_range

        selected_drink_tags = st.multiselect(
            "선호 음료 맛 태그 (최대 3개)",
            options=ui_drink_flavor_tags,
            default=[],
            max_selections=3,
            placeholder="예: 깔끔한, 쌉싸름한, 상큼한",
            key='drink_tags_multiselect'
        )
        
    with col4:
        st.markdown("#### 📊 메뉴 데이터 확인")
        # 데이터프레임의 상위 5개 행을 보여주는 간결한 요약
        st.dataframe(bakery_df.head(5).rename(columns={'name':'메뉴', 'price':'가격', 'sweetness':'당도', 'tags':'태그'}), height=200)
        st.dataframe(drink_df.head(5).rename(columns={'name':'메뉴', 'price':'가격', 'sweetness':'당도', 'tags':'태그', 'category':'카테고리'}), height=200)

    
    st.markdown("---")

    # 2. 추천 실행 버튼
    if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
        st.markdown("### 🏆 AI 추천 메뉴 조합")
        
        max_price_per_set = budget

        # --- 추천 생성 (1인 세트 기준) ---
        
        # 1. 음료 추천 (1개)
        drink_recommendations = recommend_menu(
            drink_df, 
            min_drink_sweetness, max_drink_sweetness, 
            selected_drink_tags, 1, 
            max_price=max_price_per_set, 
            selected_categories=selected_categories
        )
        
        # 2. 베이커리 추천 (n_bakery 개)
        bakery_recommendations = recommend_menu(
            bakery_df, 
            min_bakery_sweetness, max_bakery_sweetness, 
            selected_bakery_tags, n_bakery, 
            max_price=max_price_per_set
        )
        
        
        if not drink_recommendations or not bakery_recommendations:
            
            if not drink_recommendations and not bakery_recommendations:
                st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 옵션을 조정해 주세요.")
            elif not drink_recommendations:
                st.warning(f"⚠️ **음료 추천 실패:** 선택된 카테고리, 당도({min_drink_sweetness}~{max_drink_sweetness}), 또는 태그에 맞는 음료를 찾을 수 없습니다.")
            elif not bakery_recommendations:
                st.warning(f"⚠️ **베이커리 추천 실패:** 설정된 조건(당도({min_bakery_sweetness}~{max_bakery_sweetness})/태그)에 맞는 베이커리 조합이 없습니다. 개수를 줄이거나 옵션을 조정해 주세요.")


        else:
            # 3. 최종 조합 생성 및 스코어링
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations) 

            final_sets = []
            
            # 점수 계산에 사용할 태그 목록 (음료+베이커리 선호 태그 모두 합산)
            all_selected_tags_for_score = selected_drink_tags + selected_bakery_tags
            
            for drink_combo, bakery_combo in all_combinations:
                drink_price = drink_combo[0]['price']
                bakery_price_sum = sum(item['price'] for item in bakery_combo)
                total_price_per_set = drink_price + bakery_price_sum
                
                all_items = drink_combo + bakery_combo

                if max_price_per_set == float('inf') or total_price_per_set <= max_price_per_set:
                    # 가중치 점수 계산: 모든 선호 태그를 기준으로 일치도를 계산
                    weighted_score = calculate_weighted_score(all_items, all_selected_tags_for_score)
                    
                    final_sets.append({
                        "score": weighted_score,
                        "drink": drink_combo[0], 
                        "bakery": bakery_combo,
                        "total_price_per_set": total_price_per_set,
                        "total_price_for_n_people": total_price_per_set * n_people
                    })
                
                if len(final_sets) >= 200: 
                    break

            if not final_sets:
                st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 태그나 예산을 조정해 주세요.")
            else:
                # 점수 순으로 정렬하고 상위 3개만 선택
                final_sets.sort(key=lambda x: x['score'], reverse=True)
                top_3_sets = final_sets[:3]

                for i, result in enumerate(top_3_sets):
                    st.markdown(f"#### 🥇 세트 {i+1} - 추천 점수: **{result['score']}점** / 100점")
                    
                    st.markdown(f"**1인 세트 가격:** {result['total_price_per_set']:,}원")
                    st.markdown(f"**{n_people}명 예상 총 가격:** **{result['total_price_for_n_people']:,}원** (1인 세트 {n_people}개 기준)")
                    
                    # --- N-people Drink Recommendation Logic ---
                    st.markdown(f"##### 음료 🥤 ({n_people}개 추천)")

                    primary_drink = result['drink']
                    other_drinks = []
                    if n_people > 1:
                        available_drinks = drink_df[drink_df['name'] != primary_drink['name']].copy()
                        
                        # 나머지 음료 옵션 필터링: 카테고리, 음료 당도 범위, 음료 태그
                        filtered_options = available_drinks[available_drinks['category'].isin(selected_categories)].copy()
                        filtered_options = filtered_options[
                            (filtered_options['sweetness'] >= min_drink_sweetness) & 
                            (filtered_options['sweetness'] <= max_drink_sweetness)
                        ]
                        if selected_drink_tags:
                            filtered_options = filtered_options[filtered_options['tags_list'].apply(lambda tags: any(tag in selected_drink_tags for tag in tags))]

                        other_drink_options = filtered_options.sort_values(by='popularity_score', ascending=False)
                        
                        num_additional_drinks = min(n_people - 1, len(other_drink_options))
                        selected_others = other_drink_options.head(num_additional_drinks)
                        
                        other_drinks = [{
                            'name': row['name'], 
                            'price': row['price'], 
                            'tags': row['tags_list'],
                            'popularity': row['popularity_score'],
                            'sweetness': row['sweetness']
                        } for _, row in selected_others.iterrows()]
                    
                    display_drinks = [primary_drink] + other_drinks
                    
                    for j, d in enumerate(display_drinks):
                        drink_tags_str = ", ".join(f"#{t}" for t in d['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if d['popularity'] == 10 else ""
                        bullet = "★" if j == 0 else "•"
                        
                        category_info = drink_df[drink_df['name'] == d['name']]['category'].iloc[0] if not drink_df[drink_df['name'] == d['name']].empty else 'N/A'
                        
                        st.info(f"{bullet} **{d['name']}** ({d['price']:,}원) - *당도: {d['sweetness']} / 카테고리: {category_info}*{is_popular} - *태그: {drink_tags_str}*")
                    # ----------------------------------------
                    
                    st.markdown(f"##### 베이커리 🍞 ({n_bakery}개 추천)")
                    for item in result['bakery']:
                        bakery_tags_str = ", ".join(f"#{t}" for t in item['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if item['popularity'] == 10 else ""
                        st.success(f"• **{item['name']}** ({item['price']:,}원) - *당도: {item['sweetness']}*{is_popular} - *태그: {bakery_tags_str}*")
                    
                    if i < len(top_3_sets) - 1:
                        st.markdown("---")
            
    st.caption("※ 추천 점수(100점 만점)는 선택된 **선호 태그** 일치도(70%)와 메뉴의 인기 점수(30%)를 가중치로 계산한 값입니다.")

    # --- Expander added here for detailed explanation ---
    with st.expander("점수 계산 방법 자세히 보기"):
        st.markdown("""
        이 추천 점수는 사용자의 취향과 메뉴의 인기를 균형 있게 반영하기 위해 가중치를 적용하여 계산됩니다.
        
        **최종 점수 = (선호 태그 일치도 × 70%) + (인기 점수 × 30%)**
        
        #### 1. 선호 태그 일치도 (70% 반영)
        * **계산 방식:** 추천된 세트 내의 전체 메뉴 중에서, **사용자가 선택한 음료 태그 + 베이커리 태그**를 **하나라도 포함하는 메뉴의 비율**을 100점 만점으로 환산합니다.
        * **예시:** 3개의 메뉴가 포함된 세트에서 2개 메뉴만 선택 태그를 포함하면 태그 일치도는 (2/3) * 100 ≈ 66.7점입니다.
        * **참고:** 당도 슬라이더 및 음료 카테고리 필터링은 추천 대상 메뉴를 좁히는 데 사용되며, 최종 점수 계산에는 반영되지 않습니다.
        
        #### 2. 인기 점수 (30% 반영)
        * **계산 방식:** 메뉴 시트에 `#인기` 태그가 있으면 10점, 없으면 5점(기본점)이 부여됩니다. 세트 내 모든 메뉴의 **평균 인기 점수**를 100점 만점으로 환산하여 반영합니다.
        
        최종적으로 이 두 점수를 합산하여 가장 높은 점수를 받은 메뉴 조합을 상위 3개로 보여줍니다.
        """)
    # --- End of Expander ---


with tab_menu_board:
    st.title("📋 메뉴판")
    st.markdown("---")
    st.markdown("##### 🔍 CSV 파일을 직접 수정하여 메뉴, 가격, 태그를 변경할 수 있습니다.")

    # Image loading and display (in case file is missing)
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("베이커리 메뉴")
        if img1:
            st.image(img1, caption="Bakery 메뉴판 (1/2)", use_column_width=True)
        else:
            display_bakery_df = bakery_df.copy()
            display_bakery_df = display_bakery_df.rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(1-5)', 'tags': '태그'})
            display_bakery_df['인기점수'] = display_bakery_df['popularity_score']
            display_bakery_df = display_bakery_df[['메뉴', '가격', '당도(1-5)', '태그', '인기점수']]

            st.dataframe(display_bakery_df, use_container_width=True)


    with col_img2:
        st.subheader("음료 메뉴")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)
        else:
            display_drink_df = drink_df.copy()
            display_drink_df = display_drink_df.rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(1-5)', 'tags': '태그', 'category': '카테고리'})
            display_drink_df['인기점수'] = display_drink_df['popularity_score']
            display_drink_df = display_drink_df[['메뉴', '가격', '카테고리', '당도(1-5)', '태그', '인기점수']]
            
            st.dataframe(display_drink_df, use_container_width=True)

네, 요청하신 대로 음료 선호 옵션과 베이커리 선호 옵션을 완전히 분리하여 설정하고, 당도 슬라이더 및 카테고리/태그 멀티셀렉트 기능을 각각의 메뉴에 독립적으로 적용할 수 있도록 코드를 수정했습니다.
특히, 사용자가 선택한 **'유틸리티/식감 태그'**와 **'음료 맛 태그'**가 해당 메뉴 추천의 점수 계산에 강력하게 반영되도록 로직을 재정비했습니다.
아래 코드를 기존 파일에 덮어쓰거나 새로운 파일로 저장하여 실행하시면 됩니다.
import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

font = "Jua"
# --- 데이터 로드 및 컬럼 정규화 함수 ---

def normalize_columns(df, is_drink=False):
    """'sweetness' 컬럼 및 필수 컬럼을 확인합니다."""
    
    # 2. 필수 컬럼 확인
    required_cols = ['name', 'price', 'sweetness', 'tags']
    if is_drink:
        required_cols.append('category')

    # 현재 df에 없는 필수 컬럼 목록
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        col_names = ", ".join(missing_cols)
        menu_type = "음료" if is_drink else "베이커리"
        st.error(f"🚨 오류: {menu_type} 파일에 필수 컬럼({col_names})이 없습니다. 'name', 'price', 'sweetness', 'tags' 컬럼을 확인해주세요. 음료는 추가로 'category' 컬럼이 필요합니다.")
        st.stop()
    
    # 'sweetness' 컬럼이 숫자인지 확인
    try:
        df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce')
        if df['sweetness'].isnull().any():
            st.error(f"🚨 오류: {menu_type} 파일의 'sweetness' 컬럼에 숫자가 아닌 값이 포함되어 있습니다. 1~5 사이의 숫자로 입력해주세요.")
            st.stop()
    except Exception:
        st.error(f"🚨 오류: {menu_type} 파일의 'sweetness' 컬럼을 숫자로 변환할 수 없습니다.")
        st.stop()
        
    return df

try:
    # CSV 파일 로드 및 컬럼 정규화 적용
    # 'Bakery_menu.csv'와 'Drink_menu.csv' 파일이 필요합니다.
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True) 

    if drink_df.empty or bakery_df.empty:
        st.error("🚨 오류: 메뉴 데이터가 비어 있습니다. 파일 내용을 확인해주세요.")
        st.stop()

except FileNotFoundError:
    st.error("🚨 오류: 메뉴 CSV 파일을 찾을 수 없습니다. 파일 이름과 경로(Bakery_menu.csv, Drink_menu.csv)를 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 오류: 메뉴 CSV 파일을 로드하는 중 알 수 없는 오류가 발생했습니다: {e}")
    st.stop()

# --- 데이터 전처리 및 태그 추출 ---
def preprocess_tags(df):
    """CSV의 tags 컬럼을 클린징하고 리스트로 변환합니다."""
    # 태그 컬럼에서 #을 제거하고 쉼표(,)를 기준으로 분리하며 공백을 제거합니다.
    df['tags_list'] = df['tags'].fillna('').astype(str).str.strip().str.replace('#', '').str.split(r'\s*,\s*')
    df['tags_list'] = df['tags_list'].apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    return df

bakery_df = preprocess_tags(bakery_df)
drink_df = preprocess_tags(drink_df)

# --- 인기도 점수 동적 생성 로직 ---
def assign_popularity_score(df):
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(
            lambda tags: 10 if '인기' in tags else 5
        )
    return df

bakery_df = assign_popularity_score(bakery_df)
drink_df = assign_popularity_score(drink_df)
# --------------------------------------------------------

# 전체 사용 가능한 태그 및 카테고리 추출
def uniq_tags(df):
    return set(t for sub in df['tags_list'] for t in sub if t and t != '인기')

BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS = uniq_tags(drink_df)
all_drink_categories = sorted(drink_df['category'].unique())

# --- 태그 그룹 분리 ---
# NOTE: 맛/당도 태그는 이제 CSV의 'sweetness' 컬럼(숫자)과 슬라이더로 조절되지만, 
# '달콤한', '쌉싸름한' 등 문자열 태그가 남아있다면 별도로 분리해야 합니다.

# 베이커리/음료 공통으로 사용되는 '맛/당도' 관련 태그 (필터링이 아닌, 사용자 선택을 위한 리스트)
# CSV에 있는 태그 중, 맛과 관련된 태그를 정의합니다. (주로 음료나 디저트에 사용)
FLAVOR_TAGS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','씁쓸한', '초코', '치즈'} 

# 1. 음료 전용 '맛' 태그 (DRINK_TAGS 중 FLAVOR_TAGS만)
ui_drink_flavor_tags = sorted(DRINK_TAGS & FLAVOR_TAGS)

# 2. 베이커리 전용 '식감/용도' 태그 (BAKERY_TAGS 중 FLAVOR_TAGS를 제외한 나머지)
ui_bakery_utility_tags = sorted(BAKERY_TAGS - FLAVOR_TAGS) 


# --- 추천 로직 함수 ---
def recommend_menu(df, min_sweetness, max_sweetness, selected_tags, n_items, max_price=None, selected_categories=None):
    """
    주어진 조건으로 메뉴 조합을 추천합니다.
    - min_sweetness, max_sweetness: 당도 슬라이더 범위 (숫자)
    - selected_tags: 메뉴별 선호 태그 (문자열 리스트)
    """

    filtered_df = df.copy()
    is_drink_menu = 'category' in filtered_df.columns
    
    # 1. 카테고리 필터링 (음료에만 해당)
    if is_drink_menu and selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    # 2. 당도 필터링 (숫자 슬라이더)
    filtered_df = filtered_df[
        (filtered_df['sweetness'] >= min_sweetness) & 
        (filtered_df['sweetness'] <= max_sweetness)
    ]
    
    # 3. 선호 태그 필터링
    if selected_tags:
        # 선택된 태그 중 하나라도 메뉴의 tags_list에 포함되면 유지
        temp_filtered_df = filtered_df[filtered_df['tags_list'].apply(lambda tags: any(tag in selected_tags for tag in tags))]
        if not temp_filtered_df.empty:
            filtered_df = temp_filtered_df

    # 필터링 결과가 없으면 종료
    if filtered_df.empty:
        return []

    recommendations = []
    
    # --- 4. 조합 또는 단일 아이템 추천 ---
    
    if n_items == 1: # 단일 아이템 추천 (음료)
        items = filtered_df.sort_values(by=['popularity_score', 'price'], ascending=[False, True])
        
        for _, row in items.iterrows():
            if max_price is None or row['price'] <= max_price:
                recommendations.append([{
                    'name': row['name'], 
                    'price': row['price'], 
                    'tags': row['tags_list'],
                    'popularity': row['popularity_score'],
                    'sweetness': row['sweetness'] 
                }])
                if len(recommendations) >= 200: 
                    break
    else: # n_items > 1 (베이커리 조합)
        if len(filtered_df) > 30: 
            subset = filtered_df.sort_values(by='popularity_score', ascending=False).head(30)
        else:
            subset = filtered_df

        all_combinations = list(itertools.combinations(subset.itertuples(index=False), n_items))
        random.shuffle(all_combinations)

        for combo in all_combinations:
            total_price = sum(item.price for item in combo)
            if max_price is None or total_price <= max_price:
                combo_result = [{
                    'name': item.name, 
                    'price': item.price, 
                    'tags': item.tags_list,
                    'popularity': item.popularity_score,
                    'sweetness': item.sweetness 
                } for item in combo]
                recommendations.append(combo_result)
                if len(recommendations) >= 200:
                    break
    
    return recommendations


# --- 가중치 기반 점수 계산 함수 ---
# 선택된 '유틸리티/식감' 태그에 대해서만 일치도를 계산합니다.
def calculate_weighted_score(combo_items, selected_tags):
    """
    선택된 태그 일치도(70%)와 인기 점수(30%)를 가중 평균하여 최종 점수(100점 만점)를 계산
    """
    
    # --- 1. 태그 일치도 (Tag Match Score) 계산 (70% 가중치) ---
    if not selected_tags:
        # 태그를 선택하지 않은 경우, 태그 일치도를 최고점인 100점으로 간주
        tag_match_score = 100 
    else:
        total_items = len(combo_items)
        if total_items == 0:
            tag_match_score = 0
        else:
            total_matches = 0
            selected_tags_set = set(selected_tags)

            for item in combo_items:
                item_tags_set = set(item['tags'])
                # 선택된 태그 중 하나라도 메뉴의 태그에 포함되면 일치로 간주
                if not item_tags_set.isdisjoint(selected_tags_set):
                    total_matches += 1 
            
            # (일치하는 메뉴 수 / 전체 메뉴 수) * 100
            tag_match_score = (total_matches / total_items) * 100

    # --- 2. 인기 점수 (Popularity Score) 계산 (30% 가중치) ---
    if not combo_items:
        avg_popularity_score = 0
    else:
        total_popularity = sum(item['popularity'] for item in combo_items)
        avg_popularity_score = total_popularity / len(combo_items) 
    
    # 인기 점수를 100점 만점으로 변환 (인기 10점 만점 기준)
    popularity_score_100 = avg_popularity_score * 10 
    
    # --- 3. 최종 가중치 점수 계산 (100점 만점) ---
    WEIGHT_TAG = 0.7
    WEIGHT_POPULARITY = 0.3
    
    final_score = (tag_match_score * WEIGHT_TAG) + (popularity_score_100 * WEIGHT_POPULARITY)
    
    return round(final_score, 1)


# --- Streamlit 앱 구성 ---

st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# Image loading function (in case file is missing)
def load_image(image_path):
    try:
        # NOTE: 이 코드는 파일 시스템에 "menu_board_1.png"와 "menu_board_2.png" 파일이 존재해야 정상 작동합니다.
        return Image.open(image_path)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# --- 탭 구성 ---
tab_recommendation, tab_menu_board = st.tabs(["AI 메뉴 추천", "메뉴판"])


with tab_recommendation:
    st.title("💡 AI 메뉴 추천 시스템")
    st.subheader("예산, 당도, 카테고리, 취향, 인기를 고려한 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션 (5개의 컬럼으로 분할)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown("#### 👤 인원수 & 💰 예산")
        n_people = st.number_input("인원수", min_value=1, max_value=10, value=2, step=1)
        
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        if budget_unlimited:
            budget = float('inf') 
            st.slider("최대 예산 설정", min_value=5000, max_value=50000, value=50000, step=1000, disabled=True)
        else:
            budget = st.slider("최대 예산 설정 (1인 기준)", min_value=5000, max_value=50000, value=15000, step=1000)

    with col2:
        st.markdown("#### 🍞 베이커리 옵션")
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)
        
        bakery_sweetness_range = st.slider(
            "선호 베이커리 당도 레벨",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
            key='bakery_sweetness_slider'
        )
        min_bakery_sweetness, max_bakery_sweetness = bakery_sweetness_range

        selected_bakery_tags = st.multiselect(
            "선호 베이커리 태그 (식감/용도, 최대 3개)",
            options=ui_bakery_utility_tags,
            default=[],
            max_selections=3,
            placeholder="예: 든든한, 겉바속촉, 가벼운",
            key='bakery_tags_multiselect'
        )
        
    with col3:
        st.markdown("#### ☕ 음료 옵션")
        selected_categories = st.multiselect(
            "선호 음료 카테고리",
            options=all_drink_categories,
            default=all_drink_categories,
            placeholder="예: 커피, 티",
        )

        drink_sweetness_range = st.slider(
            "선호 음료 당도 레벨",
            min_value=1,
            max_value=5,
            value=(1, 5),
            step=1,
            key='drink_sweetness_slider'
        )
        min_drink_sweetness, max_drink_sweetness = drink_sweetness_range

        selected_drink_tags = st.multiselect(
            "선호 음료 맛 태그 (최대 3개)",
            options=ui_drink_flavor_tags,
            default=[],
            max_selections=3,
            placeholder="예: 깔끔한, 쌉싸름한, 상큼한",
            key='drink_tags_multiselect'
        )
        
    with col4:
        st.markdown("#### 📊 메뉴 데이터 확인")
        # 데이터프레임의 상위 5개 행을 보여주는 간결한 요약
        st.dataframe(bakery_df.head(5).rename(columns={'name':'메뉴', 'price':'가격', 'sweetness':'당도', 'tags':'태그'}), height=200)
        st.dataframe(drink_df.head(5).rename(columns={'name':'메뉴', 'price':'가격', 'sweetness':'당도', 'tags':'태그', 'category':'카테고리'}), height=200)

    
    st.markdown("---")

    # 2. 추천 실행 버튼
    if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
        st.markdown("### 🏆 AI 추천 메뉴 조합")
        
        max_price_per_set = budget

        # --- 추천 생성 (1인 세트 기준) ---
        
        # 1. 음료 추천 (1개)
        drink_recommendations = recommend_menu(
            drink_df, 
            min_drink_sweetness, max_drink_sweetness, 
            selected_drink_tags, 1, 
            max_price=max_price_per_set, 
            selected_categories=selected_categories
        )
        
        # 2. 베이커리 추천 (n_bakery 개)
        bakery_recommendations = recommend_menu(
            bakery_df, 
            min_bakery_sweetness, max_bakery_sweetness, 
            selected_bakery_tags, n_bakery, 
            max_price=max_price_per_set
        )
        
        
        if not drink_recommendations or not bakery_recommendations:
            
            if not drink_recommendations and not bakery_recommendations:
                st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 옵션을 조정해 주세요.")
            elif not drink_recommendations:
                st.warning(f"⚠️ **음료 추천 실패:** 선택된 카테고리, 당도({min_drink_sweetness}~{max_drink_sweetness}), 또는 태그에 맞는 음료를 찾을 수 없습니다.")
            elif not bakery_recommendations:
                st.warning(f"⚠️ **베이커리 추천 실패:** 설정된 조건(당도({min_bakery_sweetness}~{max_bakery_sweetness})/태그)에 맞는 베이커리 조합이 없습니다. 개수를 줄이거나 옵션을 조정해 주세요.")


        else:
            # 3. 최종 조합 생성 및 스코어링
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations) 

            final_sets = []
            
            # 점수 계산에 사용할 태그 목록 (음료+베이커리 선호 태그 모두 합산)
            all_selected_tags_for_score = selected_drink_tags + selected_bakery_tags
            
            for drink_combo, bakery_combo in all_combinations:
                drink_price = drink_combo[0]['price']
                bakery_price_sum = sum(item['price'] for item in bakery_combo)
                total_price_per_set = drink_price + bakery_price_sum
                
                all_items = drink_combo + bakery_combo

                if max_price_per_set == float('inf') or total_price_per_set <= max_price_per_set:
                    # 가중치 점수 계산: 모든 선호 태그를 기준으로 일치도를 계산
                    weighted_score = calculate_weighted_score(all_items, all_selected_tags_for_score)
                    
                    final_sets.append({
                        "score": weighted_score,
                        "drink": drink_combo[0], 
                        "bakery": bakery_combo,
                        "total_price_per_set": total_price_per_set,
                        "total_price_for_n_people": total_price_per_set * n_people
                    })
                
                if len(final_sets) >= 200: 
                    break

            if not final_sets:
                st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 태그나 예산을 조정해 주세요.")
            else:
                # 점수 순으로 정렬하고 상위 3개만 선택
                final_sets.sort(key=lambda x: x['score'], reverse=True)
                top_3_sets = final_sets[:3]

                for i, result in enumerate(top_3_sets):
                    st.markdown(f"#### 🥇 세트 {i+1} - 추천 점수: **{result['score']}점** / 100점")
                    
                    st.markdown(f"**1인 세트 가격:** {result['total_price_per_set']:,}원")
                    st.markdown(f"**{n_people}명 예상 총 가격:** **{result['total_price_for_n_people']:,}원** (1인 세트 {n_people}개 기준)")
                    
                    # --- N-people Drink Recommendation Logic ---
                    st.markdown(f"##### 음료 🥤 ({n_people}개 추천)")

                    primary_drink = result['drink']
                    other_drinks = []
                    if n_people > 1:
                        available_drinks = drink_df[drink_df['name'] != primary_drink['name']].copy()
                        
                        # 나머지 음료 옵션 필터링: 카테고리, 음료 당도 범위, 음료 태그
                        filtered_options = available_drinks[available_drinks['category'].isin(selected_categories)].copy()
                        filtered_options = filtered_options[
                            (filtered_options['sweetness'] >= min_drink_sweetness) & 
                            (filtered_options['sweetness'] <= max_drink_sweetness)
                        ]
                        if selected_drink_tags:
                            filtered_options = filtered_options[filtered_options['tags_list'].apply(lambda tags: any(tag in selected_drink_tags for tag in tags))]

                        other_drink_options = filtered_options.sort_values(by='popularity_score', ascending=False)
                        
                        num_additional_drinks = min(n_people - 1, len(other_drink_options))
                        selected_others = other_drink_options.head(num_additional_drinks)
                        
                        other_drinks = [{
                            'name': row['name'], 
                            'price': row['price'], 
                            'tags': row['tags_list'],
                            'popularity': row['popularity_score'],
                            'sweetness': row['sweetness']
                        } for _, row in selected_others.iterrows()]
                    
                    display_drinks = [primary_drink] + other_drinks
                    
                    for j, d in enumerate(display_drinks):
                        drink_tags_str = ", ".join(f"#{t}" for t in d['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if d['popularity'] == 10 else ""
                        bullet = "★" if j == 0 else "•"
                        
                        category_info = drink_df[drink_df['name'] == d['name']]['category'].iloc[0] if not drink_df[drink_df['name'] == d['name']].empty else 'N/A'
                        
                        st.info(f"{bullet} **{d['name']}** ({d['price']:,}원) - *당도: {d['sweetness']} / 카테고리: {category_info}*{is_popular} - *태그: {drink_tags_str}*")
                    # ----------------------------------------
                    
                    st.markdown(f"##### 베이커리 🍞 ({n_bakery}개 추천)")
                    for item in result['bakery']:
                        bakery_tags_str = ", ".join(f"#{t}" for t in item['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if item['popularity'] == 10 else ""
                        st.success(f"• **{item['name']}** ({item['price']:,}원) - *당도: {item['sweetness']}*{is_popular} - *태그: {bakery_tags_str}*")
                    
                    if i < len(top_3_sets) - 1:
                        st.markdown("---")
            
    st.caption("※ 추천 점수(100점 만점)는 선택된 **선호 태그** 일치도(70%)와 메뉴의 인기 점수(30%)를 가중치로 계산한 값입니다.")

    # --- Expander added here for detailed explanation ---
    with st.expander("점수 계산 방법 자세히 보기"):
        st.markdown("""
        이 추천 점수는 사용자의 취향과 메뉴의 인기를 균형 있게 반영하기 위해 가중치를 적용하여 계산됩니다.
        
        **최종 점수 = (선호 태그 일치도 × 70%) + (인기 점수 × 30%)**
        
        #### 1. 선호 태그 일치도 (70% 반영)
        * **계산 방식:** 추천된 세트 내의 전체 메뉴 중에서, **사용자가 선택한 음료 태그 + 베이커리 태그**를 **하나라도 포함하는 메뉴의 비율**을 100점 만점으로 환산합니다.
        * **예시:** 3개의 메뉴가 포함된 세트에서 2개 메뉴만 선택 태그를 포함하면 태그 일치도는 (2/3) * 100 ≈ 66.7점입니다.
        * **참고:** 당도 슬라이더 및 음료 카테고리 필터링은 추천 대상 메뉴를 좁히는 데 사용되며, 최종 점수 계산에는 반영되지 않습니다.
        
        #### 2. 인기 점수 (30% 반영)
        * **계산 방식:** 메뉴 시트에 `#인기` 태그가 있으면 10점, 없으면 5점(기본점)이 부여됩니다. 세트 내 모든 메뉴의 **평균 인기 점수**를 100점 만점으로 환산하여 반영합니다.
        
        최종적으로 이 두 점수를 합산하여 가장 높은 점수를 받은 메뉴 조합을 상위 3개로 보여줍니다.
        """)
    # --- End of Expander ---


with tab_menu_board:
    st.title("📋 메뉴판")
    st.markdown("---")
    st.markdown("##### 🔍 CSV 파일을 직접 수정하여 메뉴, 가격, 태그를 변경할 수 있습니다.")

    # Image loading and display (in case file is missing)
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("베이커리 메뉴")
        if img1:
            st.image(img1, caption="Bakery 메뉴판 (1/2)", use_column_width=True)
        else:
            display_bakery_df = bakery_df.copy()
            display_bakery_df = display_bakery_df.rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(1-5)', 'tags': '태그'})
            display_bakery_df['인기점수'] = display_bakery_df['popularity_score']
            display_bakery_df = display_bakery_df[['메뉴', '가격', '당도(1-5)', '태그', '인기점수']]

            st.dataframe(display_bakery_df, use_container_width=True)


    with col_img2:
        st.subheader("음료 메뉴")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)
        else:
            display_drink_df = drink_df.copy()
            display_drink_df = display_drink_df.rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(1-5)', 'tags': '태그', 'category': '카테고리'})
            display_drink_df['인기점수'] = display_drink_df['popularity_score']
            display_drink_df = display_drink_df[['메뉴', '가격', '카테고리', '당도(1-5)', '태그', '인기점수']]
            
            st.dataframe(display_drink_df, use_container_width=True)

