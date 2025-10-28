import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image
[theme]
font = "Jua"
# --- 데이터 로드 및 컬럼 정규화 함수 ---

def normalize_columns(df, is_drink=False):
    """'sweetness' 컬럼을 'tags'로 리네임하고 필수 컬럼을 확인합니다."""
    
    # 1. 태그 컬럼 정규화: 'sweetness'가 있으면 'tags'로 리네임
    if 'sweetness' in df.columns and 'tags' not in df.columns:
        df = df.rename(columns={'sweetness': 'tags'})
    
    # 2. 필수 컬럼 확인
    required_cols = ['name', 'price', 'tags']
    if is_drink:
        required_cols.append('category')

    # 현재 df에 없는 필수 컬럼 목록
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        col_names = ", ".join(missing_cols)
        menu_type = "음료" if is_drink else "베이커리"
        st.error(f"🚨 오류: {menu_type} 파일에 필수 컬럼({col_names})이 없습니다. 베이커리/음료 모두 'name', 'price', 'sweetness' 또는 'tags' 컬럼이 필요합니다. 음료는 추가로 'category' 컬럼이 필요합니다.")
        st.stop()
        
    return df

try:
    # CSV 파일 로드 및 컬럼 정규화 적용
    # NOTE: 'sweetness' 컬럼이 'tags'로 내부적으로 리네임되어 사용됨.
    bakery_df = normalize_columns(pd.read_csv("Bakery_menu.csv"))
    drink_df = normalize_columns(pd.read_csv("Drink_menu.csv"), is_drink=True)

    if drink_df.empty or bakery_df.empty:
        st.error("🚨 오류: 메뉴 데이터가 비어 있습니다. 파일 내용을 확인해주세요.")
        st.stop()

except FileNotFoundError:
    st.error("🚨 오류: 메뉴 CSV 파일을 찾을 수 없습니다. 파일 이름과 경로를 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"🚨 오류: 메뉴 CSV 파일을 로드하는 중 알 수 없는 오류가 발생했습니다: {e}")
    st.stop()

# --- 데이터 전처리 및 태그 추출 ---
def preprocess_tags(df):
    """CSV의 tags 컬럼을 클린징하고 리스트로 변환합니다."""
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
all_bakery_tags = sorted(list(set(tag for sublist in bakery_df['tags_list'] for tag in sublist if tag != '인기'))) 
all_drink_tags = sorted(list(set(tag for sublist in drink_df['tags_list'] for tag in sublist if tag != '인기'))) 
all_drink_categories = sorted(drink_df['category'].unique())

# --- 태그 그룹 분리 (사용자 요청에 따라) ---
# 당도/맛 태그와 베이커리 용도 태그를 수동으로 분리 정의
# 이는 CSV 파일의 내용에 따라 달라질 수 있으므로, 사용자가 추가/수정할 경우 코드를 업데이트해야 합니다.
# --- 태그 목록 동적으로 추출 ---
def uniq_tags(df):
    return set(t for sub in df['tags_list'] for t in sub if t and t != '인기')

BAKERY_TAGS = uniq_tags(bakery_df)
DRINK_TAGS = uniq_tags(drink_df)
SWEETNESS = {'달콤한','고소한','짭짤한','단백한','부드러운','깔끔한','쌉싸름한','상큼한','씁쓸한'}

ui_sweetness_tags = sorted((BAKERY_TAGS | DRINK_TAGS) & SWEETNESS)
ui_utility_tags = sorted(BAKERY_TAGS - SWEETNESS)  # 베이커리에서 당도 태그 제외한 나머지


# --- 추천 로직 함수 ---
def recommend_menu(df, selected_sweetness_tags, selected_utility_tags, n_items, max_price=None, selected_categories=None):
    """
    주어진 조건으로 메뉴 조합을 추천합니다.
    - 음료: 카테고리 AND 당도 태그 필터링
    - 베이커리: 당도 태그 OR 유틸리티 태그 필터링
    """

    filtered_df = df.copy()
    is_drink_menu = 'category' in filtered_df.columns
    
    # 1. 카테고리 필터링 (음료에만 해당)
    if is_drink_menu and selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    # 2. 태그 필터링
    selected_tags_combined = selected_sweetness_tags + selected_utility_tags

    if is_drink_menu:
        # 2-1. 음료 필터링: 카테고리 AND 당도 태그 (더 엄격하게 필터링)
        if selected_sweetness_tags:
            temp_filtered_df = filtered_df[filtered_df['tags_list'].apply(lambda tags: any(tag in selected_sweetness_tags for tag in tags))]
            if not temp_filtered_df.empty:
                filtered_df = temp_filtered_df

    else:
        # 2-2. 베이커리 필터링: 당도 태그 OR 유틸리티 태그 (하나라도 만족하면 포함)
        if selected_tags_combined:
            temp_filtered_df = filtered_df[filtered_df['tags_list'].apply(lambda tags: any(tag in selected_tags_combined for tag in tags))]
            if not temp_filtered_df.empty:
                filtered_df = temp_filtered_df


    # 필터링 결과가 없으면 종료
    if filtered_df.empty:
        return []

    recommendations = []
    
    # --- 3. 조합 또는 단일 아이템 추천 ---
    
    if n_items == 1: # 단일 아이템 추천 (음료)
        items = filtered_df.sort_values(by=['popularity_score', 'price'], ascending=[False, True])
        
        for _, row in items.iterrows():
            if max_price is None or row['price'] <= max_price:
                recommendations.append([{
                    'name': row['name'], 
                    'price': row['price'], 
                    'tags': row['tags_list'],
                    'popularity': row['popularity_score']
                }])
                if len(recommendations) >= 100:
                    break
    else: # n_items > 1 (베이커리 조합)
        if len(filtered_df) > 15:
            subset = filtered_df.sort_values(by='popularity_score', ascending=False).head(15)
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
                    'popularity': item.popularity_score
                } for item in combo]
                recommendations.append(combo_result)
                if len(recommendations) >= 200:
                    break
    
    return recommendations


# --- 가중치 기반 점수 계산 함수 ---
def calculate_weighted_score(combo_items, selected_tags):
    """
    태그 일치도(70%)와 인기 점수(30%)를 가중 평균하여 최종 점수(100점 만점)를 계산
    """
    
    # --- 1. 태그 일치도 (Tag Match Score) 계산 (70% 가중치) ---
    if not selected_tags:
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
                if item_tags_set.intersection(selected_tags_set):
                    total_matches += 1 
            
            tag_match_score = (total_matches / total_items) * 100

    # --- 2. 인기 점수 (Popularity Score) 계산 (30% 가중치) ---
    total_popularity = sum(item['popularity'] for item in combo_items)
    avg_popularity_score = total_popularity / len(combo_items) if combo_items else 0
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
        return Image.open(image_path)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# --- 탭 구성 ---
tab_recommendation, tab_menu_board = st.tabs(["AI 메뉴 추천", "메뉴판"])


with tab_recommendation:
    st.title("💡 AI 메뉴 추천 시스템")
    st.subheader("예산, 카테고리, 취향, 인기를 고려한 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션 (5개의 컬럼으로 분할)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        st.markdown("#### 👤 인원수")
        n_people = st.number_input("인원수 만큼 음료를 추천해드려요.", min_value=1, max_value=10, value=2, step=1)

    with col2:
        st.markdown("#### 💰 예산 (1인 기준)")
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        
        if budget_unlimited:
            budget = float('inf') 
            st.slider("최대 예산 설정", min_value=5000, max_value=50000, value=50000, step=1000, disabled=True)
        else:
            budget = st.slider("최대 예산 설정 (1인 기준)", min_value=5000, max_value=50000, value=15000, step=1000)

    with col3:
        st.markdown("#### 🥖 베이커리 개수")
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)

    with col4:
        st.markdown("#### ☕ 음료 카테고리")
        # --- 1. 음료 카테고리 선택 (독립) ---
        selected_categories = st.multiselect(
            "선호 음료 카테고리",
            options=all_drink_categories,
            default=all_drink_categories,
            placeholder="예: 커피, 티",
        )
        
    with col5:
        st.markdown("#### 🏷️ 선호 태그 (최대 3개씩 선택가능합니다)")
        # --- 2. 당도/맛 태그 선택 (음료/베이커리 공통) ---
        selected_sweetness_tags = st.multiselect(
            "내 취향 음료 찾기: 선호 당도/맛 태그(미선택시 랜덤으로 추천해드려요)",
            options=ui_sweetness_tags,
            default=[],
            max_selections=3,
            placeholder="예: 달콤한, 쌉싸름한",
        )
        # --- 3. 베이커리 전용 태그 선택 (베이커리 필터링 기준) ---
        selected_utility_tags = st.multiselect(
            "내 취향 베이커리 찾기: 선호 베이커리 태그 (미선택시 랜덤으로 추천해드려요)",
            options=ui_utility_tags,
            default=[],
            max_selections=3,
            placeholder="예: 든든한, 간단한",
        )
    
    st.markdown("---")

    # 2. 추천 실행 버튼
    if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
        st.markdown("### 🏆 AI 추천 메뉴 조합")
        
        max_price_per_set = budget

        # --- 추천 생성 (1인 세트 기준) ---
        
        # 1. 음료 추천 (1개) - 카테고리 및 당도 태그 필터링 적용
        # NOTE: 음료는 유틸리티 태그(든든한 등)는 무시함
        drink_recommendations = recommend_menu(drink_df, selected_sweetness_tags, [], 1, max_price=max_price_per_set, selected_categories=selected_categories)
        
        # 2. 베이커리 추천 (n_bakery 개) - 당도 태그 OR 유틸리티 태그 필터링 적용
        bakery_recommendations = recommend_menu(bakery_df, selected_sweetness_tags, selected_utility_tags, n_bakery, max_price=max_price_per_set)
        
        
        if not drink_recommendations or not bakery_recommendations:
            
            if not drink_recommendations:
                st.warning(f"⚠️ **음료 추천 실패:** 선택된 카테고리/당도 태그 및 1인 예산({max_price_per_set:,}원)에 맞는 음료를 찾을 수 없습니다.")
            if not bakery_recommendations:
                st.warning(f"⚠️ **베이커리 추천 실패:** 설정된 조건(태그/예산)에 맞는 베이커리 조합이 없습니다. 베이커리 개수를 줄이거나 예산을 높여주세요.")
            
            if drink_recommendations and bakery_recommendations:
                 st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 태그나 예산을 조정해 주세요.")


        else:
            # 3. 최종 조합 생성 및 스코어링
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations) 

            final_sets = []
            # 점수 계산에 사용할 태그 목록 (모든 선택된 태그)
            all_selected_tags_for_score = selected_sweetness_tags + selected_utility_tags
            
            for drink_combo, bakery_combo in all_combinations:
                # 1세트 가격 계산
                drink_price = drink_combo[0]['price']
                bakery_price_sum = sum(item['price'] for item in bakery_combo)
                total_price_per_set = drink_price + bakery_price_sum
                
                # 전체 아이템 (점수 계산용)
                all_items = drink_combo + bakery_combo

                if max_price_per_set == float('inf') or total_price_per_set <= max_price_per_set:
                    # 가중치 점수 계산: 모든 선택 태그를 기준으로 일치도를 계산
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
                    # 점수 표시
                    st.markdown(f"#### 🥇 세트 {i+1} - 추천 점수: **{result['score']}점** / 100점")
                    
                    # 가격 정보 표시 (총 가격은 1인 세트 가격 * N명으로 계산)
                    st.markdown(f"**1인 세트 가격:** {result['total_price_per_set']:,}원")
                    st.markdown(f"**{n_people}명 예상 총 가격:** **{result['total_price_for_n_people']:,}원** (1인 세트 {n_people}개 기준)")
                    
                    # --- N-people Drink Recommendation Logic ---
                    st.markdown(f"##### 음료 🥤 ({n_people}개 추천)")

                    # 1. 1인 세트의 대표 음료
                    primary_drink = result['drink']

                    # 2. 나머지 인원수만큼 인기 순위 기반의 다른 음료 선택
                    other_drinks = []
                    if n_people > 1:
                        # 대표 음료를 제외하고, 선택된 카테고리 및 당도 태그 내에서 필터링
                        available_drinks = drink_df[drink_df['name'] != primary_drink['name']].copy()
                        
                        # 필터링 로직을 다시 적용하여 나머지 음료 옵션을 찾음
                        filtered_options = available_drinks[available_drinks['category'].isin(selected_categories)].copy()
                        if selected_sweetness_tags:
                            filtered_options = filtered_options[filtered_options['tags_list'].apply(lambda tags: any(tag in selected_sweetness_tags for tag in tags))]

                        # 인기 점수 순으로 정렬하여 선택
                        other_drink_options = filtered_options.sort_values(by='popularity_score', ascending=False)
                        
                        num_additional_drinks = min(n_people - 1, len(other_drink_options))
                        
                        selected_others = other_drink_options.head(num_additional_drinks)
                        
                        other_drinks = [{
                            'name': row['name'], 
                            'price': row['price'], 
                            'tags': row['tags_list'],
                            'popularity': row['popularity_score']
                        } for _, row in selected_others.iterrows()]
                    
                    # 최종 추천 음료 목록 (대표 음료가 맨 앞에 오도록)
                    display_drinks = [primary_drink] + other_drinks
                    
                    for j, d in enumerate(display_drinks):
                        drink_tags_str = ", ".join(f"#{t}" for t in d['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if d['popularity'] == 10 else ""
                        bullet = "★" if j == 0 else "•" # 대표 음료에 별표 표시
                        
                        category_info = drink_df[drink_df['name'] == d['name']]['category'].iloc[0]
                        
                        st.info(f"{bullet} **{d['name']}** ({d['price']:,}원) - *카테고리: {category_info}*{is_popular} - *태그: {drink_tags_str}*")
                    # ----------------------------------------
                    
                    st.markdown(f"##### 베이커리 🍞 ({n_bakery}개 추천)")
                    # Format bakery list
                    for item in result['bakery']:
                        bakery_tags_str = ", ".join(f"#{t}" for t in item['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if item['popularity'] == 10 else ""
                        st.success(f"• **{item['name']}** ({item['price']:,}원){is_popular} - *태그: {bakery_tags_str}*")
                    
                    if i < len(top_3_sets) - 1:
                        st.markdown("---")
            
    st.caption("※ 추천 점수(100점 만점)는 태그 일치도(70%)와 메뉴의 인기 점수(30%)를 가중치로 계산한 값입니다. 인원수만큼 음료를 추천하며, 가장 점수가 높은 음료에 ★이 표시됩니다.")

    # --- Expander added here for detailed explanation ---
    with st.expander("점수 계산 방법 자세히 보기"):
        st.markdown("""
        이 추천 점수는 사용자의 취향과 메뉴의 인기를 균형 있게 반영하기 위해 가중치를 적용하여 계산됩니다.
        
        **최종 점수 = (태그 일치도 × 70%) + (인기 점수 × 30%)**
        
        #### 1. 태그 일치도 (70% 반영)
        * **계산 방식:추천된 세트 내의 전체 메뉴 중에서, 사용자가 선택한 모든 선호 해시태그(당도/맛 + 베이커리 태그)를 하나라도 포함하는 메뉴의 비율을 100점 만점으로 환산합니다.
        * **예시: 3개의 메뉴가 포함된 세트에서 2개 메뉴만 선택 태그를 포함하면 태그 일치도는 (2/3) * 100 ≈ 66.7점입니다.
        
        #### 2. 인기 점수 (30% 반영)
        * **계산 방식:** 메뉴 시트에 `#인기` 태그가 있으면 10점, 없으면 5점(기본점)이 부여됩니다. 세트 내 모든 메뉴의 **평균 인기 점수**를 100점 만점으로 환산하여 반영합니다.
        * **예시:** 세트의 평균 인기 점수가 8점이면, 8 * 10 = 80점으로 환산되어 30%의 가중치가 적용됩니다.
        
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
            # 'tags'는 원본 컬럼(sweetness) 값을 사용하도록 복사
            display_bakery_df = bakery_df.copy()
            # 원본 컬럼이 'sweetness'였을 수 있으므로 이를 '당도/태그'로 표시
            display_bakery_df = display_bakery_df.rename(columns={'name': '메뉴', 'price': '가격', 'tags': '당도/태그'})
            display_bakery_df['인기점수'] = display_bakery_df['popularity_score']
            display_bakery_df = display_bakery_df[['메뉴', '가격', '당도/태그', '인기점수']]

            st.dataframe(display_bakery_df, use_container_width=True)


    with col_img2:
        st.subheader("음료 메뉴")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)
        else:
            # 'tags'는 원본 컬럼(sweetness) 값을 사용하도록 복사
            display_drink_df = drink_df.copy()
            display_drink_df = display_drink_df.rename(columns={'name': '메뉴', 'price': '가격', 'tags': '당도/태그', 'category': '카테고리'})
            display_drink_df['인기점수'] = display_drink_df['popularity_score']
            display_drink_df = display_drink_df[['메뉴', '가격', '카테고리', '당도/태그', '인기점수']]
            
            st.dataframe(display_drink_df, use_container_width=True)


