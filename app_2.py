import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# --- 데이터 로드 ---
try:
    # CSV 파일 로드
    bakery_df = pd.read_csv("Bakery_menu.csv")
    drink_df = pd.read_csv("Drink_menu.csv")

except FileNotFoundError:
    st.error("메뉴 CSV 파일을 찾을 수 없습니다. 'Bakery_menu.csv'와 'Drink_menu.csv' 파일을 확인해주세요.")
    st.stop()
except Exception as e:
    st.error(f"메뉴 CSV 파일을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# --- 데이터 전처리 및 태그 추출 ---
def preprocess_tags(df):
    """CSV의 tags 컬럼을 클린징하고 리스트로 변환합니다."""
    # NaN 처리, 문자열 변환, 양쪽 공백 제거, 쉼표 및 샵 제거 후 분리
    df['tags_list'] = df['tags'].fillna('').astype(str).str.strip().str.replace('#', '').str.split(r'\s*,\s*')
    # 빈 문자열 및 공백 제거
    df['tags_list'] = df['tags_list'].apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    return df

bakery_df = preprocess_tags(bakery_df)
drink_df = preprocess_tags(drink_df)

# --- 인기도 점수 동적 생성 로직 (사용자 피드백 반영) ---
# 'popularity_score' 컬럼이 없으면, '인기' 태그 유무에 따라 점수를 부여합니다.
# '인기' 태그가 있으면 10점 (만점), 없으면 5점 (중간점)을 부여합니다.
def assign_popularity_score(df):
    if 'popularity_score' not in df.columns:
        df['popularity_score'] = df['tags_list'].apply(
            lambda tags: 10 if '인기' in tags else 5
        )
    return df

bakery_df = assign_popularity_score(bakery_df)
drink_df = assign_popularity_score(drink_df)
# --------------------------------------------------------

# 전체 사용 가능한 태그 추출
all_bakery_tags = sorted(list(set(tag for sublist in bakery_df['tags_list'] for tag in sublist if tag != '인기'))) # '인기' 태그는 제외
all_drink_tags = sorted(list(set(tag for sublist in drink_df['tags_list'] for tag in sublist if tag != '인기'))) # '인기' 태그는 제외
all_tags = sorted(list(set(all_bakery_tags + all_drink_tags)))


# --- 추천 로직 함수 ---
def recommend_menu(df, selected_tags, n_items, max_price=None):
    """
    주어진 예산과 태그를 기반으로 메뉴 조합을 추천합니다.
    결과에는 item name, price, tags_list, popularity_score가 포함됩니다.
    """

    # '인기' 태그는 사용자의 '선호 태그'에서는 제외되어야 하므로 필터링 전에 처리
    if selected_tags:
        # 태그 필터링 (하나라도 일치하는 태그가 있으면 선택)
        filtered_df = df[df['tags_list'].apply(lambda tags: any(tag in selected_tags for tag in tags))]
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        # 태그 필터링 결과가 없으면, 태그 무시하고 전체에서 선택
        filtered_df = df.copy()
        
    if filtered_df.empty:
        return []

    recommendations = []
    
    # 결과 포맷 정의: 리스트 오브 딕셔너리 [{name, price, tags, popularity}, ...]
    
    if n_items == 1: # 단일 아이템 추천 (음료 또는 베이커리 1개)
        # 인기 점수와 가격을 기준으로 정렬하여 다양성 확보
        items = filtered_df.sample(frac=1, random_state=42).sort_values(by=['popularity_score', 'price'], ascending=[False, True])
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
            # 메모리 및 시간 제한을 위해 조합 가능한 아이템이 너무 많으면 인기순으로 일부만 선택
            subset = filtered_df.sort_values(by='popularity_score', ascending=False).head(15)
        else:
            subset = filtered_df

        # 조합 생성
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
        tag_match_score = 100 # 태그 미선택 시 태그 일치도 100점
    else:
        total_items = len(combo_items)
        if total_items == 0:
            tag_match_score = 0
        else:
            total_matches = 0
            selected_tags_set = set(selected_tags)

            for item in combo_items:
                item_tags_set = set(item['tags'])
                # 아이템의 태그 중 하나라도 선택된 태그와 겹치면 매치로 인정
                if item_tags_set.intersection(selected_tags_set):
                    total_matches += 1 
            
            # (매치된 아이템 수 / 전체 아이템 수) * 100
            tag_match_score = (total_matches / total_items) * 100

    # --- 2. 인기 점수 (Popularity Score) 계산 (30% 가중치) ---
    total_popularity = sum(item['popularity'] for item in combo_items)
    
    # 인기 점수를 100점 만점 기준으로 환산 (인기 점수 1~10 기준이므로, 평균 인기점수 * 10)
    avg_popularity_score = total_popularity / len(combo_items) if combo_items else 0
    popularity_score_100 = avg_popularity_score * 10 
    
    # --- 3. 최종 가중치 점수 계산 (100점 만점) ---
    WEIGHT_TAG = 0.7
    WEIGHT_POPULARITY = 0.3
    
    final_score = (tag_match_score * WEIGHT_TAG) + (popularity_score_100 * WEIGHT_POPULARITY)
    
    return round(final_score, 1)


# --- Streamlit 앱 구성 ---

st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# 이미지 로드 함수 (파일이 없을 경우 대비)
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
    st.subheader("예산, 취향, 인기를 고려한 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.markdown("#### 👤 인원수")
        n_people = st.number_input("함께하는 인원수", min_value=1, max_value=10, value=2, step=1)

    with col2:
        st.markdown("#### 💰 예산 설정 (1인 기준)")
        # 예산 무제한 체크박스
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        
        # 예산 슬라이더
        if budget_unlimited:
            budget = float('inf') # 무한대로 설정
            st.slider("최대 예산 설정", min_value=5000, max_value=50000, value=50000, step=1000, disabled=True)
        else:
            # 1인 기준 예산
            budget = st.slider("최대 예산 설정 (1인 기준)", min_value=5000, max_value=50000, value=15000, step=1000)

    with col3:
        st.markdown("#### 🥖 베이커리 개수 (1인 기준)")
        # 1세트(1인) 기준의 베이커리 개수
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)
        
    with col4:
        st.markdown("#### 🏷️ 선호 해시태그 (최대 3개)")
        selected_tags = st.multiselect(
            "취향에 맞는 태그를 선택하세요.",
            options=all_tags,
            default=[],
            max_selections=3,
            placeholder="예: #달콤한, #고소한, #든든한"
        )
    
    st.markdown("---")

    # 2. 추천 실행 버튼
    if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
        st.markdown("### 🏆 AI 추천 메뉴 조합")
        
        # 1세트(1인분) 기준의 최대 가격만 고려
        max_price_per_set = budget

        # --- 추천 생성 ---
        
        # 1. 음료 추천 (1개)
        drink_recommendations = recommend_menu(drink_df, selected_tags, 1, max_price=max_price_per_set)
        
        # 2. 베이커리 추천 (n_bakery 개)
        bakery_recommendations = recommend_menu(bakery_df, selected_tags, n_bakery, max_price=max_price_per_set)
        
        
        if not drink_recommendations or not bakery_recommendations:
            st.warning("선택하신 조건에 맞는 메뉴를 찾지 못했습니다. 태그나 예산을 조정해 주세요.")
        else:
            # 3. 최종 조합 생성 및 스코어링
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations) 

            final_sets = []
            
            for drink_combo, bakery_combo in all_combinations:
                # 1세트 가격 계산
                drink_price = drink_combo[0]['price']
                bakery_price_sum = sum(item['price'] for item in bakery_combo)
                total_price_per_set = drink_price + bakery_price_sum
                
                # 전체 아이템 (점수 계산용)
                all_items = drink_combo + bakery_combo

                if max_price_per_set == float('inf') or total_price_per_set <= max_price_per_set:
                    # 가중치 점수 계산
                    weighted_score = calculate_weighted_score(all_items, selected_tags)
                    
                    final_sets.append({
                        "score": weighted_score,
                        "drink": drink_combo[0],
                        "bakery": bakery_combo,
                        "total_price_per_set": total_price_per_set,
                        "total_price_for_n_people": total_price_per_set * n_people
                    })
                
                if len(final_sets) >= 200: # 너무 많은 조합 생성 방지
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
                    
                    # 가격 정보 표시
                    st.markdown(f"**1인 세트 가격:** {result['total_price_per_set']:,}원")
                    st.markdown(f"**{n_people}명 예상 총 가격:** **{result['total_price_for_n_people']:,}원** (1인 세트 {n_people}개 기준)")
                    
                    st.markdown("##### 음료 🥤 (1개)")
                    # 태그와 인기 점수 포함하여 출력
                    drink_tags_str = ", ".join(f"#{t}" for t in result['drink']['tags'] if t != '인기')
                    is_popular = " (인기 메뉴!)" if result['drink']['popularity'] == 10 else ""
                    st.info(f"**{result['drink']['name']}** ({result['drink']['price']:,}원){is_popular} - *태그: {drink_tags_str}*")
                    
                    st.markdown(f"##### 베이커리 🍞 ({n_bakery}개)")
                    # 베이커리 목록을 문자열로 포맷팅
                    for item in result['bakery']:
                        bakery_tags_str = ", ".join(f"#{t}" for t in item['tags'] if t != '인기')
                        is_popular = " (인기 메뉴!)" if item['popularity'] == 10 else ""
                        st.success(f"• **{item['name']}** ({item['price']:,}원){is_popular} - *태그: {bakery_tags_str}*")
                    
                    if i < len(top_3_sets) - 1:
                        st.markdown("---")
            
    st.caption("※ 추천 점수(100점 만점)는 **태그 일치도(70%)**와 메뉴의 **인기 점수(30%)**를 가중치로 계산한 값입니다. '인기 메뉴!'는 메뉴 시트에 `#인기` 태그가 있는 경우 자동으로 부여됩니다.")

with tab_menu_board:
    st.title("📋 메뉴판")
    st.markdown("---")

    # 이미지 로드 및 표시 (이미지 파일이 없을 경우 대비)
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("베이커리 메뉴")
        if img1:
            st.image(img1, caption="Bakery 메뉴판 (1/2)", use_column_width=True)
        else:
            st.warning("`menu_board_1.png` 파일을 찾을 수 없어 이미지 대신 데이터 테이블을 표시합니다.")
            st.dataframe(bakery_df.drop(columns=['tags_list']).rename(columns={'name': '메뉴', 'price': '가격', 'tags': '태그', 'popularity_score': '인기점수'}), use_container_width=True)


    with col_img2:
        st.subheader("음료 메뉴")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)
        else:
            st.warning("`menu_board_2.png` 파일을 찾을 수 없어 이미지 대신 데이터 테이블을 표시합니다.")
            st.dataframe(drink_df.drop(columns=['tags_list']).rename(columns={'name': '메뉴', 'price': '가격', 'tags': '태그', 'popularity_score': '인기점수'}), use_container_width=True)

    
