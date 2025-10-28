import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# --- 데이터 로드 및 전처리 ---
def load_and_preprocess_data():
    """CSV 파일을 로드하고 태그 및 당도 컬럼을 전처리합니다."""
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
    
    # 1. 태그 전처리 함수 (기존 로직 유지)
    def preprocess_tags(df):
        """CSV의 tags 컬럼을 클린징하고 리스트로 변환합니다."""
        df['tags_list'] = df['tags'].fillna('').astype(str).str.strip().str.replace('#', '').str.split(r'\s*,\s*')
        df['tags_list'] = df['tags_list'].apply(lambda x: [tag.strip() for tag in x if tag.strip()])
        return df

    bakery_df = preprocess_tags(bakery_df)
    drink_df = preprocess_tags(drink_df)

    # 2. 당도(sweetness) 컬럼 처리
    # sweetness 컬럼이 숫자가 아닐 경우를 대비해 처리
    for df in [bakery_df, drink_df]:
        if 'sweetness' in df.columns:
            df['sweetness'] = pd.to_numeric(df['sweetness'], errors='coerce').fillna(0).astype(int)
        else:
            # sweetness 컬럼이 없는 경우 기본값 0 설정
            df['sweetness'] = 0

    # 전체 사용 가능한 태그 추출
    all_bakery_tags = sorted(list(set(tag for sublist in bakery_df['tags_list'] for tag in sublist)))
    all_drink_tags = sorted(list(set(tag for sublist in drink_df['tags_list'] for tag in sublist)))
    all_tags = sorted(list(set(all_bakery_tags + all_drink_tags)))
    
    return bakery_df, drink_df, all_tags

bakery_df, drink_df, all_tags = load_and_preprocess_data()


# --- 추천 로직 함수 ---

def recommend_menu(df, selected_tags, n_items, max_price=None, max_sweetness=None, is_drink=False):
    """
    주어진 예산, 태그, 당도를 기반으로 메뉴 조합을 추천합니다.
    (is_drink=True인 경우 n_items는 음료 수량(인원수)이 됨)
    """

    # 1. 태그 필터링
    if selected_tags:
        filtered_df = df[df['tags_list'].apply(lambda tags: any(tag in selected_tags for tag in tags))]
    else:
        filtered_df = df.copy()
        
    # 2. 당도 필터링 (음료에만 적용)
    if is_drink and max_sweetness is not None and 'sweetness' in filtered_df.columns:
        # max_sweetness 보다 당도가 낮거나 같은 메뉴만 선택
        filtered_df = filtered_df[filtered_df['sweetness'] <= max_sweetness]

    if filtered_df.empty:
        return []

    # 3. 조합 생성
    recommendations = []
    
    # 음료/베이커리 조합 생성
    if n_items == 1 and not is_drink: # 단일 베이커리 추천 (n_bakery=1)
         items = filtered_df.sample(frac=1).sort_values(by='price', ascending=True)
         for _, row in items.iterrows():
             if max_price is None or row['price'] <= max_price:
                 recommendations.append([(row['name'], row['price'])])
                 if len(recommendations) >= 100:
                     break
    elif n_items == 1 and is_drink: # 단일 음료 추천 (음료는 항상 1개 메뉴만 선택)
        items = filtered_df.sample(frac=1).sort_values(by='price', ascending=True)
        for _, row in items.iterrows():
            if max_price is None or row['price'] <= max_price:
                recommendations.append([(row['name'], row['price'])])
                if len(recommendations) >= 100:
                    break
    else:
        # 여러 아이템 조합 (베이커리 n_bakery > 1 또는 음료 n_people > 1)
        
        # itertools.combinations/product를 위한 데이터셋 준비
        if len(filtered_df) > 15 and not is_drink:
             # 베이커리 조합이 너무 많으면 일부만 선택 (메모리 제한)
            subset = filtered_df.sample(n=min(15, len(filtered_df)), random_state=42)
        elif len(filtered_df) > 10 and is_drink:
             # 음료 조합이 너무 많으면 일부만 선택
             subset = filtered_df.sample(n=min(10, len(filtered_df)), random_state=42)
        else:
            subset = filtered_df
        
        # 음료: 인원수(n_items)만큼 중복을 허용하여 선택 (itertools.product)
        # 베이커리: n_items만큼 중복 없이 선택 (itertools.combinations)
        if is_drink:
            all_combinations = list(itertools.product(subset.itertuples(index=False), repeat=n_items))
        else:
            # 베이커리는 중복 없이 선택
            all_combinations = list(itertools.combinations(subset.itertuples(index=False), n_items))


        random.shuffle(all_combinations) # 랜덤하게 섞어 다양한 결과 유도

        for combo in all_combinations:
            total_price = sum(item.price for item in combo)
            if max_price is None or total_price <= max_price:
                # 음료 조합의 경우, 어떤 메뉴를 선택했는지와 가격을 조합 리스트에 추가
                recommendations.append([(item.name, item.price) for item in combo])
                if len(recommendations) >= 100: # 최대 100개까지만 생성
                    break
    
    return recommendations


# --- Streamlit 앱 구성 ---

st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# 사이드바: 메뉴판 탭의 이미지를 위해 PIL 사용
def load_image(image_path):
    # 이미지 파일이 제공되지 않았으므로 None을 반환합니다.
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
    st.subheader("예산과 취향에 맞는 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.markdown("#### 💰 예산 설정")
        # 예산 무제한 체크박스
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        
        # 예산 슬라이더
        if budget_unlimited:
            budget = float('inf') # 무한대로 설정
            st.slider("최대 예산 설정", min_value=5000, max_value=30000, value=30000, step=1000, disabled=True)
        else:
            budget = st.slider("최대 예산 설정", min_value=5000, max_value=30000, value=15000, step=1000)

    with col2:
        st.markdown("#### 🧑‍🤝‍🧑 인원 / 🥖 베이커리 개수")
        # 인원 설정 (음료 수량)
        n_people = st.slider("인원 (음료 수량)", min_value=1, max_value=5, value=1, step=1)
        # 베이커리 개수
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)
        
    with col3:
        st.markdown("#### 🌡️ 당도 설정 (음료)")
        # 당도 설정 (0: 무당 ~ 4: 고당)
        max_sweetness = st.slider(
            "최대 당도 선호도 (0~4)", 
            min_value=0, max_value=4, value=4, step=1, 
            help="0: 무당, 4: 고당. 선택한 값 이하의 당도를 가진 음료만 추천됩니다."
        )

    with col4:
        st.markdown("#### 🏷️ 해시태그 선택 (최대 3개)")
        selected_tags = st.multiselect(
            "취향에 맞는 태그를 선택하세요.",
            options=all_tags,
            default=[],
            max_selections=3,
            placeholder="예: 달콤한, 고소한, 든든한"
        )
    
    st.markdown("---")

    # 2. 추천 실행 버튼
    if st.button("AI 추천 메뉴 조합 받기", type="primary", use_container_width=True):
        st.markdown("### 🏆 AI 추천 메뉴 조합 3세트")
        
        # 예산 할당: 전체 예산(budget) 기준으로 필터링 로직 단순화
        max_drink_price = budget
        total_max_price = budget

        # --- 추천 생성 ---
        
        # 1. 음료 추천 (인원수만큼 n_people개의 음료 조합)
        # is_drink=True로 설정하여 당도 필터링을 활성화
        drink_recommendations = recommend_menu(
            drink_df, selected_tags, n_people, 
            max_price=max_drink_price, max_sweetness=max_sweetness, 
            is_drink=True
        )
        
        # 2. 베이커리 추천 (n_bakery개의 베이커리 조합)
        bakery_recommendations = recommend_menu(
            bakery_df, selected_tags, n_bakery, 
            max_price=total_max_price, 
            is_drink=False # 베이커리는 당도 필터링 미적용 (태그로 대체)
        )
        
        
        if not drink_recommendations or not bakery_recommendations:
            st.warning("선택하신 조건에 맞는 메뉴를 찾지 못했습니다. 태그, 인원, 당도, 예산을 조정해 주세요.")
        else:
            # 3. 최종 조합 생성
            
            # 음료 조합과 베이커리 조합을 결합
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations)

            final_sets = []
            
            for drink_combo, bakery_combo in all_combinations:
                
                # 음료 가격 합산
                drink_price_sum = sum(price for name, price in drink_combo)
                
                # 베이커리 가격 합산
                bakery_price_sum = sum(price for name, price in bakery_combo)
                
                total_price = drink_price_sum + bakery_price_sum
                
                if budget == float('inf') or total_price <= budget:
                    final_sets.append({
                        "drink": drink_combo,
                        "bakery": bakery_combo,
                        "total_price": total_price
                    })
                
                if len(final_sets) >= 3:
                    break

            if not final_sets:
                st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 태그, 인원, 당도, 예산을 조정해 주세요.")
            else:
                for i, result in enumerate(final_sets):
                    st.markdown(f"#### ☕️ 세트 {i+1} (총 가격: **{result['total_price']:,}원**)")
                    
                    st.markdown(f"##### 음료 🥤 ({n_people}잔)")
                    # 음료 목록을 문자열로 포맷팅
                    drink_list_str = " / ".join([f"{name} ({price:,}원)" for name, price in result['drink']])
                    st.info(f"{drink_list_str}")
                    
                    st.markdown(f"##### 베이커리 🍞 ({n_bakery}개)")
                    # 베이커리 목록을 문자열로 포맷팅
                    bakery_list_str = " / ".join([f"{name} ({price:,}원)" for name, price in result['bakery']])
                    st.success(f"{bakery_list_str}")
                    
                    if i < len(final_sets) - 1:
                        st.markdown("---")
            
    st.caption("※ 추천 로직은 선택된 해시태그를 포함하며, 설정된 인원수와 당도 선호도를 반영하여 예산 내에서 조합을 추출합니다.")

with tab_menu_board:
    st.title("📋 메뉴판")
    st.markdown("---")

    # 이미지 로드 및 표시
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("메뉴판 1 (베이커리)")
        if img1:
            st.image(img1, caption="Bakery 메뉴판 (1/2)", use_column_width=True)
        else:
            st.warning("`menu_board_1.png` 파일을 찾을 수 없어 메뉴판 이미지를 표시할 수 없습니다. 대신 데이터 테이블을 표시합니다.")
            display_df = bakery_df.drop(columns=['tags_list', 'tags']).rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(0-4)'})
            st.dataframe(display_df, use_container_width=True)


    with col_img2:
        st.subheader("메뉴판 2 (음료)")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)
        else:
            st.warning("`menu_board_2.png` 파일을 찾을 수 없어 메뉴판 이미지를 표시할 수 없습니다. 대신 데이터 테이블을 표시합니다.")
            display_df = drink_df.drop(columns=['tags_list', 'tags']).rename(columns={'name': '메뉴', 'price': '가격', 'sweetness': '당도(0-4)'})
            st.dataframe(display_df, use_container_width=True)

  
