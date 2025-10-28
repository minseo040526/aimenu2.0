
### Python Streamlit 코드

```python
import streamlit as st
import pandas as pd
import random
import itertools
from PIL import Image

# --- 데이터 로드 ---
try:
    # 파일 경로가 실제 환경에 맞게 조정될 수 있습니다.
    bakery_df = pd.read_csv("Bakery_menu.csv")
    drink_df = pd.read_csv("Drink_menu.csv")
except FileNotFoundError:
    st.error("메뉴 CSV 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.")
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

# 전체 사용 가능한 태그 추출
all_bakery_tags = sorted(list(set(tag for sublist in bakery_df['tags_list'] for tag in sublist)))
all_drink_tags = sorted(list(set(tag for sublist in drink_df['tags_list'] for tag in sublist)))
all_tags = sorted(list(set(all_bakery_tags + all_drink_tags)))


# --- 추천 로직 함수 ---

def recommend_menu(df, selected_tags, n_items, max_price=None):
    """
    주어진 예산과 태그를 기반으로 메뉴 조합을 추천합니다.
    (음료는 1개만, 베이커리는 n_items 만큼 조합)
    """

    # 1. 태그 필터링
    if selected_tags:
        # 하나라도 일치하는 태그가 있으면 선택
        filtered_df = df[df['tags_list'].apply(lambda tags: any(tag in selected_tags for tag in tags))]
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        return []

    # 2. 조합 생성
    recommendations = []
    
    # 베이커리 추천: n_items 만큼 조합 생성
    if df is bakery_df:
        if n_items == 1:
             # 단일 베이커리 추천 (랜덤 섞기 후 가격순 정렬하여 다양성 확보)
            items = filtered_df.sample(frac=1).sort_values(by='price', ascending=True)
            for _, row in items.iterrows():
                if max_price is None or row['price'] <= max_price:
                    recommendations.append([(row['name'], row['price'])])
                    if len(recommendations) >= 100: # 최대 100개까지만 생성
                        break
        else:
            # itertools.combinations로 조합 생성 (메모리 및 시간 제한을 위해 작은 데이터셋 사용)
            if len(filtered_df) > 15: # 조합 가능한 아이템이 너무 많으면 일부만 선택
                subset = filtered_df.sample(n=min(15, len(filtered_df)))
            else:
                subset = filtered_df

            # 조합 생성
            all_combinations = list(itertools.combinations(subset.itertuples(index=False), n_items))
            random.shuffle(all_combinations) # 랜덤하게 섞어 다양한 결과 유도

            for combo in all_combinations:
                total_price = sum(item.price for item in combo)
                if max_price is None or total_price <= max_price:
                    recommendations.append([(item.name, item.price) for item in combo])
                    if len(recommendations) >= 100: # 최대 100개까지만 생성
                        break
    
    # 음료 추천: 무조건 1개만
    elif df is drink_df:
        items = filtered_df.sample(frac=1).sort_values(by='price', ascending=True)
        for _, row in items.iterrows():
            if max_price is None or row['price'] <= max_price:
                recommendations.append([(row['name'], row['price'])])
                if len(recommendations) >= 100: # 최대 100개까지만 생성
                    break
    
    return recommendations

# --- Streamlit 앱 구성 ---

st.set_page_config(page_title="AI 베이커리 메뉴 추천 시스템", layout="wide")

# 사이드바: 메뉴판 탭의 이미지를 위해 PIL 사용
def load_image(image_path):
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"이미지 파일 '{image_path}'을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"이미지 로드 중 오류 발생: {e}")
        return None


# --- 탭 구성 ---
tab_recommendation, tab_menu_board = st.tabs(["AI 메뉴 추천", "메뉴판"])


with tab_recommendation:
    st.title("💡 AI 메뉴 추천 시스템")
    st.subheader("예산과 취향에 맞는 최고의 조합을 찾아보세요!")
    st.markdown("---")

    # 1. 설정 섹션 (사이드바 대신 메인 화면에 배치)
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown("#### 💰 예산 설정")
        # 예산 무제한 체크박스
        budget_unlimited = st.checkbox("예산 무제한", value=True)
        
        # 예산 슬라이더
        if budget_unlimited:
            budget = float('inf') # 무한대로 설정
            st.slider("최대 예산 설정", min_value=5000, max_value=30000, value=20000, step=1000, disabled=True)
        else:
            budget = st.slider("최대 예산 설정", min_value=5000, max_value=30000, value=15000, step=1000)

    with col2:
        st.markdown("#### 🥖 베이커리 개수")
        n_bakery = st.slider("추천받을 베이커리 개수", min_value=1, max_value=5, value=2, step=1)
        
    with col3:
        st.markdown("#### 🏷️ 해시태그 선택 (최대 3개)")
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
        st.markdown("### 🏆 AI 추천 메뉴 조합 3세트")
        
        # 예산 분배 (간단하게 음료 최소가 4000원 가정)
        # 예산이 무제한이거나 충분하면 모두 사용 가능
        if budget == float('inf'):
            max_drink_price = float('inf')
            max_bakery_price = float('inf')
            total_max_price = float('inf')
        else:
            # 예산 내에서 음료 1개 + 베이커리 n개 조합
            # 가장 저렴한 음료(4000원 가정)를 제외하고 남은 금액을 베이커리에 할당할 수 있도록
            # 조합 추천 시에 전체 예산(budget)을 기준으로 필터링하도록 로직 단순화
            max_drink_price = budget 
            total_max_price = budget

        # --- 추천 생성 ---
        
        # 1. 음료 추천
        drink_recommendations = recommend_menu(drink_df, selected_tags, 1, max_price=max_drink_price)
        
        # 2. 베이커리 추천
        # 전체 예산에서 가장 저렴한 음료 가격(4000원)을 빼고 베이커리 예산을 잡을 수도 있지만,
        # 여기서는 음료와 베이커리를 독립적으로 추천 후 조합의 전체 가격을 필터링하는 방식으로 진행
        bakery_recommendations = recommend_menu(bakery_df, selected_tags, n_bakery, max_price=total_max_price)
        
        
        if not drink_recommendations or not bakery_recommendations:
            st.warning("선택하신 조건에 맞는 메뉴를 찾지 못했습니다. 태그나 예산을 조정해 주세요.")
        else:
            # 3. 최종 조합 생성
            # 상위 100개 음료, 100개 베이커리 조합 중에서 예산에 맞는 3세트 랜덤 추출
            
            # 음료와 베이커리 조합: (음료, 베이커리)
            all_combinations = list(itertools.product(drink_recommendations, bakery_recommendations))
            random.shuffle(all_combinations)

            final_sets = []
            
            for drink_combo, bakery_combo in all_combinations:
                # 음료는 항상 1개이므로 drink_combo[0]
                drink_name, drink_price = drink_combo[0]
                
                # 베이커리 가격 합산
                bakery_price_sum = sum(price for name, price in bakery_combo)
                
                total_price = drink_price + bakery_price_sum
                
                if budget == float('inf') or total_price <= budget:
                    final_sets.append({
                        "drink": (drink_name, drink_price),
                        "bakery": bakery_combo,
                        "total_price": total_price
                    })
                
                if len(final_sets) >= 3:
                    break

            if not final_sets:
                 st.warning("선택하신 조건에 맞는 메뉴 조합을 찾지 못했습니다. 태그나 예산을 조정해 주세요.")
            else:
                for i, result in enumerate(final_sets):
                    st.markdown(f"#### ☕️ 세트 {i+1} (총 가격: **{result['total_price']:,}원**)")
                    
                    st.markdown("##### 음료 🥤")
                    st.info(f"**{result['drink'][0]}** ({result['drink'][1]:,}원)")
                    
                    st.markdown("##### 베이커리 🍞")
                    # 베이커리 목록을 문자열로 포맷팅
                    bakery_list_str = " / ".join([f"{name} ({price:,}원)" for name, price in result['bakery']])
                    st.success(f"{bakery_list_str}")
                    
                    if i < len(final_sets) - 1:
                        st.markdown("---")
        
    st.caption("※ 추천 로직은 선택된 해시태그를 포함하며, 설정된 예산 내에서 랜덤하게 조합을 추출합니다.")

with tab_menu_board:
    st.title("📋 메뉴판")
    st.markdown("---")

    # 이미지 로드 및 표시
    img1 = load_image("menu_board_1.png")
    img2 = load_image("menu_board_2.png")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("메뉴판 1")
        if img1:
            st.image(img1, caption="Bakery 메뉴판 (1/2)", use_column_width=True)

    with col_img2:
        st.subheader("메뉴판 2")
        if img2:
            st.image(img2, caption="Drink 메뉴판 (2/2)", use_column_width=True)

    st.caption("※ 이미지 파일(menu_board_1.png, menu_board_2.png)이 앱이 실행되는 폴더에 있어야 정상적으로 표시됩니다.")

```
