# Этот код реализует интерактивный веб-интерфейс на Streamlit для получения прогноза рейтинга ресторана.
# Пользователь выбирает характеристики заведения: тип, кухню, популярное блюдо, а также дополнительные параметры
# (штат, город, количество рабочих дней и т.д.).
# Категориальные признаки передаются в API в исходном виде, а их преобразование в числовой формат
# (включая one-hot кодирование категориальных переменных) выполняется уже на стороне FastAPI.

import streamlit as st        # Streamlit — библиотека для создания интерактивных веб-приложений на Python.
                              # Позволяет строить интерфейсы без знания HTML/JS, прямо из кода.
import requests               # Requests — удобная библиотека для отправки HTTP-запросов (GET, POST и др.).
                              # Здесь используется для общения с REST API (FastAPI).
import joblib                 # Joblib — библиотека для сериализации объектов Python.
                              # В проекте применяется для загрузки сохранённых моделей и вспомогательных данных.
from pathlib import Path      # Pathlib — современный модуль для работы с файловыми путями.
                              # Делает код более читаемым и переносимым между операционными системами.

# Определяем базовую директорию проекта и путь к файлу с колонками модели
BASE_DIR = Path(__file__).resolve().parent.parent
COLUMNS_PATH = BASE_DIR / "model" / "columns.pkl"

# Загружаем список признаков модели (колонки, которые ожидает модель)
model_columns = joblib.load(COLUMNS_PATH)

# Настройка страницы Streamlit: заголовок и расположение элементов
st.set_page_config(page_title="Прогноз рейтинга ресторана", layout="centered")
st.title("Прогноз рейтинга ресторана")
st.write("Выберите характеристики заведения и получите прогноз рейтинга.")

# URL API, к которому будем обращаться
API_URL = "http://127.0.0.1:8000/predict"

# --- Справочники категорий ---
# Эти списки используются для построения выпадающих меню.
format_categories = [
    "Restaurants", "Cafes", "Bars", "Pubs", "Cafeteria", "Diners",
    "Bistros", "Buffets", "Fast Food", "Food Court", "Street Vendors",
    "Food Trucks", "Breakfast & Brunch", "Steakhouses", "Sushi Bars",
    "Cocktail Bars", "Wine Bars", "Beer Bar", "Gastropubs", "Hotel bar", "Irish Pub"
]

cuisine_categories = [
    "Italian", "French", "Chinese", "Japanese", "American (New)", "American (Traditional)",
    "Thai", "Mexican", "Indian", "Korean", "Vietnamese", "Turkish", "Greek", "Spanish",
    "Middle Eastern", "Mediterranean"
]

popular_food_categories = [
    "Burgers", "Pizza", "Sandwiches", "Hot Dogs", "Chicken Wings", "Chicken Shop", "Donuts",
    "Bagels", "Pancakes", "Waffles", "Cheesesteaks", "Wraps", "Noodles", "Soup", "Salad",
    "Poke", "Ramen", "Dumplings", "Dim Sum", "Donburi", "Tonkatsu"
]

# Список штатов формируем из колонок модели (все признаки, начинающиеся с "state_")
state_options = sorted([col.replace("state_", "") for col in model_columns if col.startswith("state_")])

# --- Интерфейс выбора ---
# Streamlit позволяет создавать интерактивные элементы: выпадающие списки, поля ввода, слайдеры.
# Пользователь выбирает значения, которые затем формируют payload для API.

selected_format = st.selectbox("Тип заведения", format_categories)
selected_cuisine = st.selectbox("Кухня", ["Не выбрано"] + cuisine_categories)
selected_food = st.selectbox("Популярное блюдо", ["Не выбрано"] + popular_food_categories)

selected_state = st.selectbox("Штат", state_options)
selected_city = st.text_input("Город", value="")

open_days_count = st.slider("Количество рабочих дней в неделю", 0, 7, 5)

# --- Дополнительные бинарные признаки ---
# Чтобы не перегружать интерфейс, часть признаков фиксируем по умолчанию:
# wheelchair = 1 (доступность для инвалидов предполагается), is_open = 1 (заведение открыто).
# Остальные признаки пользователь может выбрать вручную.
binary_features = {
    "ambience_casual": st.selectbox("Неформальная атмосфера", [0, 1], index=1),
    "RestaurantsGoodForGroups": st.selectbox("Подходит для компаний", [0, 1], index=1),
    "HasTV": st.selectbox("Есть телевизор", [0, 1])
}

# --- Формирование payload ---
# Payload — это словарь с данными, которые отправляются в API.
payload = {
    "state": selected_state,
    "city": selected_city,
    "open_days_count": open_days_count,
    "WheelchairAccessible": 1,   # фиксируем как доступное
    "is_open": 1,                # фиксируем как открытое
    **binary_features,           # добавляем остальные бинарные признаки
    "format_category": selected_format,
    "cuisine": None if selected_cuisine == "Не выбрано" else selected_cuisine,
    "popular_food": None if selected_food == "Не выбрано" else selected_food
}

# --- Запрос к API ---
# При нажатии кнопки "Получить прогноз" отправляем POST-запрос к FastAPI.
# Если ответ успешный (код 200), выводим предсказанный рейтинг и дополнительные данные.
# Если произошла ошибка — выводим сообщение об ошибке.
if st.button("Получить прогноз"):
    if not selected_city.strip():
        st.error("Введите город")
    else:
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Предсказанный рейтинг: {result['predicted_rating']}")
                st.info(f"Географический кластер: {result['geo_cluster']}")
                st.info(f"Плотность города: {result['city_density']}")
            else:
                st.error(f"Ошибка API: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error(f"Не удалось подключиться к FastAPI: {e}")
