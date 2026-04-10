# Данный код реализует REST API для предсказания рейтинга ресторанов на основе обученной модели машинного обучения.
# FastAPI используется как лёгкий и высокопроизводительный веб-фреймворк: он позволяет быстро создавать веб-сервисы,
# предоставляя удобный синтаксис для описания эндпоинтов и автоматическую генерацию документации.
# Pydantic обеспечивает строгую валидацию входных данных: каждый ресторан описывается набором признаков,
# и библиотека гарантирует, что данные будут приведены к нужному типу.
# Joblib используется для загрузки заранее обученной модели и вспомогательных объектов (списки признаков, словари).
# Pathlib - современный модуль для работы с файловыми путями: он делает код более читаемым и переносимым между ОС.

from fastapi import FastAPI, HTTPException   # FastAPI - веб-фреймворк, HTTPException — для возврата ошибок
from pydantic import BaseModel               # Pydantic - строгая валидация входных данных
import pandas as pd                          # Pandas - работа с табличными данными
import joblib                                # Joblib - загрузка/сохранение моделей и объектов
from pathlib import Path                     # Pathlib - удобная работа с путями к файлам

# Создаём приложение FastAPI с заголовком
app = FastAPI(title="Restaurant Rating Prediction API")

# Пути к модели, колонкам и словарям
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "model.pkl")                        # обученная модель
model_columns = joblib.load(BASE_DIR / "model" / "columns.pkl")              # список признаков
city_cluster_map = joblib.load(BASE_DIR / "model" / "city_cluster_map.pkl")  # словарь: (штат, город) -> кластер
city_density_map = joblib.load(BASE_DIR / "model" / "city_density_map.pkl")  # словарь: город -> плотность

# Категории форматов заведений (например, ресторан, бар, кафе)
format_categories = [
    'Restaurants', 'Cafes', 'Bars', 'Pubs', 'Cafeteria', 'Diners',
    'Bistros', 'Buffets', 'Fast Food', 'Food Court', 'Street Vendors',
    'Food Trucks', 'Breakfast & Brunch', 'Steakhouses', 'Sushi Bars',
    'Cocktail Bars', 'Wine Bars', 'Beer Bar', 'Gastropubs', 'Hotel bar', 'Irish Pub'
]

# Категории кухонь (например, итальянская, французская, японская)
cuisine_categories = [
    'Italian', 'French', 'Chinese', 'Japanese', 'American (New)', 'American (Traditional)',
    'Thai', 'Mexican', 'Indian', 'Korean', 'Vietnamese', 'Turkish', 'Greek', 'Spanish',
    'Middle Eastern', 'Mediterranean'
]

# Популярные блюда (например, пицца, бургеры, лапша)
popular_food_categories = [
    'Burgers', 'Pizza', 'Sandwiches', 'Hot Dogs', 'Chicken Wings', 'Chicken Shop', 'Donuts',
    'Bagels', 'Pancakes', 'Waffles', 'Cheesesteaks', 'Wraps', 'Noodles', 'Soup', 'Salad',
    'Poke', 'Ramen', 'Dumplings', 'Dim Sum', 'Donburi', 'Tonkatsu'
]

# Функция нормализации названий категорий:
# переводим строку в нижний регистр, заменяем пробелы и спецсимволы на "_"
def normalize_name(x: str) -> str:
    return (x.lower()
              .replace(' & ', '_')
              .replace(' ', '_')
              .replace('/', '_')
              .replace('(', '')
              .replace(')', ''))

# Описание структуры входных данных для API
# BaseModel от Pydantic гарантирует, что все поля будут проверены и приведены к нужному типу
class RestaurantFeatures(BaseModel):
    state: str                            # штат
    city: str                             # город
    open_days_count: int = 0              # количество дней работы
    WheelchairAccessible: int = 0         # доступность для инвалидов
    ambience_casual: int = 0              # неформальная атмосфера
    RestaurantsGoodForGroups: int = 0     # подходит для групп
    is_open: int = 1                      # открыт/закрыт
    HasTV: int = 0                        # наличие телевизора

    # Категориальные признаки для нового заведения
    format_category: str | None = None    # формат заведения
    cuisine: str | None = None            # кухня
    popular_food: str | None = None       # популярное блюдо

# Эндпоинт для проверки работы API
@app.get("/")
def root():
    return {"message": "API is running"}

# Эндпоинт для health-check (проверка состояния сервиса)
@app.get("/health")
def health():
    return {"status": "ok"}

# Основной эндпоинт для предсказания рейтинга
@app.post("/predict")
def predict(features: RestaurantFeatures):
    # Преобразуем входные данные в словарь
    data = features.dict()

    state = data["state"]
    city = data["city"]
    selected_format = data.get("format_category")
    selected_cuisine = data.get("cuisine")
    selected_popular_food = data.get("popular_food")

    # Проверяем допустимые значения категорий
    if selected_format and selected_format not in format_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format_category. Allowed values: {format_categories}"
        )

    if selected_cuisine and selected_cuisine not in cuisine_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cuisine. Allowed values: {cuisine_categories}"
        )

    if selected_popular_food and selected_popular_food not in popular_food_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid popular_food. Allowed values: {popular_food_categories}"
        )

    # Вычисляем плотность города автоматически через словарь
    city_density = city_density_map.get(city, 0)
    data["city_density"] = city_density

    # Вычисляем географический кластер (приближенная география)
    geo_cluster = city_cluster_map.get((state, city), 0)
    data["geo_cluster"] = int(geo_cluster)

    # One-hot кодирование для признака "state"
    for col in model_columns:
        if col.startswith("state_"):
            data[col] = 0
    state_col = f"state_{state}"
    if state_col in model_columns:
        data[state_col] = 1

    # One-hot кодирование для признака "format_category"
    for col in model_columns:
        if col.startswith("format_"):
            data[col] = 0
    if selected_format:
        normalized_format = normalize_name(selected_format)
        format_col = f"format_{normalized_format}"
        if format_col in model_columns:
            data[format_col] = 1

    # One-hot кодирование для признака "cuisine"
    for col in model_columns:
        if col.startswith("cuisine_"):
            data[col] = 0
    if selected_cuisine:
        normalized_cuisine = normalize_name(selected_cuisine)
        cuisine_col = f"cuisine_{normalized_cuisine}"
        if cuisine_col in model_columns:
            data[cuisine_col] = 1

    # One-hot кодирование для признака "popular_food"
    for col in model_columns:
        if col.startswith("food_"):
            data[col] = 0
    if selected_popular_food:
        normalized_food = normalize_name(selected_popular_food)
        food_col = f"food_{normalized_food}"
        if food_col in model_columns:
            data[food_col] = 1

    # Удаляем исходные поля, чтобы оставить только закодированные признаки
    for field in ["state", "city", "format_category", "cuisine", "popular_food"]:
        data.pop(field, None)

    # Приводим данные к формату модели: создаём DataFrame с нужными колонками
    row = pd.DataFrame([data]).reindex(columns=model_columns, fill_value=0)

    # Получаем предсказание рейтинга
    prediction = model.predict(row)[0]

    # Возвращаем результат в удобном формате
    return {
        "predicted_rating": round(float(prediction), 3),
        "geo_cluster": int(geo_cluster),
        "city_density": int(city_density)
    }
