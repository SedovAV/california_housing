import os
from flask import Flask, request
import joblib

app = Flask(__name__)

MODEL_PATH = 'california_v3.pkl'
bundle = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        print("Модели загружены")
    except Exception as e:
        print(f"Ошибка: {e}")


def describe_location(lat, lon):
    """
    Возвращает человекочитаемое описание региона Калифорнии
    по широте и долготе. Приблизительные границы.
    """
    # Сан-Франциско и залив
    if 37.0 <= lat <= 38.5 and -123.0 <= lon <= -121.5:
        return "Сан-Франциско / район залива"

    # Сакраменто и центральная долина (север)
    if 38.0 <= lat <= 40.0 and -122.5 <= lon <= -121.0:
        return "Сакраменто / северная часть Центральной долины"

    # Лос-Анджелес
    if 33.7 <= lat <= 34.3 and -118.7 <= lon <= -117.5:
        return "Лос-Анджелес"

    # Сан-Диего
    if 32.5 <= lat <= 33.3 and -117.5 <= lon <= -116.5:
        return "Сан-Диего"

    # Южная Калифорния (общий случай)
    if lat < 34.5 and lon > -120.0:
        return "Южная Калифорния"

    # Центральная долина (общий случай)
    if 35.0 <= lat <= 38.0 and -121.5 <= lon <= -118.5:
        return "Центральная долина"

    # Северная Калифорния (общий случай)
    if lat > 38.0:
        return "Северная Калифорния"

    # Центральное побережье
    if 34.5 <= lat <= 37.0 and lon <= -120.0:
        return "Центральное побережье (Монтерей, Санта-Барбара)"

    # Пустыня / восток
    if lon > -117.0:
        return "Восточная Калифорния (пустыня)"

    return "Калифорния (точный регион не определён)"


@app.route('/')
def index():
    return """
    <div style="text-align: center; margin-top: 50px;">
        <h1>Оценка стоимости дома (California)</h1>
        <p><small>Введите параметры дома — приложение предскажет стоимость
        и порекомендует подходящий район.</small></p>
        <form method="POST" action="/predict">
            <p>
                Возраст дома (HouseAge):<br>
                <input type="number" step="any" name="house_age" required>
            </p>
            <p>
                Среднее число комнат (AveRooms):<br>
                <input type="number" step="any" name="ave_rooms" required>
            </p>
            <p>
                Среднее число спален (AveBedrms):<br>
                <input type="number" step="any" name="ave_bedrms" required>
            </p>
            <p>
                <input type="submit" value="Оценить">
            </p>
        </form>
    </div>
    """


@app.route('/predict', methods=['POST'])
def predict():
    if not bundle:
        return "<div style='text-align: center; margin-top: 50px;'>Модель не загружена<br><a href='/'>Назад</a></div>"

    try:
        house_age = float(request.form.get('house_age', 0))
        ave_rooms = float(request.form.get('ave_rooms', 0))
        ave_bedrms = float(request.form.get('ave_bedrms', 0))

        if house_age < 0 or ave_rooms <= 0 or ave_bedrms <= 0:
            return "<div style='text-align: center; margin-top: 50px;'>Ошибка: значения должны быть положительными<br><a href='/'>Назад</a></div>"

        features = [[house_age, ave_rooms, ave_bedrms]]

        # Предсказание цены
        pred = bundle['main'].predict(features)[0]
        price_usd = pred * 100_000

        # Рекомендации по району
        med_inc    = bundle['aux']['med_inc'].predict(features)[0]
        population = bundle['aux']['population'].predict(features)[0]
        ave_occup  = bundle['aux']['ave_occup'].predict(features)[0]
        latitude   = bundle['aux']['latitude'].predict(features)[0]
        longitude  = bundle['aux']['longitude'].predict(features)[0]

        # Обрезка значений в разумные пределы
        med_inc    = max(0.5, min(15.0, med_inc))
        population = max(3.0, min(4000.0, population))
        ave_occup  = max(1.0, min(10.0, ave_occup))
        latitude   = max(32.0, min(42.0, latitude))
        longitude  = max(-124.0, min(-114.0, longitude))

        # Перевод координат в человекочитаемый вид
        region = describe_location(latitude, longitude)

        med_inc_usd = med_inc * 10_000

        return f"""
        <div style="text-align: center; margin-top: 50px;">
            <h2>Оценочная стоимость дома</h2>
            <h3>${price_usd:,.0f}</h3>

            <hr style="width: 450px; margin: 20px auto;">

            <h3>Рекомендации по району</h3>
            <p>Чтобы дом с такими параметрами стоил указанную сумму,
            район должен иметь примерно такие характеристики:</p>
            <table style="margin: 0 auto; text-align: left; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px 15px;">Медианный доход:</td>
                    <td style="padding: 5px 15px;"><b>${med_inc_usd:,.0f}/год</b>
                        <small>({med_inc:.2f} в единицах датасета)</small></td>
                </tr>
                <tr>
                    <td style="padding: 5px 15px;">Население квартала:</td>
                    <td style="padding: 5px 15px;"><b>{population:.0f} чел.</b></td>
                </tr>
                <tr>
                    <td style="padding: 5px 15px;">Средняя занятость:</td>
                    <td style="padding: 5px 15px;"><b>{ave_occup:.2f}</b></td>
                </tr>
                <tr>
                    <td style="padding: 5px 15px;">Широта:</td>
                    <td style="padding: 5px 15px;"><b>{latitude:.4f}</b></td>
                </tr>
                <tr>
                    <td style="padding: 5px 15px;">Долгота:</td>
                    <td style="padding: 5px 15px;"><b>{longitude:.4f}</b></td>
                </tr>
                <tr>
                    <td style="padding: 5px 15px;">Регион:</td>
                    <td style="padding: 5px 15px;"><b>{region}</b></td>
                </tr>
            </table>

            <hr style="width: 450px; margin: 20px auto;">

            <h4>Введённые параметры</h4>
            <p>Возраст дома: {house_age} лет</p>
            <p>Среднее число комнат: {ave_rooms}</p>
            <p>Среднее число спален: {ave_bedrms}</p>

            <br>
            <a href="/">Вернуться</a>
        </div>
        """
    except Exception as e:
        return f"<div style='text-align: center; margin-top: 50px;'>Ошибка: {e}<br><a href='/'>Назад</a></div>"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
