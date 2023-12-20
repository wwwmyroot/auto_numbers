# auto_numbers
NN for automobile numbers detection
------

**Разработка нейросети для обнаружения автомобильных номеров на изображениях.**  
(Регион: dataset_1: Россия; dataset_2 (self_made): Крым, Симферополь )

* Основной код (.py) - [NN_auto_numbers_MAIN.py](https://github.com/wwwmyroot/nn_auto_numbers/blob/main/NN_auto_numbers_MAIN.py)
* Ноутбук (.ipynb) - [NN_auto_numbers_MAIN.ipynb](https://github.com/wwwmyroot/nn_auto_numbers/blob/main/NN_auto_numbers_MAIN.ipynb)
* Ноутбук с output cells  (GoogleCollab) - [NN_auto_numbers_MAIN_with_output.ipynb](https://github.com/wwwmyroot/nn_auto_numbers/..... _with_output.ipynb)
* Скрины - в папке [/img](https://github.com/wwwmyroot/nn_auto_numbers/tree/main/img)

-----

* **Dataset_1:** yandexcloud, 1.82G | ( yolo.zip );   
* **Dataset_2:** yandexcloud, ? G | ( auto_numbers_self_made_01.zip ); img scale ? ;  

-----

* **Results:**

* Yolo_v5
- **Эксп. № (sum)**
  - Средняя точность на обучающей выборке: 0.84
  - Максимальная точность на обучающей выборке: 0.87
  - Средняя точность на проверочной выборке: 0.85 
  - Максимальная точность на проверочной выборке: 0.88

* Yolo_v8
- **Эксп. № (sum)**
  - Средняя точность на обучающей выборке: 0.87
  - Максимальная точность на обучающей выборке: 0.91
  - Средняя точность на проверочной выборке: 0.86
  - Максимальная точность на проверочной выборке: 0.91

----- 

**Дополнительно.**

- Разметка: CVAT  

-----
