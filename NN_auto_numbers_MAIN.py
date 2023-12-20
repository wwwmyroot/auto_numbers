# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
# ---

#
# Python file for GoogleCollab ant jupytext (pair file .ipynb)
# Original file is located at https://github.com/wwwmyroot/nn_auto_numbers/NN_auto_numbers_MAIN.py
#
# LATER: Добавить YOLO_v8

#@title ---- TRY: решение дисконнекта GoogleCollab
# ---- Не понадобилось, но сохраню на всякий случай ----
# -- v0
# from IPython.display import Audio
#   # import numpy as np
#
# Audio(np.array([0] * 2 * 3600 * 3000, dtype=np.int8), normalize=False, rate=3000, autoplay=True)
#   # Audio([None] * 2 * 3600 * 3000, normalize=False, rate=3000, autoplay=True)
#
# -- v1: F-12 and input in dev-tools console
#
# function ClickConnect(){
#  console.log("Working");
#  document.querySelector("colab-toolbar-button#connect").click()
#}
#setInterval(ClickConnect, 60000)
#
# -- v2: python solution
# import numpy as np
# import time
# import mouse
# while True:
    # random_row = np.random.random_sample()*100
    # random_col = np.random.random_sample()*10
    # random_time = np.random.random_sample()*np.random.random_sample() * 100
    # mouse.wheel(1000)
    # mouse.wheel(-1000)
    # mouse.move(random_row, random_col, absolute=False, duration=0.2)
    # mouse.move(-random_row, -random_col, absolute=False, duration = 0.2)
    # mouse.LEFT
    # time.sleep(random_time)
#

#@title ---- Создать инфраструктуру папок. Разово.
# -- папка "my_yolov5" для хранения моделей и данных
# -- v0: python lib (base commands for operating in OS);
import os
# -- Для работы с архивами в питоне
import zipfile
#
#
# полный путь к папке в переменной TRAIN_DIR. Будет использоваться и в других функциях
TRAIN_DIR = "/content/my_yolov5"
# при помощии команды "mkdir" создадим новую папку
os.mkdir(TRAIN_DIR)
print("-- Создан каталог  my_yolov5  для хранения моделей и данных.")
# посмотреть содержание каталога
print("-- Текущее содержание корневого рабочего каталога:")
print(os.listdir("/content/"))
#
# -- v1: напрямую
# NOTE: Лучше всё равно средствами python, т.к. переменная "TRAIN_DIR"
# будет использоваться и в других операциях;
#
# #!ls -la
# #!rm -rf /content/sample_data
# #!ls -la
# #!rm -rf /content/my_yolov5
# #!ls -la
# ---- аналог:
# #!mkdir {TRAIN_DIR}
# #!mkdir my_yolov5
# --- control:
# print(os.listdir("/content/"))
# #!ls -la
#

#@title ---- Загрузка подготовленного датасета (gdown; yolo.zip)
#
# -- загрузка при помощи gdown
# #!gdown https://storage.yandexcloud.net/terradev/terrayolo/cabin_data_1_classes.zip
# ---- UNCOMMENT:
# #!gdown https://storage.yandexcloud.net/aiueducation/marketing/datasets/yolo.zip
# gdown.download ("https://storage.yandexcloud.net/aiueducation/marketing/datasets/yolo.zip")
print("-- Датасет успешно загружен.")
#
# -- control:
# #!ls -la
os.listdir("/content/")
#

#@title ---- Распаковка датасета (.zip)
#
# -- средствами питона
try:
    with zipfile.ZipFile("yolo.zip", "r") as zip_ref:
        zip_ref.extractall("/content/my_yolov5")
except zipfile.BadZipFile:
    print("(!)ERROR: Файл архива поврежден или не является архивом.")
#
print("-- Датасет успешно распакован.")
# -- control:
# #!ls {TRAIN_DIR}
print(os.listdir(TRAIN_DIR))
#
# -- в коллабе:
# #!unzip -q -o yolo.zip  -d {TRAIN_DIR}
# - OR:
# #!unzip -q -o /content/yolo.zip  -d {TRAIN_DIR}
# #!unzip -q -o "yolo.zip" -d /content/my_yolov5
#
# #!unzip --h
#  -q: "тихий режим" min сообщений;
#  -o: режим перезаписи, на случай, если по путям распаковки существуют другие файлы;
#  -d: целевая директория распаковки;
#
# -- CLI python:
# python -m zipfile -e chest_xray.zip /content/data/
#

#@title ---- директории для train и valid
#
# в каталогах train и valid - тренировочная и валидационная части датасета;
# -- сохранить путь к файлу в отдельную переменную;
data_file_path = TRAIN_DIR+"dataset.yaml"
# через cat посмотрим содержимое файла data.yaml;
print("-- содержимое файла data.yaml (тренировочные и валидационные части)")
# #!cat {data_file_path}
with open(data_file_path, "r") as f:
    print(f.read())
#
#  посмотреть папку тренировочной выборки;
print("-- Папка тренировочной выборки:")
# #!ls {TRAIN_DIR+"train"}
print(os.listdir(TRAIN_DIR+"train"))
#
# ---- control:
print("-- Папка ", TRAIN_DIR+"train/labels :")
# #!ls {TRAIN_DIR+"train"}/labels
print(os.listdir(TRAIN_DIR+"train/labels"))
print("-- Папка ", TRAIN_DIR+"train/images :")
# #!ls {TRAIN_DIR+"train/images/"}
print(os.listdir(TRAIN_DIR+"train/images/"))
#
# ---- control:
# сколько картинок / сколько описаний
# NOTE: количества могут не совпадать, это нормально, так как номера могут быть не на всех картинках).
#
print("---- количество картинок: ", len(os.listdir(TRAIN_DIR+"train/images/")))
print("---- количество описаний: ", len(os.listdir(TRAIN_DIR+"train/labels/")))
#
# ---- control
# что содержится в описаниях в валидационной части;
# возьмем имя первого файла из подкаталога labels
lab_f_name = os.listdir(TRAIN_DIR+"train/labels/")[0]
# #!cat {TRAIN_DIR+"train/labels/"+lab_f_name}
with open(TRAIN_DIR+"train/labels/"+lab_f_name, "r") as f:
    print(f.read())

#@title ---- Модель.
#
# -- загрузка пакета TerraYolo
# #!pip install --index-url https://test.pypi.org/simple/ --no-deps TerraYolo --upgrade
# -- TO UNCOMMENT:
# !pip install --upgrade --force-reinstall --index-url https://test.pypi.org/simple/ --no-deps TerraYoloTest
#
# -- Импорт класса TerraYoloV5 из пакета TerraYolo модуля TerraYoloTest;
from TerraYoloTest.TerraYolo import TerraYoloV5
#
# -- создать экземпляр класса my_terra_yolov5, в качестве параметра - передать путь к рабочему каталогу;
my_terra_yolov5 = TerraYoloV5(work_dir=TRAIN_DIR)
#
# При создании класса в папку рабочем каталоге создается директория yolov5,
# в которую загружаются скрипты модели;
print("-- Папка для скриптов модели:")
# #!ls {my_terra_yolov5.work_dir}
print(os.listdir(TRAIN_DIR))
#
# Метод count_labels может посчитать число объектов в файлах меток.
# В качестве параметра ему нужно передать путь к yaml-описанию датасета.
# Проверим число объектов датасета:
print("-- Число объектов датасета:")
my_terra_yolov5.count_labels(data_file_path)
#
# -- по итогам у нас: 532 тренировочные метки ; 3 валидационные метки;
#

# @title ---- Обучение.
#
# YOLOv5 используется в трех основных режимах - тренировка, валидация и детекция.
# Они отличаются способами загрузки данных.
# Загрузчик режима train спроектирован как оптимальный вариант с точки зрения скорости загрузки данных и качества обучения.
# Режим валидации разработан c целью быстро рассчитать метрики обучения,
# а режим детекции - для лучших результатов в реальных задачах.
#
# -- Основной словарь обучения
# Для начала достаточно указать количество эпох и путь к файлу с описанием датасета;
#
train_dict=dict()
# количество эпох;
train_dict["epochs"] = 10
# путь к описанию датасета
train_dict["data"] = data_file_path
#
# ---- control
# файл с описанием путей к данным;
print ("-- Файл с описанием путей к данным:")
# #!cat {data_file_path}
with open(data_file_path, "r") as f:
    print(f.read())
#
# NOTE: указывать относительные пути вместо абсолютных удобно для переносимости;
#
# -- в режиме 'train' запустить обучение с задаными основным словарем параметрами при помощи метода "run";
#
my_terra_yolov5.run(train_dict, exp_type="train")
#

#@title ---- Детекция.
#
# Проверить результаты.
# Снова при помощи словаря, настроить параметры детектора.
# Модель возвращает вероятности обнаружения объекта.
# Указать порог уверенности 'conf'.
# Нужно также указать путь к весам обученной модели.
# По умолчанию результаты всех экспериментов хранятся в подкаталоге runs каталога модели.
#
# ---- control
# каталог с результатами экспериментов;
# для первого эксперимента в папке создается каталог 'exp',
# для следующего 'exp2', потом 'exp3' и т.д.;
print("---- Каталог с результатами экспериментов:")
# #!ls /content/my_yolov5/yolov5/runs
print(os.listdir("/content/my_yolov5/yolov5/runs"))
# каталог с результатами обучения;
print("---- Каталог с результатами обучения:")
# #!ls /content/my_yolov5/yolov5/runs/train/
print(os.listdir ("/content/my_yolov5/yolov5/runs/train/"))
#
# веса модели: best.pt (лучшие результаты на валидации),
# last.pt (рассчитанные по последней эпохе обучения);
print("---- Веса модели:")
# #!ls /content/my_yolov5/yolov5/runs/train/exp/weights
print(os.listdir ("/content/my_yolov5/yolov5/runs/train/exp/weights"))
#

#@title ---- Настройки валидации
# Параметры моделей можно посмотреть в словаре exp_dict.
# Для каждого типа эксперимента в этом словаре хранится отдельный словарь,
# доступный по ключам ["train"], ["val"], ["test"].
#
test_dict = dict()
# так как в нашем датасете мы не подготовили тестовую выборку -  укажем путь
# к изображениям валидационной выборки;
test_dict["source"] = TRAIN_DIR+"/valid/images/"
# порог вероятности обнаружения объекта
test_dict["conf"] = 0.5
# путь к весам модели
test_dict["weights"] = my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/best.pt"
#

#@title ---- Запуск.
print("---- START ----")
my_terra_yolov5.run(test_dict, exp_type="test")
print("\n ---- FINISH ----")
#

#@title ---- Просмотр результатов детекции;
#
# зафиксируем в константе число изображений:
N_SAMPLES=3
# показать n_samples изображений:
print("-- Примеры изображений:")
my_terra_yolov5.show_test_images(n_samples=N_SAMPLES, img_dir=None)
#
# Результаты по умолчанию хранятся в подкаталоге test каталога run.
# Метод show_test_images покажет из каталога img_dir n_samples случайных
# изображений, но если каталог не указывать, то метод попытается найти
# результаты последней детекции.



#@title ---- FIX_#0104: "last_exp_path"
# бывает, что колаб к этому шагу профукивает путь к последнему эксперименту.
# путь хранится в скриптах для train, detect, validate запуска самого фреймворка
# если нет смысла перезапускать предыдущие шаги или копаться во фреймверке, то
# есть смысл явно прописывать пути, заново объявить пути к рабочим
# папкам итренировочному словарю.
# NOTE: обратить внимание на номер папки 'exp' (exp; exp2; exp3 и т.д.).
# ---- uncomment FIX_№0104 (start):
#f TRAIN_DIR = "/content/my_yolov5"
#f data_file_path = TRAIN_DIR+"/dataset.yaml"
#f train_dict=dict()
# - количество эпох;
#f train_dict['epochs'] = 10
# - путь к описанию датасета
#f train_dict['data'] = data_file_path
#f train_dict['weights'] =  "/content/my_yolov5/yolov5/runs/train/exp/weights/last.pt"
#- control
# файл с описанием путей к данным;
#f print ("-- Файл с описанием путей к данным:")
# !cat {data_file_path}
#f with open(data_file_path, "r") as f:
#f    print(f.read())
#
# ---- FIX (end).


#@title ---- Дообучение.
#
# Неплохо, но попробуем улучшить.
# Для того, чтобы не повторять обучение с самого начала - взять веса,
# полученные на последней эпохе обучения.
#
# -- используются веса последней эпохи;
#
train_dict["weights"] = my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/last.pt"
#fix#0104: train_dict['weights'] =  "/content/my_yolov5/yolov5/runs/train/exp/weights/last.pt"
#
# -- NOTE: если collab к этому шагу опять профукал 'last_exp_path', см. FIX_#0104.
# плюс, скорее всего он профукал и словарь обучения и путь к тренировояной папке.
# NOTE: обратить внимание на номер папки 'exp' (exp; exp2; exp3 и т.д.).
#
# -- control
print("-- Веса:")
print(train_dict["weights"])
print("-- -- -- --")
#
# запуск скрипта train  с параметрами train_dict;
my_terra_yolov5.run(train_dict, exp_type="train")
##

#@title ---- Детекция. Следующий цикл.
#
#
# -- NOTE: если collab к этому шагу опять профукал 'last_exp_path',
# то он профукал и словарь обучения и путь к тренировояной папке.
# см. FIX_#0104
# NOTE: обратить внимание на номер папки 'exp' (exp; exp2; exp3 и т.д.).
#
# путь к лучшим весам модели;
test_dict["weights"] = my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/best.pt"
#fix#0104: test_dict['weights'] =  "/content/my_yolov5/yolov5/runs/test/exp/weights/best.pt"
#
# проверим полный путь;
print(test_dict["weights"])
#
#запускаем скрипт test с параметрами test_dict;
my_terra_yolov5.run(test_dict, exp_type="test")
#

#@title ---- Просмотр результатов (пути к файлам изменились).
#
# -- зафиксируем в константе число изображений;
N_SAMPLES=3
# -- показать n_samples изображений;
print("-- Примеры изображений:")
my_terra_yolov5.show_test_images(n_samples=N_SAMPLES, img_dir=None)
#

# @title ---- Понижение порога детекции (если не все машинки определились) -- NOTE: НЕ ПОНАДОБИЛОСЬ;
#
# -- NOTE: НЕ ПОНАДОБИЛОСЬ, после дообучения обнаружились все (точность min 0,71 ; max 0,85);
#
# --установить значение
test_dict["conf"] = 0.4
#
# -- запуск
my_terra_yolov5.run(test_dict, exp_type="test")
#
# -- просмотр
my_terra_yolov5.show_test_images(n_samples=N_SAMPLES)
#

#@title ---- Валидация.
#
# при помощи словаря настроим параметры и посмотрим результаты валидации;
val_dict=dict()
val_dict["data"] = train_dict["data"]
#
val_dict["weights"] = os.path.abspath(
        my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/last.pt")
#
# -- control path;
print("-- Путь к весам:")
print(val_dict["weights"])
#
# -- запуск;
my_terra_yolov5.run(val_dict, exp_type="val")
#
# Результаты валидации сохраняются в соответствующем подкаталоге каталога runs;
os.listdir(my_terra_yolov5.exp_dict["train"]["last_exp_path"])
#

#@title ---- Матрица ошибок.
#
# Анализ матрицы ошибок (матрица путаницы).
# Каждому классу соответствует строка матрицы и можно увидеть,
# за объекты каких классов модель принимает объекты определенного класса.
# Так как у в нашем датасете только один класс, модель может просто не
# распознать объект и в этом случае можно сказать, что она отнесла изображение
# к классу 'фон'. Вывести на экран список файлов из любого каталога
# можно при помощи метода show_val_results.
# Если не указать каталог - по умолчанию модель попытается найти изображения
# в подкаталоге 'val' каталога 'runs'
#
my_terra_yolov5.show_val_results(img_path=None, img_list=["confusion_matrix.png", "PR_curve.png"])
#

#@title ---- Сохранение модели. GoogleDrive.
#
# Чтобы иметь возможность использовать результаты обучения
# достаточно сохранить веса модели в какое-нибудь постоянное хранилище.
# Диск колаба станет недоступен по завершению сессии, поэтому сначала подключим свой google drive.
# Если запрещены всплывающие окна нужно разрешить их для colab.
# И нужно подтвердить разрешение на доступ к диску.
#
# -- подключениe google drive;
# импорт модуля:
from google.colab import drive
# монтировать свой диск в папку drive рабочего каталога;
drive.mount("/content/drive/")
#
# Посмотрим, что у нас получилось. В папке /content/drive должен появиться каталог MyDrive - это и есть наш диск google.
#
# -- control
# В папке /content/drive должен появиться каталог MyDrive.
# Это и есть наш диск google.
# сохранить путь к нашему диску в переменную:
my_drv_path = "/content/drive/MyDrive/"
#
# содержимое каталога диска:
print("-- В папке /content/drive должен появиться каталог MyDrive :")
# #!ls {my_drv_path}
print(os.listdir("/content/drive/"))
# создать каталог для хранения нашей модели:
# #!mkdir {my_drv_path}/yolo_weights/
os.mkdir("/content/drive/MyDrive/yolo_weights")
print("-- Создан кататлог для хранения модели:")
print(os.listdir("/content/drive/MyDrive"))
#
# сохраним путь к лучшим весам в переменную
best_weights_path = my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/best.pt"
# сохраним путь к последним весам в переменную
last_weights_path = my_terra_yolov5.exp_dict["train"]["last_exp_path"]+"/weights/last.pt"
# скопируем лучшие веса
# #!cp {best_weights_path} {my_drv_path+"/yolo_weights/"}
cmd_cp_01 = f"cp {best_weights_path} {my_drv_path+'/yolo_weights/'}"
os.system(cmd_cp_01)
print("-- best_weights: Скопированы.")
# скопируем последние веса
# #!cp {last_weights_path} {my_drv_path+"/yolo_weights/"}
cmd_cp_02 = f"cp {last_weights_path} {my_drv_path+'/yolo_weights/'}"
os.system(cmd_cp_02)
print("-- last_weights: Скопированы.")
#
# control
print("-- Содержание папки с весами на my drive: ")
# #!ls {my_drv_path+"/yolo_weights/"}
# print(os.listdir({my_drv_path+"/yolo_weights/"}))
print(os.listdir("/content/drive/MyDrive/yolo_weights/"))
#

#@title ---- Про варианты YOLO. Аннотация модели.
#
# YOLO v5 позволяет сделать выбор из нескольких архитектур, отличающихся сложностью,
# числом параметров и как следствие скоростью работы.
# Увеличение качества распознавания обычно сопровождается увеличением размера модели,
# и ее соответствующего замедленния. Отличия разных архитектур можно отследить в таблице.
#
# По умолчанию YOLO v5 использует модификацию yolov5s.
# При помощи метода get_annotation можно познакомиться с структурой модели.
#
# -- вывести файл аннотации для модели
print("-- Аннотация для модели:")
my_terra_yolov5.get_annotation("yolov5s")
print("-- NOTE: памятка - закомментироана в коде ячейки.")
print("---- FIN ----")
#
#
# Cообщения модели.
# Перед стартом обучения модель информирует о состоянии сервисов и параметров, в первую очередь
# нас будут интересовать метрики обучения, которые выводятся каждую эпоху в строке:
#
# **'Class     Images  Instances          P          R      mAP50   mAP50-95'**
#
# Колонки  **Class,     Images,  Instances**
# относятся к статистике классов обучающей выборки, можете свериться с нашими данными.
# В колонках  '** P          R      mAP50   mAP50-95**
# выводятся значения метрик на каждой эпохе.
#
# ### P - Precision, R-Recall, Accuracy
#
# Когда модель делает прогноз, может реализоваться одна из четырех комбинаций правильного
# ответа и предсказания модели:
# - TP = True Positive: объект действительно есть и модель его правильно обнаруживает
# - TN = True Negative: объект отсутствует и детектор его правильно не находит
# - FP = False Positive: объект отсутствует, но модель ошибочно его детектирует
# - FN = False Negative: объект имеется, но модель ошибочно его не видит
#
# ${Precision}=\frac{TP}{TP+FP}$ - в числителе у нас число правильных положительных ответов,
# а в знаменателе - число всех **ПОЛОЖИТЕЛЬНЫХ ОТВЕТОВ** модели: и верных и неверных.
# Таким образом **Precision** характеризует точность детекции положительного класса.
# Например, пусть мы диагностируем заболевание и класс Positive
# соответствует факту заболевания (что должно вызвать недумение,  так как в болезни
# обычно трудно найти что-нибудь положительное) - тогда **Presision** равняется
# вероятности обнаружить болезнь, **если пациент действительно болен**.
#
# ${Recall}=\frac{TP}{TP+FN}$ - в числителе у нас снова число правильных положительных
# ответов, а в знаменателе в этот раз - число всех **ПОЛОЖИТЕЛЬНЫХ ИСХОДОВ**.
# В примере с установкой диагноза  **Recall** будет просто равняться вероятности
# обнаружить заболевание.
#
# Таким образом, **Precision** измеряет какой процент предсказаний корректен,
# а **Recall** измеряет насколько хорошо модель находит положительные образцы
# (в нашем примере - ставит диагнозы).
#
# Cледует также упомянуть про **Accuracy** (если бы мы выше перевели Precision
# как Точность, Accuracy следовало бы перевести как Аккуратность):
#
# ${Accuracy}=\frac{TP+TN}{TP+FP+TN+FN} $
#
# Рассмотренные метрики хороши, но в случае дисбаланса в датасете могут потерять
# свою информативность. На примере Accuracy, рассмотрим ситуацию когда у нас
# среди 1000 здоровых пациентов найдется один заболевший. В этом случае, для
# модели очень велик соблазн классифицировать всех здоровыми, при этом получится
# великолепные Precision и Accuracy = 0.999, но задача постановки диагноза
# выполнена не будет, Recall будет равен 0. Если поступить наоборот и максимизировать
# Recall - модель будет стремиться выявить как можно больше больных и уменьшится
# точность прогноза. Поэтому были разработаны и используются другие показатели.
#
# ### IoU - Interception over Union
#
# Чтобы лучше понять следующие метрики нам нужно познакомиться с одним важным
# показателем, заодно подробнее рассмотрим процесс детекции:  YoloV5 использует
# метрику Interception over Union (aka IoU, индекс (расстояние) Жаккара,
# пересечение над объединением).
#
# Предположим верхнюю рамку построил детектор, а нижняя - действительный bbox объекта.
# Отношение площадей пересечения и объединения  прямоугольников всегда меньше
# или равно 1. Далее достаточно назначить порог, при превышении которого
# факт можно подтверждать детекцию.
#
# На самом деле, до расчета IoU, YOLO предсказывает вероятность обнаружить объект
# Conf, поэтому обычно настраиваются два порога -  для IoU  и для Conf.
#
# ### AP - Average Precision
#
# Чтобы посчитать следующую метрику -  Average Precision рассмотрим упрощенный
# пример из отличной статьи господина Kukil, полный текст которой можно найти по ссылке:
# https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
# Предположим наш детектор обнаружил на изображении объекты классов
# 'dog', 'person', 'teddy' и других, но нас пока будут интересовать только класс 'dog'n.
# Можем сразу отметить, что некоторые объекты классифицированы ошибочно.
#
# Отсортируем все найденные объекты класса dog в порядке возрастания вероятности детекции Conf:
#
# Во второй колонке таблицы у нас порог уверенности, в третьей - результат детекции, в четвертой
# и пятой - соответственно накопленные TP и FP для всех экпериментов, в 6 и 7 - соответственно
# Precision и Recall, которые последовательно, строка за строкой сверху вниз,
# рассчитываются после каждой детекции
#
#
# Также отметим факт, что Recall не убывает, а Precision может как расти, так и убывать.
# Пока примем к сведению, что в реальности кривую Precision-Recall дополнительно
# сглаживают, но принципиально от этого ничего не меняется.
#
# ### mAP - mean Average Precision,  mAP50, mAP50-95
#
# Теперь, вычислив значение AP для каждого класса и посчитав среднее арифметическое
# для всех классов получим значение метрики mAP (MAP) - Mean Average Precision - усредненная
# по классам средняя Precision.
# Если для усреднения брать значение Precision для порога IoU = .5 мы получим
# метрику mAP50. Чем выше значение этой метрики - тем лучше работает модель с порогом IoU=.5
# Также для оценки моделей используется метрика mAP50-95. Для ее расчета применяется
# дополнительное усреднение по сетке порогов от 0.5 до 0.95 с шагом 0.05
# (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).
# Эта метрика будет тем выше , чем увереннее (с большей вероятностью) модель детектирует объекты.
#
# Теперь можно снова посмотреть историю обучения и проанализировать поведение динамику метрик
#
