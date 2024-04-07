# НС_ЛР Хохлова Анастасия Алексеевна

1) Перед запуском программы распаковать в папку scripts\datasets набор данных cifar-10: 
CIFAR-10 по ссылке https://www.cs.toronto.edu/~kriz/cifar.html или используйте команду !bash get_datasets.sh (google colab, local ubuntu)

2) В разделе "Сверточные нейронные сети (CNN)" есть пункт "Для компиляции выполните следующую команду в директории scripts: python setup.py build_ext --inplace
Для этого:

+ удалить в папке `scripts` файлы `im2col_cython.c` и `im2col_cython.pyd`, если они существуют
+ запустить `setup.py` (лучше через консоль)
+ переименовать созданный файл с расширением `.pyd` в`im2col_cython.pyd`
+ запустить необходимые ячейки
