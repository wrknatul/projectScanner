# projectScanner

## Описание
Программа для восстановления изображений, используя МРТ.

## Установка
Убедитесь, что у вас установлены библиотеки из requirements.txt:
```bash
pip install -r requirements.txt
```

## Генерация входных данных

Для генерации фантома Шепа Логана воспользуйтесь скриптом shep_logan.py:
```bash
python3 shep_logan.py /path/to/image.jpg
```

Для генерации круга 100 на 100 воспользуйтесь скриптом photo_generator.py:

```bash
python3 photo_generator.py /path/to/image.jpg
```

## Запуск скрипта восстановления

Для генерации выходного изображения запустите main.py:

 ```bash
python3 main.py --input_image /path/to/input/image.jpg --output_image /path/to/output/image.jpg
```


### Параметры алгоритмов
| Флаг | Тип | По умолчанию | Допустимые значения | Описание |
|------|-----|--------------|----------------------|----------|
| `--bresenham` | int | `1` | `0` или `1` | Флаг использования алгоритма Брезенхема (1 - использовать, 0 - не использовать) |
| `--kaczmarz` | int | `0` | `0` или `1` | Флаг использования ART алгоритма (1 - использовать, 0 - не использовать) |

### Параметры ART алгоритма
| Флаг | Тип | По умолчанию | Допустимые значения | Описание |
|------|-----|--------------|----------------------|----------|
| `--function_type` | string | `"cicle"` | `"cicle"`, `"symART"`, `"evenART"`, `"probabilities_algorithm"` | Тип функции для алгоритма Качмажа, есть выбор из цикличного выбора строк, перебор индекса от первого до конечного и обратно, четный ART метод, вероятностный ART метод|

### Параметры сканеров
| Флаг | Тип | По умолчанию | Допустимые значения | Описание |
|------|-----|--------------|----------------------|----------|
| `--num_scanners` | int | `41` | Натуральные числа | Количество используемых сканеров при восстановлении|
| `--distance_between_scanners` | int | `6` | Натуральные числа | Расстояние между сканерами в пикселях |