# Практическая работа: спутниковая океанология — мультисенсорный подход

**Цель:** исследовать временные изменения метеорологических параметров и пространственное распределение температуры поверхности моря (ТПМ) по предоставленным данным.

## Оглавление

1. [Задание 1. Временной ход температуры воздуха и давления](#1-задание-1-временной-ход-температуры-воздуха-и-давления)
2. [Задание 2. Горизонтальное распределение ТПМ](#2-задание-2-горизонтальное-распределение-тпм)
   - [2.1. Открытие NetCDF](#21-открытие-netcdf-и-описание-переменных)
   - [2.2. Чтение SST и проверка единиц](#22-чтение-температуры-масштабирование-и-проверка-единиц)
   - [2.3. Карта выбранного района](#23-выбор-участка-океана-и-построение-карты)
3. [Выводы](#3-выводы)

---


## 1. Задание 1. Временной ход температуры воздуха и давления


```python
df = pd.read_excel(XLSX_PATH)
print('Размер таблицы:', df.shape)
print('Столбцы:', df.columns.tolist())
display(df.head())                          # в чистом Python заменить на print(df.head())
print('Типы данных:\n', df.dtypes)
print('Количество пропусков:\n', df.isna().sum())

# Приведение времени к datetime и сортировка
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

# В таблице температура указана в °C, давление — в мм рт. ст.
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(df['Time'], df['Temperature'], color='tomato', linewidth=1.8)
axes[0].set_title('Временной ход температуры воздуха')
axes[0].set_ylabel('Температура, °C')
axes[0].grid(True, alpha=0.35)

axes[1].plot(df['Time'], df['Pressure'], color='royalblue', linewidth=1.8)
axes[1].set_title('Временной ход атмосферного давления')
axes[1].set_ylabel('Давление, мм рт. ст.')
axes[1].set_xlabel('Дата и время')
axes[1].grid(True, alpha=0.35)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

print(f"Период наблюдений: {df['Time'].min()} — {df['Time'].max()}")
print(f"Температура: от {df['Temperature'].min():.1f} до {df['Temperature'].max():.1f} °C")
print(f"Давление:    от {df['Pressure'].min():.1f} до {df['Pressure'].max():.1f} мм рт. ст.")
```

### Ожидаемый результат

| Параметр | Значение |
|---|---|
| Размер таблицы | `(120, 3)` |
| Период наблюдений | 2021-09-01 00:00 — 2021-09-15 21:00 |
| Диапазон температуры | 4.9 – 19.7 °C |
| Диапазон давления | 746.9 – 765.8 мм рт. ст. |
<img width="1298" height="790" alt="image" src="https://github.com/user-attachments/assets/0a9fd462-aea6-4c77-8e09-b849c227806b" />


### Краткий вывод

В наборе 120 наблюдений за 1–15 сентября 2021 года с трёхчасовым шагом. По графикам видно, что изменения температуры воздуха и атмосферного давления частично синхронны (рост давления часто сопровождается похолоданием), однако встречаются и разновременные эпизоды. Числовые диапазоны получены кодом; интерпретировать их следует с учётом малой длительности ряда.

---

## 3. Задание 2. Горизонтальное распределение ТПМ

### 3.1. Открытие NetCDF и описание переменных

```python
with h5py.File(NC_PATH, 'r') as nc:
    print('Переменные и размерности:')
    for name, obj in nc.items():
        if isinstance(obj, h5py.Dataset):
            units = obj.attrs.get('units', 'не указаны')
            if isinstance(units, bytes):
                units = units.decode('utf-8')
            print(f'{name:8s} shape={obj.shape}, dtype={obj.dtype}, units={units}')

    print('\nГлобальные атрибуты:')
    for key in ['title', 'id', 'time_coverage_start', 'time_coverage_end']:
        value = nc.attrs.get(key, '')
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        print(f'{key}: {value}')
```

**Структура файла:**

| Переменная | Shape | Dtype | Units |
|---|---|---|---|
| `anom` | (1, 1, 720, 1440) | int16 | Celsius |
| `err` | (1, 1, 720, 1440) | int16 | Celsius |
| `ice` | (1, 1, 720, 1440) | int16 | % |
| `lat` | (720,) | float32 | degrees_north |
| `lon` | (1440,) | float32 | degrees_east |
| `sst` | (1, 1, 720, 1440) | int16 | Celsius |
| `time` | (1,) | float32 | days since 1978-01-01 12:00:00 |
| `zlev` | (1,) | float32 | meters |

Сетка: 720 × 1440 узлов, шаг 0.25° (глобальный анализ OISST).

---

### 3.2. Чтение температуры, масштабирование и проверка единиц

**Важно:** в метаданных именно этого файла переменная `sst` уже имеет единицы **Celsius**. Значения хранятся как `int16` с `scale_factor = 0.01`, поэтому:

1. Сначала маскируются пропуски (`_FillValue`).
2. Применяется `scale_factor` и `add_offset` (распаковка packed data).
3. Перевод из Кельвинов в °C выполняется **только** если `units` действительно `Kelvin`. Для данного файла вычитать 273.15 нельзя.

```python
with h5py.File(NC_PATH, 'r') as nc:
    raw_sst = nc['sst'][0, 0, :, :].astype('float32')
    lat = nc['lat'][:]
    lon = nc['lon'][:]
    sst_attrs = dict(nc['sst'].attrs)
    time_value = float(nc['time'][0])
    time_units = nc['time'].attrs['units']
    fill_value = float(np.asarray(sst_attrs['_FillValue']).squeeze())
    scale_factor = float(np.asarray(sst_attrs.get('scale_factor', 1.0)).squeeze())
    add_offset = float(np.asarray(sst_attrs.get('add_offset', 0.0)).squeeze())
    units = sst_attrs.get('units', b'')
    if isinstance(units, bytes):
        units = units.decode('utf-8')

# Маскирование пропусков и применение packed-data scale/offset
sst = np.where(raw_sst == fill_value, np.nan, raw_sst * scale_factor + add_offset)

# Приведение к °C в зависимости от единиц в метаданных
if units.strip().lower() in {'kelvin', 'kelvins', 'k'}:
    sst_c = sst - 273.15
elif units.strip().lower() in {'celsius', 'degree_celsius', 'degrees_celsius', '°c'}:
    sst_c = sst.copy()
else:
    raise ValueError(f'Неизвестные единицы температуры: {units!r}')

# Восстановление даты по времени (дни от эпохи)
origin = pd.Timestamp('1978-01-01 12:00:00')
map_date = origin + pd.to_timedelta(time_value, unit='D')

print('Единицы в файле:', units)
print('Scale factor:', scale_factor, '| Add offset:', add_offset)
print('Дата наблюдения:', map_date)
print('Размер массива SST:', sst_c.shape)
print(f'Диапазон SST: {np.nanmin(sst_c):.2f} — {np.nanmax(sst_c):.2f} °C')
```

### Ожидаемый результат

```
Единицы в файле: Celsius
Scale factor: 0.009999999776482582 | Add offset: 0.0
Дата наблюдения: 1981-09-01 12:00:00
Размер массива SST: (720, 1440)
Диапазон SST: -1.80 — 33.74 °C
```

---

### 3.3. Выбор участка океана и построение карты

В качестве примера выбран прямоугольник в западной части Атлантики и Мексиканском заливе: **15–32° с.ш., 260–280° в.д.** Долготы в файле заданы в диапазоне 0–360°.

```python
# Границы исследуемого района
lat_min, lat_max = 15, 32
lon_min, lon_max = 260, 280

lat_mask = (lat >= lat_min) & (lat <= lat_max)
lon_mask = (lon >= lon_min) & (lon <= lon_max)

region     = sst_c[np.ix_(lat_mask, lon_mask)]
region_lat = lat[lat_mask]
region_lon = lon[lon_mask]

print('Размер выбранного фрагмента:', region.shape)
print(f'Диапазон координат: {lat_min}–{lat_max}° с.ш., {lon_min}–{lon_max}° в.д.')

fig, ax = plt.subplots(figsize=(11, 6))
mesh = ax.pcolormesh(region_lon, region_lat, region,
                     shading='auto', cmap='turbo', vmin=15, vmax=32)
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label('Температура поверхности моря, °C')

ax.set_xlabel('Долгота, ° в.д.')
ax.set_ylabel('Широта, ° с.ш.')
ax.set_title(
    f'Температура поверхности моря — {map_date:%d.%m.%Y}\n'
    'Западная Атлантика и Мексиканский залив'
)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
```

### Ожидаемый результат

```
Размер выбранного фрагмента: (68, 80)
Диапазон координат: 15–32° с.ш., 260–280° в.д.
```

На карте видно тёплое Карибское море и Мексиканский залив (≈28–30 °C) и более холодные воды северной части выбранного прямоугольника (≈18–22 °C).

---

## 4. Выводы

### По заданию 1
- Загружено и отсортировано 120 наблюдений (1–15 сентября 2021).
- Пропусков нет, типы данных корректны.
- Построены два временных ряда: температура воздуха (4.9–19.7 °C) и атмосферное давление (746.9–765.8 мм рт. ст.).
- Прослеживается тенденция к падению температуры при понижении давления.

### По заданию 2
- Файл `sst.nc` — глобальный анализ NOAA/NCEI OISST v2.1, разрешение 0.25°.
- Поле `sst` имеет размерность `(1, 1, 720, 1440)`, единицы — **Celsius**, `scale_factor = 0.01`.
- После маскирования `_FillValue` и применения `scale_factor` температуры уже выражены в °C — конвертация из Кельвинов не требуется.
- Дата наблюдения, закодированная в файле: **1 сентября 1981 года**.
- Построена карта ТПМ для западной Атлантики (15–32° с.ш., 260–280° в.д.).

### Общее замечание
Два исходных набора относятся к разным годам (2021 и 1981) и независимы друг от друга. Совместный анализ возможен только как демонстрация методики, а не как сравнение одного и того же периода.

---

## Приложение. Запуск как `.py`-скрипт

Если файл сохраняется как `.py` (без Jupyter):

1. Заменить `display(df.head())` на `print(df.head())`.
2. Добавить `matplotlib.use('Agg')` перед `import matplotlib.pyplot as plt` — если запуск происходит в headless-окружении.
3. Заменить `plt.show()` на `fig.savefig('output.png', dpi=120)` для сохранения графиков.
