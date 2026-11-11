# =============================================================================
# Практическая работа: спутниковая океанология — мультисенсорный подход
# =============================================================================
# Цель: исследовать временные изменения метеорологических параметров и
# пространственное распределение температуры поверхности моря (ТПМ).
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# 0. Импорт библиотек и настройка
# =============================================================================
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.grid'] = True
DATA_DIR = Path('данные')
XLSX_PATH = DATA_DIR / 'September.xlsx'
NC_PATH = DATA_DIR / 'sst.nc'

for path in (XLSX_PATH, NC_PATH):
    if not path.exists():
        raise FileNotFoundError(f'Не найден файл: {path.resolve()}')
print('Файлы данных найдены.')


# =============================================================================
# Задание 1. Временной ход температуры воздуха и атмосферного давления
# =============================================================================
df = pd.read_excel(XLSX_PATH)
print('Размер таблицы:', df.shape)
print('Столбцы:', df.columns.tolist())
display(df.head())
print('Типы данных:\n', df.dtypes)
print('Количество пропусков:\n', df.isna().sum())

# Приведение времени к datetime и сортировка
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

# В таблице температура указана в °C, давление — в мм рт. ст. (по диапазону исходных значений).
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
print(f"Давление: от {df['Pressure'].min():.1f} до {df['Pressure'].max():.1f} гПа")


# =============================================================================
# Задание 2. Горизонтальное распределение температуры поверхности моря
# =============================================================================
# 2.1. Открытие NetCDF и описание переменных
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
        if isinstance(value, bytes): value = value.decode('utf-8')
        print(f'{key}: {value}')


# -----------------------------------------------------------------------------
# 2.2. Чтение температуры, масштабирование и проверка единиц
# -----------------------------------------------------------------------------
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
    if isinstance(units, bytes): units = units.decode('utf-8')

# Маскирование пропусков и применение packed-data scale/offset
sst = np.where(raw_sst == fill_value, np.nan, raw_sst * scale_factor + add_offset)
if units.strip().lower() in {'kelvin', 'kelvins', 'k'}:
    sst_c = sst - 273.15
    units_label = '°C'
elif units.strip().lower() in {'celsius', 'degree_celsius', 'degrees_celsius', '°c'}:
    sst_c = sst.copy()
    units_label = '°C'
else:
    raise ValueError(f'Неизвестные единицы температуры: {units!r}')

# Дата задаётся в атрибуте времени: дни от указанной эпохи
# Для файла: days since 1978-01-01 12:00:00
origin = pd.Timestamp('1978-01-01 12:00:00')
map_date = origin + pd.to_timedelta(time_value, unit='D')
print('Единицы в файле:', units)
print('Scale factor:', scale_factor, '| Add offset:', add_offset)
print('Дата наблюдения:', map_date)
print('Размер массива SST:', sst_c.shape)
print(f'Диапазон SST: {np.nanmin(sst_c):.2f} — {np.nanmax(sst_c):.2f} °C')


# -----------------------------------------------------------------------------
# 2.3. Выбор участка океана и построение карты
# -----------------------------------------------------------------------------
# Границы исследуемого района
lat_min, lat_max = 15, 32
lon_min, lon_max = 260, 280

lat_mask = (lat >= lat_min) & (lat <= lat_max)
lon_mask = (lon >= lon_min) & (lon <= lon_max)
region = sst_c[np.ix_(lat_mask, lon_mask)]
region_lat = lat[lat_mask]
region_lon = lon[lon_mask]

print('Размер выбранного фрагмента:', region.shape)
print(f'Диапазон координат: {lat_min}–{lat_max}° с.ш., {lon_min}–{lon_max}° в.д.')

fig, ax = plt.subplots(figsize=(11, 6))
mesh = ax.pcolormesh(region_lon, region_lat, region, shading='auto', cmap='turbo', vmin=15, vmax=32)
cb = fig.colorbar(mesh, ax=ax, pad=0.02)
cb.set_label('Температура поверхности моря, °C')
ax.set_xlabel('Долгота, ° в.д.')
ax.set_ylabel('Широта, ° с.ш.')
ax.set_title(f'Температура поверхности моря — {map_date:%d.%m.%Y}\nЗападная Атлантика и Мексиканский залив')
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
