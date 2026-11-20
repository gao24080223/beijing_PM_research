import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# 解决图表中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 自动读取并合并所有站点的数据
# ==========================================
print("正在读取并合并 12 个站点的数据...")
# 查找当前目录下所有以 PRSA_Data 开头的 csv 文件
file_list = glob.glob('PRSA_Data_*.csv')

if not file_list:
    print("❌ 找不到数据文件！请确保 12 个 CSV 文件都在代码同级目录下。")
    exit()

all_data = []
for file in file_list:
    # 读取每个站点的数据
    temp_df = pd.read_csv(file)
    # 从文件名中提取站点名字 (比如从 'PRSA_Data_Aotizhongxin_20130301-20170228.csv' 提取 'Aotizhongxin')
    station_name = os.path.basename(file).split('_')[2]
    temp_df['station'] = station_name
    all_data.append(temp_df)

# 合并成一个大的 DataFrame
df = pd.concat(all_data, ignore_index=True)

# ==========================================
# 2. 数据清洗与时间字段处理
# ==========================================
print("正在处理时间字段和缺失值...")
# UCI 数据集的时间是分开的列：year, month, day, hour
# 将它们合并为一个 datetime 列
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df = df.sort_values(['station', 'datetime']).reset_index(drop=True)

# 处理缺失值：针对 PM2.5 列，使用线性插值（limmit=3表示最多连续填补3小时）
# 这种方式比直接删除更平滑，适合时间序列
df['PM2.5'] = df.groupby('station')['PM2.5'].transform(lambda x: x.interpolate(method='linear', limit=3))
# 删除仍然存在缺失值的行（可能缺失太久无法插值）
df = df.dropna(subset=['PM2.5'])

print(f"数据合并完成！总共有 {len(df)} 条记录，时间跨度从 {df['datetime'].min()} 到 {df['datetime'].max()}")


# ==========================================
# 3. 识别高浓度时段 (核心逻辑)
# ==========================================
def identify_high_episodes(data, st_col, t_col, p_col, threshold=75):
    episodes = []
    # 按站点分组处理
    for station, group in data.groupby(st_col):
        group = group.sort_values(t_col).reset_index(drop=True)
        mask = group[p_col] >= threshold

        # 找连续 True 的区段
        episode_id = (mask != mask.shift()).cumsum()

        for eid, seg in group[mask].groupby(episode_id[mask]):
            if len(seg) == 0: continue
            episodes.append({
                'station': station,
                'start': seg[t_col].iloc[0],
                'end': seg[t_col].iloc[-1],
                'duration_hours': len(seg),  # UCI 是逐小时数据，所以直接是 len(seg)
                'mean_pm25': seg[p_col].mean(),
                'max_pm25': seg[p_col].max()
            })
    return pd.DataFrame(episodes)


print("正在识别高浓度时段 (PM2.5 >= 75 μg/m³)...")
episodes = identify_high_episodes(df, 'station', 'datetime', 'PM2.5', threshold=75)
print(f"✅ 成功识别到 {len(episodes)} 个高浓度时段！")

# ==========================================
# 4. 多维度规律分析可视化
# ==========================================
episodes['month'] = episodes['start'].dt.month
month_counts = episodes.groupby('month').size()

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('UCI 北京多站点 PM2.5 高浓度时段规律分析 (2013-2017)', fontsize=18)

# 图1：月份分布
axes[0, 0].bar(month_counts.index, month_counts.values, color='steelblue')
axes[0, 0].set_title('高浓度时段数量的月份分布')
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('时段数量')
axes[0, 0].set_xticks(range(1, 13))

# 图2：持续时长分布 (只画100小时以内的，防止极端值拉长X轴)
axes[0, 1].hist(episodes[episodes['duration_hours'] <= 100]['duration_hours'], bins=30, color='coral',
                edgecolor='white')
axes[0, 1].set_title('高浓度时段持续时长分布 (<=100小时)')
axes[0, 1].set_xlabel('持续时长 (小时)')
axes[0, 1].set_ylabel('频次')

# 图3：各站点高浓度时段数量
station_counts = episodes.groupby('station').size().sort_values(ascending=True)
axes[1, 0].barh(station_counts.index, station_counts.values, color='seagreen')
axes[1, 0].set_title('各站点高浓度时段总数量对比')
axes[1, 0].set_xlabel('时段数量')

# 图4：各年份高浓度时段数量 (体现长期趋势)
episodes['year'] = episodes['start'].dt.year
year_counts = episodes.groupby('year').size()
axes[1, 1].bar(year_counts.index, year_counts.values, color='goldenrod')
axes[1, 1].set_title('各年份高浓度时段数量趋势')
axes[1, 1].set_xlabel('年份')
axes[1, 1].set_ylabel('时段数量')
# 强制X轴显示整数年份
axes[1, 1].set_xticks(year_counts.index)

plt.tight_layout()
plt.savefig('1_high_episode_overview_UCI.png', dpi=150)
plt.show()

# ==========================================
# 5. 跨站点同步性分析 (挑选污染最重的年份和月份)
# ==========================================
# 找出污染时段最多的年份和月份
top_year = year_counts.idxmax()
top_month = episodes[episodes['year'] == top_year]['month'].value_counts().idxmax()
print(f"污染最集中的时间段：{top_year}年 {top_month}月")

# 筛选该时间段的数据
plot_data = df[(df['datetime'].dt.year == top_year) & (df['datetime'].dt.month == top_month)]
# 随机挑6个站点画图
stations_to_plot = plot_data['station'].unique()[:6]

plt.figure(figsize=(16, 6))
for st in stations_to_plot:
    sub = plot_data[plot_data['station'] == st]
    plt.plot(sub['datetime'], sub['PM2.5'], label=st, alpha=0.8, linewidth=1)

plt.axhline(y=75, color='red', linestyle='--', linewidth=1.5, label='阈值 75 μg/m³')
plt.title(f'{top_year}年{top_month}月 多站点 PM2.5 时间序列对比 (查看同步性)')
plt.ylabel('PM2.5 (μg/m³)')
plt.legend(fontsize=10, ncol=3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('2_multi_station_timeseries_UCI.png', dpi=150)
plt.show()

print("🎉 分析完成！两张图表已保存到当前文件夹。")