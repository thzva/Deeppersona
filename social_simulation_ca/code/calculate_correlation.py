#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算数据集之间的相关性

此脚本读取survey_country_names.csv中的国家名称，然后在三个数据集中找到对应的数据，
计算human score、LLM score和我们自己数据之间的相关性。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ks_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE

def load_data(file_path):
    """
    加载CSV文件并返回DataFrame
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        加载的DataFrame
    """
    df = pd.read_csv(file_path)
    print(f"从 {file_path} 加载了 {len(df)} 条记录")
    return df

def filter_by_countries(df, country_names):
    """
    根据国家名称过滤DataFrame
    
    Args:
        df: 输入DataFrame
        country_names: 国家名称列表
        
    Returns:
        过滤后的DataFrame
    """
    return df[df['country_name'].isin(country_names)]

def calculate_metrics(df1, df2):
    """
    计算两个DataFrame之间的统计分布差异指标
    
    Args:
        df1: 第一个DataFrame
        df2: 第二个DataFrame
    
    Returns:
        字典，包含每列的各种统计分布差异指标
    """
    results = {}
    
    # 获取两个DataFrame中共同的列
    common_columns = [col for col in df1.columns if col in df2.columns and col.startswith('Q')]
    
    for col in common_columns:
        # 确保两个数据集都有这个列且不全是NaN
        if col in df1.columns and col in df2.columns:
            values1 = df1[col].dropna()
            values2 = df2.loc[values1.index, col].dropna()
            
            # 确保有共同的索引
            common_idx = values1.index.intersection(values2.index)
            if len(common_idx) > 0:
                values1 = values1.loc[common_idx].astype(float)
                values2 = values2.loc[common_idx].astype(float)
                
                # 1. KS统计量 (Kolmogorov-Smirnov Statistic)
                # 评估两个分布之间的最大差异
                ks_stat, ks_pvalue = ks_2samp(values1, values2)
                
                # 2. Wasserstein距离 (Earth Mover's Distance)
                # 衡量将一个分布转换为另一个分布所需的最小“工作量”
                wasserstein_dist = stats.wasserstein_distance(values1, values2)
                
                # 3. JS散度 (Jensen-Shannon Divergence)
                # 衡量两个概率分布之间的相似性
                # 需要将数据转换为概率分布
                # 首先将数据分箱并计算频率
                min_val = min(values1.min(), values2.min())
                max_val = max(values1.max(), values2.max())
                bins = np.linspace(min_val, max_val, 20)  # 使用20个区间
                
                hist1, _ = np.histogram(values1, bins=bins, density=True)
                hist2, _ = np.histogram(values2, bins=bins, density=True)
                
                # 处理零频率，避免计算JS散度时出现问题
                hist1 = np.clip(hist1, 1e-10, None)  # 将零值替换为小正数
                hist2 = np.clip(hist2, 1e-10, None)
                
                # 归一化使其成为概率分布
                hist1 = hist1 / hist1.sum()
                hist2 = hist2 / hist2.sum()
                
                # 计算JS散度
                js_divergence = jensenshannon(hist1, hist2)
                
                # 4. 相关系数 (Pearson Correlation)
                # 衡量两个变量之间的线性相关性
                try:
                    pearson_corr, pearson_pvalue = pearsonr(values1, values2)
                except:
                    pearson_corr = pearson_pvalue = np.nan
                
                # 计算平均值和平均差异
                mean_1 = values1.mean()
                mean_2 = values2.mean()
                mean_diff = abs(mean_1 - mean_2)
                
                results[col] = {
                    'KS_Statistic': ks_stat,
                    'KS_pvalue': ks_pvalue,
                    'Wasserstein_Distance': wasserstein_dist,
                    'JS_Divergence': js_divergence,
                    'Pearson_Correlation': pearson_corr,
                    'Pearson_pvalue': pearson_pvalue,
                    'Mean_1': mean_1,
                    'Mean_2': mean_2,
                    'Mean_Diff': mean_diff
                }
    
    return results

def create_bar_charts(metrics_df, country_diffs, question_diffs, output_dir):
    """
    Create bar charts to visualize evaluation metrics
    
    Args:
        metrics_df: Evaluation metrics DataFrame
        country_diffs: Country differences DataFrame
        question_diffs: Question differences DataFrame
        output_dir: Output directory path
    """
    # Set chart style
    sns.set(style="whitegrid")
    
    # 设置支持中文的字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        print("警告：无法设置中文字体，图表中的中文可能无法正确显示")
    
    # 1. Create country differences bar chart (using the existing data for now)
    plt.figure(figsize=(14, 8))
    
    # Load country differences data
    country_diff_path = output_dir / "country_average_differences.csv"
    if country_diff_path.exists():
        country_diffs = pd.read_csv(country_diff_path)
        
        # Prepare data
        countries = country_diffs['Country'].tolist()
        our_wvs_diffs = country_diffs['Abs Our-Survey Diff'].tolist()
        llm_wvs_diffs = country_diffs['Abs LLM-Survey Diff'].tolist()
        llm_our_diffs = country_diffs['Abs LLM-Our Diff'].tolist()
        
        # Set bar positions
        x = np.arange(len(countries))
        width = 0.25
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width, our_wvs_diffs, width, label='Abs Our-Survey Diff')
        ax.bar(x, llm_wvs_diffs, width, label='Abs LLM-Survey Diff')
        ax.bar(x + width, llm_our_diffs, width, label='Abs LLM-Our Diff')
        
        # Add text labels and axis labels
        ax.set_title('Average Absolute Differences by Country', fontsize=15)
        ax.set_ylabel('Average Absolute Difference', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(countries, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        country_chart_path = output_dir / "country_differences.png"
        plt.savefig(country_chart_path)
        plt.close()
        print(f"\nCountry differences chart saved to: {country_chart_path}")
    
    # 2. Create statistical measures bar charts from metrics_df
    if not metrics_df.empty:
        # Extract questions
        questions = metrics_df['Question'].tolist()
        
        # Create KS Statistic bar chart
        plt.figure(figsize=(14, 8))
        x = np.arange(len(questions))
        width = 0.25
        # 使用所有问题，包括Q57
        filtered_metrics_df = metrics_df
        filtered_questions = metrics_df['Question'].tolist()
        filtered_x = np.arange(len(filtered_questions))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ours_wvs_ks = filtered_metrics_df['Ours-WVS KS'].tolist()
        llm_wvs_ks = filtered_metrics_df['LLM-WVS KS'].tolist()
        
        # 使用三个条形图
        oc_wvs_ks = filtered_metrics_df['OC-WVS KS'].tolist() if 'OC-WVS KS' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        
        ax.bar(filtered_x - width, ours_wvs_ks, width, label='Ours-WVS KS')
        ax.bar(filtered_x, llm_wvs_ks, width, label='LLM-WVS KS')
        ax.bar(filtered_x + width, oc_wvs_ks, width, label='OC-WVS KS')
        
        ax.set_title('Kolmogorov-Smirnov Statistics by Question', fontsize=15)
        ax.set_ylabel('KS Statistic', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        ks_chart_path = output_dir / "ks_statistics.png"
        plt.savefig(ks_chart_path)
        plt.close()
        print(f"KS Statistics chart saved to: {ks_chart_path}")
        
        # Create Wasserstein Distance bar chart
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 使用过滤后的数据
        ours_wvs_wasserstein = filtered_metrics_df['Ours-WVS Wasserstein'].tolist()
        llm_wvs_wasserstein = filtered_metrics_df['LLM-WVS Wasserstein'].tolist()
        
        # 使用三个条形图
        oc_wvs_wasserstein = filtered_metrics_df['OC-WVS Wasserstein'].tolist() if 'OC-WVS Wasserstein' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        
        ax.bar(filtered_x - width, ours_wvs_wasserstein, width, label='Ours-WVS Wasserstein')
        ax.bar(filtered_x, llm_wvs_wasserstein, width, label='LLM-WVS Wasserstein')
        ax.bar(filtered_x + width, oc_wvs_wasserstein, width, label='OC-WVS Wasserstein')
        
        ax.set_title('Wasserstein Distances by Question', fontsize=15)
        ax.set_ylabel('Wasserstein Distance', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        wasserstein_chart_path = output_dir / "wasserstein_distances.png"
        plt.savefig(wasserstein_chart_path)
        plt.close()
        print(f"Wasserstein Distances chart saved to: {wasserstein_chart_path}")
        
        # Create JS Divergence bar chart
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 使用过滤后的数据
        ours_wvs_js = filtered_metrics_df['Ours-WVS JS'].tolist()
        llm_wvs_js = filtered_metrics_df['LLM-WVS JS'].tolist()
        
        # 使用三个条形图
        oc_wvs_js = filtered_metrics_df['OC-WVS JS'].tolist() if 'OC-WVS JS' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        
        ax.bar(filtered_x - width, ours_wvs_js, width, label='Ours-WVS JS')
        ax.bar(filtered_x, llm_wvs_js, width, label='LLM-WVS JS')
        ax.bar(filtered_x + width, oc_wvs_js, width, label='OC-WVS JS')
        
        ax.set_title('Jensen-Shannon Divergences by Question', fontsize=15)
        ax.set_ylabel('JS Divergence', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        js_chart_path = output_dir / "js_divergences.png"
        plt.savefig(js_chart_path)
        plt.close()
        print(f"JS Divergences chart saved to: {js_chart_path}")
        
        # 创建KS p值条形图
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 过滤掉Q57
        filtered_questions = [q for q in questions if q != 'Q57']
        filtered_indices = [i for i, q in enumerate(questions) if q != 'Q57']
        filtered_x = np.array([x[i] for i in filtered_indices])
        
        # 获取过滤后的数据
        filtered_metrics_df = metrics_df[metrics_df['Question'] != 'Q57']
        
        ours_wvs_ks_pvalue = filtered_metrics_df['Ours-WVS KS_pvalue'].tolist()
        llm_wvs_ks_pvalue = filtered_metrics_df['LLM-WVS KS_pvalue'].tolist()
        
        # 使用三个条形图
        oc_wvs_ks_pvalue = filtered_metrics_df['OC-WVS KS_pvalue'].tolist() if 'OC-WVS KS_pvalue' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        
        # 设置更合适的y轴范围
        max_pvalue = max(max(ours_wvs_ks_pvalue), max(llm_wvs_ks_pvalue), max(oc_wvs_ks_pvalue)) * 1.1
        ax.set_ylim(0, max_pvalue)
        
        ax.bar(filtered_x - width, ours_wvs_ks_pvalue, width, label='Ours-WVS KS p-value')
        ax.bar(filtered_x, llm_wvs_ks_pvalue, width, label='LLM-WVS KS p-value')
        ax.bar(filtered_x + width, oc_wvs_ks_pvalue, width, label='OC-WVS KS p-value')
        
        ax.set_title('Kolmogorov-Smirnov p-values by Question', fontsize=15)
        ax.set_ylabel('KS p-value', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        ks_pvalue_chart_path = output_dir / "ks_pvalues.png"
        plt.savefig(ks_pvalue_chart_path)
        plt.close()
        print(f"KS p-values chart saved to: {ks_pvalue_chart_path}")
        
        # 创建平均差异条形图
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 使用过滤后的数据
        ours_wvs_mean_diff = filtered_metrics_df['Ours-WVS Mean Diff'].tolist() if 'Ours-WVS Mean Diff' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        llm_wvs_mean_diff = filtered_metrics_df['LLM-WVS Mean Diff'].tolist() if 'LLM-WVS Mean Diff' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        oc_wvs_mean_diff = filtered_metrics_df['OC-WVS Mean Diff'].tolist() if 'OC-WVS Mean Diff' in filtered_metrics_df.columns else [0] * len(filtered_questions)
        
        ax.bar(filtered_x - width, ours_wvs_mean_diff, width, label='Ours-WVS Mean Diff')
        ax.bar(filtered_x, llm_wvs_mean_diff, width, label='LLM-WVS Mean Diff')
        ax.bar(filtered_x + width, oc_wvs_mean_diff, width, label='OC-WVS Mean Diff')
        
        ax.set_title('Mean Differences by Question', fontsize=15)
        ax.set_ylabel('Mean Difference', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        mean_diff_chart_path = output_dir / "mean_differences.png"
        plt.savefig(mean_diff_chart_path)
        plt.close()
        print(f"Mean Differences chart saved to: {mean_diff_chart_path}")
        
        # 创建问题差异图表
        plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 每个问题的所有指标平均值
        metrics_by_question = []
        for q in filtered_questions:
            # 获取当前问题的行
            q_row = filtered_metrics_df[filtered_metrics_df['Question']==q]
            
            # 如果找不到对应的行，跳过这个问题
            if len(q_row) == 0:
                continue
                
            # 安全获取指标值，如果不存在则使用0
            ours_ks = q_row['Ours-WVS KS'].values[0] if 'Ours-WVS KS' in q_row.columns and not pd.isna(q_row['Ours-WVS KS'].values[0]) else 0
            ours_wasserstein = q_row['Ours-WVS Wasserstein'].values[0] if 'Ours-WVS Wasserstein' in q_row.columns and not pd.isna(q_row['Ours-WVS Wasserstein'].values[0]) else 0
            ours_js = q_row['Ours-WVS JS'].values[0] if 'Ours-WVS JS' in q_row.columns and not pd.isna(q_row['Ours-WVS JS'].values[0]) else 0
            
            llm_ks = q_row['LLM-WVS KS'].values[0] if 'LLM-WVS KS' in q_row.columns and not pd.isna(q_row['LLM-WVS KS'].values[0]) else 0
            llm_wasserstein = q_row['LLM-WVS Wasserstein'].values[0] if 'LLM-WVS Wasserstein' in q_row.columns and not pd.isna(q_row['LLM-WVS Wasserstein'].values[0]) else 0
            llm_js = q_row['LLM-WVS JS'].values[0] if 'LLM-WVS JS' in q_row.columns and not pd.isna(q_row['LLM-WVS JS'].values[0]) else 0
            
            oc_ks = q_row['OC-WVS KS'].values[0] if 'OC-WVS KS' in q_row.columns and not pd.isna(q_row['OC-WVS KS'].values[0]) else 0
            oc_wasserstein = q_row['OC-WVS Wasserstein'].values[0] if 'OC-WVS Wasserstein' in q_row.columns and not pd.isna(q_row['OC-WVS Wasserstein'].values[0]) else 0
            oc_js = q_row['OC-WVS JS'].values[0] if 'OC-WVS JS' in q_row.columns and not pd.isna(q_row['OC-WVS JS'].values[0]) else 0
            
            q_metrics = {
                'Question': q,
                'Ours-WVS': np.mean([ours_ks, ours_wasserstein, ours_js]),
                'LLM-WVS': np.mean([llm_ks, llm_wasserstein, llm_js]),
                'OC-WVS': np.mean([oc_ks, oc_wasserstein, oc_js])
            }
            metrics_by_question.append(q_metrics)
        
        # 转换为DataFrame
        q_metrics_df = pd.DataFrame(metrics_by_question)
        
        # 绘制图表
        ours_wvs_avg = q_metrics_df['Ours-WVS'].tolist()
        llm_wvs_avg = q_metrics_df['LLM-WVS'].tolist()
        oc_wvs_avg = q_metrics_df['OC-WVS'].tolist()
        
        ax.bar(filtered_x - width, ours_wvs_avg, width, label='Ours-WVS Avg Metrics')
        ax.bar(filtered_x, llm_wvs_avg, width, label='LLM-WVS Avg Metrics')
        ax.bar(filtered_x + width, oc_wvs_avg, width, label='OC-WVS Avg Metrics')
        
        ax.set_title('Average Metrics by Question (KS, Wasserstein, JS)', fontsize=15)
        ax.set_ylabel('Average Metric Value', fontsize=12)
        ax.set_xticks(filtered_x)
        ax.set_xticklabels(filtered_questions, fontsize=8, rotation=90)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        question_chart_path = output_dir / "question_differences.png"
        plt.savefig(question_chart_path)
        plt.close()
        print(f"Question differences chart saved to: {question_chart_path}")
    
    print(f"\nBar charts saved to directory: {output_dir}")


def create_markdown_tables(metrics_df, country_diffs, question_diffs, avg_metrics, output_dir):
    """
    创建漂亮的Markdown格式表格
    
    Args:
        metrics_df: 统计分布差异指标结果DataFrame
        country_diffs: 国家差异数据
        question_diffs: 问题差异数据
        avg_metrics: 平均统计分布差异指标字典
        output_dir: 输出目录路径
    """
    # 创建Markdown文件内容
    markdown_content = "# 数据集统计分布差异分析结果\n\n"
    
    # 汇总表
    markdown_content += "## 汇总统计分布差异指标\n\n"
    markdown_content += "| 比较 | KS统计量 | KS p值 | Wasserstein距离 | JS散度 | 平均差异 |\n"
    markdown_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    # 添加平均指标行
    markdown_content += f"| Ours vs WVS | {avg_metrics['our_vs_survey_ks']:.3f} | {avg_metrics['our_vs_survey_ks_pvalue']:.3f} | {avg_metrics['our_vs_survey_wasserstein']:.3f} | {avg_metrics['our_vs_survey_js']:.3f} | {avg_metrics['avg_our_vs_wvs_diff']:.3f} |\n"
    markdown_content += f"| LLM vs WVS | {avg_metrics['llm_vs_survey_ks']:.3f} | {avg_metrics['llm_vs_survey_ks_pvalue']:.3f} | {avg_metrics['llm_vs_survey_wasserstein']:.3f} | {avg_metrics['llm_vs_survey_js']:.3f} | {avg_metrics['avg_llm_vs_wvs_diff']:.3f} |\n"
    markdown_content += f"| OC vs WVS | {avg_metrics['oc_vs_survey_ks']:.3f} | {avg_metrics['oc_vs_survey_ks_pvalue']:.3f} | {avg_metrics['oc_vs_survey_wasserstein']:.3f} | {avg_metrics['oc_vs_survey_js']:.3f} | {avg_metrics['oc_vs_wvs_ind_diff']:.3f} |\n"
    
    # 各问题统计分布差异指标详细
    markdown_content += "\n## 各问题统计分布差异指标详细\n\n"
    
    # 创建一个表格，显示KS统计量、Wasserstein距离和JS散度
    markdown_content += "| 问题 | Ours-WVS KS | LLM-WVS KS | Ours-WVS Wasserstein | LLM-WVS Wasserstein | Ours-WVS JS | LLM-WVS JS | OC-WVS Mean Diff |\n"
    markdown_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for _, row in metrics_df.iterrows():
        question = row['Question']
        ours_wvs_ks = f"{row['Ours-WVS KS']:.3f}" if pd.notnull(row['Ours-WVS KS']) else 'N/A'
        llm_wvs_ks = f"{row['LLM-WVS KS']:.3f}" if pd.notnull(row['LLM-WVS KS']) else 'N/A'
        
        ours_wvs_wasserstein = f"{row['Ours-WVS Wasserstein']:.3f}" if pd.notnull(row['Ours-WVS Wasserstein']) else 'N/A'
        llm_wvs_wasserstein = f"{row['LLM-WVS Wasserstein']:.3f}" if pd.notnull(row['LLM-WVS Wasserstein']) else 'N/A'
        
        ours_wvs_js = f"{row['Ours-WVS JS']:.3f}" if pd.notnull(row['Ours-WVS JS']) else 'N/A'
        llm_wvs_js = f"{row['LLM-WVS JS']:.3f}" if pd.notnull(row['LLM-WVS JS']) else 'N/A'
        
        # 添加CAN- vs WVS (Canada)的平均差异
        oc_wvs_mean_diff = f"{row['OC-WVS Mean Diff']:.3f}" if pd.notnull(row['OC-WVS Mean Diff']) else 'N/A'
        
        markdown_content += f"| {question} | {ours_wvs_ks} | {llm_wvs_ks} | {ours_wvs_wasserstein} | {llm_wvs_wasserstein} | {ours_wvs_js} | {llm_wvs_js} | {oc_wvs_mean_diff} |\n"
    
    # 保存Markdown文件
    markdown_path = output_dir / "metrics_results.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\nMarkdown format results saved to: {markdown_path}")


def create_country_question_comparison(our_df, survey_df, llm_df, opencharacter_df, survey_usa_df, output_dir):
    """
    创建每个国家每个问题的逐行比较表格
    
    Args:
        our_df: 我们的数据DataFrame
        survey_df: 世界价值观调查DataFrame
        llm_df: LLM数据DataFrame
        opencharacter_df: CAN-角色数据DataFrame
        survey_usa_df: 世界价值观调查Canada数据DataFrame
        output_dir: 输出目录路径
    """
    # 获取共同的问题列
    common_questions = sorted([col for col in our_df.columns if col.startswith('Q') and 
                               col in survey_df.columns and 
                               col in llm_df.columns])
    
    print(f"\nCreating country-question comparison tables...")
    
    # 创建一个大型比较表格
    comparison_data = []
    
    # 对每个国家进行循环
    for country in our_df.index:
        if country in survey_df.index and country in llm_df.index:
            # 对每个问题进行循环
            for question in common_questions:
                # 获取三个数据集中该国家该问题的值
                try:
                    our_value = pd.to_numeric(our_df.loc[country, question], errors='coerce')
                    survey_value = pd.to_numeric(survey_df.loc[country, question], errors='coerce')
                    llm_value = pd.to_numeric(llm_df.loc[country, question], errors='coerce')
                    
                    # 跳过包含NaN值的行
                    if pd.isna(our_value) or pd.isna(survey_value) or pd.isna(llm_value):
                        continue
                    
                    # 计算差异，顺序：1. our-wvs, 2. llm-wvs, 3. llm-ours
                    our_survey_diff = our_value - survey_value
                    llm_survey_diff = llm_value - survey_value
                    llm_our_diff = llm_value - our_value
                    
                    # 添加到比较数据中
                    comparison_data.append({
                        'Country': country,
                        'Question': question,
                        'Our Value': float(our_value),
                        'Survey Value': float(survey_value),
                        'LLM Value': float(llm_value),
                        'Our-Survey Diff': float(our_survey_diff),
                        'LLM-Survey Diff': float(llm_survey_diff),
                        'LLM-Our Diff': float(llm_our_diff),
                        'Abs Our-Survey Diff': float(abs(our_survey_diff)),
                        'Abs LLM-Survey Diff': float(abs(llm_survey_diff)),
                        'Abs LLM-Our Diff': float(abs(llm_our_diff))
                    })
                except (ValueError, TypeError):
                    # 如果转换失败，跳过这一行
                    continue
    
    # 创建比较DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        print("Warning: No valid comparison data found.")
        return pd.DataFrame()
    
    # 确保数值列是浮点数类型
    numeric_cols = ['Our Value', 'Survey Value', 'LLM Value', 
                    'Our-Survey Diff', 'LLM-Survey Diff', 'LLM-Our Diff',
                    'Abs Our-Survey Diff', 'Abs LLM-Survey Diff', 'Abs LLM-Our Diff']
    for col in numeric_cols:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
    
    # Calculate average differences by country
    country_avg_diff = comparison_df.groupby('Country')[
        ['Abs Our-Survey Diff', 'Abs LLM-Survey Diff', 'Abs LLM-Our Diff']
    ].mean().reset_index()
    
    # Calculate average differences by question
    question_avg_diff = comparison_df.groupby('Question')[
        ['Abs Our-Survey Diff', 'Abs LLM-Survey Diff', 'Abs LLM-Our Diff']
    ].mean().reset_index()
    
    # Calculate overall average differences
    overall_avg_diff = comparison_df[
        ['Abs Our-Survey Diff', 'Abs LLM-Survey Diff', 'Abs LLM-Our Diff']
    ].mean(numeric_only=True)
    
    # 将数值列四舍五入为3位小数
    for col in ['Our-Survey Diff', 'LLM-Survey Diff', 'LLM-Our Diff', 
               'Abs Our-Survey Diff', 'Abs LLM-Survey Diff', 'Abs LLM-Our Diff']:
        comparison_df[col] = comparison_df[col].round(3)
        if col in country_avg_diff.columns:
            country_avg_diff[col] = country_avg_diff[col].round(3)
        if col in question_avg_diff.columns:
            question_avg_diff[col] = question_avg_diff[col].round(3)
    
    # 保存比较数据到CSV文件
    comparison_output_path = output_dir / "country_question_comparison.csv"
    comparison_df.to_csv(comparison_output_path, index=False)
    print(f"  Country-question comparison data saved to: {comparison_output_path}")
    
    # Save country average differences to CSV
    country_avg_output_path = output_dir / "country_average_differences.csv"
    country_avg_diff.to_csv(country_avg_output_path, index=False)
    print(f"  Country average differences saved to: {country_avg_output_path}")
    
    # Save question average differences to CSV
    question_avg_output_path = output_dir / "question_average_differences.csv"
    question_avg_diff.to_csv(question_avg_output_path, index=False)
    print(f"  Question average differences saved to: {question_avg_output_path}")
    
    # 不再保存overall_average_differences.csv文件
    # 只在控制台输出结果
    pass
    
    # Print overall average differences
    print("\nOverall Mean Absolute Differences:")
    print(f"  Ours-WVS: {overall_avg_diff['Abs Our-Survey Diff']:.4f}")
    print(f"  LLM-WVS: {overall_avg_diff['Abs LLM-Survey Diff']:.4f}")
    print(f"  LLM-Ours: {overall_avg_diff['Abs LLM-Our Diff']:.4f}")
    
    # Return overall average differences
    return {
        'Abs Our-Survey Diff': overall_avg_diff['Abs Our-Survey Diff'],
        'Abs LLM-Survey Diff': overall_avg_diff['Abs LLM-Survey Diff'],
        'Abs LLM-Our Diff': overall_avg_diff['Abs LLM-Our Diff']
    }
    

    
    return comparison_df

def visualize_tsne(survey_df, llm_df, our_df, opencharacter_df, output_dir):
    """
    使用t-SNE对数据进行可视化
    
    Args:
        survey_df: 调查数据DataFrame
        llm_df: LLM数据DataFrame
        our_df: 我们的数据DataFrame
        opencharacter_df: CAN-数据DataFrame
        output_dir: 输出目录路径
    """
    # 选择指定的列进行t-SNE分析
    columns_to_use = ['Q45', 'Q46', 'Q57', 'Q184', 'Q218', 'Q254']
    
    # 创建一个包含来源标签的组合数据集
    combined_data = []
    
    # 处理调查数据
    survey_data = survey_df[columns_to_use].copy()
    survey_data = survey_data.apply(pd.to_numeric, errors='coerce')
    survey_data['source'] = 'Survey'
    combined_data.append(survey_data)
    
    # 处理LLM数据
    llm_data = llm_df[columns_to_use].copy()
    llm_data = llm_data.apply(pd.to_numeric, errors='coerce')
    llm_data['source'] = 'LLM'
    combined_data.append(llm_data)
    
    # 处理我们的数据
    our_data = our_df[columns_to_use].copy()
    our_data = our_data.apply(pd.to_numeric, errors='coerce')
    our_data['source'] = 'Ours'
    combined_data.append(our_data)
    
    # 处理CAN-数据
    ind_data = opencharacter_df[columns_to_use].copy()
    ind_data = ind_data.apply(pd.to_numeric, errors='coerce')
    ind_data['source'] = 'CAN-'
    combined_data.append(ind_data)
    
    # 合并所有数据
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # 删除包含NaN值的行
    combined_df = combined_df.dropna()
    
    if len(combined_df) == 0:
        print("错误：删除NaN值后没有有效数据。")
        return
    
    # 提取用于t-SNE的特征
    features = combined_df[columns_to_use].values
    
    # 执行t-SNE
    print("\n执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    tsne_results = tsne.fit_transform(features)
    
    # 将t-SNE结果添加到DataFrame
    combined_df['tsne_x'] = tsne_results[:, 0]
    combined_df['tsne_y'] = tsne_results[:, 1]
    
    # 创建散点图
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")
    
    # 设置支持中文的字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        print("警告：无法设置中文字体，图表中的中文可能无法正确显示")
    
    # 为每个来源定义颜色
    colors = {'Survey': 'blue', 'LLM': 'green', 'Ours': 'red', 'CAN-': 'purple'}
    
    # 用不同的颜色绘制每个来源
    for source, group in combined_df.groupby('source'):
        plt.scatter(
            group['tsne_x'], 
            group['tsne_y'], 
            label=source, 
            alpha=0.7,
            color=colors[source],
            edgecolors='w',
            s=100
        )
    
    plt.title('t-SNE Visualization of Survey Data', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'tsne_visualization.png'
    plt.savefig(output_path, dpi=300)
    print(f"t-SNE可视化已保存至: {output_path}")
    
    # 创建3D版本以获得更详细的可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用3个组件执行t-SNE
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(features)-1))
    tsne_results_3d = tsne_3d.fit_transform(features)
    
    # 将3D t-SNE结果添加到DataFrame
    combined_df['tsne_x_3d'] = tsne_results_3d[:, 0]
    combined_df['tsne_y_3d'] = tsne_results_3d[:, 1]
    combined_df['tsne_z_3d'] = tsne_results_3d[:, 2]
    
    # 在3D中用不同的颜色绘制每个来源
    for source, group in combined_df.groupby('source'):
        ax.scatter(
            group['tsne_x_3d'], 
            group['tsne_y_3d'], 
            group['tsne_z_3d'],
            label=source, 
            alpha=0.7,
            color=colors[source],
            s=50
        )
    
    ax.set_title('3D t-SNE Visualization of Survey Data', fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_zlabel('t-SNE Component 3', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存3D图表
    output_path_3d = output_dir / 'tsne_visualization_3d.png'
    plt.savefig(output_path_3d, dpi=300)
    print(f"3D t-SNE可视化已保存至: {output_path_3d}")
    
    # 为每个来源创建单独的图表
    plt.figure(figsize=(20, 5))
    
    # 绘制调查数据
    plt.subplot(1, 4, 1)
    survey_group = combined_df[combined_df['source'] == 'Survey']
    plt.scatter(survey_group['tsne_x'], survey_group['tsne_y'], color=colors['Survey'], alpha=0.7, s=100, edgecolors='w')
    plt.title('Survey Data', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    
    # 绘制LLM数据
    plt.subplot(1, 4, 2)
    llm_group = combined_df[combined_df['source'] == 'LLM']
    plt.scatter(llm_group['tsne_x'], llm_group['tsne_y'], color=colors['LLM'], alpha=0.7, s=100, edgecolors='w')
    plt.title('LLM Data', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    
    # 绘制我们的数据
    plt.subplot(1, 4, 3)
    our_group = combined_df[combined_df['source'] == 'Ours']
    plt.scatter(our_group['tsne_x'], our_group['tsne_y'], color=colors['Ours'], alpha=0.7, s=100, edgecolors='w')
    plt.title('Our Data', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    
    # 绘制CAN-数据
    plt.subplot(1, 4, 4)
    ind_group = combined_df[combined_df['source'] == 'CAN-']
    plt.scatter(ind_group['tsne_x'], ind_group['tsne_y'], color=colors['CAN-'], alpha=0.7, s=100, edgecolors='w')
    plt.title('CAN- Data', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=10)
    plt.ylabel('t-SNE Component 2', fontsize=10)
    
    plt.tight_layout()
    
    # 保存单独的图表
    output_path_individual = output_dir / 'tsne_visualization_individual.png'
    plt.savefig(output_path_individual, dpi=300)
    print(f"单独的t-SNE可视化已保存至: {output_path_individual}")
    
    # 将t-SNE坐标保存到CSV以供进一步分析
    tsne_output_path = output_dir / 'tsne_coordinates.csv'
    combined_df.to_csv(tsne_output_path, index=False)
    print(f"t-SNE坐标已保存至: {tsne_output_path}")

def main():
    # 获取文件路径
    base_dir = Path(__file__).parent.parent
    # 使用指定的数据源
    survey_file = base_dir / "output" / "all_Canda_survey_responses.csv"
    our_file = base_dir / "output" / "our_responses.csv"
    llm_file = base_dir / "output" / "llm_responses.csv"
    opencharacter_file = base_dir / "output" / "opencharacter_responses.csv"  # 添加ind_responses.csv
    output_dir = base_dir / "results"
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    # 使用所有数据集中出现的国家作为过滤条件
    print("\nLoading datasets to extract country names...")
    our_df_temp = pd.read_csv(our_file)
    survey_df_temp = pd.read_csv(survey_file)
    llm_df_temp = pd.read_csv(llm_file)
    opencharacter_df_temp = pd.read_csv(opencharacter_file)  # 添加ind_responses.csv
    
    # 获取所有数据集中出现的国家名称的交集
    our_countries = set(our_df_temp['country_name'].values)
    survey_countries = set(survey_df_temp['country_name'].values)
    llm_countries = set(llm_df_temp['country_name'].values)
    opencharacter_countries = set(opencharacter_df_temp['country_name'].values)  # 添加us国家
    
    # 取交集，只保留所有数据集中都出现的国家
    # 注意：ind_responses.csv只包含Canada数据，所以我们不要求所有数据集都包含相同的国家
    filter_country_names = our_countries & survey_countries & llm_countries
    print(f"\nFound {len(filter_country_names)} countries that appear in all datasets")
    
    # Load datasets
    print("\nLoading all datasets...")
    our_df = load_data(our_file)
    survey_df = load_data(survey_file)
    llm_df = load_data(llm_file)
    opencharacter_df = load_data(opencharacter_file)  # 添加ind_responses.csv
    
    # 直接使用"IND"来过滤our_responses.csv
    our_df_filtered = our_df[our_df['country_name'] == 'CAN']
    # 直接使用"Canada"来过滤survey_df
    survey_df_filtered = survey_df[survey_df['country_name'] == 'CAN']
    # 直接使用"Canada"来过滤llm_df
    llm_df_filtered = llm_df[llm_df['country_name'] == 'CAN']
    
    # 对于opencharacter_df，我们只保留IND的数据
    opencharacter_df_filtered = opencharacter_df[opencharacter_df['country_name'] == 'CAN']
    
    # 从原始的survey_df中提取Canada数据作为比较基准
    survey_usa_df = survey_df[survey_df['country_name'] == 'CAN']
    
    print(f"\nFiltered dataset sizes:")
    print(f"  Ours: {len(our_df_filtered)} records")
    print(f"  WVS: {len(survey_df_filtered)} records")
    print(f"  LLM: {len(llm_df_filtered)} records")
    print(f"  CAN-: {len(opencharacter_df_filtered)} records")
    print(f"  WVS (Canada only): {len(survey_usa_df)} records")
    
    # 不再需要设置索引，因为calculate_metrics函数已经不依赖于共同的索引
    # 保留这些注释作为参考
    # our_df_filtered.set_index('country_name', inplace=True)
    # survey_df_filtered.set_index('country_name', inplace=True)
    # llm_df_filtered.set_index('country_name', inplace=True)
    # opencharacter_df_filtered.set_index('character_id', inplace=True)
    
    # 对于Canada数据，我们需要将survey数据转换为相同的格式
    # 这里我们假设两个数据集都可以直接比较
    
    # 计算多种评估指标
    print("\nCalculating evaluation metrics...")
    our_vs_survey_metrics = calculate_metrics(our_df_filtered, survey_df_filtered)
    llm_vs_survey_metrics = calculate_metrics(llm_df_filtered, survey_df_filtered)
    
    # 添加CAN-与WVS的比较
    # 注意：这里我们不能直接使用calculate_metrics函数，因为索引不同
    # 我们需要单独处理CAN-与survey_usa的比较
    
    # 首先获取共同的问题ID
    opencharacter_question_ids = [col for col in opencharacter_df_filtered.columns if col.startswith('Q')]
    survey_question_ids = [col for col in survey_df[survey_df['country_name'] == 'CAN'].columns if col.startswith('Q')]
    common_question_ids = list(set(opencharacter_question_ids) & set(survey_question_ids))
    
    print(f"\nCommon question IDs between CAN- and WVS (Canada): {len(common_question_ids)}")
    
    # 对每个共同的问题ID，我们计算平均值和标准差
    opencharacter_vs_survey_usa_stats = {}
    opencharacter_vs_survey_usa_metrics = {}
    
    for qid in common_question_ids:
        opencharacter_values = opencharacter_df_filtered[qid].dropna().astype(float)
        survey_values = survey_df[survey_df['country_name'] == 'CAN'][qid].dropna().astype(float)
        
        opencharacter_mean = opencharacter_values.mean()
        survey_mean = survey_values.mean()
        opencharacter_std = opencharacter_values.std()
        survey_std = survey_values.std()
        
        opencharacter_vs_survey_usa_stats[qid] = {
            'CAN- Mean': opencharacter_mean,
            'WVS Canada Mean': survey_mean,
            'CAN- Std': opencharacter_std,
            'WVS Canada Std': survey_std,
            'Mean_Diff': abs(opencharacter_mean - survey_mean)
        }
        
        # 计算KS统计量
        ks_stat, ks_pvalue = ks_2samp(opencharacter_values, survey_values)
        
        # 计算Wasserstein距离
        from scipy.stats import wasserstein_distance
        wasserstein_dist = wasserstein_distance(opencharacter_values, survey_values)
        
        # 计算JS散度
        # 需要将数据转换为概率分布
        min_val = min(opencharacter_values.min(), survey_values.min())
        max_val = max(opencharacter_values.max(), survey_values.max())
        bins = np.linspace(min_val, max_val, 20)  # 使用20个区间
        
        hist1, _ = np.histogram(opencharacter_values, bins=bins, density=True)
        hist2, _ = np.histogram(survey_values, bins=bins, density=True)
        
        # 处理零频率
        hist1 = np.clip(hist1, 1e-10, None)  # 将零值替换为小正数
        hist2 = np.clip(hist2, 1e-10, None)
        
        # 归一化使其成为概率分布
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # 计算JS散度
        js_divergence = jensenshannon(hist1, hist2)
        
        # 保存指标
        opencharacter_vs_survey_usa_metrics[qid] = {
            'KS_Statistic': ks_stat,
            'KS_pvalue': ks_pvalue,
            'Wasserstein_Distance': wasserstein_dist,
            'JS_Divergence': js_divergence
        }
    
    # 打印统计结果
    print("\nCAN- vs WVS (Canada) Statistics:")
    for qid, stats in opencharacter_vs_survey_usa_stats.items():
        print(f"{qid}: CAN- Mean = {stats['CAN- Mean']:.2f}, WVS Canada Mean = {stats['WVS Canada Mean']:.2f}, Diff = {stats['Mean_Diff']:.2f}")
    
    # 将统计结果保存到CSV文件
    oc_vs_wvs_usa_stats_df = pd.DataFrame.from_dict(opencharacter_vs_survey_usa_stats, orient='index')
    oc_vs_wvs_usa_stats_df.index.name = 'Question_ID'
    oc_vs_wvs_usa_stats_df.reset_index(inplace=True)
    
    # 将指标结果保存到CSV文件
    oc_vs_wvs_usa_metrics_df = pd.DataFrame.from_dict(opencharacter_vs_survey_usa_metrics, orient='index')
    oc_vs_wvs_usa_metrics_df.index.name = 'Question_ID'
    oc_vs_wvs_usa_metrics_df.reset_index(inplace=True)
    
    # 合并统计和指标结果
    oc_vs_wvs_usa_combined_df = pd.merge(oc_vs_wvs_usa_stats_df, oc_vs_wvs_usa_metrics_df, on='Question_ID')
    oc_vs_wvs_usa_combined_df.to_csv(output_dir / 'can_vs_wvs_ind_stats.csv', index=False)
    print(f"CAN- vs WVS (Canada) statistics and metrics saved to: {output_dir / 'can_vs_wvs_ind_stats.csv'}")
    
    # 计算平均指标
    avg_ks_stat = oc_vs_wvs_usa_metrics_df['KS_Statistic'].mean()
    avg_wasserstein = oc_vs_wvs_usa_metrics_df['Wasserstein_Distance'].mean()
    avg_js = oc_vs_wvs_usa_metrics_df['JS_Divergence'].mean()
    avg_mean_diff = oc_vs_wvs_usa_stats_df['Mean_Diff'].mean()
    
    print(f"\nCAN- vs WVS (Canada) Average Metrics:")
    print(f"Average KS Statistic: {avg_ks_stat:.4f}")
    print(f"Average Wasserstein Distance: {avg_wasserstein:.4f}")
    print(f"Average JS Divergence: {avg_js:.4f}")
    print(f"Average Mean Difference: {avg_mean_diff:.4f}")
    # 我们不再llm与our的比较
    # llm_vs_our_metrics = calculate_metrics(llm_df_filtered, our_df_filtered)
    
    # 创建评估指标结果矩阵
    metrics_results = []
    for col in sorted(set(our_vs_survey_metrics.keys()) | set(llm_vs_survey_metrics.keys())):
        row = {'Question': col}
        
        # 添加our vs survey的指标
        if col in our_vs_survey_metrics:
            row['Ours-WVS KS'] = our_vs_survey_metrics[col]['KS_Statistic']
            row['Ours-WVS KS_pvalue'] = our_vs_survey_metrics[col]['KS_pvalue']
            row['Ours-WVS Wasserstein'] = our_vs_survey_metrics[col]['Wasserstein_Distance']
            row['Ours-WVS JS'] = our_vs_survey_metrics[col]['JS_Divergence']
            row['Ours Mean'] = our_vs_survey_metrics[col]['Mean_1']
            row['WVS Mean'] = our_vs_survey_metrics[col]['Mean_2']
            row['Ours-WVS Mean Diff'] = our_vs_survey_metrics[col]['Mean_Diff']
        else:
            row['Ours-WVS KS'] = np.nan
            row['Ours-WVS KS_pvalue'] = np.nan
            row['Ours-WVS Wasserstein'] = np.nan
            row['Ours-WVS JS'] = np.nan
            row['Ours Mean'] = np.nan
            row['WVS Mean'] = np.nan
            row['Ours-WVS Mean Diff'] = np.nan
        
        # 添加llm vs survey的指标
        if col in llm_vs_survey_metrics:
            row['LLM-WVS KS'] = llm_vs_survey_metrics[col]['KS_Statistic']
            row['LLM-WVS KS_pvalue'] = llm_vs_survey_metrics[col]['KS_pvalue']
            row['LLM-WVS Wasserstein'] = llm_vs_survey_metrics[col]['Wasserstein_Distance']
            row['LLM-WVS JS'] = llm_vs_survey_metrics[col]['JS_Divergence']
            row['LLM Mean'] = llm_vs_survey_metrics[col]['Mean_1']
            row['LLM-WVS Mean Diff'] = llm_vs_survey_metrics[col]['Mean_Diff']
        else:
            row['LLM-WVS KS'] = np.nan
            row['LLM-WVS KS_pvalue'] = np.nan
            row['LLM-WVS Wasserstein'] = np.nan
            row['LLM-WVS JS'] = np.nan
            row['LLM Mean'] = np.nan
            row['LLM-WVS Mean Diff'] = np.nan
        
        # 添加CAN- vs WVS (Canada)的比较结果
        if col in opencharacter_vs_survey_usa_stats and col in opencharacter_vs_survey_usa_metrics:
            row['OC-WVS Mean Diff'] = opencharacter_vs_survey_usa_stats[col]['Mean_Diff']
            row['CAN- Mean'] = opencharacter_vs_survey_usa_stats[col]['CAN- Mean']
            row['WVS Canada Mean'] = opencharacter_vs_survey_usa_stats[col]['WVS Canada Mean']
            row['OC-WVS KS'] = opencharacter_vs_survey_usa_metrics[col]['KS_Statistic']
            row['OC-WVS KS_pvalue'] = opencharacter_vs_survey_usa_metrics[col]['KS_pvalue']
            row['OC-WVS Wasserstein'] = opencharacter_vs_survey_usa_metrics[col]['Wasserstein_Distance']
            row['OC-WVS JS'] = opencharacter_vs_survey_usa_metrics[col]['JS_Divergence']
        else:
            row['OC-WVS Mean Diff'] = np.nan
            row['CAN- Mean'] = np.nan
            row['WVS Canada Mean'] = np.nan
            row['OC-WVS KS'] = np.nan
            row['OC-WVS KS_pvalue'] = np.nan
            row['OC-WVS Wasserstein'] = np.nan
            row['OC-WVS JS'] = np.nan
        
        metrics_results.append(row)
    
    # 创建评估指标结果DataFrame
    metrics_df = pd.DataFrame(metrics_results)
    
    # 检查metrics_df是否为空
    if not metrics_df.empty and 'Question' in metrics_df.columns:
        # 定义正确的问题顺序
        question_order = ['Q45', 'Q46', 'Q57', 'Q184', 'Q218', 'Q254']
        
        # 按照指定顺序排序
        metrics_df['Question'] = pd.Categorical(metrics_df['Question'], categories=question_order, ordered=True)
        metrics_df = metrics_df.sort_values('Question').reset_index(drop=True)
    
    # Save metrics results to CSV with 3 decimal places
    metrics_output_path = output_dir / "metrics_results.csv"
    # 对数值列应用3位小数格式化
    for col in metrics_df.columns:
        if col != 'Question':
            metrics_df[col] = metrics_df[col].apply(lambda x: round(x, 3) if pd.notnull(x) else x)
    
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"\nEvaluation metrics saved to: {metrics_output_path}")
    
    # 打印评估指标结果
    print("\nEvaluation metrics results:")
    print(metrics_df.to_string(index=False))
    
    # 计算平均评估指标
    # 1. 平均KS统计量
    avg_our_vs_survey_ks = np.nanmean([our_vs_survey_metrics[col]['KS_Statistic'] for col in our_vs_survey_metrics])
    avg_llm_vs_survey_ks = np.nanmean([llm_vs_survey_metrics[col]['KS_Statistic'] for col in llm_vs_survey_metrics])
    avg_oc_vs_survey_ks = np.nanmean([opencharacter_vs_survey_usa_metrics[col]['KS_Statistic'] for col in opencharacter_vs_survey_usa_metrics])
    
    # 2. 平均KS p值
    avg_our_vs_survey_ks_pvalue = np.nanmean([our_vs_survey_metrics[col]['KS_pvalue'] for col in our_vs_survey_metrics])
    avg_llm_vs_survey_ks_pvalue = np.nanmean([llm_vs_survey_metrics[col]['KS_pvalue'] for col in llm_vs_survey_metrics])
    avg_oc_vs_survey_ks_pvalue = np.nanmean([opencharacter_vs_survey_usa_metrics[col]['KS_pvalue'] for col in opencharacter_vs_survey_usa_metrics])
    
    # 3. 平均Wasserstein距离
    avg_our_vs_survey_wasserstein = np.nanmean([our_vs_survey_metrics[col]['Wasserstein_Distance'] for col in our_vs_survey_metrics])
    avg_llm_vs_survey_wasserstein = np.nanmean([llm_vs_survey_metrics[col]['Wasserstein_Distance'] for col in llm_vs_survey_metrics])
    avg_oc_vs_survey_wasserstein = np.nanmean([opencharacter_vs_survey_usa_metrics[col]['Wasserstein_Distance'] for col in opencharacter_vs_survey_usa_metrics])
    
    # 4. 平均JS散度
    avg_our_vs_survey_js = np.nanmean([our_vs_survey_metrics[col]['JS_Divergence'] for col in our_vs_survey_metrics])
    avg_llm_vs_survey_js = np.nanmean([llm_vs_survey_metrics[col]['JS_Divergence'] for col in llm_vs_survey_metrics])
    avg_oc_vs_survey_js = np.nanmean([opencharacter_vs_survey_usa_metrics[col]['JS_Divergence'] for col in opencharacter_vs_survey_usa_metrics])
    
    # 5. 计算平均差异
    avg_our_vs_wvs_diff = np.nanmean([our_vs_survey_metrics[col]['Mean_Diff'] for col in our_vs_survey_metrics])
    avg_llm_vs_wvs_diff = np.nanmean([llm_vs_survey_metrics[col]['Mean_Diff'] for col in llm_vs_survey_metrics])
    avg_oc_vs_wvs_ind_diff = np.nanmean([stats['Mean_Diff'] for stats in opencharacter_vs_survey_usa_stats.values()])
    
    print("\nAverage Kolmogorov-Smirnov Statistic (KS):")
    print(f"  Ours vs WVS: {avg_our_vs_survey_ks:.3f}")
    print(f"  LLM vs WVS: {avg_llm_vs_survey_ks:.3f}")
    print(f"  OC vs WVS: {avg_oc_vs_survey_ks:.3f}")
    
    print("\nAverage KS p-value:")
    print(f"  Ours vs WVS: {avg_our_vs_survey_ks_pvalue:.3f}")
    print(f"  LLM vs WVS: {avg_llm_vs_survey_ks_pvalue:.3f}")
    print(f"  OC vs WVS: {avg_oc_vs_survey_ks_pvalue:.3f}")
    
    print("\nAverage Wasserstein Distance:")
    print(f"  Ours vs WVS: {avg_our_vs_survey_wasserstein:.3f}")
    print(f"  LLM vs WVS: {avg_llm_vs_survey_wasserstein:.3f}")
    print(f"  OC vs WVS: {avg_oc_vs_survey_wasserstein:.3f}")
    
    print("\nAverage Jensen-Shannon Divergence:")
    print(f"  Ours vs WVS: {avg_our_vs_survey_js:.3f}")
    print(f"  LLM vs WVS: {avg_llm_vs_survey_js:.3f}")
    print(f"  OC vs WVS: {avg_oc_vs_survey_js:.3f}")
    
    print("\nAverage Mean Difference:")
    print(f"  Ours vs WVS: {avg_our_vs_wvs_diff:.3f}")
    print(f"  LLM vs WVS: {avg_llm_vs_wvs_diff:.3f}")
    print(f"  OC vs WVS: {avg_oc_vs_wvs_ind_diff:.3f}")
    

    
    # 创建每个国家每个问题的逐行比较
    create_country_question_comparison(our_df_filtered, survey_df_filtered, llm_df_filtered, opencharacter_df_filtered, survey_usa_df, output_dir)
    
    # 加载国家和问题差异数据
    country_diff_path = output_dir / "country_average_differences.csv"
    question_diff_path = output_dir / "question_average_differences.csv"
    
    country_diffs = pd.DataFrame()
    question_diffs = pd.DataFrame()
    
    if country_diff_path.exists():
        country_diffs = pd.read_csv(country_diff_path)
    
    if question_diff_path.exists():
        question_diffs = pd.read_csv(question_diff_path)
    
    # 收集平均指标数据
    avg_metrics = {
        'our_vs_survey_ks': avg_our_vs_survey_ks,
        'llm_vs_survey_ks': avg_llm_vs_survey_ks,
        'oc_vs_survey_ks': avg_oc_vs_survey_ks,
        'our_vs_survey_ks_pvalue': avg_our_vs_survey_ks_pvalue,
        'llm_vs_survey_ks_pvalue': avg_llm_vs_survey_ks_pvalue,
        'oc_vs_survey_ks_pvalue': avg_oc_vs_survey_ks_pvalue,
        'our_vs_survey_wasserstein': avg_our_vs_survey_wasserstein,
        'llm_vs_survey_wasserstein': avg_llm_vs_survey_wasserstein,
        'oc_vs_survey_wasserstein': avg_oc_vs_survey_wasserstein,
        'our_vs_survey_js': avg_our_vs_survey_js,
        'llm_vs_survey_js': avg_llm_vs_survey_js,
        'oc_vs_survey_js': avg_oc_vs_survey_js,
        'avg_our_vs_wvs_diff': avg_our_vs_wvs_diff,
        'avg_llm_vs_wvs_diff': avg_llm_vs_wvs_diff,
        'oc_vs_wvs_ind_diff': avg_oc_vs_wvs_ind_diff
    }
    
    # 创建Markdown格式的评估指标结果表格
    create_markdown_tables(metrics_df, country_diffs, question_diffs, avg_metrics, output_dir)
    
    # 创建柱状图可视化
    create_bar_charts(metrics_df, country_diffs, question_diffs, output_dir)
    
    # 使用t-SNE进行可视化
    visualize_tsne(survey_df_filtered, llm_df_filtered, our_df_filtered, opencharacter_df_filtered, output_dir)
    
    # Print summary of metrics
    print("\nEvaluation Complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
