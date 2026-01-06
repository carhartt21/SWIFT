#!/usr/bin/env python3
"""
Weather Dataset Visualization Pipeline with Plotting
====================================================

A comprehensive data visualization pipeline for analyzing weather-classified image datasets.
Now includes matplotlib-based visualizations for improved visual analysis.

This pipeline processes subdirectories containing weather-classified images and generates
detailed statistics, comparison tables, and visual plots.

Author: Generated for Computer Vision Research
Date: October 2025
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WeatherDatasetVisualizer:
    """
    A comprehensive pipeline for analyzing and visualizing weather-classified image datasets.

    The pipeline discovers subdirectories containing 'categories' folders, loads image metadata,
    and generates various statistical analyses, comparison tables, and visual plots.

    Attributes:
        parent_folder (Path): Path to the parent folder containing dataset subdirectories
        datasets (dict): Dictionary storing loaded dataset information
        weather_categories (list): List of weather category names
        output_folder (Path): Path to the output folder for generated files
        plots_folder (Path): Path to the plots subfolder

    Example:
        >>> visualizer = WeatherDatasetVisualizer('/path/to/datasets')
        >>> visualizer.analyze_all_datasets()
        >>> visualizer.generate_all_visualizations()
    """

    def __init__(self, parent_folder):
        """
        Initialize the visualizer with a parent folder containing subdirectories.

        Parameters:
            parent_folder (str): Path to the parent folder containing dataset subdirectories
        """
        self.parent_folder = Path(parent_folder)
        self.datasets = {}
        self.weather_categories = ['clear_day', 'foggy', 'snowy', 'night', 'rainy', 'dawn_dusk', 'cloudy']
        self.excluded_datasets = ['SEVERE', 'SEVERE_WEATHER']  # Datasets to ignore
        self.output_folder = Path(parent_folder) / 'analysis_output'
        self.output_folder.mkdir(exist_ok=True)

        # Create plots subfolder
        self.plots_folder = self.output_folder / 'plots'
        self.plots_folder.mkdir(exist_ok=True)

        # Color palette for consistent plotting
        self.category_colors = {
            'clear_day': '#FFD700',
            'foggy': '#A9A9A9',
            'snowy': '#E0FFFF',
            'night': '#191970',
            'rainy': '#4682B4',
            'dawn_dusk': '#FF8C00',
            'cloudy': '#778899'
        }

        # Display names for weather categories (proper names)
        self.category_display_names = {
            'clear_day': 'Clear Day',
            'cloudy': 'Cloudy',
            'dawn_dusk': 'Dawn/Dusk',
            'foggy': 'Foggy',
            'night': 'Night',
            'rainy': 'Rainy',
            'snowy': 'Snowy'
        }

    def discover_datasets(self):
        """
        Discover all subdirectories containing weather category folders.
        Checks for 'images', 'categories', or 'categories/original_size' folders.

        Returns:
            list: Names of discovered datasets
        """
        discovered = []
        for subdir in self.parent_folder.iterdir():
            if subdir.is_dir():
                # Skip excluded datasets
                if subdir.name in self.excluded_datasets:
                    continue
                # Check various possible folder structures
                possible_paths = [
                    subdir / 'images',
                    subdir / 'categories' / 'original_size',
                    subdir / 'categories',
                ]
                for path in possible_paths:
                    if path.exists() and path.is_dir():
                        # Check if it contains any weather category subfolders
                        has_weather_category = False
                        for category in self.weather_categories:
                            if (path / category).exists():
                                has_weather_category = True
                                break
                        if has_weather_category:
                            discovered.append(subdir.name)
                            break  # Found valid path, don't check other paths
        print(f"Discovered {len(discovered)} datasets: {discovered}")
        return discovered

    def _get_images_root(self, dataset_name):
        """
        Find the correct images root path for a dataset.
        Checks for 'images', 'categories/original_size', or 'categories' folders.

        Parameters:
            dataset_name (str): Name of the dataset

        Returns:
            Path: Path to the images root, or None if not found
        """
        possible_paths = [
            self.parent_folder / dataset_name / 'images',
            self.parent_folder / dataset_name / 'categories' / 'original_size',
            self.parent_folder / dataset_name / 'categories',
        ]
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # Verify it has at least one weather category
                for category in self.weather_categories:
                    if (path / category).exists():
                        return path
        return None

    def load_dataset_stats(self, dataset_name):
        """
        Generate dataset statistics by scanning the dataset folder instead of loading from file.

        Parameters:
            dataset_name (str): Name of the dataset

        Returns:
            dict: Dataset statistics including total_images, total_categories, and category_counts
        """
        images_root = self._get_images_root(dataset_name)
        if images_root is None:
            return None

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}
        category_counts = {}

        for category in self.weather_categories:
            cat_dir = images_root / category
            if not cat_dir.exists() or not cat_dir.is_dir():
                continue

            # Prefer counting metadata JSON files if present
            json_count = len(list(cat_dir.glob('*.json')))

            if json_count > 0:
                count = json_count
            else:
                # Fallback: count image files by common extensions
                try:
                    count = sum(1 for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
                except Exception:
                    count = 0

            if count > 0:
                category_counts[category] = int(count)

        total_images = int(sum(category_counts.values()))
        total_categories = int(len([c for c in category_counts if category_counts[c] > 0]))

        stats = {
            'dataset': dataset_name,
            'total_images': total_images,
            'total_categories': total_categories,
            'category_counts': category_counts,
            # Optional fields for compatibility with prior stats.json
            'confidence_threshold': None,
            'margin_threshold': None,
        }

        return stats

    def load_image_metadata(self, dataset_name, category):
        """
        Load all metadata files for images in a specific category.

        Parameters:
            dataset_name (str): Name of the dataset
            category (str): Weather category name

        Returns:
            list: List of metadata dictionaries
        """
        images_root = self._get_images_root(dataset_name)
        if images_root is None:
            return []
        category_path = images_root / category
        metadata_list = []

        if not category_path.exists():
            return metadata_list

        for meta_file in category_path.glob('*.json'):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    metadata['dataset'] = dataset_name
                    metadata_list.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load {meta_file}: {e}")
                continue

        return metadata_list

    def analyze_all_datasets(self):
        """
        Load and analyze all discovered datasets.

        Returns:
            dict: Dictionary of analyzed datasets
        """
        dataset_names = self.discover_datasets()

        for dataset_name in dataset_names:
            print(f"\nAnalyzing dataset: {dataset_name}")
            stats = self.load_dataset_stats(dataset_name)

            all_metadata = []
            for category in self.weather_categories:
                metadata = self.load_image_metadata(dataset_name, category)
                all_metadata.extend(metadata)

            self.datasets[dataset_name] = {
                'stats': stats,
                'metadata': all_metadata,
                'metadata_df': pd.DataFrame(all_metadata) if all_metadata else pd.DataFrame()
            }

            if stats:
                print(f"  Total images: {stats['total_images']}")
                print(f"  Categories: {stats['total_categories']}")

        return self.datasets

    def create_category_distribution_table(self):
        """
        Create a table showing category distribution across all datasets.

        Returns:
            DataFrame: Category distribution table
        """
        distribution_data = []

        for dataset_name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                row = {'Dataset': dataset_name}
                row.update(data['stats']['category_counts'])
                row['Total'] = data['stats']['total_images']
                distribution_data.append(row)

        df = pd.DataFrame(distribution_data)
        cols = ['Dataset'] + self.weather_categories + ['Total']
        cols = [c for c in cols if c in df.columns]
        df = df[cols].fillna(0)

        # Convert to integers
        for col in df.columns:
            if col != 'Dataset':
                df[col] = df[col].astype(int)

        output_path = self.output_folder / 'category_distribution.csv'
        df.to_csv(output_path, index=False)
        print(f"\nSaved category distribution table to: {output_path}")
        return df

    def create_category_percentages_table(self):
        """
        Create a table showing category percentages across all datasets.

        Returns:
            DataFrame: Category percentages table
        """
        percentage_data = []

        for dataset_name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                row = {'Dataset': dataset_name}
                total = data['stats']['total_images']

                for category, count in data['stats']['category_counts'].items():
                    row[category] = round(100 * count / total, 2) if total > 0 else 0

                percentage_data.append(row)

        df = pd.DataFrame(percentage_data)
        cols = ['Dataset'] + self.weather_categories
        cols = [c for c in cols if c in df.columns]
        df = df[cols].fillna(0)

        output_path = self.output_folder / 'category_percentages.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved category percentages table to: {output_path}")
        return df

    def create_confidence_statistics_table(self):
        """
        Create a table with confidence and margin statistics for each dataset and category.

        Returns:
            DataFrame: Confidence statistics table
        """
        stats_data = []

        for dataset_name, data in self.datasets.items():
            if data['metadata_df'].empty:
                continue

            df = data['metadata_df']

            overall_stats = {
                'Dataset': dataset_name,
                'Category': 'Overall',
                'Count': len(df),
                'Avg_Confidence': df['confidence'].mean() if 'confidence' in df.columns else None,
                'Std_Confidence': df['confidence'].std() if 'confidence' in df.columns else None,
                'Avg_Margin': df['margin'].mean() if 'margin' in df.columns else None,
                'Std_Margin': df['margin'].std() if 'margin' in df.columns else None
            }
            stats_data.append(overall_stats)

            if 'category' in df.columns:
                for category in df['category'].unique():
                    cat_df = df[df['category'] == category]
                    cat_stats = {
                        'Dataset': dataset_name,
                        'Category': category,
                        'Count': len(cat_df),
                        'Avg_Confidence': cat_df['confidence'].mean() if 'confidence' in cat_df.columns else None,
                        'Std_Confidence': cat_df['confidence'].std() if 'confidence' in cat_df.columns else None,
                        'Avg_Margin': cat_df['margin'].mean() if 'margin' in cat_df.columns else None,
                        'Std_Margin': cat_df['margin'].std() if 'margin' in cat_df.columns else None
                    }
                    stats_data.append(cat_stats)

        df = pd.DataFrame(stats_data).round(4)
        output_path = self.output_folder / 'confidence_statistics.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved confidence statistics table to: {output_path}")
        return df

    def create_fog_analysis_table(self):
        """
        Analyze fog-related metrics across datasets.

        Returns:
            DataFrame: Fog analysis table or None
        """
        fog_data = []

        for dataset_name, data in self.datasets.items():
            if data['metadata_df'].empty:
                continue

            df = data['metadata_df']

            if 'is_foggy' in df.columns and 'fog_score' in df.columns:
                fog_stats = {
                    'Dataset': dataset_name,
                    'Total_Images': len(df),
                    'Foggy_Images': int(df['is_foggy'].sum()) if 'is_foggy' in df.columns else 0,
                    'Foggy_Percentage': round(100 * df['is_foggy'].sum() / len(df), 2) if len(df) > 0 else 0,
                    'Avg_Fog_Score': df['fog_score'].mean(),
                    'Std_Fog_Score': df['fog_score'].std(),
                    'Avg_Fog_Margin': df['fog_margin'].mean() if 'fog_margin' in df.columns else None,
                    'Std_Fog_Margin': df['fog_margin'].std() if 'fog_margin' in df.columns else None
                }
                fog_data.append(fog_stats)

        if fog_data:
            df = pd.DataFrame(fog_data).round(4)
            output_path = self.output_folder / 'fog_analysis.csv'
            df.to_csv(output_path, index=False)
            print(f"Saved fog analysis table to: {output_path}")
            return df
        return None

    def create_outdoor_confidence_table(self):
        """
        Analyze outdoor confidence metrics.

        Returns:
            DataFrame: Outdoor confidence table or None
        """
        outdoor_data = []

        for dataset_name, data in self.datasets.items():
            if data['metadata_df'].empty:
                continue

            df = data['metadata_df']

            if 'is_outdoor' in df.columns and 'outdoor_confidence' in df.columns:
                outdoor_stats = {
                    'Dataset': dataset_name,
                    'Total_Images': len(df),
                    'Outdoor_Images': int(df['is_outdoor'].sum()),
                    'Outdoor_Percentage': round(100 * df['is_outdoor'].sum() / len(df), 2) if len(df) > 0 else 0,
                    'Avg_Outdoor_Confidence': df['outdoor_confidence'].mean(),
                    'Std_Outdoor_Confidence': df['outdoor_confidence'].std(),
                    'Min_Outdoor_Confidence': df['outdoor_confidence'].min(),
                    'Max_Outdoor_Confidence': df['outdoor_confidence'].max()
                }
                outdoor_data.append(outdoor_stats)

        if outdoor_data:
            df = pd.DataFrame(outdoor_data).round(4)
            output_path = self.output_folder / 'outdoor_confidence.csv'
            df.to_csv(output_path, index=False)
            print(f"Saved outdoor confidence table to: {output_path}")
            return df
        return None

    def create_category_confusion_analysis(self, dataset_name):
        """
        Analyze which categories have the highest secondary scores (potential confusion).

        Parameters:
            dataset_name (str): Name of the dataset to analyze

        Returns:
            DataFrame: Confusion analysis table or None
        """
        data = self.datasets.get(dataset_name)
        if not data or data['metadata_df'].empty:
            return None

        df = data['metadata_df']
        confusion_data = []

        for _, row in df.iterrows():
            if 'all_scores' in row and isinstance(row['all_scores'], dict):
                assigned_category = row.get('category', 'unknown')
                all_scores = row['all_scores']

                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

                if len(sorted_scores) >= 2:
                    top_category, top_score = sorted_scores[0]
                    second_category, second_score = sorted_scores[1]

                    confusion_data.append({
                        'assigned_category': assigned_category,
                        'top_category': top_category,
                        'top_score': top_score,
                        'second_category': second_category,
                        'second_score': second_score,
                        'score_difference': top_score - second_score
                    })

        if confusion_data:
            df = pd.DataFrame(confusion_data)
            output_path = self.output_folder / f'{dataset_name}_confusion_analysis.csv'
            df.to_csv(output_path, index=False)
            print(f"Saved confusion analysis for {dataset_name} to: {output_path}")
            return df
        return None

    def create_average_category_scores_table(self):
        """
        Create average score matrix showing how each category scores on all weather types.
        """
        for dataset_name, data in self.datasets.items():
            if data['metadata_df'].empty:
                continue

            df = data['metadata_df']

            if 'category' not in df.columns or 'all_scores' not in df.columns:
                continue

            score_matrix = defaultdict(lambda: defaultdict(list))

            for _, row in df.iterrows():
                assigned_category = row['category']
                if isinstance(row['all_scores'], dict):
                    for score_category, score_value in row['all_scores'].items():
                        score_matrix[assigned_category][score_category].append(score_value)

            avg_matrix = {}
            for assigned_cat, scores_dict in score_matrix.items():
                avg_matrix[assigned_cat] = {
                    score_cat: round(np.mean(scores), 4) 
                    for score_cat, scores in scores_dict.items()
                }

            matrix_df = pd.DataFrame(avg_matrix).T
            cols = [c for c in self.weather_categories if c in matrix_df.columns]
            matrix_df = matrix_df[cols]

            output_path = self.output_folder / f'{dataset_name}_category_score_matrix.csv'
            matrix_df.to_csv(output_path)
            print(f"Saved category score matrix for {dataset_name} to: {output_path}")

    # ==================== PLOTTING METHODS ====================

    def plot_category_distribution_bars(self):
        """
        Create bar plots showing category distribution across datasets.
        """
        print("\nGenerating category distribution bar plots...")

        # Prepare data
        distribution_data = []
        for dataset_name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                for category, count in data['stats']['category_counts'].items():
                    distribution_data.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Count': count
                    })

        if not distribution_data:
            print("No distribution data available for plotting")
            return

        df = pd.DataFrame(distribution_data)
        datasets = df['Dataset'].unique()

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(self.weather_categories))
        width = 0.8 / len(datasets)

        for i, dataset in enumerate(datasets):
            dataset_data = df[df['Dataset'] == dataset]
            counts = [dataset_data[dataset_data['Category'] == cat]['Count'].values[0] 
                     if len(dataset_data[dataset_data['Category'] == cat]) > 0 else 0
                     for cat in self.weather_categories]

            ax.bar(x + i * width, counts, width, label=dataset, alpha=0.8)

        ax.set_xlabel('Weather Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Category Distribution Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(self.weather_categories, rotation=45, ha='right')
        ax.legend(title='Dataset')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.plots_folder / 'category_distribution_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved bar plot to: {output_path}")

    def plot_category_distribution_stacked(self):
        """
        Create stacked bar plots showing percentage distribution.
        """
        print("Generating stacked percentage distribution plot...")

        # Prepare percentage data
        percentage_data = {}
        dataset_names = []

        for dataset_name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                dataset_names.append(dataset_name)
                total = data['stats']['total_images']

                for category in self.weather_categories:
                    count = data['stats']['category_counts'].get(category, 0)
                    percentage = 100 * count / total if total > 0 else 0

                    if category not in percentage_data:
                        percentage_data[category] = []
                    percentage_data[category].append(percentage)

        if not percentage_data:
            print("No percentage data available for plotting")
            return

        # Sort datasets alphabetically
        sorted_indices = sorted(range(len(dataset_names)), key=lambda i: dataset_names[i].lower())
        dataset_names = [dataset_names[i] for i in sorted_indices]
        for category in percentage_data:
            percentage_data[category] = [percentage_data[category][i] for i in sorted_indices]

        # Sort weather categories alphabetically by display name
        sorted_categories = sorted(self.weather_categories, 
                                   key=lambda c: self.category_display_names.get(c, c).lower())

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        bottom = np.zeros(len(dataset_names))

        for category in sorted_categories:
            if category in percentage_data:
                values = percentage_data[category]
                color = self.category_colors.get(category, '#808080')
                display_name = self.category_display_names.get(category, category)
                ax.bar(dataset_names, values, bottom=bottom, label=display_name, 
                      color=color, alpha=0.8, edgecolor='white', linewidth=1)
                bottom += values

        ax.set_xlabel('Dataset', fontsize=16, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
        # No title (header removed as requested)
        ax.legend(title='Weather Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', labelsize=14)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = self.plots_folder / 'category_distribution_stacked.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved stacked plot to: {output_path}")

    def plot_confidence_distributions(self):
        """
        Create box plots and histograms for confidence distributions.
        """
        print("Generating confidence distribution plots...")

        # Collect confidence data
        confidence_data = []
        for dataset_name, data in self.datasets.items():
            if not data['metadata_df'].empty and 'confidence' in data['metadata_df'].columns:
                df = data['metadata_df']
                for _, row in df.iterrows():
                    confidence_data.append({
                        'Dataset': dataset_name,
                        'Category': row.get('category', 'unknown'),
                        'Confidence': row['confidence']
                    })

        if not confidence_data:
            print("No confidence data available for plotting")
            return

        conf_df = pd.DataFrame(confidence_data)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Box plot by dataset
        ax1 = fig.add_subplot(gs[0, 0])
        datasets = conf_df['Dataset'].unique()
        data_by_dataset = [conf_df[conf_df['Dataset'] == ds]['Confidence'].values 
                          for ds in datasets]
        # Filter out empty arrays
        valid_datasets = [(ds, data) for ds, data in zip(datasets, data_by_dataset) if len(data) > 0]
        if valid_datasets:
            valid_labels, valid_data = zip(*valid_datasets)
            bp1 = ax1.boxplot(valid_data, labels=valid_labels, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
        ax1.set_xlabel('Dataset', fontweight='bold')
        ax1.set_ylabel('Confidence', fontweight='bold')
        ax1.set_title('Confidence Distribution by Dataset', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Box plot by category
        ax2 = fig.add_subplot(gs[0, 1])
        categories = [cat for cat in self.weather_categories if cat in conf_df['Category'].unique()]
        data_by_category = [conf_df[conf_df['Category'] == cat]['Confidence'].values 
                           for cat in categories]
        # Filter out empty arrays
        valid_categories = [(cat, data) for cat, data in zip(categories, data_by_category) if len(data) > 0]
        if valid_categories:
            valid_cat_labels, valid_cat_data = zip(*valid_categories)
            bp2 = ax2.boxplot(valid_cat_data, labels=valid_cat_labels, patch_artist=True)
            for i, patch in enumerate(bp2['boxes']):
                color = self.category_colors.get(valid_cat_labels[i], '#808080')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax2.set_xlabel('Category', fontweight='bold')
        ax2.set_ylabel('Confidence', fontweight='bold')
        ax2.set_title('Confidence Distribution by Category', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Histogram of all confidences
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(conf_df['Confidence'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.axvline(conf_df['Confidence'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {conf_df["Confidence"].mean():.3f}')
        ax3.axvline(conf_df['Confidence'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {conf_df["Confidence"].median():.3f}')
        ax3.set_xlabel('Confidence', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Overall Confidence Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Violin plot by category
        ax4 = fig.add_subplot(gs[1, 1])
        if valid_categories:
            positions = range(len(valid_cat_labels))
            parts = ax4.violinplot(valid_cat_data, positions=positions, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                color = self.category_colors.get(valid_cat_labels[i], '#808080')
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax4.set_xticks(positions)
            ax4.set_xticklabels(valid_cat_labels, rotation=45, ha='right')
        ax4.set_xlabel('Category', fontweight='bold')
        ax4.set_ylabel('Confidence', fontweight='bold')
        ax4.set_title('Confidence Distribution (Violin Plot)', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.plots_folder / 'confidence_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confidence plots to: {output_path}")

    def plot_margin_analysis(self):
        """
        Create plots analyzing classification margins.
        """
        print("Generating margin analysis plots...")

        margin_data = []
        for dataset_name, data in self.datasets.items():
            if not data['metadata_df'].empty and 'margin' in data['metadata_df'].columns:
                df = data['metadata_df']
                for _, row in df.iterrows():
                    margin_data.append({
                        'Dataset': dataset_name,
                        'Category': row.get('category', 'unknown'),
                        'Margin': row['margin'],
                        'Confidence': row.get('confidence', 0)
                    })

        if not margin_data:
            print("No margin data available for plotting")
            return

        margin_df = pd.DataFrame(margin_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Margin distribution by dataset
        datasets = margin_df['Dataset'].unique()
        data_by_dataset = [margin_df[margin_df['Dataset'] == ds]['Margin'].values 
                          for ds in datasets]
        bp = axes[0, 0].boxplot(data_by_dataset, labels=datasets, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.7)
        axes[0, 0].set_xlabel('Dataset', fontweight='bold')
        axes[0, 0].set_ylabel('Margin', fontweight='bold')
        axes[0, 0].set_title('Classification Margin by Dataset', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Margin vs Confidence scatter
        for dataset in datasets:
            ds_data = margin_df[margin_df['Dataset'] == dataset]
            axes[0, 1].scatter(ds_data['Confidence'], ds_data['Margin'], 
                             alpha=0.5, s=20, label=dataset)
        axes[0, 1].set_xlabel('Confidence', fontweight='bold')
        axes[0, 1].set_ylabel('Margin', fontweight='bold')
        axes[0, 1].set_title('Margin vs Confidence', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # 3. Margin histogram
        axes[1, 0].hist(margin_df['Margin'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(margin_df['Margin'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {margin_df["Margin"].mean():.3f}')
        axes[1, 0].set_xlabel('Margin', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Overall Margin Distribution', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Margin by category
        categories = [cat for cat in self.weather_categories if cat in margin_df['Category'].unique()]
        avg_margins = [margin_df[margin_df['Category'] == cat]['Margin'].mean() 
                      for cat in categories]
        colors = [self.category_colors.get(cat, '#808080') for cat in categories]
        axes[1, 1].bar(categories, avg_margins, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Category', fontweight='bold')
        axes[1, 1].set_ylabel('Average Margin', fontweight='bold')
        axes[1, 1].set_title('Average Margin by Category', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.plots_folder / 'margin_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved margin analysis to: {output_path}")

    def plot_category_pie_charts(self):
        """
        Create pie charts for each dataset showing category proportions.
        """
        print("Generating category pie charts...")

        datasets_with_data = [(name, data) for name, data in self.datasets.items() 
                             if data['stats'] and 'category_counts' in data['stats']]

        if not datasets_with_data:
            print("No data available for pie charts")
            return

        n_datasets = len(datasets_with_data)
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_datasets == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_datasets > 1 else axes

        for idx, (dataset_name, data) in enumerate(datasets_with_data):
            ax = axes[idx] if n_datasets > 1 else axes

            counts = data['stats']['category_counts']
            categories = list(counts.keys())
            values = list(counts.values())
            colors = [self.category_colors.get(cat, '#808080') for cat in categories]

            ax.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                  startangle=90, textprops={'fontsize': 9})
            ax.set_title(f'{dataset_name}\n({data["stats"]["total_images"]} images)', 
                        fontweight='bold')

        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = self.plots_folder / 'category_pie_charts.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pie charts to: {output_path}")

    def plot_heatmap_score_matrix(self):
        """
        Create heatmaps for category score matrices.
        """
        print("Generating category score matrix heatmaps...")

        for dataset_name, data in self.datasets.items():
            if data['metadata_df'].empty:
                continue

            df = data['metadata_df']

            if 'category' not in df.columns or 'all_scores' not in df.columns:
                continue

            # Build score matrix
            score_matrix = defaultdict(lambda: defaultdict(list))

            for _, row in df.iterrows():
                assigned_category = row['category']
                if isinstance(row['all_scores'], dict):
                    for score_category, score_value in row['all_scores'].items():
                        score_matrix[assigned_category][score_category].append(score_value)

            # Calculate averages
            avg_matrix = {}
            for assigned_cat, scores_dict in score_matrix.items():
                avg_matrix[assigned_cat] = {
                    score_cat: np.mean(scores) 
                    for score_cat, scores in scores_dict.items()
                }

            matrix_df = pd.DataFrame(avg_matrix).T
            cols = [c for c in self.weather_categories if c in matrix_df.columns]
            matrix_df = matrix_df[cols]

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(matrix_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Set ticks
            ax.set_xticks(np.arange(len(matrix_df.columns)))
            ax.set_yticks(np.arange(len(matrix_df.index)))
            ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(matrix_df.index)

            # Add text annotations
            for i in range(len(matrix_df.index)):
                for j in range(len(matrix_df.columns)):
                    value = matrix_df.values[i, j]
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha="center", va="center", 
                                 color="black" if value > 0.5 else "white",
                                 fontsize=8)

            ax.set_xlabel('Score Category', fontweight='bold', fontsize=11)
            ax.set_ylabel('Assigned Category', fontweight='bold', fontsize=11)
            ax.set_title(f'Category Score Matrix - {dataset_name}', fontweight='bold', fontsize=13)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Average Score', fontweight='bold')

            plt.tight_layout()
            output_path = self.plots_folder / f'{dataset_name}_score_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved heatmap for {dataset_name} to: {output_path}")

    def plot_fog_analysis(self):
        """
        Create plots for fog detection analysis.
        """
        print("Generating fog analysis plots...")

        fog_data = []
        for dataset_name, data in self.datasets.items():
            if not data['metadata_df'].empty:
                df = data['metadata_df']
                if 'fog_score' in df.columns:
                    for _, row in df.iterrows():
                        fog_data.append({
                            'Dataset': dataset_name,
                            'Category': row.get('category', 'unknown'),
                            'Fog_Score': row['fog_score'],
                            'Is_Foggy': row.get('is_foggy', False),
                            'Fog_Margin': row.get('fog_margin', 0)
                        })

        if not fog_data:
            print("No fog data available for plotting")
            return

        fog_df = pd.DataFrame(fog_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Fog score distribution
        axes[0, 0].hist(fog_df['Fog_Score'], bins=50, color='#95a5a6', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(fog_df['Fog_Score'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {fog_df["Fog_Score"].mean():.3f}')
        axes[0, 0].set_xlabel('Fog Score', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Fog Score Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Fog score by category
        categories = [cat for cat in self.weather_categories if cat in fog_df['Category'].unique()]
        avg_fog_scores = [fog_df[fog_df['Category'] == cat]['Fog_Score'].mean() 
                         for cat in categories]
        colors = [self.category_colors.get(cat, '#808080') for cat in categories]
        axes[0, 1].bar(categories, avg_fog_scores, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Category', fontweight='bold')
        axes[0, 1].set_ylabel('Average Fog Score', fontweight='bold')
        axes[0, 1].set_title('Average Fog Score by Category', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Foggy vs Non-foggy counts by dataset
        datasets = fog_df['Dataset'].unique()
        foggy_counts = [fog_df[(fog_df['Dataset'] == ds) & (fog_df['Is_Foggy'] == True)].shape[0] 
                       for ds in datasets]
        non_foggy_counts = [fog_df[(fog_df['Dataset'] == ds) & (fog_df['Is_Foggy'] == False)].shape[0] 
                           for ds in datasets]

        x = np.arange(len(datasets))
        width = 0.35
        axes[1, 0].bar(x - width/2, foggy_counts, width, label='Foggy', color='#7f8c8d', alpha=0.8)
        axes[1, 0].bar(x + width/2, non_foggy_counts, width, label='Non-Foggy', color='#3498db', alpha=0.8)
        axes[1, 0].set_xlabel('Dataset', fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontweight='bold')
        axes[1, 0].set_title('Foggy vs Non-Foggy Images by Dataset', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Fog margin distribution
        if 'Fog_Margin' in fog_df.columns:
            axes[1, 1].hist(fog_df['Fog_Margin'], bins=50, color='#34495e', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
            axes[1, 1].set_xlabel('Fog Margin', fontweight='bold')
            axes[1, 1].set_ylabel('Frequency', fontweight='bold')
            axes[1, 1].set_title('Fog Margin Distribution', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.plots_folder / 'fog_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved fog analysis to: {output_path}")

    def plot_outdoor_confidence_analysis(self):
        """
        Create plots for outdoor confidence analysis.
        """
        print("Generating outdoor confidence plots...")

        outdoor_data = []
        for dataset_name, data in self.datasets.items():
            if not data['metadata_df'].empty:
                df = data['metadata_df']
                if 'outdoor_confidence' in df.columns:
                    for _, row in df.iterrows():
                        outdoor_data.append({
                            'Dataset': dataset_name,
                            'Category': row.get('category', 'unknown'),
                            'Outdoor_Confidence': row['outdoor_confidence'],
                            'Is_Outdoor': row.get('is_outdoor', False)
                        })

        if not outdoor_data:
            print("No outdoor confidence data available for plotting")
            return

        outdoor_df = pd.DataFrame(outdoor_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Outdoor confidence distribution
        axes[0, 0].hist(outdoor_df['Outdoor_Confidence'], bins=50, color='#27ae60', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(outdoor_df['Outdoor_Confidence'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {outdoor_df["Outdoor_Confidence"].mean():.3f}')
        axes[0, 0].set_xlabel('Outdoor Confidence', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title('Outdoor Confidence Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Outdoor confidence by category
        categories = [cat for cat in self.weather_categories if cat in outdoor_df['Category'].unique()]
        avg_outdoor_conf = [outdoor_df[outdoor_df['Category'] == cat]['Outdoor_Confidence'].mean() 
                           for cat in categories]
        colors = [self.category_colors.get(cat, '#808080') for cat in categories]
        axes[0, 1].bar(categories, avg_outdoor_conf, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Category', fontweight='bold')
        axes[0, 1].set_ylabel('Average Outdoor Confidence', fontweight='bold')
        axes[0, 1].set_title('Average Outdoor Confidence by Category', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Outdoor vs Indoor counts by dataset
        datasets = outdoor_df['Dataset'].unique()
        outdoor_counts = [outdoor_df[(outdoor_df['Dataset'] == ds) & (outdoor_df['Is_Outdoor'] == True)].shape[0] 
                         for ds in datasets]
        indoor_counts = [outdoor_df[(outdoor_df['Dataset'] == ds) & (outdoor_df['Is_Outdoor'] == False)].shape[0] 
                        for ds in datasets]

        x = np.arange(len(datasets))
        width = 0.35
        axes[1, 0].bar(x - width/2, outdoor_counts, width, label='Outdoor', color='#2ecc71', alpha=0.8)
        axes[1, 0].bar(x + width/2, indoor_counts, width, label='Indoor', color='#e67e22', alpha=0.8)
        axes[1, 0].set_xlabel('Dataset', fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontweight='bold')
        axes[1, 0].set_title('Outdoor vs Indoor Images by Dataset', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Box plot by dataset
        data_by_dataset = [outdoor_df[outdoor_df['Dataset'] == ds]['Outdoor_Confidence'].values 
                          for ds in datasets]
        bp = axes[1, 1].boxplot(data_by_dataset, labels=datasets, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#16a085')
            patch.set_alpha(0.7)
        axes[1, 1].set_xlabel('Dataset', fontweight='bold')
        axes[1, 1].set_ylabel('Outdoor Confidence', fontweight='bold')
        axes[1, 1].set_title('Outdoor Confidence Distribution by Dataset', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.plots_folder / 'outdoor_confidence_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved outdoor confidence analysis to: {output_path}")

    def plot_dataset_comparison_summary(self):
        """
        Create a comprehensive comparison summary across all datasets.
        """
        print("Generating dataset comparison summary...")

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Total images comparison
        ax1 = fig.add_subplot(gs[0, 0])
        dataset_names = []
        total_images = []
        for name, data in self.datasets.items():
            if data['stats']:
                dataset_names.append(name)
                total_images.append(data['stats']['total_images'])

        ax1.barh(dataset_names, total_images, color='#3498db', alpha=0.7)
        ax1.set_xlabel('Total Images', fontweight='bold')
        ax1.set_title('Dataset Sizes', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 2. Category diversity (number of categories with >0 images)
        ax2 = fig.add_subplot(gs[0, 1])
        category_diversity = []
        for name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                diversity = sum(1 for count in data['stats']['category_counts'].values() if count > 0)
                category_diversity.append(diversity)

        ax2.bar(range(len(dataset_names)), category_diversity, color='#e74c3c', alpha=0.7)
        ax2.set_xticks(range(len(dataset_names)))
        ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax2.set_ylabel('Number of Categories', fontweight='bold')
        ax2.set_title('Category Diversity', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Average confidence by dataset
        ax3 = fig.add_subplot(gs[0, 2])
        avg_confidences = []
        for name in dataset_names:
            data = self.datasets[name]
            if not data['metadata_df'].empty and 'confidence' in data['metadata_df'].columns:
                avg_confidences.append(data['metadata_df']['confidence'].mean())
            else:
                avg_confidences.append(0)

        ax3.bar(range(len(dataset_names)), avg_confidences, color='#2ecc71', alpha=0.7)
        ax3.set_xticks(range(len(dataset_names)))
        ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax3.set_ylabel('Average Confidence', fontweight='bold')
        ax3.set_title('Average Classification Confidence', fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Category balance (entropy or std deviation)
        ax4 = fig.add_subplot(gs[1, 0])
        balance_scores = []
        for name, data in self.datasets.items():
            if data['stats'] and 'category_counts' in data['stats']:
                counts = list(data['stats']['category_counts'].values())
                total = sum(counts)
                proportions = [c/total for c in counts if c > 0]
                # Calculate entropy as balance measure
                entropy = -sum(p * np.log(p) for p in proportions if p > 0)
                balance_scores.append(entropy)

        ax4.bar(range(len(dataset_names)), balance_scores, color='#9b59b6', alpha=0.7)
        ax4.set_xticks(range(len(dataset_names)))
        ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax4.set_ylabel('Entropy', fontweight='bold')
        ax4.set_title('Dataset Balance (higher = more balanced)', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # 5. Foggy image percentage (if available)
        ax5 = fig.add_subplot(gs[1, 1])
        foggy_percentages = []
        for name in dataset_names:
            data = self.datasets[name]
            if not data['metadata_df'].empty and 'is_foggy' in data['metadata_df'].columns:
                total = len(data['metadata_df'])
                foggy = data['metadata_df']['is_foggy'].sum()
                foggy_percentages.append(100 * foggy / total if total > 0 else 0)
            else:
                foggy_percentages.append(0)

        ax5.bar(range(len(dataset_names)), foggy_percentages, color='#95a5a6', alpha=0.7)
        ax5.set_xticks(range(len(dataset_names)))
        ax5.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax5.set_ylabel('Percentage (%)', fontweight='bold')
        ax5.set_title('Foggy Images Percentage', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # 6. Average margin by dataset
        ax6 = fig.add_subplot(gs[1, 2])
        avg_margins = []
        for name in dataset_names:
            data = self.datasets[name]
            if not data['metadata_df'].empty and 'margin' in data['metadata_df'].columns:
                avg_margins.append(data['metadata_df']['margin'].mean())
            else:
                avg_margins.append(0)

        ax6.bar(range(len(dataset_names)), avg_margins, color='#f39c12', alpha=0.7)
        ax6.set_xticks(range(len(dataset_names)))
        ax6.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax6.set_ylabel('Average Margin', fontweight='bold')
        ax6.set_title('Average Classification Margin', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)

        plt.suptitle('Dataset Comparison Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = self.plots_folder / 'dataset_comparison_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved dataset comparison summary to: {output_path}")

    def generate_all_plots(self):
        """
        Generate all visualization plots.
        """
        print("\n" + "=" * 80)
        print("GENERATING ALL PLOTS")
        print("=" * 80)

        try:
            self.plot_category_distribution_bars()
        except Exception as e:
            print(f"Error generating category distribution bars: {e}")

        try:
            self.plot_category_distribution_stacked()
        except Exception as e:
            print(f"Error generating stacked distribution: {e}")

        try:
            self.plot_category_pie_charts()
        except Exception as e:
            print(f"Error generating pie charts: {e}")

        try:
            self.plot_confidence_distributions()
        except Exception as e:
            print(f"Error generating confidence distributions: {e}")

        try:
            self.plot_margin_analysis()
        except Exception as e:
            print(f"Error generating margin analysis: {e}")

        try:
            self.plot_heatmap_score_matrix()
        except Exception as e:
            print(f"Error generating score heatmaps: {e}")

        try:
            self.plot_fog_analysis()
        except Exception as e:
            print(f"Error generating fog analysis: {e}")

        try:
            self.plot_outdoor_confidence_analysis()
        except Exception as e:
            print(f"Error generating outdoor confidence analysis: {e}")

        try:
            self.plot_dataset_comparison_summary()
        except Exception as e:
            print(f"Error generating dataset comparison summary: {e}")

        print("\n" + "=" * 80)
        print("ALL PLOTS GENERATED!")
        print("=" * 80)

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of all analyses.

        Returns:
            str: Summary report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("WEATHER DATASET ANALYSIS SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Parent Folder: {self.parent_folder}")
        report_lines.append(f"Output Folder: {self.output_folder}")
        report_lines.append(f"Plots Folder: {self.plots_folder}")
        report_lines.append(f"\nTotal Datasets Analyzed: {len(self.datasets)}")
        report_lines.append("\n" + "-" * 80)

        for dataset_name, data in self.datasets.items():
            report_lines.append(f"\nDataset: {dataset_name}")
            report_lines.append("-" * 40)

            if data['stats']:
                stats = data['stats']
                report_lines.append(f"  Total Images: {stats.get('total_images', 'N/A')}")
                report_lines.append(f"  Total Categories: {stats.get('total_categories', 'N/A')}")
                report_lines.append(f"  Confidence Threshold: {stats.get('confidence_threshold', 'N/A')}")
                report_lines.append(f"  Margin Threshold: {stats.get('margin_threshold', 'N/A')}")

                if 'category_counts' in stats:
                    report_lines.append("\n  Category Distribution:")
                    for category, count in sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True):
                        percentage = 100 * count / stats['total_images'] if stats['total_images'] > 0 else 0
                        report_lines.append(f"    {category:12s}: {count:5d} ({percentage:5.2f}%)")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("GENERATED CSV FILES:")
        report_lines.append("=" * 80)

        csv_files = sorted(self.output_folder.glob('*.csv'))
        for i, file_path in enumerate(csv_files, 1):
            report_lines.append(f"{i:2d}. {file_path.name}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("GENERATED PLOT FILES:")
        report_lines.append("=" * 80)

        plot_files = sorted(self.plots_folder.glob('*.png'))
        for i, file_path in enumerate(plot_files, 1):
            report_lines.append(f"{i:2d}. {file_path.name}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        report_path = self.output_folder / 'analysis_summary_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)
        print(f"\nSummary report saved to: {report_path}")

        return report_text

    def generate_all_visualizations(self):
        """
        Main method to generate all tables, plots, and visualizations.
        """
        print("\n" + "=" * 80)
        print("GENERATING ALL VISUALIZATIONS AND TABLES")
        print("=" * 80)

        # Generate CSV tables
        print("\nGenerating CSV tables...")
        self.create_category_distribution_table()
        self.create_category_percentages_table()
        self.create_confidence_statistics_table()
        self.create_fog_analysis_table()
        self.create_outdoor_confidence_table()
        self.create_average_category_scores_table()

        # Generate confusion analysis for each dataset
        for dataset_name in self.datasets.keys():
            self.create_category_confusion_analysis(dataset_name)

        # Generate all plots
        self.generate_all_plots()

        # Generate summary report
        self.generate_summary_report()

        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 80)


def main():
    """
    Main execution function with example usage.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python weather_dataset_visualizer.py <parent_folder_path>")
        print("\nExample:")
        print("  python weather_dataset_visualizer.py /path/to/datasets")
        sys.exit(1)

    parent_folder = sys.argv[1]

    if not os.path.exists(parent_folder):
        print(f"Error: Folder '{parent_folder}' does not exist!")
        sys.exit(1)

    visualizer = WeatherDatasetVisualizer(parent_folder)
    visualizer.analyze_all_datasets()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
