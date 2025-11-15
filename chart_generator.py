"""
Chart Generator for Automated Report Generation
Generates (chart PNG, JSON) pairs with matching filenames
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ChartGenerator:
    """
    Generate charts with corresponding JSON metadata files.
    Each chart is saved as a PNG with a matching JSON file.
    """
    
    SUPPORTED_CHART_TYPES = ['line', 'bar', 'pie', 'scatter', 'horizontal_bar', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'cumulative_distribution', 'time_series_histogram','hist2d', 'cohere', 'signal_pair', 'timeline', 'heatmap', 'streamplot']
    
    def __init__(self, output_dir: str = "./charts"):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory where chart files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        filename_root: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[str, str]:
        """
        Generate a chart and save both PNG and JSON files.
        
        Args:
            chart_type: Type of chart ('line', 'bar', 'pie', 'scatter', 'horizontal_bar', 'grouped_bar', 'stacked_bar', 'box', 'area', 'discrete_distribution', 'hist2d', 'cohere', 'signal_pair')
            data: Chart data dictionary (format depends on chart_type)
            filename_root: Base filename without extension
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            metadata: Additional metadata to include in JSON
            **kwargs: Additional arguments passed to the chart function
                - auto_generate_signal_pair: (bool, default=True) For 'cohere' charts, 
                  automatically generate a companion signal_pair chart
                - Other kwargs are chart-specific
            
        Returns:
            Tuple of (png_path, json_path)
            
        Note:
            When chart_type='cohere', a companion signal_pair chart is automatically
            generated with filename '{filename_root}_signals' unless 
            auto_generate_signal_pair=False is specified.
        """
        if chart_type not in self.SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}. "
                           f"Supported types: {self.SUPPORTED_CHART_TYPES}")
        
        # Generate the chart
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        chart_method = getattr(self, f"_create_{chart_type}_chart")
        chart_method(ax, data, title, xlabel, ylabel, **kwargs)
        
        # Save PNG
        png_path = self.output_dir / f"{filename_root}.png"
        plt.tight_layout()
        # Add extra space at bottom for rotated labels
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(png_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
        
        # Create and save JSON metadata
        json_data = {
            "filename_root": filename_root,
            "chart_type": chart_type,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "data": data,
            "generated_at": datetime.now().isoformat(),
            "png_file": f"{filename_root}.png",
            "metadata": metadata or {}
        }
        
        json_path = self.output_dir / f"{filename_root}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Auto-generate companion signal_pair chart for coherence
        if chart_type == 'cohere' and kwargs.get('auto_generate_signal_pair', True):
            # Generate companion signal pair chart
            signal_pair_filename = f"{filename_root}_signals"
            signal_pair_title = title.replace('Coherence', 'Signals').replace('coherence', 'signals')
            if signal_pair_title == title:  # If no replacement happened
                signal_pair_title = f"Time Domain: {title}"
            
            try:
                companion_png, companion_json = self.generate_chart(
                    chart_type='signal_pair',
                    data=data,
                    filename_root=signal_pair_filename,
                    title=signal_pair_title,
                    xlabel='Time (seconds)',
                    ylabel='Amplitude',
                    metadata=metadata,
                    auto_generate_signal_pair=False,  # Prevent recursion
                    **{k: v for k, v in kwargs.items() if k != 'auto_generate_signal_pair'}
                )
                print(f"  ✓ Auto-generated companion signal pair: {signal_pair_filename}.png")
            except Exception as e:
                # Don't fail the main chart if signal pair generation fails
                print(f"  ⚠ Could not auto-generate signal pair chart: {e}")
        
        return str(png_path), str(json_path)
    
    def _create_line_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a line chart."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        if isinstance(y[0], list):
            # Multiple lines
            labels = data.get('labels', [f"Series {i+1}" for i in range(len(y))])
            for i, y_series in enumerate(y):
                ax.plot(x, y_series, marker='o', label=labels[i], 
                       linewidth=kwargs.get('linewidth', 2))
            ax.legend()
        else:
            # Single line
            ax.plot(x, y, marker='o', linewidth=kwargs.get('linewidth', 2))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _create_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a vertical bar chart."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        colors = kwargs.get('colors', None)
        ax.bar(x, y, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if they're strings and there are many
        if len(x) > 5 or any(len(str(label)) > 8 for label in x):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add extra bottom padding if labels are rotated
        if len(x) > 5:
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_horizontal_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a horizontal bar chart."""
        categories = data.get('categories', [])
        values = data.get('values', [])
        
        colors = kwargs.get('colors', None)
        ax.barh(categories, values, color=colors, alpha=0.8, 
                edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_discrete_distribution_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a discrete probability distribution chart as horizontal bars.
        Perfect for showing probability mass functions, frequency distributions,
        or categorical probability distributions.
        
        Data format:
        {
            'values': [0, 1, 2, 3, 4, 5],  # Discrete values (x-axis in vertical orientation)
            'probabilities': [0.05, 0.15, 0.25, 0.30, 0.20, 0.05],  # P(X=x)
            'labels': ['X=0', 'X=1', 'X=2', 'X=3', 'X=4', 'X=5']  # Optional custom labels
        }
        
        OR for named categories:
        {
            'categories': ['Excellent', 'Good', 'Fair', 'Poor'],
            'probabilities': [0.30, 0.40, 0.20, 0.10]
        }
        """
        import numpy as np
        
        # Check if we have categories or numeric values
        if 'categories' in data:
            categories = data['categories']
            labels = categories
        else:
            values = data.get('values', [])
            labels = data.get('labels', [f'X={v}' for v in values])
            categories = labels
        
        probabilities = data.get('probabilities', [])
        
        # Validate probabilities sum to approximately 1.0 (allow small rounding errors)
        prob_sum = sum(probabilities)
        if not (0.99 <= prob_sum <= 1.01):
            # Normalize if not already normalized
            probabilities = [p / prob_sum for p in probabilities]
        
        # Create color gradient based on probability values
        colors = kwargs.get('colors', None)
        if colors is None:
            # Create a color gradient from light to dark blue based on probability
            cmap = plt.cm.Blues
            norm = plt.Normalize(vmin=min(probabilities), vmax=max(probabilities))
            colors = [cmap(norm(p)) for p in probabilities]
        
        # Create horizontal bars
        y_positions = np.arange(len(categories))
        bars = ax.barh(y_positions, probabilities, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1)
        
        # Add probability labels on the bars
        show_values = kwargs.get('show_values', True)
        if show_values:
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                # Position label inside bar if it's wide enough, otherwise outside
                if width > 0.1:
                    ax.text(width - 0.02, bar.get_y() + bar.get_height()/2,
                           f'{prob:.3f}',
                           ha='right', va='center', fontsize=10, fontweight='bold',
                           color='white' if prob > 0.5 else 'black')
                else:
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.3f}',
                           ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Set labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel or 'Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set x-axis limits to show full probability range
        ax.set_xlim(0, max(1.0, max(probabilities) * 1.1))
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add a vertical line at x=1.0 to show the full probability
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Total = 1.0')
        
        # Invert y-axis so highest value is at top (optional, controlled by kwargs)
        if kwargs.get('invert_y', False):
            ax.invert_yaxis()
        
        # Add mean/expected value line if requested
        if kwargs.get('show_expected_value', False) and 'values' in data:
            values = data['values']
            expected_value = sum(v * p for v, p in zip(values, probabilities))
            # Add as text annotation
            ax.text(0.98, 0.02, f'E[X] = {expected_value:.2f}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
    
    def _create_cumulative_distribution_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a cumulative distribution function (CDF) chart.
        Shows cumulative probability: P(X ≤ x)
        
        Data format:
        {
            'values': [0, 1, 2, 3, 4, 5],  # Discrete values (x-axis)
            'probabilities': [0.05, 0.15, 0.25, 0.30, 0.20, 0.05],  # P(X=x)
            'labels': ['X=0', 'X=1', ...],  # Optional custom labels
        }
        
        OR for continuous approximation:
        {
            'x': [0, 1, 2, 3, ...],  # Values
            'cdf': [0.05, 0.20, 0.45, 0.75, ...]  # Pre-calculated CDF values
        }
        """
        import numpy as np
        
        # Check if CDF is pre-calculated or needs to be computed
        if 'cdf' in data and 'x' in data:
            # Pre-calculated CDF with x values
            x_values = np.array(data.get('x', []))
            cdf_values = np.array(data.get('cdf', []))
        elif 'cdf' in data and 'values' in data:
            # CDF calculated from discrete distribution with numeric values
            x_values = np.array(data.get('values', []))
            cdf_values = np.array(data.get('cdf', []))
        elif 'cdf' in data and 'categories' in data:
            # CDF calculated from categorical distribution (e.g., ratings, scores)
            # Use index positions for categories
            categories = data.get('categories', [])
            x_values = np.arange(len(categories))
            cdf_values = np.array(data.get('cdf', []))
            # Store categories for later use in labels
            category_labels = categories
        else:
            # Calculate CDF from probabilities
            values = np.array(data.get('values', []))
            probabilities = np.array(data.get('probabilities', []))
            
            # Normalize probabilities if needed
            prob_sum = sum(probabilities)
            if not (0.99 <= prob_sum <= 1.01):
                probabilities = probabilities / prob_sum
            
            # Calculate cumulative distribution
            cdf_values = np.cumsum(probabilities)
            x_values = values
        
        # Plot style
        plot_style = kwargs.get('plot_style', 'step')  # 'step', 'line', or 'both'
        
        if plot_style == 'step' or plot_style == 'both':
            # Step plot (typical for discrete distributions)
            ax.step(x_values, cdf_values, where='post', linewidth=2.5, 
                   color='#2E86AB', label='CDF', alpha=0.9)
            
            # Add markers at data points
            ax.plot(x_values, cdf_values, 'o', color='#2E86AB', 
                   markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        if plot_style == 'line' or plot_style == 'both':
            # Smooth line (for continuous approximation)
            ax.plot(x_values, cdf_values, linewidth=2.5, 
                   color='#A23B72', linestyle='--', alpha=0.7, label='Smooth CDF')
        
        # Add horizontal line at y=1.0
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, 
                  alpha=0.5, label='Total = 1.0')
        
        # Add horizontal line at y=0.5 (median)
        ax.axhline(y=0.5, color='green', linestyle=':', linewidth=1, 
                  alpha=0.5, label='Median (50%)')
        
        # Shade area under curve
        if kwargs.get('shade_area', True):
            ax.fill_between(x_values, 0, cdf_values, alpha=0.2, color='#2E86AB', 
                           step='post' if plot_style == 'step' else None)
        
        # Add value labels at key points
        if kwargs.get('show_values', True):
            # Show values at quartiles
            quartile_indices = [int(len(cdf_values) * q) for q in [0.25, 0.5, 0.75]]
            for idx in quartile_indices:
                if 0 <= idx < len(x_values):
                    ax.annotate(f'{cdf_values[idx]:.2f}',
                              xy=(x_values[idx], cdf_values[idx]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='yellow', alpha=0.5))
        
        # Set labels and title
        ax.set_xlabel(xlabel or 'Value', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(min(x_values) - 0.5, max(x_values) + 0.5)
        ax.set_ylim(-0.05, 1.1)
        
        # Set x-tick labels for categorical data
        if 'categories' in data and 'category_labels' in locals():
            ax.set_xticks(x_values)
            ax.set_xticklabels(category_labels)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        if plot_style == 'both' or kwargs.get('show_legend', True):
            ax.legend(loc='lower right', fontsize=10)
        
        # Add statistics text box
        if kwargs.get('show_stats', True):
            # Check if we have numeric values for statistics
            if 'values' in data and 'probabilities' in data:
                values = np.array(data['values'])
                probs = np.array(data['probabilities'])
                
                mean = np.sum(values * probs)
                variance = np.sum((values - mean)**2 * probs)
                std_dev = np.sqrt(variance)
                
                # Find median (where CDF ≈ 0.5)
                median_idx = np.argmin(np.abs(cdf_values - 0.5))
                median = x_values[median_idx]
                
                stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}'
                ax.text(0.02, 0.98, stats_text,
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, family='monospace')
            elif 'categories' in data:
                # For categorical data, just show median category
                median_idx = np.argmin(np.abs(cdf_values - 0.5))
                if 'category_labels' in locals():
                    median_cat = category_labels[median_idx]
                    stats_text = f'Median: {median_cat}'
                    ax.text(0.02, 0.98, stats_text,
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=9, family='monospace')
    
    def _create_time_series_histogram_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a time series histogram showing distribution evolution over time.
        
        Data format:
        {
            'time_points': [0, 1, 2, 3, ...],  # Time steps or dates
            'data_series': [                    # Data at each time point
                [10, 15, 12, 18, ...],         # Data at time 0
                [12, 16, 14, 20, ...],         # Data at time 1
                ...
            ],
            'bins': 20,  # Optional: number of bins (default: auto)
            'labels': ['Jan', 'Feb', ...]  # Optional: time labels
        }
        """
        import numpy as np
        
        time_points = data.get('time_points', [])
        data_series = data.get('data_series', [])
        num_bins = data.get('bins', kwargs.get('bins', 20))
        time_labels = data.get('labels', None)
        
        # Determine global min/max for consistent binning
        all_data = np.concatenate(data_series)
        data_min, data_max = all_data.min(), all_data.max()
        
        # Create bin edges
        bin_edges = np.linspace(data_min, data_max, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate histogram for each time point
        histogram_matrix = np.zeros((num_bins, len(time_points)))
        
        for i, time_data in enumerate(data_series):
            counts, _ = np.histogram(time_data, bins=bin_edges)
            histogram_matrix[:, i] = counts
        
        # Create meshgrid for pcolormesh
        T, Y = np.meshgrid(time_points, bin_centers)
        
        # Plot using pcolormesh
        cmap = kwargs.get('cmap', 'viridis')
        pcm = ax.pcolormesh(T, Y, histogram_matrix, 
                           cmap=cmap, 
                           shading='auto',
                           alpha=0.9)
        
        # Add colorbar
        cbar = ax.figure.colorbar(pcm, ax=ax, label='Frequency')
        cbar.ax.tick_params(labelsize=10)
        
        # Set labels and title
        ax.set_xlabel(xlabel or 'Time', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set custom time labels if provided
        if time_labels and len(time_labels) == len(time_points):
            ax.set_xticks(time_points)
            ax.set_xticklabels(time_labels, rotation=45, ha='right')
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle='--', color='white', linewidth=0.5)
        
        # Optional: Add contour lines to show density
        if kwargs.get('show_contours', True):
            contour_levels = kwargs.get('contour_levels', 5)
            contours = ax.contour(T, Y, histogram_matrix, 
                                 levels=contour_levels,
                                 colors='white', 
                                 alpha=0.4, 
                                 linewidths=1)
        
        # Optional: Overlay mean line
        if kwargs.get('show_mean', True):
            means = [np.mean(time_data) for time_data in data_series]
            ax.plot(time_points, means, 'r-', linewidth=2.5, 
                   label='Mean', alpha=0.8)
            ax.legend(loc='upper right', fontsize=10)
        
        # Optional: Overlay median line
        if kwargs.get('show_median', False):
            medians = [np.median(time_data) for time_data in data_series]
            ax.plot(time_points, medians, 'y--', linewidth=2.5, 
                   label='Median', alpha=0.8)
            ax.legend(loc='upper right', fontsize=10)
    
    def _create_pie_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a pie chart."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        colors = kwargs.get('colors', None)
        explode = data.get('explode', None)
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors, explode=explode, textprops={'fontsize': 10})
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
    
    def _create_scatter_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """Create a scatter plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        
        sizes = data.get('sizes', kwargs.get('marker_size', 50))
        colors = data.get('colors', kwargs.get('color', 'blue'))
        
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _create_grouped_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a grouped bar chart for multi-dimensional data.
        
        Data format:
        {
            'categories': ['Q1', 'Q2', 'Q3', 'Q4'],  # x-axis categories (e.g., time periods, regions)
            'groups': ['Product A', 'Product B', 'Product C'],  # different groups
            'values': [
                [100, 120, 115, 140],  # Product A values for each category
                [80, 95, 110, 105],     # Product B values
                [60, 70, 85, 95]        # Product C values
            ]
        }
        """
        import numpy as np
        
        categories = data.get('categories', [])
        groups = data.get('groups', [])
        values = data.get('values', [])
        
        x = np.arange(len(categories))
        width = 0.8 / len(groups)  # Width of bars
        
        colors = kwargs.get('colors', None)
        if not colors:
            # Default color palette
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Plot each group
        for i, (group_name, group_values) in enumerate(zip(groups, values)):
            offset = (i - len(groups)/2 + 0.5) * width
            ax.bar(x + offset, group_values, width, label=group_name,
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(categories) > 5 or any(len(str(cat)) > 8 for cat in categories):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_stacked_bar_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a stacked bar chart for multi-dimensional data.
        
        Data format:
        {
            'categories': ['Q1', 'Q2', 'Q3', 'Q4'],  # x-axis categories (e.g., time periods, regions)
            'groups': ['Product A', 'Product B', 'Product C'],  # different groups
            'values': [
                [100, 120, 115, 140],  # Product A values for each category
                [80, 95, 110, 105],     # Product B values
                [60, 70, 85, 95]        # Product C values
            ]
        }
        """
        import numpy as np
        
        categories = data.get('categories', [])
        groups = data.get('groups', [])
        values = data.get('values', [])
        
        x = np.arange(len(categories))
        
        colors = kwargs.get('colors', None)
        if not colors:
            # Default color palette
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Plot each group as a stacked segment
        bottom = np.zeros(len(categories))
        for i, (group_name, group_values) in enumerate(zip(groups, values)):
            ax.bar(x, group_values, bottom=bottom, label=group_name,
                  color=colors[i % len(colors)], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += np.array(group_values)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(categories) > 5 or any(len(str(cat)) > 8 for cat in categories):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_box_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a box plot (box and whisker chart) for distribution analysis.
        
        Data format:
        {
            'labels': ['Group A', 'Group B', 'Group C'],  # Category labels
            'data': [
                [10, 15, 13, 17, 20, 25, 18, 16],  # Group A data points
                [8, 12, 11, 14, 19, 16, 13, 15],    # Group B data points
                [12, 18, 15, 20, 22, 19, 17, 21]    # Group C data points
            ]
        }
        """
        labels = data.get('labels', [])
        values = data.get('data', [])
        
        # Create box plot
        box_parts = ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,  # Fill boxes with color
            notch=kwargs.get('notch', False),  # Add notch for median confidence interval
            showmeans=kwargs.get('showmeans', True),  # Show mean as well as median
            meanline=kwargs.get('meanline', False)  # Show mean as line or point
        )
        
        # Color the boxes
        colors = kwargs.get('colors', None)
        if not colors:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style the other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            if element in box_parts:
                plt.setp(box_parts[element], color='black', linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if len(labels) > 5 or any(len(str(label)) > 8 for label in labels):
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.gcf().subplots_adjust(bottom=0.2)
    
    def _create_area_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create an area chart using fill_between.
        
        Data format:
        {
            'x': [1, 2, 3, 4, 5],  # x-axis values
            'y': [10, 15, 13, 17, 20],  # Single area (fills from 0)
            
            # OR for range/band:
            'y1': [10, 12, 11, 14, 16],  # Lower bound
            'y2': [15, 18, 16, 20, 22]   # Upper bound
            
            # OR for multiple areas (stacked, stacked_100, overlapping):
            'areas': [
                {'y': [10, 15, 13, 17, 20], 'label': 'Series 1'},
                {'y': [5, 8, 7, 9, 11], 'label': 'Series 2'}
            ],
            'area_type': 'stacked' | 'stacked_100' | 'overlapping'
            
            # OR for step area:
            'y': [10, 15, 13, 17, 20],
            'step': True  # Creates step transitions
        }
        """
        import numpy as np
        x = data.get('x', [])
        
        colors = kwargs.get('colors', None)
        if not colors:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#FFD93D', '#6BCB77', '#C56CF0', '#17C0EB', '#F8B739']
        
        # Determine if step mode
        is_step = data.get('step', False)
        step_where = 'post' if is_step else None
        
        # Check which type of area chart
        if 'areas' in data:
            areas = data['areas']
            area_type = data.get('area_type', 'stacked')  # Default to stacked
            
            if area_type == 'stacked':
                # Traditional stacked areas
                for i, area in enumerate(areas):
                    y = area['y']
                    label = area.get('label', f'Series {i+1}')
                    ax.fill_between(x, y, alpha=0.7, color=colors[i % len(colors)], 
                                   label=label, step=step_where)
                    ax.plot(x, y, color=colors[i % len(colors)], linewidth=2)
                ax.legend(loc='best', fontsize=10)
                
            elif area_type == 'stacked_100':
                # 100% stacked - normalize to percentages
                # Convert to numpy arrays
                y_arrays = [np.array(area['y']) for area in areas]
                
                # Calculate total at each x point
                totals = np.zeros(len(x))
                for y_arr in y_arrays:
                    totals += y_arr
                
                # Calculate percentages
                percentages = []
                for y_arr in y_arrays:
                    pct = (y_arr / totals) * 100
                    percentages.append(pct)
                
                # Plot stacked from bottom to top
                bottom = np.zeros(len(x))
                for i, (pct, area) in enumerate(zip(percentages, areas)):
                    label = area.get('label', f'Series {i+1}')
                    ax.fill_between(x, bottom, bottom + pct, alpha=0.7, 
                                   color=colors[i % len(colors)], label=label,
                                   step=step_where)
                    ax.plot(x, bottom + pct, color=colors[i % len(colors)], linewidth=1.5)
                    bottom += pct
                
                ax.set_ylim(0, 100)
                ax.set_ylabel('Percentage (%)', fontsize=12)
                ax.legend(loc='best', fontsize=10)
                
            elif area_type == 'overlapping':
                # Overlapping areas with transparency
                for i, area in enumerate(areas):
                    y = area['y']
                    label = area.get('label', f'Series {i+1}')
                    ax.fill_between(x, 0, y, alpha=0.4, color=colors[i % len(colors)], 
                                   label=label, step=step_where)
                    ax.plot(x, y, color=colors[i % len(colors)], linewidth=2.5)
                ax.legend(loc='best', fontsize=10)
            
        elif 'y1' in data and 'y2' in data:
            # Range/band chart (between y1 and y2)
            y1 = data['y1']
            y2 = data['y2']
            ax.fill_between(x, y1, y2, alpha=0.3, color=colors[0], label='Range',
                           step=step_where)
            ax.plot(x, y1, color=colors[0], linewidth=2, linestyle='--', label='Lower Bound')
            ax.plot(x, y2, color=colors[1], linewidth=2, linestyle='--', label='Upper Bound')
            # Calculate and plot average
            avg = [(a + b) / 2 for a, b in zip(y1, y2)]
            ax.plot(x, avg, color='black', linewidth=2.5, label='Average')
            ax.legend(loc='best', fontsize=10)
            
        else:
            # Single area (from 0 to y)
            y = data.get('y', [])
            ax.fill_between(x, 0, y, alpha=0.6, color=colors[0], step=step_where)
            ax.plot(x, y, color=colors[0], linewidth=2.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        if 'y1' not in data:  # Don't override ylabel for stacked_100
            ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Start y-axis at 0 for better visualization (except stacked_100)
        if data.get('area_type') != 'stacked_100':
            ax.set_ylim(bottom=0)
    
    def _create_hist2d_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a 2D histogram (heatmap) showing the density of points in 2D space.
        Perfect for visualizing the distribution and concentration of bivariate data.
        
        Data format:
        {
            'x': [1, 2, 3, 4, ...],  # x-coordinates of points
            'y': [5, 6, 7, 8, ...],  # y-coordinates of points
            'bins': [20, 20]  # Optional: number of bins for x and y (default: [20, 20])
        }
        """
        x = data.get('x', [])
        y = data.get('y', [])
        bins = data.get('bins', kwargs.get('bins', [20, 20]))
        
        # Get colormap
        cmap = kwargs.get('cmap', 'YlOrRd')  # Yellow-Orange-Red by default
        
        # Create 2D histogram
        hist, xedges, yedges, image = ax.hist2d(
            x, y, 
            bins=bins,
            cmap=cmap,
            alpha=0.9,
            edgecolors='none'
        )
        
        # Add colorbar to show density scale
        from matplotlib import pyplot as plt
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
        
        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        # Add grid for better readability (behind the heatmap)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
        
        # Add statistics as text annotation if requested
        if kwargs.get('show_stats', True):
            import numpy as np
            stats_text = f'n = {len(x)}\nμx = {np.mean(x):.1f}\nμy = {np.mean(y):.1f}'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')

    def _create_cohere_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a coherence plot showing the correlation between two signals
        as a function of frequency.
        
        Perfect for: Signal processing, correlation analysis, frequency-domain analysis
        
        Data format:
        {
            'x': [0.1, 0.2, 0.3, ...],      # First signal (time series)
            'y': [0.15, 0.25, 0.32, ...],   # Second signal (time series)
            'Fs': 1000,                      # Sampling frequency (optional, default: 1000)
            'NFFT': 256                      # FFT length (optional, default: 256)
        }
        """
        import numpy as np
        
        x = data.get('x', [])
        y = data.get('y', [])
        Fs = data.get('Fs', kwargs.get('Fs', 1000))  # Sampling frequency
        NFFT = data.get('NFFT', kwargs.get('NFFT', 256))  # FFT length
        
        # Compute coherence
        Cxy, freqs = ax.cohere(
            x, y,
            NFFT=NFFT,
            Fs=Fs,
            noverlap=NFFT // 2,  # 50% overlap
            scale_by_freq=True
        )
        
        # Enhance the plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or 'Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or 'Coherence', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal line at coherence = 0.5 for reference
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                   alpha=0.5, label='Coherence = 0.5')
        
        # Add statistics if requested
        if kwargs.get('show_stats', True):
            mean_coherence = np.mean(Cxy)
            max_coherence = np.max(Cxy)
            max_freq = freqs[np.argmax(Cxy)]
            
            stats_text = (f'Mean: {mean_coherence:.3f}\n'
                         f'Max: {max_coherence:.3f}\n'
                         f'Peak: {max_freq:.1f} Hz')
            
            ax.text(0.98, 0.97, stats_text,
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.8, edgecolor='gray'),
                   fontsize=9, family='monospace')
        
        ax.legend(loc='lower left', fontsize=9)

    def _create_signal_pair_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a dual plot showing two time-domain signals (typically used with coherence data).
        Perfect for: Visualizing signal pairs, comparing waveforms, time-series comparison
        
        Data format:
        {
            'x': [0.1, 0.2, 0.3, ...],      # First signal (time series)
            'y': [0.15, 0.25, 0.32, ...],   # Second signal (time series)
            'Fs': 1000,                      # Sampling frequency (optional, default: 1)
            'labels': ['Signal X', 'Signal Y']  # Optional custom labels
        }
        """
        import numpy as np
        
        x = np.array(data.get('x', []))
        y = np.array(data.get('y', []))
        Fs = data.get('Fs', kwargs.get('Fs', 1))
        labels = data.get('labels', kwargs.get('labels', ['Signal X', 'Signal Y']))
        
        # Create time array
        duration = len(x) / Fs
        t = np.linspace(0, duration, len(x))
        
        # Determine plot style
        plot_style = kwargs.get('plot_style', 'stacked')  # 'stacked' or 'overlaid'
        
        if plot_style == 'overlaid':
            # Plot both signals on the same axes
            ax.plot(t, x, 'b-', linewidth=1.5, alpha=0.7, label=labels[0])
            ax.plot(t, y, 'r-', linewidth=1.5, alpha=0.7, label=labels[1])
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel or 'Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel or 'Amplitude', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Add correlation coefficient
            if len(x) > 1 and len(y) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, family='monospace')
        else:
            # This is a special case - we need to create subplots
            # Since we only have one ax, we'll plot them overlaid with different y-scales
            # But note this in the kwargs handling
            
            ax1 = ax
            ax2 = ax.twinx()
            
            color1 = 'tab:blue'
            color2 = 'tab:red'
            
            ax1.plot(t, x, color=color1, linewidth=1.5, alpha=0.8, label=labels[0])
            ax1.set_xlabel(xlabel or 'Time (seconds)', fontsize=12, fontweight='bold')
            ax1.set_ylabel(labels[0], fontsize=11, fontweight='bold', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(t, y, color=color2, linewidth=1.5, alpha=0.8, label=labels[1])
            ax2.set_ylabel(labels[1], fontsize=11, fontweight='bold', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add statistics
            stats_text = (f'{labels[0]}: μ={np.mean(x):.2f}, σ={np.std(x):.2f}\n'
                         f'{labels[1]}: μ={np.mean(y):.2f}, σ={np.std(y):.2f}')
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8, family='monospace')

    def _create_timeline_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a timeline chart with dates and event text.
        
        Data format:
        {
            'dates': ['2024-01', '2024-03', '2024-06', '2024-09'],
            'events': ['Event 1', 'Event 2', 'Event 3', 'Event 4'],
            'categories': ['Type A', 'Type B', 'Type A', 'Type C']  # Optional
        }
        """
        import matplotlib.dates as mdates
        from datetime import datetime
        
        dates = data.get('dates', [])
        events = data.get('events', [])
        categories = data.get('categories', None)
        
        # Convert dates to datetime objects
        date_objects = []
        for d in dates:
            if isinstance(d, str):
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        date_objects.append(datetime.strptime(d, fmt))
                        break
                    except ValueError:
                        continue
            else:
                date_objects.append(d)
        
        # Assign y-positions based on categories or alternate
        if categories:
            unique_cats = list(set(categories))
            cat_to_y = {cat: i for i, cat in enumerate(unique_cats)}
            y_positions = [cat_to_y[cat] for cat in categories]
            y_labels = unique_cats
        else:
            # Alternate positions for better readability
            y_positions = [i % 3 for i in range(len(dates))]
            y_labels = None
        
        # Get colors
        colors = kwargs.get('colors', None)
        if not colors:
            default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            if categories:
                unique_cats = list(set(categories))
                cat_colors = {cat: default_colors[i % len(default_colors)] 
                            for i, cat in enumerate(unique_cats)}
                colors = [cat_colors[cat] for cat in categories]
            else:
                colors = [default_colors[i % len(default_colors)] for i in range(len(dates))]
        
        # Plot timeline
        ax.scatter(date_objects, y_positions, s=200, c=colors, 
                  alpha=0.8, edgecolors='black', linewidth=2, zorder=3)
        
        # Add event text
        for i, (date, event, y_pos) in enumerate(zip(date_objects, events, y_positions)):
            # Position text above or below point
            text_y = y_pos + 0.3 if i % 2 == 0 else y_pos - 0.3
            va = 'bottom' if i % 2 == 0 else 'top'
            
            ax.text(date, text_y, event, 
                   ha='center', va=va, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           edgecolor='gray', alpha=0.8))
        
        # Draw horizontal line
        if y_labels:
            for y in range(len(y_labels)):
                ax.axhline(y=y, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)
        else:
            ax.axhline(y=1, color='gray', linestyle='-', linewidth=2, alpha=0.5, zorder=1)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Set labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel or 'Date', fontsize=12)
        
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
            ax.set_ylabel(ylabel or 'Category', fontsize=12)
        else:
            ax.set_yticks([])
            ax.set_ylabel('')
        
        # Set limits
        ax.set_ylim(-0.5, max(y_positions) + 1)
        ax.grid(True, alpha=0.3, axis='x')

    def _create_heatmap_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create an annotated heatmap with values displayed in each cell.
        
        Data format:
        {
            'values': [
                [23, 45, 12, 67],  # Row 1 values
                [89, 34, 56, 23],  # Row 2 values
                [45, 78, 90, 12]   # Row 3 values
            ],
            'row_labels': ['Product A', 'Product B', 'Product C'],
            'col_labels': ['Q1', 'Q2', 'Q3', 'Q4'],
            'annotations': None  # Optional: custom text for cells (defaults to values)
        }
        """
        import numpy as np
        
        values = np.array(data.get('values', []))
        row_labels = data.get('row_labels', [f'Row {i+1}' for i in range(len(values))])
        col_labels = data.get('col_labels', [f'Col {i+1}' for i in range(len(values[0]))])
        annotations = data.get('annotations', None)
        
        # Get colormap
        cmap = kwargs.get('cmap', 'YlOrRd')
        
        # Create heatmap
        im = ax.imshow(values, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', rotation=270, labelpad=20, fontsize=11)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate x labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add annotations in each cell
        show_annotations = kwargs.get('show_annotations', True)
        if show_annotations:
            # Use provided annotations or values
            annot_data = annotations if annotations is not None else values
            
            # Determine text color based on value (light text on dark cells, dark on light)
            threshold = (values.max() + values.min()) / 2
            
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    # Choose text color based on cell brightness
                    text_color = 'white' if values[i, j] > threshold else 'black'
                    
                    # Format annotation
                    if isinstance(annot_data[i][j], (int, np.integer)):
                        text = f'{annot_data[i][j]}'
                    else:
                        text = f'{annot_data[i][j]:.1f}'
                    
                    ax.text(j, i, text, 
                           ha="center", va="center", 
                           color=text_color, fontsize=10, fontweight='bold')
        
        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Remove grid
        ax.grid(False)
        
        # Add frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)

    def _create_streamplot_chart(
        self,
        ax,
        data: Dict[str, Any],
        title: str,
        xlabel: str,
        ylabel: str,
        **kwargs
    ):
        """
        Create a streamplot showing vector field flow patterns.
        Used for: fluid dynamics, wind patterns, electromagnetic fields, gradient flows.
        
        Data format:
        {
            'x': [0, 1, 2, 3, 4],           # X-axis grid coordinates
            'y': [0, 1, 2, 3, 4],           # Y-axis grid coordinates
            'u': [[...], [...], ...],       # X-component of velocity (2D array)
            'v': [[...], [...], ...],       # Y-component of velocity (2D array)
            'magnitude': [[...], [...]]     # Optional: flow magnitude for coloring
            'mask': [[bool, ...], ...]      # Optional: mask array (True = hide)
            'start_points': [[x, y], ...]   # Optional: custom starting points
        }
        
        kwargs options:
            density: float or (float, float) - streamline spacing
            linewidth: float or 'variable' - line width
            color: 'velocity', 'uniform', or hex color
            arrowsize: float - arrow size multiplier
            cmap: str - colormap name
            broken_streamlines: bool - allow broken lines (default True)
            start_points: array - manual starting positions
            mask: array - mask region
        """
        import numpy as np
        
        # Check if we have explicit data or need to generate
        if 'u' in data and 'v' in data:
            # User provided explicit flow data
            x = np.array(data.get('x', []))
            y = np.array(data.get('y', []))
            u = np.array(data.get('u', []))
            v = np.array(data.get('v', []))
            magnitude = data.get('magnitude', None)
            mask = data.get('mask', None)
            
            # If x, y are 1D arrays, create meshgrid
            if x.ndim == 1 and y.ndim == 1:
                X, Y = np.meshgrid(x, y)
            else:
                X, Y = x, y
        else:
            # This shouldn't happen in normal use, but provide fallback
            grid_size = data.get('grid_size', 20)
            x = np.linspace(-2, 2, grid_size)
            y = np.linspace(-2, 2, grid_size)
            X, Y = np.meshgrid(x, y)
            u = -Y  # Simple circular flow
            v = X
            magnitude = None
            mask = None
        
        # Calculate magnitude if not provided (for coloring)
        if magnitude is None:
            magnitude = np.sqrt(u**2 + v**2)
        else:
            magnitude = np.array(magnitude)
        
        # Get streamplot parameters
        density = kwargs.get('density', 1.5)
        linewidth = kwargs.get('linewidth', 'variable')
        color_scheme = kwargs.get('color', 'velocity')
        arrowsize = kwargs.get('arrowsize', 1.2)
        cmap = kwargs.get('cmap', 'viridis')
        broken_streamlines = kwargs.get('broken_streamlines', True)
        start_points = kwargs.get('start_points', data.get('start_points', None))
        
        # Apply mask if provided
        if mask is not None:
            mask = np.array(mask)
            u = np.ma.array(u, mask=mask)
            v = np.ma.array(v, mask=mask)
        
        # Determine line width
        if linewidth == 'variable':
            # Normalize linewidth based on magnitude
            lw = 0.5 + 2.5 * magnitude / magnitude.max()
        elif linewidth == 'uniform':
            lw = 1.5
        else:
            lw = linewidth
        
        # Prepare streamplot kwargs
        stream_kwargs = {
            'density': density,
            'arrowsize': arrowsize,
            'arrowstyle': '->',
        }
        
        # Add broken_streamlines parameter if matplotlib supports it
        if not broken_streamlines:
            stream_kwargs['broken_streamlines'] = False
        
        # Add starting points if provided
        if start_points is not None:
            stream_kwargs['start_points'] = np.array(start_points)
        
        # Create streamplot with appropriate coloring
        if color_scheme == 'velocity':
            # Color by velocity magnitude
            strm = ax.streamplot(X, Y, u, v, 
                                color=magnitude, 
                                linewidth=lw,
                                cmap=cmap,
                                norm=plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max()),
                                **stream_kwargs)
            
            # Add colorbar
            cbar = plt.colorbar(strm.lines, ax=ax)
            cbar.set_label('Flow Speed', rotation=270, labelpad=20, fontsize=11)
            
        elif color_scheme == 'uniform':
            # Single color
            strm = ax.streamplot(X, Y, u, v,
                                color='#1f77b4',
                                linewidth=lw if isinstance(lw, (int, float)) else 1.5,
                                **stream_kwargs)
        elif color_scheme == 'field':
            # Color by a different field (use magnitude as default)
            field = data.get('color_field', magnitude)
            strm = ax.streamplot(X, Y, u, v,
                                color=field,
                                linewidth=lw,
                                cmap=cmap,
                                **stream_kwargs)
            cbar = plt.colorbar(strm.lines, ax=ax)
            cbar.set_label('Field Value', rotation=270, labelpad=20, fontsize=11)
        else:
            # Custom color (hex or named)
            strm = ax.streamplot(X, Y, u, v,
                                color=color_scheme,
                                linewidth=lw if isinstance(lw, (int, float)) else 1.5,
                                **stream_kwargs)
        
        # Plot starting points if provided and requested
        if start_points is not None and kwargs.get('show_start_points', False):
            start_points = np.array(start_points)
            ax.scatter(start_points[:, 0], start_points[:, 1], 
                      s=100, c='blue', marker='o', zorder=10,
                      edgecolors='white', linewidths=2)
        
        # Draw masked region as a visual obstacle if mask is provided
        if mask is not None and kwargs.get('show_mask_region', True):
            from matplotlib.patches import Rectangle, Circle
            mask_shape = data.get('mask_shape', 'rectangle')
            mask_color = kwargs.get('mask_color', 'gray')
            
            if mask_shape == 'circle':
                mask_params = data.get('mask_params', {})
                center_x = mask_params.get('x', 0)
                center_y = mask_params.get('y', 0)
                radius = mask_params.get('r', 1)
                circle = Circle((center_x, center_y), radius, 
                               facecolor=mask_color, edgecolor='black', 
                               linewidth=2, alpha=0.7, zorder=5)
                ax.add_patch(circle)
            else:  # rectangle
                mask_params = data.get('mask_params', {})
                x_min = mask_params.get('x_min', -1)
                x_max = mask_params.get('x_max', 1)
                y_min = mask_params.get('y_min', -1)
                y_max = mask_params.get('y_max', 1)
                width = x_max - x_min
                height = y_max - y_min
                rect = Rectangle((x_min, y_min), width, height,
                                facecolor=mask_color, edgecolor='black',
                                linewidth=2, alpha=0.7, zorder=5)
                ax.add_patch(rect)
        
        # Set labels and title
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Set equal aspect ratio for undistorted flow
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

    
    def batch_generate(
        self,
        chart_specs: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """
        Generate multiple charts at once.
        
        Args:
            chart_specs: List of chart specification dictionaries
            
        Returns:
            List of (png_path, json_path) tuples
        """
        results = []
        for spec in chart_specs:
            png_path, json_path = self.generate_chart(**spec)
            results.append((png_path, json_path))
        return results